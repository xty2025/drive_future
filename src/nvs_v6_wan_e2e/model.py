"""NAVSIM v6-wan generator: DrivoR scene perception + Wan2.2 Action DiT.

Three streams keep the v6 directed contract ``GROUP_ALLOW = ((1,0,0),(1,1,1),(1,0,1))``:

* ``eta`` -- DrivoR's scene perception (DINOv3 with prepended scene queries and
  q/v LoRA, see :mod:`nvs_v6_wan_e2e.vision`) plus DrivoR's ``hist_encoding``
  ego token, fused by ``scene_blocks``.
* ``phi`` -- the Future Register, reading ``[S, R]`` per layer and initialised
  from the metric source trajectory, so one scene yields a different future per
  candidate trajectory.
* ``theta`` -- SimWAM's Wan2.2 ActionDiT at ``hidden_dim=256``
  (:mod:`nvs_v6_wan_e2e.wan_dit`), reading ``[S, A, sg(R)]`` in Wan's
  self+cross split.

Trajectories are DrivoR-native metric ``[x, y, heading]`` in metres and radians
with no rescaling.  The Action DiT is a rectified-flow model over them
(:mod:`nvs_v6_wan_e2e.flow`): its ``Head`` emits a velocity, and one forward is
inverted algebraically to a trajectory, so there is no rollout at train or
inference time.  ``sigma = 0`` makes the flow state the clean action (Stage 1);
``sigma = 1`` makes it the anchor prior, which is what
:meth:`DirectedPairDiT.forward_pairs` deploys.

The Register consumes the trajectory as an *initialisation feature*, never as an
attention key: ``GROUP_ALLOW`` still forbids ``R<-A``, and Scene is still never
updated by Action or Register.  Action and Register therefore share the same
flow state as an input while remaining separate streams.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

import torch
from torch import nn

from .data import Observation
from .flow import INFERENCE_SIGMA, ActionFlow
from .vision import (
    DINOV3_MODEL_NAME,
    DRIVOR_OFFICIAL_IMAGE_HW,
    DRIVOR_LORA_RANK,
    DRIVOR_NUM_SCENE_TOKENS,
    DrivoRDinoV3Student,
    freeze_backbone_keep_lora,
)
from .wan_dit import NvsActionDiT
from .wan_init import load_action_dit_backbone

GROUP_ALLOW = ((1, 0, 0), (1, 1, 1), (1, 0, 1))

# DrivoR's observable ego vector: pose3, velocity2, acceleration2, command4.
EGO_OBSERVABLE_DIM = 11
EGO_STATUS_DIM = 7
TRAJECTORY_STATE_DIM = 3


def build_group_mask(ns: int, na: int, nr: int, device: torch.device | None = None) -> torch.Tensor:
    sizes = (ns, na, nr)
    mask = torch.zeros(sum(sizes), sum(sizes), dtype=torch.bool, device=device)
    row_start = 0
    for row, row_size in enumerate(sizes):
        col_start = 0
        for col, col_size in enumerate(sizes):
            if GROUP_ALLOW[row][col]:
                mask[row_start : row_start + row_size, col_start : col_start + col_size] = True
            col_start += col_size
        row_start += row_size
    return mask


@dataclass
class PairOutputs:
    scene_context: torch.Tensor
    endpoint: torch.Tensor
    trajectory: torch.Tensor
    predicted_register: torch.Tensor


@dataclass
class FlowOutputs:
    """One noised training forward: the recovered trajectory plus its flow targets.

    ``pairs`` carries exactly what the inference path returns, so the scorer and
    the oracle consume the same object in both stages; the extra fields are what
    :func:`nvs_v6_wan_e2e.losses.flow_velocity_loss` needs.
    """

    pairs: PairOutputs
    noisy_action: torch.Tensor
    velocity: torch.Tensor
    velocity_target: torch.Tensor
    timestep: torch.Tensor


class SpatialStudent(nn.Module):
    """Checkpoint-free stand-in emitting the DrivoR scene-query token budget."""

    def __init__(self, output_dim: int, num_frames: int = 5, num_scene_tokens: int = DRIVOR_NUM_SCENE_TOKENS) -> None:
        super().__init__()
        self.num_frames = num_frames
        self.num_scene_tokens = num_scene_tokens
        self.patch = nn.Conv2d(3, output_dim, kernel_size=16, stride=16)
        self.pool = nn.AdaptiveAvgPool1d(num_scene_tokens)
        for parameter in self.patch.parameters():
            parameter.requires_grad_(False)
        self.neck = nn.Linear(output_dim, output_dim)

    def train(self, mode: bool = True) -> "SpatialStudent":
        super().train(mode)
        self.patch.eval()
        return self

    def token_layout(self) -> dict[str, int]:
        return {
            "scene_queries_per_frame": self.num_scene_tokens,
            "frames": self.num_frames,
            "output_tokens": self.num_frames * self.num_scene_tokens,
            "position_encoding": "synthetic_conv_patch_no_dinov3",
        }

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 5:
            raise ValueError(f"scene images must be [B,N,3,H,W], got {tuple(images.shape)}")
        batch, frames = images.shape[0], images.shape[1]
        if frames != self.num_frames:
            raise ValueError(f"scene encoder was built for {self.num_frames} frames, got {frames}")
        tokens = self.patch(images.reshape(batch * frames, *images.shape[2:])).flatten(2)
        tokens = self.pool(tokens).transpose(1, 2)
        tokens = self.neck(tokens)
        return tokens.reshape(batch, frames * self.num_scene_tokens, -1)


class SceneEncoder(nn.Module):
    """DrivoR's scene features: per-frame scene queries plus one ego token.

    DrivoR builds ``ego_token = hist_encoding(ego_status)[:, None]`` from a single
    ``Linear(11, tf_d_model)`` -- one token, no attention fusion.  NVS keeps that
    exactly and only adds a learnable per-frame embedding, because DrivoR consumes
    one timestamp across four cameras while NVS consumes five timestamps of one
    camera and still has to tell them apart.
    """

    def __init__(
        self,
        d_model: int,
        frames: int = 5,
        views: int = 1,
        num_scene_tokens: int = DRIVOR_NUM_SCENE_TOKENS,
        dino_checkpoint: str | None = None,
        dino_model_name: str = DINOV3_MODEL_NAME,
        dino_image_hw: Sequence[int] = DRIVOR_OFFICIAL_IMAGE_HW,
        dino_lora_rank: int = DRIVOR_LORA_RANK,
        dino_grad_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.frames = frames
        self.views = views
        self.num_scene_tokens = num_scene_tokens
        self.frame_axis = frames * views
        self.student = (
            DrivoRDinoV3Student(
                checkpoint=dino_checkpoint,
                model_name=dino_model_name,
                image_hw=tuple(dino_image_hw),
                num_scene_tokens=num_scene_tokens,
                num_frames=self.frame_axis,
                lora_rank=dino_lora_rank,
                output_dim=d_model,
                grad_checkpointing=dino_grad_checkpointing,
            )
            if dino_checkpoint
            else SpatialStudent(d_model, num_frames=self.frame_axis, num_scene_tokens=num_scene_tokens)
        )
        self.time_embedding = nn.Parameter(torch.zeros(self.frame_axis, d_model))
        self.hist_encoding = nn.Linear(EGO_OBSERVABLE_DIM, d_model)

    @property
    def visual_tokens(self) -> int:
        return self.frame_axis * self.num_scene_tokens

    @property
    def scene_tokens(self) -> int:
        return self.visual_tokens + 1

    def token_layout(self) -> dict[str, object]:
        layout = dict(self.student.token_layout())
        layout.update(
            {
                "ego_tokens": 1,
                "ego_observable_dim": EGO_OBSERVABLE_DIM,
                "scene_tokens": self.scene_tokens,
                "frames": self.frames,
                "views": self.views,
            }
        )
        return layout

    def encode_clean_ego(self, observation: Observation) -> torch.Tensor:
        """DrivoR's ``ego_status``: ``pose3, velocity2, acceleration2, command4``."""

        observation.validate()
        current = observation.ego_history[:, -1]
        ego_status = torch.cat((current, observation.route_command), dim=-1)
        if ego_status.shape[-1] != EGO_OBSERVABLE_DIM:
            raise ValueError(f"ego observable must be {EGO_OBSERVABLE_DIM}-wide")
        return self.hist_encoding(ego_status)

    def forward(self, observation: Observation, clean_ego: torch.Tensor | None = None) -> torch.Tensor:
        observation.validate()
        batch, frames, views = observation.rgb_history.shape[:3]
        if (frames, views) != (self.frames, self.views):
            raise ValueError(f"observation must carry {self.frames}x{self.views} frames, got {frames}x{views}")
        images = observation.rgb_history.reshape(batch, frames * views, *observation.rgb_history.shape[3:])
        # Frame-major, matching the student's `(b n) t c -> b (n t) c` output: each
        # frame's embedding is shared by its `num_scene_tokens` queries.
        time_embedding = self.time_embedding.unsqueeze(0).repeat_interleave(self.num_scene_tokens, dim=1)
        visual = self.student(images) + time_embedding
        if clean_ego is None:
            clean_ego = self.encode_clean_ego(observation)
        if clean_ego.shape != (batch, self.hist_encoding.out_features):
            raise ValueError("clean ego encoding must be [B,d_model]")
        return torch.cat((visual, clean_ego.unsqueeze(1)), dim=1)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, ffn_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, ffn_dim), nn.GELU(), nn.Linear(ffn_dim, d_model))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class SceneBlock(nn.Module):
    def __init__(self, d_model: int, heads: int, ffn_dim: int) -> None:
        super().__init__()
        self.norm_attn = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, ffn_dim)

    def forward(self, scene: torch.Tensor) -> torch.Tensor:
        normalized = self.norm_attn(scene)
        scene = scene + self.attn(normalized, normalized, normalized, need_weights=False)[0]
        return scene + self.ffn(self.norm_ffn(scene))


class RegisterBlock(nn.Module):
    def __init__(self, d_model: int, heads: int, ffn_dim: int) -> None:
        super().__init__()
        self.norm_attn = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, ffn_dim)

    def forward(self, register: torch.Tensor, scene: torch.Tensor) -> torch.Tensor:
        b, m, nr, d = register.shape
        scene = scene.unsqueeze(1).expand(-1, m, -1, -1).reshape(b * m, scene.shape[1], d)
        register = register.reshape(b * m, nr, d)
        query = self.norm_attn(register)
        key_value = torch.cat((scene, register), dim=1)
        register = register + self.attn(query, key_value, key_value, need_weights=False)[0]
        register = register + self.ffn(self.norm_ffn(register))
        return register.reshape(b, m, nr, d)


class DirectedPairDiT(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        heads: int = 4,
        ffn_dim: int = 1024,
        depth: int = 8,
        action_tokens: int = 8,
        register_tokens: int = 4,
        frames: int = 5,
        views: int = 1,
        num_scene_tokens: int = DRIVOR_NUM_SCENE_TOKENS,
        dino_checkpoint: str | None = None,
        dino_model_name: str = DINOV3_MODEL_NAME,
        dino_image_hw: Sequence[int] = DRIVOR_OFFICIAL_IMAGE_HW,
        dino_lora_rank: int = DRIVOR_LORA_RANK,
        dino_grad_checkpointing: bool = False,
        wan_backbone_path: str | None = None,
        attn_head_dim: int = 64,
        freq_dim: int = 256,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if d_model % heads:
            raise ValueError("d_model must be divisible by heads")
        if freq_dim != d_model:
            raise ValueError("the Wan time_embedding input is the NVS condition vector at d_model")
        self.d_model = d_model
        self.action_tokens = action_tokens
        self.register_tokens = register_tokens
        self.depth = depth
        self.eta = SceneEncoder(
            d_model,
            frames=frames,
            views=views,
            num_scene_tokens=num_scene_tokens,
            dino_checkpoint=dino_checkpoint,
            dino_model_name=dino_model_name,
            dino_image_hw=dino_image_hw,
            dino_lora_rank=dino_lora_rank,
            dino_grad_checkpointing=dino_grad_checkpointing,
        )
        self.scene_tokens = self.eta.scene_tokens
        self.phi = nn.ModuleDict(
            {
                "mode_projection": nn.Linear(2, d_model),
                # The metric trajectory is what makes one candidate's future
                # differ from another's inside the same scene: 8 poses x [x,y,heading].
                "trajectory_projection": nn.Linear(action_tokens * TRAJECTORY_STATE_DIM, d_model),
                "blocks": nn.ModuleList([RegisterBlock(d_model, heads, ffn_dim) for _ in range(depth)]),
            }
        )
        self.phi_time_queries = nn.Parameter(torch.randn(register_tokens, d_model) * 0.02)
        # The trajectory arrives in metres, so its coordinates reach ~60 while the
        # compressed targets sit near 1.  A default-initialised projection therefore
        # starts the register ~5x larger than anything it has to predict, and the
        # first steps only undo that.  Zeroing it makes the trajectory a residual the
        # branch grows into, which leaves the anchor-only initialisation exactly as
        # it was and still passes gradient: the input is non-zero, so both weight and
        # bias receive one from step 0.
        nn.init.zeros_(self.phi["trajectory_projection"].weight)
        nn.init.zeros_(self.phi["trajectory_projection"].bias)
        self.theta = NvsActionDiT(
            hidden_dim=d_model,
            action_dim=TRAJECTORY_STATE_DIM,
            ffn_dim=ffn_dim,
            text_dim=d_model,
            freq_dim=freq_dim,
            eps=eps,
            num_heads=heads,
            attn_head_dim=attn_head_dim,
            num_layers=depth,
            action_tokens=action_tokens,
            mode_dim=2,
            ego_dim=d_model,
        )
        self.wan_backbone_path = None
        self.wan_backbone_info: dict[str, object] = {"loaded": False}
        if wan_backbone_path:
            self.wan_backbone_info = load_action_dit_backbone(self.theta, wan_backbone_path)
            self.wan_backbone_info["loaded"] = True
            self.wan_backbone_path = str(wan_backbone_path)
        self.scene_blocks = nn.ModuleList([SceneBlock(d_model, heads, ffn_dim) for _ in range(depth)])
        self.register_readout = nn.Identity()
        # Stateless schedule, not an nn.Module on purpose: it must contribute no
        # Parameter that parameter_groups()/the Wan payload key check would see.
        self.flow = ActionFlow()

    def token_layout(self) -> dict[str, object]:
        return self.eta.token_layout()

    def _initialize_register(self, modes: torch.Tensor | None, source: torch.Tensor) -> torch.Tensor:
        """``R0 = time_queries + projected_trajectory [+ projected_mode_code]``.

        ``modes=None`` drops the mode term.  Stage 1 trains one register per GT
        trajectory, so a mode code that is constant per row index would let the
        network answer "which GT is this" from the row alone and never read the
        trajectory -- the exact failure the anchor-only Stage 1 had.
        """

        if source.ndim != 4 or source.shape[-2:] != (self.action_tokens, TRAJECTORY_STATE_DIM):
            raise ValueError(f"source must be [B,M,{self.action_tokens},{TRAJECTORY_STATE_DIM}] in metres")
        b, m = source.shape[0], source.shape[1]
        if modes is not None and modes.shape[:2] != (b, m):
            raise ValueError("modes must align with source on [B,M]")
        query = self.phi_time_queries.view(1, 1, self.register_tokens, self.d_model)
        register = query.expand(b, m, -1, -1) + self.phi["trajectory_projection"](source.flatten(-2)).unsqueeze(2)
        if modes is not None:
            register = register + self.phi["mode_projection"](modes).unsqueeze(2)
        return register

    def _directed_pass(
        self,
        scene: torch.Tensor,
        action: torch.Tensor | None,
        register: torch.Tensor,
        condition: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        b, m, nr, d = register.shape
        ns = scene.shape[1]
        raw_contexts: list[torch.Tensor] = []
        for scene_block, register_block in zip(self.scene_blocks, self.phi["blocks"]):
            scene = scene_block(scene)
            register = register_block(register, scene)
            if action is not None:
                scene_flat = scene.unsqueeze(1).expand(b, m, ns, d).reshape(b * m, ns, d)
                register_flat = self.register_readout(register).detach().reshape(b * m, nr, d)
                raw_contexts.append(torch.cat((scene_flat, register_flat), dim=1))
        if action is None:
            return scene, register, None
        # Wan embeds the cross-attention context once in `pre_dit`; NVS's Action
        # stream reads the Register at every layer, so each layer gets its own.
        contexts = [self.theta.text_embedding(context) for context in raw_contexts]
        flat_condition = condition.reshape(b * m, d)
        velocity = self.theta(action.reshape(b * m, self.action_tokens, d), flat_condition, contexts)
        return scene, register, velocity.reshape(b, m, self.action_tokens, -1)

    def forward_register_only(self, observation: Observation, source: torch.Tensor, modes: torch.Tensor | None = None) -> torch.Tensor:
        clean_ego = self.eta.encode_clean_ego(observation)
        scene = self.eta(observation, clean_ego=clean_ego)
        register = self._initialize_register(modes, source)
        _, register, _ = self._directed_pass(scene, None, register)
        return self.register_readout(register)

    def _flow_pass(
        self,
        observation: Observation,
        noisy_action: torch.Tensor,
        modes: torch.Tensor,
        timestep: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Shared trunk: encode ``x_sigma`` and return ``(scene, register, velocity)``."""
        clean_ego = self.eta.encode_clean_ego(observation)
        scene = self.eta(observation, clean_ego=clean_ego)
        action = self.theta.encode_action(noisy_action, modes, clean_ego)
        condition = self.theta.encode_condition(modes, clean_ego, timestep)
        # The Register is conditioned on the same flow state the Action stream
        # sees, so Stage 1 (sigma=0) reads the clean GT trajectory and Stage 2
        # reads the noised one -- never a clean answer the DiT has to rediscover.
        register = self._initialize_register(modes, noisy_action)
        scene, register, velocity = self._directed_pass(scene, action, register, condition)
        return scene, self.register_readout(register), velocity

    def forward_pairs(self, observation: Observation, source: torch.Tensor, modes: torch.Tensor) -> PairOutputs:
        """Deployed inference: one forward at ``sigma = 1``, no rollout.

        ``source`` is the clean anchor grid, which at ``sigma = 1`` *is* the flow
        state, so this path never needs a GT trajectory.  With the near-zero
        ``Head`` initialization the recovered endpoint equals ``source``, i.e. the
        anchor grid's own oracle-best PDMS is the starting point of training.
        """
        self._check_source(source)
        timestep = torch.full(
            (source.shape[0],),
            INFERENCE_SIGMA * self.flow.num_train_timesteps,
            device=source.device,
            dtype=source.dtype,
        )
        scene, register, velocity = self._flow_pass(observation, source, modes, timestep)
        endpoint = self.flow.recover(source, velocity, timestep)
        return PairOutputs(scene, endpoint, endpoint, register)

    def forward_flow(
        self,
        observation: Observation,
        source: torch.Tensor,
        modes: torch.Tensor,
        data: torch.Tensor,
        timestep: torch.Tensor,
    ) -> FlowOutputs:
        """Training forward at a sampled ``timestep``: noise the action, denoise once.

        ``source`` is the anchor prior (the flow's ``sigma = 1`` endpoint) and
        ``data`` is the per-candidate target trajectory (its ``sigma = 0``
        endpoint).  The two are interpolated by :class:`ActionFlow`, the DiT
        predicts the velocity, and the trajectory is recovered algebraically in the
        same forward -- so training never rolls out ``step1 -> step2 -> ...``.
        """
        self._check_source(source)
        if data.shape != source.shape:
            raise ValueError(f"flow data must be {tuple(source.shape)}, got {tuple(data.shape)}")
        if tuple(timestep.shape) != (source.shape[0],):
            raise ValueError(f"timestep must be [B], got {tuple(timestep.shape)}")
        noisy_action, velocity_target = self.flow.noisy_action(source, data, timestep)
        scene, register, velocity = self._flow_pass(observation, noisy_action, modes, timestep)
        endpoint = self.flow.recover(noisy_action, velocity, timestep)
        return FlowOutputs(
            PairOutputs(scene, endpoint, endpoint, register),
            noisy_action,
            velocity,
            velocity_target,
            timestep,
        )

    def _check_source(self, source: torch.Tensor) -> None:
        if source.ndim != 4 or source.shape[-1] != TRAJECTORY_STATE_DIM:
            raise ValueError(f"source must be [B,M,{self.action_tokens},{TRAJECTORY_STATE_DIM}] in metres")

    def named_trainable_parameters(self) -> Iterator[tuple[str, nn.Parameter]]:
        return ((name, p) for name, p in self.named_parameters() if p.requires_grad)


def parameter_groups(model: DirectedPairDiT) -> dict[str, list[nn.Parameter]]:
    groups = {"eta": [], "theta": [], "phi": []}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("eta.") or name.startswith("scene_blocks."):
            groups["eta"].append(parameter)
        elif name.startswith("theta."):
            groups["theta"].append(parameter)
        elif name.startswith("phi.") or name == "phi_time_queries":
            groups["phi"].append(parameter)
        else:
            raise ValueError(f"parameter `{name}` is not assigned to an eta/theta/phi group")
    return groups


__all__ = [
    "EGO_OBSERVABLE_DIM",
    "EGO_STATUS_DIM",
    "GROUP_ALLOW",
    "TRAJECTORY_STATE_DIM",
    "DirectedPairDiT",
    "FlowOutputs",
    "PairOutputs",
    "RegisterBlock",
    "SceneBlock",
    "SceneEncoder",
    "SpatialStudent",
    "build_group_mask",
    "freeze_backbone_keep_lora",
    "parameter_groups",
]
