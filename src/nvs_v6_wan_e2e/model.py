"""NAVSIM v6-wan generator: DrivoR scene perception + Wan2.2 Action DiT.

Two streams, and the future representation is an *auxiliary output* of the second
one -- never an input to it:

* ``eta`` -- DrivoR's scene perception (DINOv3 with prepended scene queries and
  q/v LoRA, see :mod:`nvs_v6_wan_e2e.vision`) plus DrivoR's ``hist_encoding``
  ego token, refined by ``scene_blocks`` into the 81 Scene tokens.
* ``theta`` -- SimWAM's Wan2.2 ActionDiT at ``hidden_dim=256``
  (:mod:`nvs_v6_wan_e2e.wan_dit`) over the single 121-token sequence
  ``[scene(81) | action(8) | register(32)]``.  Wan's head reads the eight action
  tokens as the rectified-flow velocity; ``future_head`` reads the thirty-two
  register tokens -- four DINOv3 prefix-query readouts per future frame -- as the
  predicted future.

Trajectories are DrivoR-native metric ``[x, y, heading]`` in metres and radians
with no rescaling.  The Action DiT is a rectified-flow model over them
(:mod:`nvs_v6_wan_e2e.flow`): its ``Head`` emits a velocity, and one forward is
inverted algebraically to a trajectory, so there is no rollout at train or
inference time.  ``sigma = 0`` makes the flow state the clean action (Stage 1);
``sigma = 1`` makes it the anchor prior, which is what
:meth:`DirectedPairDiT.forward_pairs` deploys.

Direction of information flow
-----------------------------
The three segments sit in one sequence under **one mask**, identical for every
stage and every entry point (see
:attr:`nvs_v6_wan_e2e.wan_dit.NvsActionDiT.SEGMENT_VISIBILITY`):

=================  ======  ======  ========
query \\ key        Scene   Action  Register
=================  ======  ======  ========
Scene              yes     no      no
Action             yes     yes     no
Register           yes     yes     yes
=================  ======  ======  ========

**Action never reads Register.**  That is a structural invariant rather than a
training choice: the predicted future cannot reach the velocity head through the
shared trunk, and this module deliberately provides no channel around it either.
An earlier revision had one -- ``future_fuse``, which cross-attended an EMA copy of
``theta``'s register into the action tokens during Stage 2 -- and it was removed
precisely because it *was* a ``Register -> Action`` bypass of the mask the contract
exists to enforce.  It had a second, independent defect: the register segment
receives no mode code (:meth:`nvs_v6_wan_e2e.wan_dit.NvsActionDiT.encode_future`),
so its output was identical across all candidates of a scene and the fusion could
not have ranked them.

What is left is a pure auxiliary readout.  The register is supervised by the cached
Teacher target and its gradient flows into the block stack the action stream shares,
so predicting the future still *shapes* the generator -- V-JEPA's predictor shaping
its encoder -- without conditioning a single emitted waypoint.  That is a weaker
coupling than future-conditioned planning, and it is the honest description of what
this model does.

``Action -> Register`` staying open is what keeps the register non-trivial: since
``encode_future`` supplies neither trajectory nor mode, closing it would leave the
register a function of ``(Scene, ego)`` alone and therefore **bitwise identical
across the K Multi-GT rows of one scene**, pinning ``paired_future_l1`` at the
``scene_mean_target_l1`` floor and forcing ``gain_over_trajectory_blind`` to exactly
zero on every step.

Scene reads only itself, so the ActionDiT never writes to the tokens DrivoR's
scorer takes as K/V; ``PairOutputs.scene_context`` is exactly eta's refined Scene.
No GT ever enters the sequence -- the register slots start as ``future_queries``
plus the RMS-clamped ego encoding, and the cached Teacher target is only ever
compared against ``future_head``'s output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

import torch
from torch import nn

from .config import EXPECTED_REGISTER_TARGET_DIM, EXPECTED_REGISTER_TOKENS
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

# DrivoR's observable ego vector: pose3, velocity2, acceleration2, command4.
EGO_OBSERVABLE_DIM = 11
EGO_STATUS_DIM = 7
TRAJECTORY_STATE_DIM = 3


@dataclass
class PairOutputs:
    scene_context: torch.Tensor
    endpoint: torch.Tensor
    trajectory: torch.Tensor
    #: ``None`` when the model was built with ``predict_future=False``, i.e. when
    #: no Teacher target cache exists and the future tokens are left out of the
    #: DiT sequence altogether.
    predicted_register: torch.Tensor | None


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
        dino_vision_bf16: bool = False,
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
                vision_bf16=dino_vision_bf16,
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


class DirectedPairDiT(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        heads: int = 4,
        ffn_dim: int = 1024,
        depth: int = 8,
        action_tokens: int = 8,
        register_tokens: int = EXPECTED_REGISTER_TOKENS,
        #: The future head's output width = the Teacher's own feature width, not the
        #: ``d_model`` above; the two were one parameter before the query readout.
        register_target_dim: int = EXPECTED_REGISTER_TARGET_DIM,
        frames: int = 5,
        views: int = 1,
        num_scene_tokens: int = DRIVOR_NUM_SCENE_TOKENS,
        dino_checkpoint: str | None = None,
        dino_model_name: str = DINOV3_MODEL_NAME,
        dino_image_hw: Sequence[int] = DRIVOR_OFFICIAL_IMAGE_HW,
        dino_lora_rank: int = DRIVOR_LORA_RANK,
        dino_grad_checkpointing: bool = False,
        dino_vision_bf16: bool = False,
        wan_backbone_path: str | None = None,
        attn_head_dim: int = 64,
        freq_dim: int = 256,
        eps: float = 1e-6,
        predict_future: bool = True,
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
        #: ``False`` drops the future tokens from the DiT sequence entirely, which
        #: is what a joint run does when no Teacher target cache exists.  The
        #: parameters stay allocated so a Stage 1 checkpoint still loads; callers
        #: freeze them, and :meth:`_flow_pass` then returns ``register=None``.
        self.predict_future = predict_future
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
            dino_vision_bf16=dino_vision_bf16,
        )
        self.scene_tokens = self.eta.scene_tokens
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
            future_tokens=register_tokens,
            target_dim=register_target_dim,
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
        # Stateless schedule, not an nn.Module on purpose: it must contribute no
        # Parameter that parameter_groups()/the Wan payload key check would see.
        self.flow = ActionFlow()

    def token_layout(self) -> dict[str, object]:
        return self.eta.token_layout()

    def _refine_scene(self, scene: torch.Tensor) -> torch.Tensor:
        for block in self.scene_blocks:
            scene = block(scene)
        return scene

    def forward_register_only(self, observation: Observation, source: torch.Tensor, modes: torch.Tensor) -> torch.Tensor:
        """Stage 1: the clean action at ``sigma = 0``, read back through the future head.

        ``modes`` must be a *zero* code, not ``None``: ``encode_action`` and
        ``encode_condition`` require ``[B,M,2]``, and a code that is constant per
        row index would let the network answer "which GT is this" from the row
        alone and never read the trajectory -- the exact failure the anchor-only
        Stage 1 had.  Zeros carry no row information, so the shortcut is
        structurally unavailable and both stages share one code path.
        """
        if not self.predict_future:
            raise ValueError("forward_register_only needs a model built with predict_future=True")
        timestep = torch.zeros(source.shape[0], device=source.device, dtype=source.dtype)
        _, register, _ = self._flow_pass(observation, source, modes, timestep)
        assert register is not None
        return register

    def _flow_pass(
        self,
        observation: Observation,
        noisy_action: torch.Tensor,
        modes: torch.Tensor,
        timestep: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Shared trunk: run the DiT once and return ``(scene, register, velocity)``.

        One call, one mask, for all three public entry points -- there is no stage
        argument to get wrong, because the information flow no longer depends on
        which objective is being computed.  ``register`` is ``None`` when
        ``predict_future`` is off.

        The register reads the action segment beside it and the action cannot read
        back, so ``(Scene, Action) -> Future`` is one wiring for Stage 1, Stage 2 and
        the deployed path alike.  Same wiring is not same operating point: Stage 1
        runs ``timestep = 0`` on the clean GT trajectory while Stage 2 runs a sampled
        ``sigma`` on the noised one, and ``timestep`` reaches the register through the
        adaLN modulation of every block -- measured on the production geometry, a
        condition perturbation of its own magnitude moves the register output by 44%
        of ``|register|``.  So the two stages' ``loss_future`` come from the same
        function evaluated at different inputs and their absolute values are not
        directly comparable; a Stage-2 rise is not evidence Stage 1 was wasted.

        The future therefore never conditions the velocity: it is an auxiliary head
        whose gradient shapes the block stack the action stream shares and nothing
        more.

        The returned ``scene`` is eta's refined Scene, untouched by the DiT: Scene
        reads only itself.  That is what keeps the scorer's K/V exactly DrivoR's.
        """
        clean_ego = self.eta.encode_clean_ego(observation)
        scene = self._refine_scene(self.eta(observation, clean_ego=clean_ego))
        b, m = noisy_action.shape[:2]
        action = self.theta.encode_action(noisy_action, modes, clean_ego)
        future = self.theta.encode_future(action, clean_ego) if self.predict_future else None
        condition = self.theta.encode_condition(modes, clean_ego, timestep)
        ns, d = scene.shape[1], self.d_model
        context = scene.unsqueeze(1).expand(b, m, ns, d).reshape(b * m, ns, d)
        velocity, register = self.theta(
            action.reshape(b * m, self.action_tokens, d),
            None if future is None else future.reshape(b * m, self.register_tokens, d),
            condition.reshape(b * m, d),
            context,
        )
        return scene, None if register is None else register.reshape(b, m, self.register_tokens, -1), velocity.reshape(b, m, self.action_tokens, -1)

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
    groups = {"eta": [], "theta": []}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("eta.") or name.startswith("scene_blocks."):
            groups["eta"].append(parameter)
        elif name.startswith("theta."):
            groups["theta"].append(parameter)
        else:
            raise ValueError(f"parameter `{name}` is not assigned to an eta/theta group")
    return groups


__all__ = [
    "EGO_OBSERVABLE_DIM",
    "EGO_STATUS_DIM",
    "TRAJECTORY_STATE_DIM",
    "DirectedPairDiT",
    "FlowOutputs",
    "PairOutputs",
    "SceneBlock",
    "SceneEncoder",
    "SpatialStudent",
    "freeze_backbone_keep_lora",
    "parameter_groups",
]
