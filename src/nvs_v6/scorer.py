"""Scratch DrivoR scorer driven exclusively by NVS DINOv3 Scene tokens.

The decoder and six score heads are instantiated from the official DrivoR source
modules, but every scorer parameter is randomly initialized.  No DrivoR
checkpoint, native DINOv2 feature tower, or cached native ego token is used.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from .oracle import OracleLabels
from .selection import select_best_candidate


HEAD_NAMES = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "driving_direction_compliance",
    "time_to_collision_within_bound",
    "ego_progress",
    "comfort",
)


def _drivor_config(d_model: int, ffn_dim: int, heads: int) -> SimpleNamespace:
    """Minimal config consumed by DrivoR's official decoder and scorer heads."""

    return SimpleNamespace(
        b2d=False,
        proposal_num=64,
        num_poses=8,
        tf_d_model=d_model,
        tf_d_ffn=ffn_dim,
        refiner_num_heads=heads,
        refiner_ls_values=0.0,
        double_score=False,
        agent_pred=False,
        area_pred=False,
        bev_map=False,
        bev_agent=False,
        one_token_per_traj=True,
    )


class ScratchDrivoRScorer(nn.Module):
    """Official DrivoR scorer architecture, scratch initialized on NVS Scene K/V.

    The trajectory projection is exactly ``Linear(8*3,1024)-ReLU-
    Linear(1024,256)`` at production defaults.  Candidate self-attention and
    cross-attention are the official ``TransformerDecoderScorer``.  Cross K/V
    is the detached ``DirectedPairDiT.forward_pairs().scene_context``; the mean
    of its final two ego-fused tokens is added before the six official heads.
    Future Register is intentionally not a scorer input under the v6 contract.
    """

    initialization = "scratch_random"
    feature_source = "nvs_dinov3_scene_context"
    checkpoint_loaded = False
    native_feature_cache = False
    strict_drivor_features = False

    def __init__(
        self,
        d_model: int = 256,
        heads: int = 1,
        ffn_dim: int = 1024,
        decoder_layers: int = 4,
        scene_tokens: int = 66,
        future_layers: int | None = None,
    ) -> None:
        super().__init__()
        del future_layers  # compatibility only; Future Register is not consumed
        # Expected NVS Scene K/V length.  66 is the compressed-query layout;
        # 1007 is the all-native layout (5x201 DINOv3 tokens + 2 ego-fused).
        # The scorer is length-agnostic except that the final two tokens are
        # the ego-fused pair pooled into ``ego_token``.
        self.scene_tokens = scene_tokens
        try:
            from navsim.agents.drivoR.layers.losses.drivor_loss import DrivoRLoss
            from navsim.agents.drivoR.score_module.scorer import Scorer
            from navsim.agents.drivoR.transformer_decoder import TransformerDecoderScorer
        except ImportError as error:
            raise ImportError(
                "DrivoR source modules are required; add vendor/DrivoR and "
                "nuplan-devkit to PYTHONPATH"
            ) from error
        config = _drivor_config(d_model, ffn_dim, heads)
        self.config = config
        self.trajectory_projection = nn.Sequential(
            nn.Linear(config.num_poses * 3, config.tf_d_ffn),
            nn.ReLU(),
            nn.Linear(config.tf_d_ffn, config.tf_d_model),
        )
        self.scorer_attention = TransformerDecoderScorer(
            num_layers=decoder_layers,
            d_model=d_model,
            proj_drop=0.1,
            drop_path=0.2,
            config=config,
        )
        self.scorer = Scorer(config)
        self.loss_fn = DrivoRLoss()
        self.feature_audit = {
            "initialization": self.initialization,
            "checkpoint_loaded": self.checkpoint_loaded,
            "feature_source": self.feature_source,
            "native_dinov2_scene_cache": False,
            "native_ego_cache": False,
            "scene_kv_detached": True,
            "ego_pool": "mean(scene_context[:, -2:, :])",
            "predicted_register_input": False,
        }

    def forward(
        self,
        scene: torch.Tensor,
        trajectory: torch.Tensor,
        predicted_register: torch.Tensor | None = None,
        detach_inputs: bool = True,
    ) -> torch.Tensor:
        del predicted_register
        if trajectory.ndim != 4 or trajectory.shape[-2:] != (8, 3):
            raise ValueError("DrivoR scorer expects trajectory [B,M,8,3]")
        if scene.ndim != 3:
            raise ValueError("NVS scene_context must be [B,S,D]")
        batch, candidates = trajectory.shape[:2]
        if scene.shape[0] != batch:
            raise ValueError("scene_context and trajectory batch axes must align")
        if scene.shape[1] != self.scene_tokens:
            raise ValueError(
                f"scratch scorer requires NVS Scene [B,{self.scene_tokens},D], got {tuple(scene.shape)}"
            )
        if scene.shape[-1] != self.config.tf_d_model:
            raise ValueError("scene_context width must equal scorer d_model")
        # The score objective is psi-only.  Keep detachment unconditional even
        # for callers that retain the historical detach_inputs argument.
        del detach_inputs
        scene_kv = scene.detach()
        proposals = trajectory.detach()
        embedded = self.trajectory_projection(proposals.reshape(batch, candidates, 8 * 3))
        ego_token = scene_kv[:, -2:].mean(dim=1, keepdim=True)
        score_features = self.scorer_attention(embedded, scene_kv) + ego_token
        pred_logit, *_ = self.scorer(proposals, score_features)
        return torch.stack([pred_logit[name] for name in HEAD_NAMES], dim=-1)

    def source_loss(self, logits: torch.Tensor, labels: OracleLabels) -> torch.Tensor:
        """Call the official ``DrivoRLoss.score_loss`` and return its utility loss."""

        if labels.failure_diagnostics or not labels.final_valid.all():
            raise RuntimeError("do not train scratch scorer on failed oracle evaluations")
        pred_logit = {name: logits[..., index] for index, name in enumerate(HEAD_NAMES)}
        # The source maps NC/DDC classes in-place; use an independent float64
        # tensor so OracleLabels remains immutable to the rest of the pipeline.
        target_scores = labels.raw7.to(device=logits.device, dtype=torch.float64).clone()
        return self.loss_fn.score_loss(
            pred_logit,
            None,
            None,
            None,
            target_scores,
            None,
            None,
            None,
            None,
        )[1]

    def select(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the official DrivoR utility and argmax selection formula."""

        return select_best_candidate(logits)


# Compatibility aliases now resolve to the production scratch scorer.  The
# explicit native/pretrained comparison tool remains NativeDrivoRScorer.
DrivoRSourceScorer = ScratchDrivoRScorer
PairedFutureScorer = ScratchDrivoRScorer


def scorer_parameter_ids(scorer: ScratchDrivoRScorer) -> set[int]:
    return {id(parameter) for parameter in scorer.parameters() if parameter.requires_grad}
