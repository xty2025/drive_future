"""NAVSIM v6-wan-e2e: DrivoR scene perception + Wan2.2 Action DiT generator.

The Scene stream is DrivoR's exact recipe on DINOv3: ``num_scene_tokens``
learnable queries are prepended to every frame's patch tokens, the backbone
keeps its native RoPE position encoding (no interpolation), and only q/v LoRA
adapters are trainable.  The Action stream is SimWAM's Wan2.2 ``ActionDiT`` at
``hidden_dim=256``, initialized from the released Wan2.2-TI2V-5B weights with
SimWAM's own last-dimension interpolation.  Trajectories stay DrivoR-native
metric ``[x, y, heading]`` -- nothing is rescaled.

This remains the DrivoR-exact end-to-end scorer variant: the Scene K/V fed to
``ScratchDrivoRScorer`` is NOT detached, so the PDM score loss also shapes the
Scene representation (eta), exactly like DrivoR."""

from .data import Observation, PairTargets, drivor_observable_ego_feature, drivor_route_command
from .model import (
    EGO_OBSERVABLE_DIM,
    GROUP_ALLOW,
    TRAJECTORY_STATE_DIM,
    DirectedPairDiT,
    PairOutputs,
    build_group_mask,
    parameter_groups,
)
from .modes import ImmutableModeBuffer, StructuredSourceInitializer
from .scorer import PairedFutureScorer, ScratchDrivoRScorer
from .vision import (
    DRIVOR_IMAGE_HW,
    DRIVOR_LORA_RANK,
    DRIVOR_NUM_SCENE_TOKENS,
    DRIVOR_OFFICIAL_IMAGE_HW,
    DrivoRDinoV3Student,
)
from .wan_dit import NvsActionDiT

__all__ = [
    "DRIVOR_IMAGE_HW",
    "DRIVOR_LORA_RANK",
    "DRIVOR_NUM_SCENE_TOKENS",
    "DRIVOR_OFFICIAL_IMAGE_HW",
    "EGO_OBSERVABLE_DIM",
    "GROUP_ALLOW",
    "TRAJECTORY_STATE_DIM",
    "DirectedPairDiT",
    "DrivoRDinoV3Student",
    "ImmutableModeBuffer",
    "NvsActionDiT",
    "Observation",
    "PairOutputs",
    "PairTargets",
    "PairedFutureScorer",
    "ScratchDrivoRScorer",
    "StructuredSourceInitializer",
    "build_group_mask",
    "drivor_observable_ego_feature",
    "drivor_route_command",
    "parameter_groups",
]
