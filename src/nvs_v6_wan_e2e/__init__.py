"""NAVSIM v6-wan-e2e: Multi-GT / Future Register generator whose Action DiT
blocks follow the Wan2.2 DiTBlock design (randomly initialized).  Directed S/A/R
reads and the one-pass residual-velocity endpoint output are unchanged from v6.

This is the DrivoR-exact end-to-end scorer variant of ``nvs_v6_wan``: the Scene
K/V fed to ``ScratchDrivoRScorer`` is NOT detached, so the PDM score loss also
shapes the Scene representation (eta), exactly like DrivoR.  Proposals stay
detached; no register is fed; token counts, model, selection and optimizer are
identical to ``nvs_v6_wan``."""

from .data import Observation, PairTargets, decode_trajectory, encode_trajectory
from .model import GROUP_ALLOW, DirectedPairDiT, PairOutputs, build_group_mask
from .modes import ImmutableModeBuffer, StructuredSourceInitializer
from .scorer import PairedFutureScorer

__all__ = [
    "GROUP_ALLOW",
    "DirectedPairDiT",
    "ImmutableModeBuffer",
    "Observation",
    "PairOutputs",
    "PairTargets",
    "PairedFutureScorer",
    "StructuredSourceInitializer",
    "build_group_mask",
    "decode_trajectory",
    "encode_trajectory",
]
