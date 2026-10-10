"""NAVSIM v6-wan: Multi-GT / Future Register generator whose Action DiT blocks
follow the Wan2.2 DiTBlock design (randomly initialized).  Directed S/A/R reads
and the one-pass residual-velocity endpoint output are unchanged from v6."""

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
