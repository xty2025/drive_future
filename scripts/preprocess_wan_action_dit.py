"""Build the NVS Action DiT backbone payload from a local Wan2.2-TI2V-5B checkout.

Mirrors SimWAM's ``preprocess_action_dit_backbone.py`` policy (sequential 1D
linear interpolation with ``align_corners=True`` + ``alpha=sqrt(d_src/d_tgt)``)
but targets the 256-dim NVS Action DiT and reads the safetensors shards directly,
so the 5B video DiT is never instantiated.

Example:
    python scripts/preprocess_wan_action_dit.py \
        --wan-root /mnt/LLM_weight/Wan-AI/Wan2.2-TI2V-5B \
        --output checkpoints/ActionDiT_Wan22_256hdim_3outdim.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from nvs_v6_wan_e2e.wan_dit import NvsActionDiT  # noqa: E402
from nvs_v6_wan_e2e.wan_init import build_action_dit_backbone  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wan-root", default="/mnt/LLM_weight/Wan-AI/Wan2.2-TI2V-5B")
    parser.add_argument("--output", required=True)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--ffn-dim", type=int, default=1024)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--attn-head-dim", type=int, default=64)
    parser.add_argument("--freq-dim", type=int, default=256)
    parser.add_argument("--action-dim", type=int, default=3)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--no-alpha-scaling", action="store_true")
    args = parser.parse_args()

    target = NvsActionDiT(
        hidden_dim=args.hidden_dim,
        action_dim=args.action_dim,
        ffn_dim=args.ffn_dim,
        text_dim=args.hidden_dim,
        freq_dim=args.freq_dim,
        eps=args.eps,
        num_heads=args.num_heads,
        attn_head_dim=args.attn_head_dim,
        num_layers=args.num_layers,
    )
    payload = build_action_dit_backbone(args.wan_root, target, apply_alpha_scaling=not args.no_alpha_scaling)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # `stats` is informational only; drop it so the payload stays weights-only loadable.
    stats = payload.pop("stats")
    torch.save(payload, output)
    print(
        f"[INFO] wrote {output} (copied={stats['copied']} interpolated={stats['interpolated']} "
        f"layer_map={payload['policy']['layer_map']})"
    )


if __name__ == "__main__":
    main()
