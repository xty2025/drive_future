#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="v6 evaluation entrypoint with explicit cache separation")
    parser.add_argument("--mode", choices=("drivor-online", "fastwam-offline"), required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = Path(args.cache_root)
    if not root.exists():
        raise SystemExit(f"cache root does not exist: {root}")
    resolved = str(root.resolve()).lower()
    is_fastwam = "fastwam" in resolved or "trajectory" in resolved
    if args.mode == "fastwam-offline" and not is_fastwam:
        raise SystemExit("FastWAM offline evaluation requires an explicitly trajectory-keyed FastWAM cache path")
    if args.mode == "drivor-online" and is_fastwam:
        raise SystemExit("DrivoR online mode must use a real scene metric_cache, not the FastWAM trajectory-keyed cache symlink")
    result = {
        "mode": args.mode,
        "cache_root": str(root.resolve()),
        "cache_kind": "fastwam_trajectory_keyed" if is_fastwam else "drivor_scene_metric_cache",
        "dry_run": args.dry_run,
        "training_cache_reuse": False,
        "status": "validated_path_only" if args.dry_run else "adapter_required",
    }
    print(json.dumps(result, indent=2))
    if not args.dry_run:
        raise SystemExit("wire the audited NAVSIM/DrivoR evaluator adapter before real evaluation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
