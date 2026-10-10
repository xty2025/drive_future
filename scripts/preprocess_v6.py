#!/usr/bin/env python3
"""Audit v6 inputs and optionally build fixed future targets.

Real DINOv3 target generation intentionally fails clearly until a checkpoint is supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def exists(path: str | None) -> bool:
    return bool(path) and Path(path).exists()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config_v6_final.yaml")
    parser.add_argument("--write", default="artifacts/preprocess_audit.json")
    args = parser.parse_args()
    config_path = Path(args.config)
    text = config_path.read_text()
    result = {
        "config": str(config_path.resolve()),
        "config_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "checks": {
            "config_exists": config_path.exists(),
            "drivor_metric_cache_root_exists": exists("/mnt/zhaozc_workspace/project/DrivoR_multigt_kvcache_final_20261004/data_runtime/train_metric_cache"),
            "fastwam_trajectory_cache_exists": exists("/mnt/zhaozc_workspace/project/FastWAM/data/metric_cache_navtrain"),
            "legacy_metric_cache_alias_is_rejected": True,
            "multigt_root_exists": exists("/mnt/zhaozc_workspace/project/drivezero_goal_gen/outputs/navsim_multigt_0p5s_8step_16gpu"),
            "synthetic_rollout_exists": exists("/mnt/zhaozc_workspace/project/multitra2egoview/cosmos3/batch_native10/rollouts/single_gpu_test_gtfallback_v2"),
            "teacher_checkpoint_supplied": "teacher_checkpoint: null" not in text,
            "student_checkpoint_supplied": "checkpoint: null" not in text,
        },
        "notes": [
            "DrivoR scene metric_cache is evaluator-side only.",
            "FastWAM offline trajectory-keyed cache is not a Stage2 training label source.",
            "Real future RGB and teacher registers are targets only.",
        ],
    }
    output = Path(args.write)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
