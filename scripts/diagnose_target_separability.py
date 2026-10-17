"""Measure whether the Future Register target is discriminable across GT trajectories.

Stage 1's shuffle control (`gain_over_shuffled`) came out at 0.4% on the two real
runs, versus the 2% margin the trainer itself demands.  Two explanations produce
that number and they need completely different fixes:

1. the model cannot tell the trajectories apart (register has no trajectory
   input, only a 2-dim immutable anchor code) -- fix the conditioning;
2. the *target* cannot tell them apart -- no architecture change can help.

This separates them without training anything.  For each scene it Teacher-encodes
every GT's own `future.mp4` and reduces each frame with the same CLS-query
attention the cache builder uses, then reports the L1 distance between two GTs'
targets inside one scene against the L1 distance between two different scenes,
once in the raw 8x1024 query space and once after the per-dimension
standardization the trainer fits.  If standardization retains the across-scene
distance but destroys the within-scene one, it is discarding exactly the signal
Stage 1 is supposed to learn.

Two further numbers are reported because a distance ratio on its own is not a
sufficient statistic: the readout's normalized attention entropy (a uniform
attention *is* the mean pool this target replaced, and every distance below stays
finite), and the correlation between target distance and GT trajectory ADE.

Run with the repository's standard PYTHONPATH::

    PYTHONPATH="$PWD/src:$PWD/vendor/DrivoR:$PWD/vendor/DrivoR/nuplan-devkit" \
      /mnt/zhaozc_workspace/envs/driov_dpo/bin/python \
      scripts/diagnose_target_separability.py --scenes 16
"""

from __future__ import annotations

import argparse
import gc
import itertools
import json
import math
import pickle
from pathlib import Path

import torch

from nvs_v6_wan_e2e.multiview_data import (
    DEFAULT_CACHE_ROOT,
    DEFAULT_LOG_ROOT,
    DEFAULT_MULTIVIEW_ROOT,
    DEFAULT_SENSOR_ROOT,
    iter_ready_scenes,
    load_multiview_sample,
)
from nvs_v6_wan_e2e.teacher import (
    MAX_QUERY_ATTENTION_ENTROPY,
    NATIVE_FUTURE_IMAGE_HW,
    QUERY_HEADS,
    QUERY_TEMPERATURE,
    FutureQueryExtractor,
    TargetStandardizer,
    TimmDinoV3Teacher,
    measure_target_quality,
    query_future_register,
)
from nvs_v6_wan_e2e.vision import DRIVOR_OFFICIAL_IMAGE_HW

DEFAULT_DINO_CHECKPOINT = "/mnt/LLM_weight/timm/vit_large_patch16_dinov3.lvd1689m/model.safetensors"


def _l1(left: torch.Tensor, right: torch.Tensor) -> float:
    """The ``mean_paired_future_l1`` metric convention: mean over features, then frames."""

    return float((left - right).abs().mean(dim=-1).mean())


def _pairwise(blocks: list[torch.Tensor]) -> dict[str, object]:
    """Within-scene across-GT distance, across-scene distance, and across-chunk."""

    within, across, chunk = [], [], []
    for block in blocks:
        gt_count = block.shape[0]
        for i, j in itertools.combinations(range(gt_count), 2):
            for c in range(block.shape[1]):
                within.append(_l1(block[i, c], block[j, c]))
        for c1, c2 in itertools.combinations(range(block.shape[1]), 2):
            chunk.append(_l1(block[:, c1].mean(dim=0), block[:, c2].mean(dim=0)))
    for left, right in itertools.combinations(blocks, 2):
        pairs = min(left.shape[0], right.shape[0])
        for i in range(pairs):
            for c in range(left.shape[1]):
                across.append(_l1(left[i, c], right[i, c]))

    def summary(values: list[float]) -> dict[str, float]:
        if not values:
            return {"mean": float("nan"), "min": float("nan"), "max": float("nan"), "pairs": 0}
        tensor = torch.tensor(values)
        return {
            "mean": float(tensor.mean()),
            "min": float(tensor.min()),
            "max": float(tensor.max()),
            "pairs": len(values),
        }

    return {
        "within_scene_across_gt": summary(within),
        "across_scene": summary(across),
        "within_scene_across_chunk": summary(chunk),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--multiview-root", default=DEFAULT_MULTIVIEW_ROOT)
    parser.add_argument("--log-root", default=DEFAULT_LOG_ROOT)
    parser.add_argument("--sensor-root", default=DEFAULT_SENSOR_ROOT)
    parser.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--scenes", type=int, default=16)
    parser.add_argument("--dino-checkpoint", default=DEFAULT_DINO_CHECKPOINT)
    parser.add_argument("--image-hw", type=int, nargs=2, default=list(DRIVOR_OFFICIAL_IMAGE_HW))
    parser.add_argument("--teacher-image-hw", type=int, nargs=2, default=list(NATIVE_FUTURE_IMAGE_HW))
    parser.add_argument("--query-temperature", type=float, default=QUERY_TEMPERATURE,
                        help="cosine temperature of the CLS query; lower is more selective")
    parser.add_argument("--query-heads", type=int, default=QUERY_HEADS,
                        help="independent query readouts concatenated into the target")
    parser.add_argument("--skip-degenerate", action="store_true", default=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--output", default="artifacts/diagnose_target_separability.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("the frozen Teacher needs the configured CUDA/PPU device")

    scene_dirs = list(
        iter_ready_scenes(
            args.multiview_root,
            args.cache_root,
            limit=args.scenes + 8,
            start=args.start,
            require_distinct_labels=args.skip_degenerate,
        )
    )
    if not scene_dirs:
        raise FileNotFoundError(f"no ready Multi-GT scenes under {args.multiview_root}")

    log_cache: dict[str, list[dict]] = {}
    blocks: list[torch.Tensor] = []
    trajectories: list[torch.Tensor] = []
    scene_ids: list[str] = []
    gt_counts: list[int] = []
    entropies: list[torch.Tensor] = []
    traj_within: list[float] = []
    skipped: list[dict[str, str]] = []
    teacher = TimmDinoV3Teacher(checkpoint=args.dino_checkpoint, image_hw=tuple(args.teacher_image_hw)).to(device)
    extractor = FutureQueryExtractor(teacher.feature_dim, args.query_temperature, args.query_heads)
    for scene_dir in scene_dirs:
        if len(blocks) >= args.scenes:
            break
        metadata = json.loads((scene_dir / "scene_metadata.json").read_text())
        log_name = str(metadata["log_name"])
        if log_name not in log_cache:
            with (Path(args.log_root) / f"{log_name}.pkl").open("rb") as handle:
                log_cache[log_name] = pickle.load(handle)
        try:
            sample = load_multiview_sample(
                scene_dir,
                log_root=args.log_root,
                sensor_root=args.sensor_root,
                cache_root=args.cache_root,
                image_hw=tuple(args.image_hw),
                future_image_hw=tuple(args.teacher_image_hw),
                log_frames=log_cache[log_name],
            )
        except (FileNotFoundError, ValueError) as error:
            skipped.append({"scene": scene_dir.name, "reason": str(error)})
            continue
        if args.skip_degenerate and sample.metadata["degenerate_multi_gt"]:
            skipped.append({"scene": scene_dir.name, "reason": "degenerate_multi_gt"})
            continue

        future_rgb = sample.future_rgb
        if future_rgb is None:
            skipped.append({"scene": scene_dir.name, "reason": "future_rgb_not_loaded"})
            continue
        block, block_entropies = query_future_register(teacher, extractor, future_rgb, device)
        blocks.append(block)
        entropies.append(block_entropies)
        scene_ids.append(sample.scene_id)
        gt_counts.append(int(sample.targets.gt_trajectory.shape[1]))
        trajectories.append(sample.targets.gt_trajectory[0].cpu())
        # The trajectory-side separation, for contrast: how far apart the GT
        # trajectories themselves are, in metres of ADE.
        gt = sample.targets.gt_trajectory[0]
        valid = sample.targets.gt_valid[0]
        for i, j in itertools.combinations(range(int(valid.sum())), 2):
            traj_within.append(float((gt[i, :, :2] - gt[j, :, :2]).norm(dim=-1).mean()))
        sample.future_rgb = None
        del sample
        gc.collect()
        print(json.dumps({"stage": "encode", "scene": scene_ids[-1], "gt": gt_counts[-1],
                          "block": list(blocks[-1].shape),
                          "attention_entropy": round(float(block_entropies.mean()), 4)}), flush=True)
    grid_hw = teacher.grid_hw
    feature_dim = teacher.feature_dim
    del teacher
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if len(blocks) < 2:
        raise RuntimeError(f"need at least two scenes, got {len(blocks)}; skipped={skipped[:5]}")

    all_entropies = torch.cat(entropies, dim=0)
    entropy_mean = float(all_entropies.mean())
    raw = _pairwise(blocks)

    # Exactly what the builder feeds `TargetStandardizer.fit`: every GT's query
    # vectors pooled, so each dimension's mean/std is estimated over (scene, GT,
    # frame) rows rather than over the scenes alone.
    fit_rows = torch.cat(blocks, dim=0)
    standardizer = TargetStandardizer(feature_dim, args.query_temperature, args.query_heads)
    standardizer.fit(fit_rows, source_hash="+".join(scene_ids), split="train")
    standardized_blocks = [standardizer.transform(block) for block in blocks]
    standardized_metrics = _pairwise(standardized_blocks)
    quality = measure_target_quality(standardized_blocks, trajectories)

    raw_within = raw["within_scene_across_gt"]["mean"]
    std_within = standardized_metrics["within_scene_across_gt"]["mean"]
    raw_across = raw["across_scene"]["mean"]
    std_across = standardized_metrics["across_scene"]["mean"]

    retention_within = std_within / raw_within if raw_within else float("nan")
    retention_across = std_across / raw_across if raw_across else float("nan")
    verdict = {
        "raw_within_scene_gt_l1": raw_within,
        "standardized_within_scene_gt_l1": std_within,
        "raw_across_scene_l1": raw_across,
        "standardized_across_scene_l1": std_across,
        "standardization_retention_within_scene_gt": retention_within,
        "standardization_retention_across_scene": retention_across,
        "query_attention_entropy_mean": entropy_mean,
        "query_attention_entropy_max": float(all_entropies.max()),
        "query_attention_entropy_ceiling": MAX_QUERY_ATTENTION_ENTROPY,
        "separation_ratio": quality["separation_ratio"],
        "trajectory_distance_pearson": quality["trajectory_distance_pearson"],
        "trajectory_distance_spearman": quality["trajectory_distance_spearman"],
        "gt_trajectory_within_scene_ade_m": (
            float(torch.tensor(traj_within).mean()) if traj_within else float("nan")
        ),
    }
    # A retention ratio far below 1 for within-scene while across-scene stays near 1
    # is the standardizer reweighting away the trajectory signal specifically.  It
    # cannot zero a dimension the way the PCA projector it replaced did, so this is
    # now a loss-weighting concern rather than a dead-target one.
    verdict["standardization_destroys_within_scene_signal"] = bool(
        retention_within < 0.5 * retention_across
    )
    # Uniform attention makes the readout the mean pool it replaced: every distance
    # above stays finite and 81% of the GT-discriminating variance is gone.  This is
    # the same gate the cache builder refuses to persist past.
    verdict["query_is_a_mean_pool"] = bool(entropy_mean > MAX_QUERY_ATTENTION_ENTROPY)
    # Separation alone is not sufficient: the random-query control matched the CLS
    # query's ratio (0.4957 against 0.4934) while correlating only 0.4121 with
    # trajectory ADE against the CLS query's 0.6647.  Both numbers have to be read;
    # see ``measure_target_quality``.
    pearson = quality["trajectory_distance_pearson"]
    verdict["target_carries_trajectory_signal"] = bool(
        not math.isnan(pearson) and pearson > 0.5
    )

    summary = {
        "device": str(device),
        "scenes": scene_ids,
        "gt_counts": gt_counts,
        "skipped": skipped[:20],
        "teacher_image_hw": [int(v) for v in args.teacher_image_hw],
        "teacher_grid_hw": [int(grid_hw[0]), int(grid_hw[1])],
        "query_temperature": float(args.query_temperature),
        "query_heads": int(args.query_heads),
        "register_target_dim": int(blocks[0].shape[-1]),
        "standardizer_fit_rows": int(fit_rows.shape[0] * fit_rows.shape[1]),
        "standardizer_clamped_components": int(standardizer.state.clamped_components),
        "target_fingerprint": standardizer.fingerprint(),
        "raw_space": raw,
        "standardized_space": standardized_metrics,
        "verdict": verdict,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"stage": "verdict", **verdict}, indent=2), flush=True)
    print(json.dumps({"stage": "raw_space", **raw}, indent=2), flush=True)
    print(json.dumps({"stage": "standardized_space", **standardized_metrics}, indent=2), flush=True)
    print(f"wrote {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
