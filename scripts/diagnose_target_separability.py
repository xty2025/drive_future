"""Measure whether the Future Register target is discriminable across GT trajectories.

Stage 1's shuffle control (`gain_over_shuffled`) came out at 0.4% on the two real
runs, versus the 2% margin the trainer itself demands.  Two explanations produce
that number and they need completely different fixes:

1. the model cannot tell the trajectories apart (register has no trajectory
   input, only a 2-dim immutable anchor code) -- fix the conditioning;
2. the *target* cannot tell them apart -- no architecture change can help.

This separates them without training anything.  For each scene it Teacher-encodes
every GT's own `future.mp4`, and reports the L1 distance between two GTs' targets
inside one scene against the L1 distance between two different scenes, once in the
raw 4x4096 ordered-quadrant space and once after the 256-dim PCA the trainer fits.
If PCA retains the across-scene distance but destroys the within-scene one, the
compressor is discarding exactly the signal Stage 1 is supposed to learn.

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
    NATIVE_FUTURE_IMAGE_HW,
    Ordered2x2Compressor,
    TimmDinoV3Teacher,
    ordered_quadrant_vectors,
)
from nvs_v6_wan_e2e.vision import DRIVOR_OFFICIAL_IMAGE_HW

DEFAULT_DINO_CHECKPOINT = "/mnt/LLM_weight/timm/vit_large_patch16_dinov3.lvd1689m/model.safetensors"

# Reference numbers from the two real Stage 1 runs that motivated this script:
# artifacts/stage1_zerointerp (44 scenes, 400 steps) and
# artifacts/stage1_multiview_aligned (20 scenes, 120 steps).
OBSERVED = {
    "stage1_zerointerp": {"paired_future_l1": 0.4461728883907199, "shuffled_pair_l1": 0.4478922775015235},
    "stage1_multiview_aligned": {"paired_future_l1": 0.34815371222794056, "shuffled_pair_l1": 0.3496891111135483},
}


def _l1(left: torch.Tensor, right: torch.Tensor) -> float:
    """Same convention as `paired_future_l1`: mean over features, then mean over chunks."""

    return float((left - right).abs().mean(dim=-1).mean())


def _scene_block(
    teacher: TimmDinoV3Teacher,
    sample,
    device: torch.device,
) -> torch.Tensor:
    """One scene's ordered quadrant vectors, `[K, 4, 4D]`, on CPU.

    The patch grids are reduced as they are produced and never retained: a grid is
    50 MB float32 while its ordered vectors are a few hundred KB, and this script
    walks many scenes on a 50 GB machine.
    """

    future_rgb = sample.future_rgb
    if future_rgb is None:
        raise ValueError(f"{sample.scene_id}: future RGB must be loaded")
    per_gt = []
    for gt_index in range(future_rgb.shape[1]):
        images = future_rgb[:, gt_index].to(device)
        grid = teacher(images).float().cpu()
        per_gt.append(ordered_quadrant_vectors(grid)[0])
        del grid, images
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return torch.stack(per_gt, dim=0)


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
    parser.add_argument("--target-dim", type=int, default=256)
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
    scene_ids: list[str] = []
    gt_counts: list[int] = []
    traj_within: list[float] = []
    skipped: list[dict[str, str]] = []
    teacher = TimmDinoV3Teacher(checkpoint=args.dino_checkpoint, image_hw=tuple(args.teacher_image_hw)).to(device)
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

        block = _scene_block(teacher, sample, device)
        blocks.append(block)
        scene_ids.append(sample.scene_id)
        gt_counts.append(int(sample.targets.gt_trajectory.shape[1]))
        # The trajectory-side separation, for contrast: how far apart the GT
        # trajectories themselves are, in metres of ADE.
        gt = sample.targets.gt_trajectory[0]
        valid = sample.targets.gt_valid[0]
        for i, j in itertools.combinations(range(int(valid.sum())), 2):
            traj_within.append(float((gt[i, :, :2] - gt[j, :, :2]).norm(dim=-1).mean()))
        sample.future_rgb = None
        del sample
        gc.collect()
        print(json.dumps({"stage": "encode", "scene": scene_ids[-1], "gt": gt_counts[-1], "block": list(blocks[-1].shape)}), flush=True)
    del teacher
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if len(blocks) < 2:
        raise RuntimeError(f"need at least two scenes, got {len(blocks)}; skipped={skipped[:5]}")

    raw = _pairwise(blocks)

    # Exactly what the trainer feeds `Ordered2x2Compressor.fit`: every GT's ordered
    # vectors concatenated, so the SVD rows are (GT, chunk) pairs pooled over scenes.
    fit_rows = torch.cat(blocks, dim=0)
    compressor = Ordered2x2Compressor(output_dim=args.target_dim)
    compressor.fit(fit_rows, source_hash="+".join(scene_ids), split="train")
    compressed_blocks = [compressor.transform_ordered(block) for block in blocks]
    compressed_metrics = _pairwise(compressed_blocks)

    raw_within = raw["within_scene_across_gt"]["mean"]
    comp_within = compressed_metrics["within_scene_across_gt"]["mean"]
    raw_across = raw["across_scene"]["mean"]
    comp_across = compressed_metrics["across_scene"]["mean"]

    verdict = {
        "raw_within_scene_gt_l1": raw_within,
        "compressed_within_scene_gt_l1": comp_within,
        "raw_across_scene_l1": raw_across,
        "compressed_across_scene_l1": comp_across,
        # What a mean-future predictor would score, and what the shuffle control
        # can therefore possibly return.  If a model predicts the scene's average
        # future, its paired and shuffled errors both sit near half the within-scene
        # GT distance, so the achievable shuffle gain is bounded by that distance
        # relative to the error level the real runs actually reached.
        "within_gt_over_observed_paired": {
            name: comp_within / value["paired_future_l1"] for name, value in OBSERVED.items()
        },
        "pca_retention_within_scene_gt": comp_within / raw_within if raw_within else float("nan"),
        "pca_retention_across_scene": comp_across / raw_across if raw_across else float("nan"),
        "gt_trajectory_within_scene_ade_m": (
            float(torch.tensor(traj_within).mean()) if traj_within else float("nan")
        ),
    }
    # A retention ratio far below 1 for within-scene while across-scene stays near
    # 1 is the compressor discarding the trajectory signal specifically.
    verdict["pca_destroys_within_scene_signal"] = bool(
        verdict["pca_retention_within_scene_gt"] < 0.5 * verdict["pca_retention_across_scene"]
    )
    # If two GTs' compressed targets are within a few percent of the error level the
    # trainer already reached, there is no headroom for any model.
    verdict["target_has_no_headroom"] = bool(
        max(verdict["within_gt_over_observed_paired"].values()) < 0.05
    )

    summary = {
        "device": str(device),
        "scenes": scene_ids,
        "gt_counts": gt_counts,
        "skipped": skipped[:20],
        "teacher_image_hw": [int(v) for v in args.teacher_image_hw],
        "ordered_vector_dim": int(blocks[0].shape[-1]),
        "target_dim": args.target_dim,
        "compressor_fit_rows": int(fit_rows.shape[0] * fit_rows.shape[1]),
        "compressor_rank": int(min(args.target_dim, fit_rows.shape[0] * fit_rows.shape[1])),
        "raw_space": raw,
        "compressed_space": compressed_metrics,
        "observed_stage1_reference": OBSERVED,
        "verdict": verdict,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"stage": "verdict", **verdict}, indent=2), flush=True)
    print(json.dumps({"stage": "raw_space", **raw}, indent=2), flush=True)
    print(json.dumps({"stage": "compressed_space", **compressed_metrics}, indent=2), flush=True)
    print(f"wrote {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
