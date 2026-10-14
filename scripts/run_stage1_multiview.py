"""Stage-1 training on the real /mnt/data/multi_view Multi-GT future videos.

The observation is DrivoR-shaped (five real front-camera history frames, the
7-wide ego status plus the four-way command, metric trajectories).  The Future
Register target is built from the generated ``trajectory_k/future.mp4`` frames by
the frozen DINOv3 Teacher and the ordered 2x2 compressor, so Stage 1 trains
``phi`` to predict, for each real GT trajectory of the scene, what the world will
look like under that trajectory.  The dataset supplies the pairing: register row
``k`` is supervised by GT ``k``'s compressed future, so no anchor codebook and no
Hungarian assignment takes part.

Run with the repository's standard PYTHONPATH, e.g.::

    PYTHONPATH="$PWD/src:$PWD/vendor/DrivoR:$PWD/vendor/DrivoR/nuplan-devkit" \
      /mnt/zhaozc_workspace/envs/driov_dpo/bin/python scripts/run_stage1_multiview.py \
      --scenes 4 --steps 20 --check-scorer
"""

from __future__ import annotations

import argparse
import gc
import itertools
import json
import pickle
import random
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from nvs_v6_wan_e2e.data import Observation, PairTargets
from nvs_v6_wan_e2e.model import DirectedPairDiT, parameter_groups
from nvs_v6_wan_e2e.modes import ImmutableModeBuffer, StructuredSourceInitializer
from nvs_v6_wan_e2e.multiview_data import (
    DEFAULT_CACHE_ROOT,
    DEFAULT_LOG_ROOT,
    DEFAULT_MULTIVIEW_ROOT,
    DEFAULT_SENSOR_ROOT,
    MultiViewSample,
    iter_ready_scenes,
    load_multiview_sample,
    with_future_register,
)
from nvs_v6_wan_e2e.stages import build_optimizers, stage1_step
from nvs_v6_wan_e2e.teacher import (
    NATIVE_FUTURE_IMAGE_HW,
    Ordered2x2Compressor,
    TimmDinoV3Teacher,
    ordered_quadrant_vectors,
)
from nvs_v6_wan_e2e.vision import (
    DRIVOR_LORA_RANK,
    DRIVOR_NUM_SCENE_TOKENS,
    DRIVOR_OFFICIAL_IMAGE_HW,
)

DEFAULT_DINO_CHECKPOINT = "/mnt/LLM_weight/timm/vit_large_patch16_dinov3.lvd1689m/model.safetensors"
DEFAULT_WAN_BACKBONE = "checkpoints/ActionDiT_Wan22_256hdim_3outdim.pt"
# The Teacher encodes the future videos at the resolution they were generated in.
# 464x832 is a whole number of patch16 tiles (29x52), so nothing is resampled and
# every target patch summarises pixels the generator actually rendered.  Its grid
# is coarser than the Student's 42x72 and that is fine: DINOv3's RoPE divides
# coordinates by H and W separately and maps them to [-1, 1], so a token's
# position depends on where it sits in the frame, not on how many tokens the frame
# was cut into.  The ordered 2x2 quadrants are fractions of the frame for the same
# reason, which keeps target and scene tokens talking about the same image regions.
TEACHER_IMAGE_HW = NATIVE_FUTURE_IMAGE_HW


def _teacher_ordered_vectors(
    samples: list[MultiViewSample],
    checkpoint: str,
    image_hw: tuple[int, int],
    device: torch.device,
) -> tuple[list[torch.Tensor], list[int], list[int]]:
    """Quadrant-pooled frozen DINOv3 features, one [gt,4,8192] block per scene.

    The 42x72 grids are reduced as they are produced and never retained: a grid
    is 103 MB float32 while its ordered vectors are 131 KB, which is the
    difference between fitting a full-rank compressor and running out of RAM.
    """

    teacher = TimmDinoV3Teacher(checkpoint=checkpoint, image_hw=image_hw).to(device)
    blocks = []
    grid_hw: list[int] = []
    feature_dims: list[int] = []
    for sample in samples:
        if sample.future_rgb is None:
            raise ValueError("future RGB must be loaded to build register targets")
        per_scene = []
        for gt_index in range(sample.future_rgb.shape[1]):
            images = sample.future_rgb[:, gt_index].to(device)
            grid = teacher(images).float().cpu()
            grid_hw = [int(grid.shape[2]), int(grid.shape[3])]
            feature_dims.append(int(grid.shape[-1]))
            per_scene.append(ordered_quadrant_vectors(grid))
            del grid
        blocks.append(torch.cat(per_scene, dim=0))
    del teacher
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return blocks, grid_hw, feature_dims


def _build_future_targets(
    samples: list[MultiViewSample],
    checkpoint: str,
    image_hw: tuple[int, int],
    device: torch.device,
    target_dim: int,
    fit_count: int | None = None,
) -> tuple[list[MultiViewSample], dict[str, object]]:
    blocks, grid_hw, feature_dims = _teacher_ordered_vectors(samples, checkpoint, image_hw, device)
    compressor = Ordered2x2Compressor(output_dim=target_dim)
    fit_blocks = blocks if fit_count is None else blocks[:fit_count]
    fit_vectors = torch.cat(fit_blocks, dim=0)
    compressor.fit(
        fit_vectors,
        source_hash="+".join(sample.scene_id for sample in samples[: len(fit_blocks)]),
        split="train",
    )
    updated = []
    for sample, block in zip(samples, blocks):
        register = compressor.transform_ordered(block).unsqueeze(0)
        # The decoded frames have served their purpose; holding them would cost
        # 297 MB per scene at the aligned 672x1152 Teacher resolution.
        updated.append(replace(sample, future_rgb=None, targets=with_future_register(sample.targets, register)))
    fit_rows = fit_vectors.shape[0] * fit_vectors.shape[1]
    audit = {
        "teacher_checkpoint": checkpoint,
        "teacher_image_hw": [int(image_hw[0]), int(image_hw[1])],
        "teacher_grid_hw": grid_hw,
        "teacher_feature_dim": feature_dims[0],
        "compressor_input_dim": int(compressor.state.input_dim),
        "compressor_output_dim": int(compressor.state.output_dim),
        # The SVD sees every ordered chunk of every candidate as its own row, so
        # the target's rank is 4 * fitted candidates, not the candidate count.
        "compressor_rank": int(min(target_dim, fit_rows)),
        "compressor_fit_rows": int(fit_rows),
        "compressor_fit_scenes": len(fit_blocks),
        "compressor_full_rank": bool(min(target_dim, fit_rows) >= target_dim),
        "compressor_fingerprint": compressor.fingerprint(),
        "register_target_shape": list(updated[0].targets.future_register.shape),
    }
    return updated, audit


def _collate(samples: list[MultiViewSample]) -> tuple[Observation, PairTargets]:
    """Pad only the GT axis; every scene keeps all of its own Multi-GT pairs."""

    observation = Observation(
        rgb_history=torch.cat([sample.observation.rgb_history for sample in samples], dim=0),
        history_valid=torch.cat([sample.observation.history_valid for sample in samples], dim=0),
        ego_history=torch.cat([sample.observation.ego_history for sample in samples], dim=0),
        route_command=torch.cat([sample.observation.route_command for sample in samples], dim=0),
        relative_ego_poses=torch.cat([sample.observation.relative_ego_poses for sample in samples], dim=0),
        camera_metadata=[sample.observation.camera_metadata for sample in samples],
    )
    max_gt = max(sample.targets.gt_trajectory.shape[1] for sample in samples)

    stacked = {}
    for value_name in ("gt_trajectory", "gt_valid", "future_register", "future_chunk_valid", "future_quality"):
        values = [getattr(sample.targets, value_name) for sample in samples]
        fill = False if values[0].dtype == torch.bool else 0
        padded = values[0].new_full((len(values), max_gt, *values[0].shape[2:]), fill)
        for index, value in enumerate(values):
            padded[index, : value.shape[1]] = value[0]
        stacked[value_name] = padded
    targets = PairTargets(
        pair_ids=[sample.targets.pair_ids[0] for sample in samples],
        **stacked,
    )
    observation.validate()
    targets.validate(max_gt=max_gt)
    return observation, targets


def _move_to_device(sample: MultiViewSample, device: torch.device) -> MultiViewSample:
    observation = replace(
        sample.observation,
        **{
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in sample.observation.__dict__.items()
        },
    )
    targets = replace(
        sample.targets,
        **{
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in sample.targets.__dict__.items()
        },
    )
    return replace(sample, observation=observation, targets=targets, future_rgb=None)


def _trajectory_errors(predicted: torch.Tensor, ground_truth: torch.Tensor) -> dict[str, float]:
    """Metric ADE/FDE of proposals against the Multi-GT set (no rescaling)."""

    per_pair_ade = (predicted[:, None, :, :2] - ground_truth[None, :, :, :2]).norm(dim=-1).mean(dim=-1)
    per_pair_fde = (predicted[:, None, -1, :2] - ground_truth[None, :, -1, :2]).norm(dim=-1)
    return {
        "min_ade_over_gt_m": float(per_pair_ade.min(dim=1).values.mean()),
        "best_pair_ade_m": float(per_pair_ade.min()),
        "best_pair_fde_m": float(per_pair_fde.min()),
        "mean_speed_mps": float(predicted[:, 1:, :2].sub(predicted[:, :-1, :2]).norm(dim=-1).mean() / 0.5),
    }


def _trajectory_spread(trajectories: torch.Tensor) -> float:
    """Mean pairwise ADE, in metres, between one scene's own GT trajectories.

    This is how much the *inputs* of a scene differ.  Compared against
    ``target_spread_across_gt`` it says whether the compressed futures separate by
    at least as much as the trajectories that produced them do.  A single-GT scene
    has no pair to compare and reports NaN.
    """

    if trajectories.shape[0] < 2:
        return float("nan")
    xy = trajectories[:, :, :2]
    per_step = (xy[:, None] - xy[None, :]).norm(dim=-1)
    upper = torch.triu_indices(per_step.shape[0], per_step.shape[0], offset=1)
    return float(per_step[upper[0], upper[1]].mean())


@torch.no_grad()
def _future_fit_metrics(
    samples: list[MultiViewSample],
    model: DirectedPairDiT,
    mean_target: torch.Tensor | None = None,
    cross_samples: list[MultiViewSample] | None = None,
) -> dict[str, object]:
    """Stage-1's objective plus the controls that make a low loss meaningful.

    The register is conditioned on each GT trajectory, so row ``k`` is scored
    against GT ``k``'s own compressed future.  A low paired error alone cannot
    distinguish "predicts this trajectory's future" from "predicts the average
    future", so the controls are recorded in two families.

    Model-independent baselines, the only ones ``paired_future_l1`` can honestly be
    compared against: ``scene_mean_target_l1`` is the error of predicting this
    scene's own per-chunk average future -- perfect scene reading, zero trajectory
    reading -- and ``constant_target_l1`` is ``|mean_target - own_target|``, the
    dataset-average predictor.  ``scene_mean_target_l1`` is the tighter of the two
    and is the one that decides whether the trajectory input earned its place.

    Model-side controls: ``trajectory_blind_l1`` re-runs the scene with every row
    conditioned on GT 0's trajectory, ``shuffled_pair_l1`` scores row ``k`` against
    GT ``k+1``, and ``scene_blind_l1`` scores it against another scene's targets.
    ``distance_to_mean_target_l1`` is not a baseline at all -- a small value is the
    signature of collapse toward the dataset average.

    ``trajectory_conditioning_fidelity`` and ``trajectory_response_alignment`` say
    how the register reacts to swapping the trajectory: how far it moves relative to
    how far the target moves, and the cosine between the two displacements.  Both
    are scale-free; fidelity is 0 for a trajectory-blind register and 1.0 for one
    that reproduces the targets exactly, and alignment is 0 for a displacement in a
    random direction of 256 dimensions.
    """

    was_training = model.training
    model.eval()
    rows = []
    for index, sample in enumerate(samples):
        targets = sample.targets
        gt_count = int(targets.gt_valid[0].sum())
        source = targets.gt_trajectory[:, :gt_count]
        predicted = model.forward_register_only(sample.observation, source)[0]
        own_target = targets.future_register[0, :gt_count]
        chunk_valid = targets.future_chunk_valid[0, :gt_count]
        weights = targets.future_quality[0, :gt_count] * chunk_valid.to(targets.future_quality.dtype)

        def weighted_error(prediction: torch.Tensor, target: torch.Tensor) -> float:
            error = (prediction - target).abs().mean(dim=-1)
            return float((error * weights).sum() / weights.sum().clamp_min(1e-8))

        blind_source = source[:, :1].expand(-1, gt_count, -1, -1).contiguous()
        blind_predicted = model.forward_register_only(sample.observation, blind_source)[0]

        scene_blind = float("nan")
        if cross_samples is not None:
            cross = cross_samples[index]
            pairs = min(gt_count, int(cross.targets.gt_valid[0].sum()))
            if pairs:
                cross_weights = weights[:pairs]
                cross_target = cross.targets.future_register[0, :pairs].to(predicted.device)
                error = (predicted[:pairs] - cross_target).abs().mean(dim=-1)
                scene_blind = float((error * cross_weights).sum() / cross_weights.sum().clamp_min(1e-8))

        response, spread, alignment = [], [], []
        for left, right in itertools.combinations(range(gt_count), 2):
            both = chunk_valid[left] & chunk_valid[right]
            if not bool(both.any()):
                continue
            moved = (predicted[left, both] - predicted[right, both]).flatten()
            differs = (own_target[left, both] - own_target[right, both]).flatten()
            response.append(float(moved.abs().mean()))
            spread.append(float(differs.abs().mean()))
            alignment.append(float(torch.nn.functional.cosine_similarity(moved, differs, dim=0)))

        # The per-chunk average of this scene's own GT futures: what a register that
        # read the scene perfectly and the trajectory not at all would output.
        scene_mean = (own_target * weights.unsqueeze(-1)).sum(dim=0) / weights.sum(dim=0).clamp_min(1e-8).unsqueeze(-1)

        mean_errors, constant_errors = [], []
        if mean_target is not None:
            reference = mean_target.to(predicted.device)
            for row in range(gt_count):
                valid = chunk_valid[row]
                if not bool(valid.any()):
                    continue
                mean_errors.append(float((predicted[row, valid] - reference[valid]).abs().mean()))
                constant_errors.append(float((reference[valid] - own_target[row, valid]).abs().mean()))

        rows.append(
            {
                "scene_id": sample.scene_id,
                "gt_count": gt_count,
                "paired_future_l1": weighted_error(predicted, own_target),
                "scene_mean_target_l1": weighted_error(scene_mean.expand_as(own_target), own_target),
                "constant_target_l1": float(np.mean(constant_errors)) if constant_errors else float("nan"),
                "shuffled_pair_l1": (
                    weighted_error(predicted, torch.roll(own_target, -1, dims=0)) if gt_count > 1 else float("nan")
                ),
                "trajectory_blind_l1": (
                    weighted_error(blind_predicted, own_target) if gt_count > 1 else float("nan")
                ),
                "scene_blind_l1": scene_blind,
                "distance_to_mean_target_l1": float(np.mean(mean_errors)) if mean_errors else float("nan"),
                "register_response_to_trajectory": float(np.mean(response)) if response else float("nan"),
                "target_spread_across_gt": float(np.mean(spread)) if spread else float("nan"),
                "trajectory_conditioning_fidelity": (
                    float(np.sum(response) / np.sum(spread)) if spread and float(np.sum(spread)) > 0 else float("nan")
                ),
                "trajectory_response_alignment": float(np.mean(alignment)) if alignment else float("nan"),
                "register_std_across_candidates": float(predicted.std(dim=0).mean()),
                "register_abs_mean": float(predicted.abs().mean()),
                "gt_trajectory_spread_ade_m": _trajectory_spread(source[0].cpu()),
            }
        )
    model.train(was_training)

    keys = [key for key in rows[0] if key != "scene_id"]
    summary = {f"mean_{key}": float(np.nanmean([row[key] for row in rows])) for key in keys}
    summary["scenes"] = rows
    return summary


@torch.no_grad()
def _scorer_audit(
    model: DirectedPairDiT,
    observation: Observation,
    targets: PairTargets,
    modes: ImmutableModeBuffer,
    initializer: StructuredSourceInitializer,
    device: torch.device,
) -> dict[str, object]:
    """Structural check of the DrivoR scorer on the real NVS Scene K/V."""

    from nvs_v6_wan_e2e.scorer import HEAD_NAMES, ScratchDrivoRScorer

    scorer = ScratchDrivoRScorer(scene_tokens=model.scene_tokens).to(device)
    scorer.eval()
    single = replace(
        observation,
        rgb_history=observation.rgb_history[:1],
        history_valid=observation.history_valid[:1],
        ego_history=observation.ego_history[:1],
        route_command=observation.route_command[:1],
        relative_ego_poses=observation.relative_ego_poses[:1],
    )
    mode_codes = modes.expand_for_batch(1, device)
    source = initializer.clean(single.ego_history, mode_codes)
    outputs = model.forward_pairs(single, source, mode_codes)
    logits = scorer(outputs.scene_context, outputs.trajectory)
    selected, utility = scorer.select(logits)
    gt_count = int(targets.gt_valid[0].sum())
    return {
        "scene_context_shape": list(outputs.scene_context.shape),
        "scene_tokens_expected": int(model.scene_tokens),
        "trajectory_shape": list(outputs.trajectory.shape),
        "trajectory_units": "metres/radians, DrivoR-native, no rescaling",
        "predicted_register_shape": list(outputs.predicted_register.shape),
        "score_logits_shape": list(logits.shape),
        "score_heads": list(HEAD_NAMES),
        "logits_finite": bool(torch.isfinite(logits).all()),
        "selected_candidate": int(selected.reshape(-1)[0]),
        "selected_utility": float(utility.reshape(-1)[0]),
        "proposals_vs_multigt": _trajectory_errors(
            outputs.trajectory[0].cpu(), targets.gt_trajectory[0, :gt_count].cpu()
        ),
        "scorer_parameters": sum(parameter.numel() for parameter in scorer.parameters()),
        "feature_audit": scorer.feature_audit,
    }


def _register_verdict(train: dict[str, object], evaluation: dict[str, object] | None, margin: float = 0.02) -> dict[str, object]:
    """Turn the controls into a single honest statement about the Future Register.

    A margin is required because "paired < baseline" by a fraction of a percent is
    what an untrained register also produces; the earlier four-scene run sat at 1.8%
    and was not evidence of scene-conditional prediction.

    Every ``gain_over_*`` here compares ``mean_paired_future_l1`` against a
    model-independent baseline or a deliberately degraded re-run of the same model,
    never against ``distance_to_mean_target_l1``.  That distance is a collapse
    detector -- how far the register drifted toward the dataset average -- and is
    reported separately as ``collapse_toward_mean``, the ratio of that drift to the
    paired error.  The old ``gain_over_mean_predictor`` compared against it and was
    meaningless as a skill claim.

    ``gain_over_scene_mean_target`` carries the Stage-1 claim: the scene's own
    per-chunk average future is what a register that read the scene perfectly and
    the trajectory not at all would output, so beating it is the only evidence that
    the trajectory input changed the prediction for the better.  The two
    scale-free numbers must agree with it -- a register can lower its L1 by
    shrinking toward the average, which shows up as fidelity below 1 and alignment
    below 0 even when the paired error falls.
    """

    gains = (
        "scene_mean_target_l1",
        "constant_target_l1",
        "scene_blind_l1",
        "shuffled_pair_l1",
        "trajectory_blind_l1",
    )

    def beats(metrics: dict[str, object], key: str) -> float:
        paired = float(metrics["mean_paired_future_l1"])
        baseline = float(metrics[f"mean_{key}"])
        return (baseline - paired) / baseline if baseline > 0 else float("nan")

    def collapse(metrics: dict[str, object]) -> float:
        paired = float(metrics["mean_paired_future_l1"])
        drift = float(metrics["mean_distance_to_mean_target_l1"])
        return drift / paired if paired > 0 else float("nan")

    def describe(metrics: dict[str, object], prefix: str) -> dict[str, object]:
        reported: dict[str, object] = {
            f"{prefix}_gain_over_{key.removesuffix('_l1')}": beats(metrics, key) for key in gains
        }
        reported[f"{prefix}_collapse_toward_mean"] = collapse(metrics)
        reported[f"{prefix}_trajectory_conditioning_fidelity"] = float(metrics["mean_trajectory_conditioning_fidelity"])
        reported[f"{prefix}_trajectory_response_alignment"] = float(metrics["mean_trajectory_response_alignment"])
        reported[f"{prefix}_beats_trajectory_blind_baseline"] = bool(
            float(reported[f"{prefix}_gain_over_scene_mean_target"]) >= margin
        )
        reported[f"{prefix}_learned_trajectory_conditional_future"] = bool(
            reported[f"{prefix}_beats_trajectory_blind_baseline"]
            and all(float(reported[f"{prefix}_gain_over_{key.removesuffix('_l1')}"]) >= margin for key in gains)
            and float(reported[f"{prefix}_trajectory_response_alignment"]) >= 0.5
            and float(reported[f"{prefix}_trajectory_conditioning_fidelity"]) >= 0.25
        )
        return reported

    verdict: dict[str, object] = {"margin_required": margin}
    verdict.update(describe(train, "train"))
    if evaluation is not None:
        verdict.update(describe(evaluation, "eval"))
        verdict["generalizes_beyond_fitted_scenes"] = bool(verdict["eval_learned_trajectory_conditional_future"])
    return verdict


def _parameter_norm(parameters) -> float:
    values = [parameter.detach().float().norm() for parameter in parameters]
    return float(torch.stack(values).norm()) if values else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--multiview-root", default=DEFAULT_MULTIVIEW_ROOT)
    parser.add_argument("--log-root", default=DEFAULT_LOG_ROOT)
    parser.add_argument("--sensor-root", default=DEFAULT_SENSOR_ROOT)
    parser.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--scenes", type=int, default=4, help="training scenes; >=64 candidates are needed for a full-rank target")
    parser.add_argument("--eval-scenes", type=int, default=0, help="extra scenes held out of training and of the compressor fit")
    parser.add_argument("--batch-size", type=int, default=4, help="scenes per optimizer step; decoupled from --scenes")
    parser.add_argument("--tokens", nargs="*", default=None)
    parser.add_argument("--skip-degenerate", action="store_true", help="drop scenes whose Multi-GT candidates collapse")
    parser.add_argument("--dino-checkpoint", default=DEFAULT_DINO_CHECKPOINT)
    parser.add_argument("--wan-backbone", default=DEFAULT_WAN_BACKBONE)
    parser.add_argument("--image-hw", type=int, nargs=2, default=list(DRIVOR_OFFICIAL_IMAGE_HW), help="DrivoR's official history/current frame size; the loader bicubic-downscales 1920x1080 to it and the Student zero-pads the patch16 remainder")
    parser.add_argument("--teacher-image-hw", type=int, nargs=2, default=list(TEACHER_IMAGE_HW), help="must equal the future videos' native size; the Teacher refuses to interpolate")
    parser.add_argument("--num-scene-tokens", type=int, default=DRIVOR_NUM_SCENE_TOKENS)
    parser.add_argument("--lora-rank", type=int, default=DRIVOR_LORA_RANK)
    parser.add_argument("--grad-checkpointing", action="store_true")
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--candidates", type=int, default=64)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--target-dim", type=int, default=256)
    parser.add_argument("--check-scorer", action="store_true")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output-dir", default="artifacts/stage1_multiview")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Stage 1 on real Multi-GT videos requires the configured CUDA/PPU device")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wanted = args.scenes + args.eval_scenes
    if args.tokens:
        scene_dirs = [Path(args.multiview_root) / token for token in args.tokens]
        wanted = len(scene_dirs)
    else:
        # Degenerate scenes are rejected from their trajectory labels alone, so
        # `--skip-degenerate` never pays for a video decode it will throw away.
        # 19% of Multi-GT scenes are degenerate, so the candidate list needs real
        # slack over `wanted` or the compressor fit silently lands below full rank.
        scene_dirs = list(
            iter_ready_scenes(
                args.multiview_root,
                args.cache_root,
                limit=int(wanted * 1.5) + 8,
                require_distinct_labels=args.skip_degenerate,
            )
        )
    if not scene_dirs:
        raise FileNotFoundError(f"no ready Multi-GT scenes under {args.multiview_root}")

    log_cache: dict[str, list[dict]] = {}
    samples: list[MultiViewSample] = []
    skipped: list[dict[str, str]] = []
    for scene_dir in scene_dirs:
        if len(samples) >= wanted and not args.tokens:
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
        samples.append(sample)
    if not samples:
        raise RuntimeError(f"every candidate scene was skipped: {skipped[:5]}")
    log_cache.clear()
    train_count = max(1, len(samples) - args.eval_scenes) if args.eval_scenes else len(samples)
    train_count = min(train_count, len(samples))
    print(
        json.dumps(
            {
                "stage": "load",
                "scenes": [sample.scene_id for sample in samples],
                "train_scenes": [sample.scene_id for sample in samples[:train_count]],
                "eval_scenes": [sample.scene_id for sample in samples[train_count:]],
                "history_policies": [sample.metadata["history_policy"] for sample in samples],
                "gt_counts": [sample.metadata["gt_count"] for sample in samples],
                "labels": [sample.metadata["labels"] for sample in samples],
                "skipped": skipped[:10],
            }
        ),
        flush=True,
    )

    samples, teacher_audit = _build_future_targets(
        samples, args.dino_checkpoint, tuple(args.teacher_image_hw), device, args.target_dim, fit_count=train_count
    )
    print(json.dumps({"stage": "teacher_targets", **teacher_audit}), flush=True)
    samples = [_move_to_device(sample, device) for sample in samples]
    train_samples = samples[:train_count]
    eval_samples = samples[train_count:]
    batch_size = max(1, min(args.batch_size, len(train_samples)))
    batched = [
        _collate(train_samples[start : start + batch_size])
        for start in range(0, len(train_samples), batch_size)
    ]
    observation, targets = batched[0]
    # The PCA target is mean-centred, so this constant is the "learn nothing"
    # baseline the trained register has to beat.
    mean_target = torch.stack([sample.targets.future_register[0] for sample in train_samples]).mean(dim=(0, 1))
    cross_train = [train_samples[(index + 1) % len(train_samples)] for index in range(len(train_samples))]
    cross_eval = [eval_samples[(index + 1) % len(eval_samples)] for index in range(len(eval_samples))] if eval_samples else None

    model = DirectedPairDiT(
        d_model=args.target_dim,
        heads=args.heads,
        ffn_dim=4 * args.target_dim,
        depth=args.depth,
        num_scene_tokens=args.num_scene_tokens,
        dino_checkpoint=args.dino_checkpoint,
        dino_image_hw=tuple(args.image_hw),
        dino_lora_rank=args.lora_rank,
        dino_grad_checkpointing=args.grad_checkpointing,
        wan_backbone_path=args.wan_backbone if Path(args.wan_backbone).exists() else None,
    ).to(device)
    model_audit = {
        "token_layout": model.token_layout(),
        "scene_tokens": int(model.scene_tokens),
        "action_tokens": int(model.action_tokens),
        "register_tokens": int(model.register_tokens),
        "wan_backbone": model.wan_backbone_info,
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
    # Stage 1 itself never touches the anchor codebook: the register is conditioned
    # on the scene's real GT trajectories and paired by row index.  The codebook is
    # built here only for `--check-scorer`, which exercises the Stage-2 anchor path.
    modes = ImmutableModeBuffer().subset(args.candidates).to(device)
    initializer = StructuredSourceInitializer(jitter_std=0.0)
    from nvs_v6_wan_e2e.scorer import ScratchDrivoRScorer

    scorer = ScratchDrivoRScorer(scene_tokens=model.scene_tokens).to(device)
    optimizer, _ = build_optimizers(model, scorer, stage=1, lr=args.lr)
    trainable = {name: parameter for name, parameter in model.named_trainable_parameters()}
    model_audit["trainable_parameters"] = sum(parameter.numel() for parameter in trainable.values())
    model_audit["trainable_tensors"] = len(trainable)
    groups = parameter_groups(model)
    norms_before = {key: _parameter_norm(value) for key, value in groups.items()}
    print(json.dumps({"stage": "model", **model_audit}), flush=True)

    future_before = _future_fit_metrics(train_samples, model, mean_target, cross_train)
    eval_before = (
        _future_fit_metrics(eval_samples, model, mean_target, cross_eval) if eval_samples else None
    )
    print(
        json.dumps(
            {
                "stage": 1,
                "phase": "before",
                "train": {key: value for key, value in future_before.items() if key != "scenes"},
                "eval": None if eval_before is None else {key: value for key, value in eval_before.items() if key != "scenes"},
            }
        ),
        flush=True,
    )
    phi_queries_before = model.phi_time_queries.detach().clone()

    started = time.time()
    history = []
    supervised_rows = []
    for step in range(args.steps):
        step_observation, step_targets = batched[step % len(batched)]
        result = stage1_step(step_observation, step_targets, model, optimizer)
        history.append(float(result.loss_future))
        supervised_rows.append(int(result.supervised_rows))
        if (step + 1) % max(1, args.steps // 10) == 0 or step == args.steps - 1:
            print(
                json.dumps(
                    {
                        "stage": 1,
                        "step": step + 1,
                        "loss_future": history[-1],
                        "mean_loss_future": float(np.mean(history)),
                        "batch_scenes": len(step_observation.ego_history),
                        "supervised_gt_rows": supervised_rows[-1],
                        "elapsed_s": round(time.time() - started, 1),
                    }
                ),
                flush=True,
            )
    future_after = _future_fit_metrics(train_samples, model, mean_target, cross_train)
    eval_after = (
        _future_fit_metrics(eval_samples, model, mean_target, cross_eval) if eval_samples else None
    )
    norms_after = {key: _parameter_norm(value) for key, value in parameter_groups(model).items()}
    phi_query_delta = float((model.phi_time_queries.detach() - phi_queries_before).float().norm())

    scorer_audit = (
        _scorer_audit(model, observation, targets, modes, initializer, device) if args.check_scorer else None
    )

    summary = {
        "device": str(device),
        "stage": 1,
        "steps": args.steps,
        "candidates": args.candidates,
        "lr": args.lr,
        "stage1_pairing": "identity_one_register_row_per_gt",
        "stage1_register_conditioning": "metric_gt_trajectory_no_mode_code",
        "scene_ids": [sample.scene_id for sample in samples],
        "gt_counts": [sample.metadata["gt_count"] for sample in samples],
        "history_policies": [sample.metadata["history_policy"] for sample in samples],
        "degenerate_multi_gt": [sample.metadata["degenerate_multi_gt"] for sample in samples],
        "paired_pdms": [sample.metadata["paired_pdms"] for sample in samples],
        "skipped_scenes": skipped[:20],
        "model": model_audit,
        "teacher_targets": teacher_audit,
        "loss_future_first": history[0],
        "loss_future_last": history[-1],
        "loss_future_history": history,
        "loss_decreased": history[-1] < history[0],
        "all_losses_finite": bool(np.isfinite(history).all()),
        "supervised_gt_rows_per_step": supervised_rows,
        "future_before": future_before,
        "future_after": future_after,
        "eval_before": eval_before,
        "eval_after": eval_after,
        "batch_size": batch_size,
        "batches": len(batched),
        "register_verdict": _register_verdict(future_after, eval_after),
        "trajectory_conditioning_before": {
            key: future_before[key]
            for key in ("mean_paired_future_l1", "mean_scene_mean_target_l1", "mean_trajectory_blind_l1",
                        "mean_trajectory_conditioning_fidelity", "mean_trajectory_response_alignment")
        },
        "trajectory_conditioning_after": {
            key: future_after[key]
            for key in ("mean_paired_future_l1", "mean_scene_mean_target_l1", "mean_trajectory_blind_l1",
                        "mean_trajectory_conditioning_fidelity", "mean_trajectory_response_alignment")
        },
        "parameter_norms_before": norms_before,
        "parameter_norms_after": norms_after,
        "parameter_groups_updated": {key: norms_after[key] != norms_before[key] for key in norms_before},
        "phi_time_query_update_l2": phi_query_delta,
        "scorer_audit": scorer_audit,
        "elapsed_s": time.time() - started,
    }
    (output_dir / "stage1_multiview_result.json").write_text(json.dumps(summary, indent=2) + "\n")
    torch.save({"model": model.state_dict(), "stage": 1, "scene_ids": summary["scene_ids"]}, output_dir / "stage1_checkpoint.pt")
    print(
        json.dumps(
            {
                "stage": 1,
                "phase": "after",
                "train": {key: value for key, value in future_after.items() if key != "scenes"},
                "eval": None if eval_after is None else {key: value for key, value in eval_after.items() if key != "scenes"},
                "register_verdict": summary["register_verdict"],
                "loss_future_first": history[0],
                "loss_future_last": history[-1],
                "parameter_groups_updated": summary["parameter_groups_updated"],
                "output": str(output_dir / "stage1_multiview_result.json"),
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
