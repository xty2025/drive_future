"""Stage-1 training on the real /mnt/data/multi_view Multi-GT future videos.

The observation is DrivoR-shaped (five real front-camera history frames, the
7-wide ego status plus the four-way command, metric trajectories).  The Future
Register target is built from the generated ``trajectory_k/future.mp4`` frames by
the frozen DINOv3 Teacher and its CLS-query extractor, so Stage 1 trains the
Action DiT -- clean GT trajectory at ``sigma = 0``, future query tokens read out of
its last layer -- to predict, for each real GT trajectory of the scene, what the
world will look like under that trajectory.  The dataset supplies the pairing:
register row ``k`` is supervised by GT ``k``'s extracted future, so no anchor
codebook and no Hungarian assignment takes part.

Run with the repository's standard PYTHONPATH, e.g.::

    PYTHONPATH="$PWD/src:$PWD/vendor/DrivoR:$PWD/vendor/DrivoR/nuplan-devkit" \
      /mnt/zhaozc_workspace/envs/driov_dpo/bin/python scripts/run_stage1_multiview.py \
      --scenes 4 --steps 20 --check-scorer
"""

from __future__ import annotations

import argparse
import gc
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
from nvs_v6_wan_e2e.stage1_eval import (
    CONDITIONING_SUMMARY_KEYS,
    future_fit_metrics,
    register_verdict,
    trajectory_errors,
)
from nvs_v6_wan_e2e.stages import build_optimizers, stage1_step
from nvs_v6_wan_e2e.target_cache import (
    MIN_GT_SPREAD_M,
    is_trajectory_degenerate,
    multi_gt_spread_metres,
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
# was cut into.  The CLS query attends over that 29x52 grid, so the regions it can
# select are fractions of the frame for the same reason, which keeps target and
# scene tokens talking about the same image regions.
TEACHER_IMAGE_HW = NATIVE_FUTURE_IMAGE_HW


def _build_future_targets(
    samples: list[MultiViewSample],
    checkpoint: str,
    image_hw: tuple[int, int],
    device: torch.device,
    query_temperature: float = QUERY_TEMPERATURE,
    query_heads: int = QUERY_HEADS,
    fit_count: int | None = None,
) -> tuple[list[MultiViewSample], dict[str, object]]:
    """Prefix-query-extracted, standardized register targets, one ``[1,K,32,D]`` per scene.

    ``D`` is the Teacher's own ``feature_dim`` and is *reported*, not requested: the
    extractor returns a convex combination of patch tokens and the standardizer is
    diagonal, so nothing between the checkpoint and the register head can change the
    width.  A caller-supplied ``--target-dim`` would only be a way to get it wrong,
    which is why the model is built from ``audit["register_target_dim"]`` below.

    The 29x52 patch grids are reduced as they are produced and never retained -- a
    grid is 49 MB float32 per GT while its thirty-two query tokens are 128 KB -- and the
    Teacher is freed before the Student is built, so the two never share the device.
    """

    teacher = TimmDinoV3Teacher(checkpoint=checkpoint, image_hw=image_hw).to(device)
    extractor = FutureQueryExtractor(int(teacher.feature_dim), query_temperature, query_heads)
    blocks: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []
    for sample in samples:
        if sample.future_rgb is None:
            raise ValueError(f"{sample.scene_id}: future RGB must be loaded to build register targets")
        block, entropy = query_future_register(teacher, extractor, sample.future_rgb, device)
        blocks.append(block)
        entropies.append(entropy)
    feature_dim = int(teacher.feature_dim)
    grid_hw = [int(value) for value in teacher.grid_hw]
    del teacher
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    fit_blocks = blocks if fit_count is None else blocks[:fit_count]
    fit_rows = torch.cat([block.reshape(-1, feature_dim) for block in fit_blocks], dim=0)
    standardizer = TargetStandardizer(feature_dim, query_temperature, query_heads)
    standardizer.fit(
        fit_rows,
        source_hash="+".join(sample.scene_id for sample in samples[: len(fit_blocks)]),
        split="train",
    )
    updated = []
    for sample, block in zip(samples, blocks):
        register = standardizer.transform(block).unsqueeze(0)
        # The decoded frames have served their purpose; holding them would cost
        # 148 MB per scene at the futures' native 464x832.
        updated.append(replace(sample, future_rgb=None, targets=with_future_register(sample.targets, register)))
    fitted = len(fit_blocks)
    audit = {
        "teacher_checkpoint": checkpoint,
        "teacher_image_hw": [int(image_hw[0]), int(image_hw[1])],
        "teacher_grid_hw": grid_hw,
        "teacher_feature_dim": feature_dim,
        "register_target_dim": feature_dim,
        "query_temperature": float(query_temperature),
        "query_heads": int(query_heads),
        # Entropy is over every encoded frame, not just the fit scenes: it is a
        # property of the readout, and this script encodes too few scenes to split.
        "query_attention_entropy_mean": float(torch.cat(entropies).mean()),
        "query_attention_entropy_max": float(torch.cat(entropies).max()),
        "standardizer_fit_rows": int(fit_rows.shape[0]),
        "standardizer_fit_scenes": fitted,
        "standardizer_clamped_components": int(standardizer.state.clamped_components),
        "standardizer_fingerprint": standardizer.fingerprint(),
        "register_target_shape": list(updated[0].targets.future_register.shape),
    }
    # Quality is measured on the train subset's standardized targets, so the numbers
    # mean the same thing as the offline cache builder's.
    audit.update(
        measure_target_quality(
            [sample.targets.future_register[0] for sample in updated[:fitted]],
            [sample.targets.gt_trajectory[0] for sample in updated[:fitted]],
        )
    )
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
        "proposals_vs_multigt": trajectory_errors(
            outputs.trajectory[0].cpu(), targets.gt_trajectory[0, :gt_count].cpu()
        ),
        "scorer_parameters": sum(parameter.numel() for parameter in scorer.parameters()),
        "feature_audit": scorer.feature_audit,
    }


def _parameter_norm(parameters) -> float:
    values = [parameter.detach().float().norm() for parameter in parameters]
    return float(torch.stack(values).norm()) if values else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--multiview-root", default=DEFAULT_MULTIVIEW_ROOT)
    parser.add_argument("--log-root", default=DEFAULT_LOG_ROOT)
    parser.add_argument("--sensor-root", default=DEFAULT_SENSOR_ROOT)
    parser.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--scenes", type=int, default=4, help="training scenes; every one contributes K*8 rows to the target mean/std")
    parser.add_argument("--eval-scenes", type=int, default=0, help="extra scenes held out of training and of the standardizer fit")
    parser.add_argument("--batch-size", type=int, default=4, help="scenes per optimizer step; decoupled from --scenes")
    parser.add_argument("--tokens", nargs="*", default=None)
    parser.add_argument("--skip-degenerate", action="store_true", help="drop scenes whose Multi-GT candidates collapse, or whose K trajectories are closer than --min-gt-spread-m")
    parser.add_argument("--min-gt-spread-m", type=float, default=MIN_GT_SPREAD_M, help="same threshold the cached-DDP trainer uses; the two Stage-1 paths must agree on what a degenerate scene is or a number in one log does not mean what it means in the other")
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
    parser.add_argument("--target-dim", type=int, default=256, help="d_model: the DiT hidden width. The register target width is the Teacher's feature_dim and is reported in the audit, not requested here")
    parser.add_argument("--query-temperature", type=float, default=QUERY_TEMPERATURE, help="cosine-attention temperature of the CLS query readout; part of the target fingerprint")
    parser.add_argument("--query-heads", type=int, default=QUERY_HEADS, help="query heads the patch grid is split over; part of the target fingerprint")
    parser.add_argument("--check-scorer", action="store_true")
    parser.add_argument(
        "--allow-degenerate-query",
        action="store_true",
        help="run even when the query attention is close to uniform, i.e. the readout has become a mean pool",
    )
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
        # slack over `wanted` or the run silently trains on fewer scenes than asked.
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
        if args.skip_degenerate and is_trajectory_degenerate(
            sample.targets.gt_trajectory, sample.targets.gt_valid, args.min_gt_spread_m
        ):
            spread = multi_gt_spread_metres(sample.targets.gt_trajectory, sample.targets.gt_valid)
            skipped.append({"scene": scene_dir.name, "reason": f"gt_spread_{spread:.3f}_m"})
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
        samples,
        args.dino_checkpoint,
        tuple(args.teacher_image_hw),
        device,
        query_temperature=args.query_temperature,
        query_heads=args.query_heads,
        fit_count=train_count,
    )
    print(json.dumps({"stage": "teacher_targets", **teacher_audit}), flush=True)
    # The one failure mode the query readout has is silent.  If the attention comes out
    # uniform the extractor *is* the mean pool it replaced: every number below stays
    # finite, the loss still falls, and 81% of the GT-discriminating variance -- the
    # part living in the spatial pattern rather than the per-frame average -- is gone.
    # ``separation_ratio`` and ``trajectory_distance_pearson`` in the audit line above
    # say how much survived; the entropy says whether the mechanism ran at all.
    # Checked here rather than inside ``_build_future_targets`` so that audit line is
    # always emitted first.
    entropy_mean = float(teacher_audit["query_attention_entropy_mean"])
    if entropy_mean > MAX_QUERY_ATTENTION_ENTROPY and not args.allow_degenerate_query:
        raise ValueError(
            f"the query readout's attention is {entropy_mean:.4f} of uniform (worst frame "
            f"{float(teacher_audit['query_attention_entropy_max']):.4f}, ceiling "
            f"{MAX_QUERY_ATTENTION_ENTROPY}), so the target has degenerated into a mean pool over "
            f"{teacher_audit['teacher_grid_hw']} patches. Lower --query-temperature below "
            f"{args.query_temperature}, or pass --allow-degenerate-query to run the plumbing anyway."
        )
    samples = [_move_to_device(sample, device) for sample in samples]
    train_samples = samples[:train_count]
    eval_samples = samples[train_count:]
    batch_size = max(1, min(args.batch_size, len(train_samples)))
    batched = [
        _collate(train_samples[start : start + batch_size])
        for start in range(0, len(train_samples), batch_size)
    ]
    observation, targets = batched[0]
    # The standardized target has zero mean over the fit scenes, so this constant is the
    # "learn nothing" baseline the trained register has to beat.
    mean_target = torch.stack([sample.targets.future_register[0] for sample in train_samples]).mean(dim=(0, 1))
    cross_train = [train_samples[(index + 1) % len(train_samples)] for index in range(len(train_samples))]
    cross_eval = [eval_samples[(index + 1) % len(eval_samples)] for index in range(len(eval_samples))] if eval_samples else None

    register_target_dim = int(teacher_audit["register_target_dim"])
    model = DirectedPairDiT(
        d_model=args.target_dim,
        heads=args.heads,
        ffn_dim=4 * args.target_dim,
        depth=args.depth,
        register_target_dim=register_target_dim,
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
        # Recorded separately from d_model because they are no longer equal: the DiT
        # stays 256-wide while the future head emits the Teacher's own feature_dim.
        # Stage 2 reads this to rebuild the head at the right width.
        "register_target_dim": register_target_dim,
        "dit_token_order": ["scene", "action", "register"],
        "dit_sequence_tokens": int(model.scene_tokens + model.action_tokens + model.register_tokens),
        "self_attention_mask": "single_segment_block",
        "segment_visibility": [list(row) for row in model.theta.SEGMENT_VISIBILITY],
        "action_reads_register": bool(model.theta.SEGMENT_VISIBILITY[1][2]),
        "register_reads_action": bool(model.theta.SEGMENT_VISIBILITY[2][1]),
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
    # Per-group counts, because the split is what a doc number like "eta 9.8M /
    # theta 9.5M" actually claims, and it moves when a module is frozen: a frozen
    # parameter is skipped by ``parameter_groups`` entirely, so Stage 1's theta group
    # has no velocity head in it and the two counts no longer sum to the
    # ``trainable_parameters`` of an unfrozen build.
    model_audit["trainable_parameters_by_group"] = {
        key: sum(parameter.numel() for parameter in value) for key, value in groups.items()
    }
    norms_before = {key: _parameter_norm(value) for key, value in groups.items()}
    print(json.dumps({"stage": "model", **model_audit}), flush=True)

    future_before = future_fit_metrics(train_samples, model, mean_target, cross_train)
    eval_before = (
        future_fit_metrics(eval_samples, model, mean_target, cross_eval) if eval_samples else None
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
    future_queries_before = model.theta.future_queries.detach().clone()

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
    future_after = future_fit_metrics(train_samples, model, mean_target, cross_train)
    eval_after = (
        future_fit_metrics(eval_samples, model, mean_target, cross_eval) if eval_samples else None
    )
    norms_after = {key: _parameter_norm(value) for key, value in parameter_groups(model).items()}
    future_query_delta = float((model.theta.future_queries.detach() - future_queries_before).float().norm())

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
        "stage1_register_conditioning": "clean_gt_trajectory_at_sigma_zero_zeroed_mode_code",
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
        "register_verdict": register_verdict(future_after, eval_after),
        "trajectory_conditioning_before": {key: future_before[key] for key in CONDITIONING_SUMMARY_KEYS},
        "trajectory_conditioning_after": {key: future_after[key] for key in CONDITIONING_SUMMARY_KEYS},
        "parameter_norms_before": norms_before,
        "parameter_norms_after": norms_after,
        "parameter_groups_updated": {key: norms_after[key] != norms_before[key] for key in norms_before},
        "future_query_update_l2": future_query_delta,
        "scorer_audit": scorer_audit,
        "elapsed_s": time.time() - started,
    }
    (output_dir / "stage1_multiview_result.json").write_text(json.dumps(summary, indent=2) + "\n")
    # ``register_target_dim`` rides in the payload because it is a property of the
    # Teacher checkpoint these weights were trained against, not of a command line:
    # Stage 2 has to rebuild a future head of exactly this width before it can load
    # them, and d_model no longer tells it what that is.
    torch.save(
        {
            "model": model.state_dict(),
            "stage": 1,
            "register_target_dim": register_target_dim,
            "scene_ids": summary["scene_ids"],
        },
        output_dir / "stage1_checkpoint.pt",
    )
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
