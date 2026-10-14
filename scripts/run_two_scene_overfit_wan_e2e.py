from __future__ import annotations

import argparse
import gc
import json
import lzma
import pickle
import random
import time
from pathlib import Path

import numpy as np
import torch

from nvs_v6_wan_e2e.data import Observation, PairTargets
from nvs_v6_wan_e2e.losses import hungarian_xy, paired_future_l1
from nvs_v6_wan_e2e.modes import ImmutableModeBuffer, StructuredSourceInitializer
from nvs_v6_wan_e2e.model import DirectedPairDiT, TimmDinoV3Student, parameter_groups
from nvs_v6_wan_e2e.native_drivor import NativeTrainOracle
from nvs_v6_wan_e2e.rollout_data import PairedRolloutSample, load_paired_rollout_sample
from nvs_v6_wan_e2e.scorer import ScratchDrivoRScorer
from nvs_v6_wan_e2e.stages import build_optimizers, stage1_step, stage2_step
from nvs_v6_wan_e2e.teacher import Ordered2x2Compressor, TimmDinoV3Teacher


DEFAULT_ROLLOUT_ROOT = "/mnt/zhaozc_workspace/project/multitra2egoview/cosmos3/batch_native10/rollouts/single_gpu_test_gtfallback_v2"
DEFAULT_CACHE = "/mnt/zhaozc_workspace/official_DrivoR_clean/data_runtime/official_metric_cache_navtrain_20261005/metric_cache"
DEFAULT_DINO = "/mnt/LLM_weight/timm/vit_large_patch16_dinov3.lvd1689m/model.safetensors"
DEFAULT_SCENES = ("8f4a85d43ef351f9", "b00d83c03bbf5b47")


def _ordered_vectors(teacher_grid: torch.Tensor) -> torch.Tensor:
    _, _, height, width, _ = teacher_grid.shape
    vectors = []
    for chunk in range(4):
        regions = []
        for frame_index in (2 * chunk, 2 * chunk + 1):
            frame = teacher_grid[:, frame_index]
            for row in range(2):
                for col in range(2):
                    region = frame[:, row * height // 2 : (row + 1) * height // 2, col * width // 2 : (col + 1) * width // 2]
                    regions.append(region.mean(dim=(1, 2)))
        vectors.append(torch.cat(regions, dim=-1))
    return torch.stack(vectors, dim=1)


def _build_paired_teacher_targets(
    samples: list[PairedRolloutSample], checkpoint: str, device: torch.device, target_dim: int
) -> str:
    teacher = TimmDinoV3Teacher(checkpoint=checkpoint).to(device)
    grids = []
    for sample in samples:
        per_scene = []
        for target_index in range(sample.future_rgb.shape[1]):
            images = sample.future_rgb[:, target_index].to(device)
            per_scene.append(teacher(images).float().cpu())
        grids.append(torch.cat(per_scene, dim=0))
    del teacher
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    compressor = Ordered2x2Compressor(output_dim=target_dim)
    compressor.fit(
        torch.cat([_ordered_vectors(grid) for grid in grids], dim=0),
        source_hash="+".join(sample.scene_id for sample in samples),
        split="train",
    )
    for sample, grid in zip(samples, grids):
        sample.targets.future_register = compressor.transform(grid).unsqueeze(0)
    return compressor.fingerprint()


def _move_sample(sample: PairedRolloutSample, device: torch.device) -> PairedRolloutSample:
    for key, value in sample.observation.__dict__.items():
        if isinstance(value, torch.Tensor):
            setattr(sample.observation, key, value.to(device))
    for key, value in sample.targets.__dict__.items():
        if isinstance(value, torch.Tensor):
            setattr(sample.targets, key, value.to(device))
    sample.future_rgb = torch.empty(0)
    return sample


def _collate_samples(samples: list[PairedRolloutSample]) -> tuple[Observation, PairTargets]:
    """Pad only the GT axis; every scene still contributes all of its GT pairs."""
    observation = Observation(
        rgb_history=torch.cat([sample.observation.rgb_history for sample in samples], dim=0),
        history_valid=torch.cat([sample.observation.history_valid for sample in samples], dim=0),
        ego_history=torch.cat([sample.observation.ego_history for sample in samples], dim=0),
        route_command=torch.cat([sample.observation.route_command for sample in samples], dim=0),
        relative_ego_poses=torch.cat([sample.observation.relative_ego_poses for sample in samples], dim=0),
        camera_metadata=[sample.observation.camera_metadata for sample in samples],
    )
    max_gt = max(sample.targets.gt_trajectory.shape[1] for sample in samples)

    def pad(value: torch.Tensor, fill: float | bool = 0) -> torch.Tensor:
        padded = value.new_full((1, max_gt, *value.shape[2:]), fill)
        padded[:, : value.shape[1]] = value
        return padded

    targets = PairTargets(
        gt_trajectory=torch.cat([pad(sample.targets.gt_trajectory) for sample in samples], dim=0),
        gt_encoded=torch.cat([pad(sample.targets.gt_encoded) for sample in samples], dim=0),
        gt_valid=torch.cat([pad(sample.targets.gt_valid, False) for sample in samples], dim=0),
        future_register=torch.cat([pad(sample.targets.future_register) for sample in samples], dim=0),
        future_chunk_valid=torch.cat([pad(sample.targets.future_chunk_valid, False) for sample in samples], dim=0),
        future_quality=torch.cat([pad(sample.targets.future_quality) for sample in samples], dim=0),
        pair_ids=[sample.targets.pair_ids[0] for sample in samples],
    )
    return observation, targets


def _build_oracle(samples: list[PairedRolloutSample], cache_root: str) -> NativeTrainOracle:
    return NativeTrainOracle(samples, cache_root=cache_root)


def _parameter_norm(parameters) -> float:
    values = [parameter.detach().float().norm() for parameter in parameters]
    return float(torch.stack(values).norm()) if values else 0.0


def _vision_runtime_audit(
    model: DirectedPairDiT,
    observation: Observation,
    samples: list[PairedRolloutSample],
) -> dict[str, object]:
    student = model.eta.student
    if not isinstance(student, TimmDinoV3Student):
        raise TypeError("real overfit must use the checkpoint-backed TimmDinoV3Student")
    patch_size = student.backbone.patch_embed.patch_size
    if isinstance(patch_size, int):
        patch_size = (patch_size, patch_size)
    current = student.token_layout(model.eta.current_tokens)
    history = student.token_layout(model.eta.history_tokens)
    return {
        "rollout_tensor_before_dinov3": list(observation.rgb_history.shape),
        "batch": int(observation.rgb_history.shape[0]),
        "history_plus_current_frames": int(observation.rgb_history.shape[1]),
        "history_frames": int(observation.rgb_history.shape[1] - 1),
        "cameras": int(observation.rgb_history.shape[2]),
        "loaded_image_hw": list(observation.rgb_history.shape[-2:]),
        "source_image_hw": [sample.metadata.get("source_image_hw") for sample in samples],
        "dinov3_interpolate_hw": [student.image_size, student.image_size],
        "backbone_patch_size": [int(patch_size[0]), int(patch_size[1])],
        "image_divisible_by_patch": bool(
            student.image_size % int(patch_size[0]) == 0
            and student.image_size % int(patch_size[1]) == 0
        ),
        "current_frame_layout": current,
        "history_frame_layout": history,
        "view_aggregation": "mean_over_camera_axis_after_backbone",
        "time_embedding": "5 learned embeddings: 4 history + current",
        "visual_tokens": model.eta.current_tokens + 4 * model.eta.history_tokens,
        "ego_fused_tokens": 2,
        "final_scene_tokens": model.scene_tokens,
    }


def _trajectory_errors(predicted: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    xy = torch.cdist(predicted[..., :2].reshape(predicted.shape[0], -1), targets[..., :2].reshape(targets.shape[0], -1))
    nearest = xy.min(dim=1).values
    per_pair_ade = (predicted[:, None, :, :2] - targets[None, :, :, :2]).norm(dim=-1).mean(dim=-1)
    per_pair_fde = (predicted[:, None, -1, :2] - targets[None, :, -1, :2]).norm(dim=-1)
    return {
        "mean_candidate_min_ade_m": float(per_pair_ade.min(dim=1).values.mean()),
        "best_candidate_min_ade_m": float(per_pair_ade.min()),
        "best_candidate_min_fde_m": float(per_pair_fde.min()),
        "flattened_xy_nearest_mean": float(nearest.mean()),
    }


@torch.no_grad()
def _future_fit_metrics(
    samples: list[PairedRolloutSample],
    model: DirectedPairDiT,
) -> dict[str, object]:
    """Stage-1's identity-paired future objective plus a shuffled and a blind control.

    Row ``k`` of the register is conditioned on GT ``k``'s own metric trajectory and
    scored against GT ``k``'s compressed future, exactly as ``stage1_step`` trains it.
    ``shuffled_pair_l1`` scores row ``k`` against GT ``k+1`` and ``trajectory_blind_l1``
    re-runs the scene with every row conditioned on GT 0's trajectory, so an overfit
    that came from the row index or from the scene alone cannot pass for one that came
    from the trajectory.
    """

    was_training = model.training
    model.eval()
    rows = []
    for sample in samples:
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
        rows.append(
            {
                "scene_id": sample.scene_id,
                "gt_count": gt_count,
                "supervised_rows": gt_count,
                "paired_future_l1": weighted_error(predicted, own_target),
                "shuffled_pair_l1": (
                    weighted_error(predicted, torch.roll(own_target, -1, dims=0)) if gt_count > 1 else float("nan")
                ),
                "trajectory_blind_l1": (
                    weighted_error(blind_predicted, own_target) if gt_count > 1 else float("nan")
                ),
                "register_mode_std": float(predicted.std(dim=0).mean()),
            }
        )
    model.train(was_training)
    return {
        "mean_paired_future_l1": float(np.mean([row["paired_future_l1"] for row in rows])),
        "mean_shuffled_pair_l1": float(np.nanmean([row["shuffled_pair_l1"] for row in rows])),
        "mean_trajectory_blind_l1": float(np.nanmean([row["trajectory_blind_l1"] for row in rows])),
        "scenes": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None, help="deprecated alias that sets both stages")
    parser.add_argument("--stage1-epochs", type=int, default=50)
    parser.add_argument("--stage2-epochs", type=int, default=50)
    parser.add_argument("--steps-per-epoch", type=int, default=10)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--rollout-root", default=DEFAULT_ROLLOUT_ROOT)
    parser.add_argument("--cache-root", default=DEFAULT_CACHE)
    parser.add_argument("--dino-checkpoint", default=DEFAULT_DINO)
    parser.add_argument("--output-dir", default="artifacts/two_scene_overfit_10epoch")
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--scorer-only", action="store_true")
    parser.add_argument("--stop-selected-pdms", type=float, default=None)
    parser.add_argument("--use-all-native-tokens", action="store_true",
                        help="output all 201 DINOv3 native tokens per frame (5x201+2=1007 scene tokens)")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.epochs is not None:
        args.stage1_epochs = args.epochs
        args.stage2_epochs = args.epochs

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("the real two-scene overfit run requires the configured CUDA/PPU device")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rollout_root = Path(args.rollout_root)
    samples = [load_paired_rollout_sample(rollout_root / scene, args.cache_root) for scene in DEFAULT_SCENES]
    compressor_hash = _build_paired_teacher_targets(samples, args.dino_checkpoint, device, target_dim=256)
    samples = [_move_sample(sample, device) for sample in samples]
    batch_observation, batch_targets = _collate_samples(samples)
    oracle = _build_oracle(samples, args.cache_root)
    scene_ids = [sample.scene_id for sample in samples]

    model = DirectedPairDiT(
        d_model=256,
        heads=8,
        ffn_dim=1024,
        depth=args.depth,
        dino_checkpoint=args.dino_checkpoint,
        dino_model_name="vit_large_patch16_dinov3.lvd1689m",
        dino_lora_rank=16,
        dino_lora_alpha=16.0,
        use_all_native_tokens=args.use_all_native_tokens,
    ).to(device)
    vision_runtime_audit = _vision_runtime_audit(model, batch_observation, samples)
    print(json.dumps({"vision_runtime_audit": vision_runtime_audit}), flush=True)
    scorer = ScratchDrivoRScorer(
        d_model=256,
        heads=1,
        ffn_dim=1024,
        decoder_layers=4,
        scene_tokens=model.scene_tokens,
    ).to(device)
    if args.resume_checkpoint is not None:
        saved = torch.load(args.resume_checkpoint, map_location="cpu", mmap=True, weights_only=False)
        model.load_state_dict(saved["model"], strict=True)
        if saved.get("scorer_contract") != "scratch_drivor_nvs_dinov3_scene_v1":
            raise ValueError("resume checkpoint is not a scratch NVS-scene scorer checkpoint")
        scorer.load_state_dict(saved["scorer"], strict=True)
        del saved
    modes = ImmutableModeBuffer().subset(64).to(device)
    initializer = StructuredSourceInitializer(jitter_std=0.0)
    groups = parameter_groups(model)
    norms_before = {
        "eta": _parameter_norm(groups["eta"]),
        "phi": _parameter_norm(groups["phi"]),
        "theta": _parameter_norm(groups["theta"]),
        "psi": _parameter_norm(scorer.parameters()),
    }
    phi_time_queries_before = model.phi_time_queries.detach().clone()
    stage1_future_before = _future_fit_metrics(samples, model)

    # DrivoR scales its 5e-4 base LR by sqrt(global_batch/64).
    scorer_lr = 5e-4 * (len(samples) / 64.0) ** 0.5
    stage1_optimizer, _ = build_optimizers(model, scorer, stage=1, lr=2e-4, scorer_lr=scorer_lr)
    stage1_epochs = []
    stage2_epochs = []
    start = time.time()
    for epoch in range(args.stage1_epochs):
        losses = []
        for _ in range(args.steps_per_epoch):
            result = stage1_step(batch_observation, batch_targets, model, stage1_optimizer)
            losses.append(float(result.loss_future))
        rows = [
            {
                "scene_id": sample.scene_id,
                "gt_count": sample.metadata["gt_count"],
                "supervised_rows": int(sample.targets.gt_valid.sum()),
            }
            for sample in samples
        ]
        stage1_epochs.append(
            {
                "epoch": epoch + 1,
                "steps": args.steps_per_epoch,
                "mean_loss_future": float(np.mean(losses)),
                "last_loss_future": losses[-1],
                "batch_supervised_rows": int(result.supervised_rows),
                "scenes": rows,
            }
        )
        print(json.dumps({"stage": 1, **stage1_epochs[-1]}), flush=True)

    stage1_future_after = _future_fit_metrics(samples, model)
    stage1_future_audit = {
        "before": stage1_future_before,
        "after": stage1_future_after,
        "loss_decreased": stage1_future_after["mean_paired_future_l1"] < stage1_future_before["mean_paired_future_l1"],
        "paired_better_than_shuffled": stage1_future_after["mean_paired_future_l1"] < stage1_future_after["mean_shuffled_pair_l1"],
        "paired_better_than_trajectory_blind": (
            stage1_future_after["mean_paired_future_l1"] < stage1_future_after["mean_trajectory_blind_l1"]
        ),
        "phi_time_query_update_l2": float((model.phi_time_queries.detach() - phi_time_queries_before).float().norm()),
    }
    print(json.dumps({"stage": 1, "future_audit": stage1_future_audit}), flush=True)

    generator_optimizer, scorer_optimizer = build_optimizers(model, scorer, stage=2, lr=2e-4, scorer_lr=scorer_lr)
    if args.scorer_only:
        model.eval().requires_grad_(False)
        scorer.requires_grad_(True)
        mode_codes = modes.expand_for_batch(len(samples), device)
        with torch.no_grad():
            fixed_source = initializer.clean(batch_observation.ego_history, mode_codes)
            fixed_outputs = model.forward_pairs(batch_observation, fixed_source, mode_codes)
            fixed_oracle = oracle.evaluate(scene_ids, fixed_outputs.trajectory, batch_id="two-scene-scorer-only-fixed")
            fixed_assignment = hungarian_xy(fixed_outputs.trajectory, batch_targets.gt_trajectory, batch_targets.gt_valid)
            fixed_future_loss = float(paired_future_l1(fixed_outputs.predicted_register, batch_targets, fixed_assignment))
        for epoch in range(args.stage2_epochs):
            score_losses = []
            for _ in range(args.steps_per_epoch):
                scorer.train()
                scorer_optimizer.zero_grad(set_to_none=True)
                logits = scorer(fixed_outputs.scene_context, fixed_outputs.trajectory)
                loss_score = scorer.source_loss(logits, fixed_oracle)
                loss_score.backward()
                scorer_optimizer.step()
                score_losses.append(float(loss_score.detach()))
            scorer.eval()
            with torch.no_grad():
                logits = scorer(fixed_outputs.scene_context, fixed_outputs.trajectory)
                selected, _ = scorer.select(logits)
                selected_pdms = [
                    float(fixed_oracle.final_score[index, selected[index]])
                    for index in range(len(samples))
                ]
            rows = [
                {
                    "scene_id": sample.scene_id,
                    "gt_count": sample.metadata["gt_count"],
                    "matched": len(fixed_assignment[index].as_pairs()),
                    "oracle_final_mean": float(fixed_oracle.final_score[index].nanmean()),
                    "oracle_final_max": float(fixed_oracle.final_score[index].nan_to_num(-1.0).max()),
                    "selected_pdms": selected_pdms[index],
                    "oracle_failures": int(sum(1 for failure in fixed_oracle.failure_diagnostics if failure["scene_id"] == sample.scene_id)),
                }
                for index, sample in enumerate(samples)
            ]
            stage2_epochs.append(
                {
                    "epoch": epoch + 1,
                    "steps": args.steps_per_epoch,
                    "training_mode": "scorer_only_fixed_proposals",
                    "generator_loss_optimized": False,
                    "mean_loss_action": 0.0,
                    "mean_loss_future": fixed_future_loss,
                    "mean_loss_score": float(np.mean(score_losses)),
                    "mean_oracle_final": float(np.mean([row["oracle_final_mean"] for row in rows])),
                    "scenes": rows,
                }
            )
            print(json.dumps({"stage": 2, **stage2_epochs[-1]}), flush=True)
            if args.stop_selected_pdms is not None and all(
                pdms >= args.stop_selected_pdms for pdms in selected_pdms
            ):
                break
    else:
        for epoch in range(args.stage2_epochs):
            step_rows = []
            for step in range(args.steps_per_epoch):
                result = stage2_step(
                    batch_observation,
                    batch_targets,
                    scene_ids,
                    model,
                    scorer,
                    oracle,
                    generator_optimizer,
                    scorer_optimizer,
                    modes,
                    initializer,
                    batch_id=f"two-scene-e{epoch + 1}-s{step + 1}",
                )
                step_rows.append(result)
            rows = [
                {
                    "scene_id": sample.scene_id,
                    "gt_count": sample.metadata["gt_count"],
                    "matched": len(step_rows[-1].assignment[index].as_pairs()),
                    "oracle_final_mean": float(np.mean([row.oracle.final_score[index].nanmean().item() for row in step_rows])),
                    "oracle_final_max": float(max(row.oracle.final_score[index].nan_to_num(-1.0).max().item() for row in step_rows)),
                    "oracle_failures": int(sum(1 for row in step_rows for failure in row.oracle.failure_diagnostics if failure["scene_id"] == sample.scene_id)),
                }
                for index, sample in enumerate(samples)
            ]
            stage2_epochs.append(
                {
                    "epoch": epoch + 1,
                    "steps": args.steps_per_epoch,
                    "mean_loss_action": float(np.mean([float(row.loss_action) for row in step_rows])),
                    "mean_loss_future": float(np.mean([float(row.loss_future) for row in step_rows])),
                    "mean_loss_score": float(np.mean([float(row.loss_score) for row in step_rows])),
                    "mean_oracle_final": float(np.mean([row["oracle_final_mean"] for row in rows])),
                    "scenes": rows,
                }
            )
            print(json.dumps({"stage": 2, **stage2_epochs[-1]}), flush=True)

    generated = []
    selected_trajectories = []
    learned_selected_trajectories = []
    raw_scores = []
    scorer_logits = []
    evaluations = []
    clean_initializer = StructuredSourceInitializer(jitter_std=0.0)
    model.eval()
    scorer.eval()
    for sample in samples:
        with torch.no_grad():
            mode_codes = modes.expand_for_batch(1, device)
            source = clean_initializer.clean(sample.observation.ego_history, mode_codes)
            outputs = model.forward_pairs(sample.observation, source, mode_codes)
            logits = scorer(outputs.scene_context, outputs.trajectory)
            scorer_selected, scorer_utility = scorer.select(logits)
            oracle_result = oracle.evaluate([sample.scene_id], outputs.trajectory, batch_id="two-scene-final")
        oracle_scores = oracle_result.final_score[0]
        oracle_best = int(oracle_scores.nan_to_num(-1.0).argmax())
        learned_best = int(scorer_selected.item())
        generated.append(outputs.trajectory[0].detach().cpu())
        selected_trajectories.append(outputs.trajectory[0, oracle_best].detach().cpu())
        learned_selected_trajectories.append(outputs.trajectory[0, learned_best].detach().cpu())
        raw_scores.append(oracle_result.raw7[0].cpu())
        scorer_logits.append(logits[0].cpu())
        errors = _trajectory_errors(outputs.trajectory[0].detach().cpu(), sample.targets.gt_trajectory[0].detach().cpu())
        selected_target_ade = (
            outputs.trajectory[0, oracle_best, :, :2].detach().cpu()[None]
            - sample.targets.gt_trajectory[0, :, :, :2].detach().cpu()
        ).norm(dim=-1).mean(dim=-1).min()
        final_assignment = hungarian_xy(
            outputs.trajectory,
            sample.targets.gt_trajectory,
            sample.targets.gt_valid,
        )[0]
        matched_gt_ade = (
            outputs.trajectory[0, final_assignment.candidate_indices, :, :2]
            - sample.targets.gt_trajectory[0, final_assignment.gt_indices, :, :2]
        ).norm(dim=-1).mean(dim=-1)
        evaluations.append(
            {
                "scene_id": sample.scene_id,
                "matched_gt_mean_ade_m": float(matched_gt_ade.mean()),
                "matched_gt_max_ade_m": float(matched_gt_ade.max()),
                "matched_gt_ade_m": matched_gt_ade.detach().cpu().tolist(),
                "oracle_best_candidate": oracle_best,
                "oracle_best_pdms": float(oracle_scores[oracle_best]),
                "oracle_mean_pdms": float(oracle_scores.nanmean()),
                "learned_scorer_candidate": learned_best,
                "learned_scorer_candidate_pdms": float(oracle_scores[learned_best]),
                "learned_scorer_utility": float(scorer_utility[0, learned_best]),
                "learned_scorer_oracle_rank": int(
                    (oracle_scores.nan_to_num(-1.0) > oracle_scores[learned_best].nan_to_num(-1.0)).sum().item() + 1
                ),
                "learned_scorer_probabilities": logits[0, learned_best].sigmoid().cpu().tolist(),
                "oracle_best_scorer_probabilities": logits[0, oracle_best].sigmoid().cpu().tolist(),
                "scorer_matches_oracle": learned_best == oracle_best,
                "oracle_failures": oracle_result.failure_diagnostics,
                "selected_min_ade_m": float(selected_target_ade),
                "paired_target_pdms": sample.metadata["paired_pdms"],
                "paired_target_best_pdms": max(sample.metadata["paired_pdms"]),
                **errors,
            }
        )

    generated_tensor = torch.stack(generated)
    raw_score_tensor = torch.stack(raw_scores)
    scorer_logit_tensor = torch.stack(scorer_logits)
    np.save(output_dir / "generated_trajectories.npy", generated_tensor.numpy())
    np.save(output_dir / "selected_oracle_best_trajectories.npy", torch.stack(selected_trajectories).numpy())
    np.save(output_dir / "selected_learned_scorer_trajectories.npy", torch.stack(learned_selected_trajectories).numpy())
    np.save(output_dir / "official_raw7_scores.npy", raw_score_tensor.numpy())
    np.save(output_dir / "scorer_logits.npy", scorer_logit_tensor.numpy())
    topk = {
        sample.scene_id: [
            {"candidate": int(index), "pdms": float(raw_score_tensor[scene_index, index, 6])}
            for index in torch.argsort(raw_score_tensor[scene_index, :, 6].nan_to_num(-1.0), descending=True)[:10]
        ]
        for scene_index, sample in enumerate(samples)
    }
    (output_dir / "topk_official_pdms.json").write_text(json.dumps(topk, indent=2) + "\n")
    torch.save(
        {
            "model": model.state_dict(),
            "scorer": scorer.state_dict(),
            "stage1_epochs": len(stage1_epochs),
            "stage2_epochs": len(stage2_epochs),
            "scorer_only": args.scorer_only,
            "scene_ids": list(DEFAULT_SCENES),
            "compressor_hash": compressor_hash,
            "dino_checkpoint": args.dino_checkpoint,
            "scorer_contract": "scratch_drivor_nvs_dinov3_scene_v1",
        },
        output_dir / "checkpoint.pt",
    )
    norms_after = {
        "eta": _parameter_norm(groups["eta"]),
        "phi": _parameter_norm(groups["phi"]),
        "theta": _parameter_norm(groups["theta"]),
        "psi": _parameter_norm(scorer.parameters()),
    }
    summary = {
        "device": str(device),
        "stage1_epochs": len(stage1_epochs),
        "stage2_epochs": len(stage2_epochs),
        "requested_stage2_epochs": args.stage2_epochs,
        "scorer_only": args.scorer_only,
        "steps_per_epoch": args.steps_per_epoch,
        "optimizer_steps": {
            "stage1": len(stage1_epochs) * args.steps_per_epoch,
            "stage2": len(stage2_epochs) * args.steps_per_epoch,
        },
        "scorer_lr": scorer_lr,
        "scorer_lr_policy": "DrivoR base_lr * sqrt(global_batch_size / base_batch_size)",
        "scene_ids": list(DEFAULT_SCENES),
        "gt_counts": {sample.scene_id: sample.metadata["gt_count"] for sample in samples},
        "depth": args.depth,
        "candidate_count": 64,
        "teacher_checkpoint": args.dino_checkpoint,
        "compressor_hash": compressor_hash,
        "vision_runtime_audit": vision_runtime_audit,
        "scorer_contract": {
            "implementation": "ScratchDrivoRScorer",
            "initialization": "random_from_source_architecture",
            "checkpoint_loaded": False,
            "scene_kv": "detached NVS DINOv3 scene_context [B,66,256]",
            "ego_addition": "mean of final two NVS ego-fused Scene tokens",
            "native_dinov2_scene_cache": False,
            "native_ego_cache": False,
            "predicted_register_input": False,
        },
        "data_contract": {
            "student_observation": "initial_front.jpg repeated across five history slots",
            "teacher_only_targets": "trajectory_i/future.mp4 frames at 0.5s intervals",
            "trajectory_targets": "trajectory_i/ego_pose.npy sampled at 0.5s intervals",
            "online_oracle": args.cache_root,
        },
        "score_supervision_contract": {
            "proposal_shape": [len(samples), 64, 8, 3],
            "online_raw_label_shape": [len(samples), 64, 7],
            "online_six_head_shape": [len(samples), 64, 6],
            "multi_gt_shape": [len(samples), "K<=4", 8, 3],
            "hungarian_cost_shape": [len(samples), "K<=4", 64],
            "pdms_is_pairwise_64_by_k": False,
        },
        "stage1": stage1_epochs,
        "stage1_future_audit": stage1_future_audit,
        "stage2": stage2_epochs,
        "evaluation": evaluations,
        "parameter_norms_before": norms_before,
        "parameter_norms_after": norms_after,
        "parameter_updates": {key: norms_after[key] != norms_before[key] for key in norms_before},
        "metric_cache_audit": oracle.audit,
        "oracle_source": oracle.source_file,
        "score_feature_audit": scorer.feature_audit,
        "finite_losses": bool(
            np.isfinite([epoch["mean_loss_future"] for epoch in stage1_epochs]).all()
            and np.isfinite(
                [epoch[key] for epoch in stage2_epochs for key in ("mean_loss_action", "mean_loss_future", "mean_loss_score")]
            ).all()
        ),
        "elapsed_s": time.time() - start,
        "artifacts": {
            "checkpoint": str(output_dir / "checkpoint.pt"),
            "generated_trajectories": str(output_dir / "generated_trajectories.npy"),
            "selected_oracle_best_trajectories": str(output_dir / "selected_oracle_best_trajectories.npy"),
            "selected_learned_scorer_trajectories": str(output_dir / "selected_learned_scorer_trajectories.npy"),
            "official_raw7_scores": str(output_dir / "official_raw7_scores.npy"),
            "scorer_logits": str(output_dir / "scorer_logits.npy"),
            "topk_official_pdms": str(output_dir / "topk_official_pdms.json"),
        },
    }
    (output_dir / "result.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
