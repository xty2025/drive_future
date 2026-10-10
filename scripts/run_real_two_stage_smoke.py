from __future__ import annotations

import argparse
import gc
import json
import lzma
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from nvs_v6.data import PairTargets
from nvs_v6.modes import ImmutableModeBuffer, StructuredSourceInitializer
from nvs_v6.model import DirectedPairDiT, parameter_groups
from nvs_v6.real_data import RealNavsimSample, load_real_navsim_sample
from nvs_v6.scorer import ScratchDrivoRScorer
from nvs_v6.stages import build_optimizers, stage1_step, stage2_step
from nvs_v6.teacher import Ordered2x2Compressor, TimmDinoV3Teacher
from nvs_v6.oracle import DrivoRMetricCacheOracle


DEFAULT_LOG = "/mnt/data/navsim/trainval_navsim_logs/trainval/2021.08.17.17.17.01_veh-45_00207_00594.pkl"
DEFAULT_SENSOR = "/mnt/data/navsim/trainval_full_sensor_blobs/trainval"
DEFAULT_MULTIGT = "/mnt/zhaozc_workspace/project/drivezero_goal_gen/outputs/navsim_multigt_0p5s_8step_16gpu"
DEFAULT_CACHE = "/mnt/zhaozc_workspace/project/DrivoR_multigt_kvcache_final_20261004/data_runtime/train_metric_cache"
DEFAULT_DINO = "/mnt/LLM_weight/timm/vit_large_patch16_dinov3.lvd1689m/model.safetensors"
DRIVOR_ROOT = "/mnt/zhaozc_workspace/DrivoR_official_clean_20261005"
NUPLAN_ROOT = f"{DRIVOR_ROOT}/nuplan-devkit"


def parameter_norm(parameters) -> float:
    values = [parameter.detach().float().norm() for parameter in parameters if parameter.requires_grad]
    return float(torch.stack(values).norm()) if values else 0.0


def _ordered_vectors(teacher_grid: torch.Tensor) -> torch.Tensor:
    if teacher_grid.ndim != 5 or teacher_grid.shape[1] != 8:
        raise ValueError("teacher_grid must be [B,8,H,W,D]")
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


def build_targets_with_teacher(sample: RealNavsimSample, checkpoint: str, device: torch.device, target_dim: int) -> PairTargets:
    teacher = TimmDinoV3Teacher(checkpoint=checkpoint).to(device)
    teacher_grid = teacher(sample.future_rgb.to(device)).float().cpu()
    del teacher
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    ordered = _ordered_vectors(teacher_grid)
    compressor = Ordered2x2Compressor(output_dim=target_dim)
    compressor.fit(ordered, source_hash=sample.frame_token, split="train")
    target_register = compressor.transform(teacher_grid)
    targets = sample.targets
    targets.future_register = target_register.unsqueeze(1).repeat(1, targets.gt_trajectory.shape[1], 1, 1)
    targets.future_chunk_valid = torch.ones_like(targets.future_chunk_valid)
    targets.future_quality = torch.ones_like(targets.future_quality)
    return targets


def build_official_drivor_oracle(sample: RealNavsimSample, cache_root: str) -> DrivoRMetricCacheOracle:
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import pdm_score
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

    sampling = TrajectorySampling(num_poses=8, time_horizon=4.0, interval_length=0.5)
    simulator = PDMSimulator(sampling)
    scorer = PDMScorer(sampling)
    cache_root_path = Path(cache_root)

    def loader(scene_id: str):
        path = cache_root_path / sample.log_name / "unknown" / scene_id / "metric_cache.pkl"
        with lzma.open(path, "rb") as handle:
            return pickle.load(handle)

    def evaluator(_scene_id: str, proposal: torch.Tensor, metric_cache):
        trajectory = Trajectory(poses=proposal.detach().cpu().numpy().astype(np.float32), trajectory_sampling=sampling)
        result = pdm_score(metric_cache, trajectory, sampling, simulator, scorer)
        return [
            result.no_at_fault_collisions,
            result.drivable_area_compliance,
            result.ego_progress,
            result.time_to_collision_within_bound,
            result.comfort,
            result.driving_direction_compliance,
            result.score,
        ]

    return DrivoRMetricCacheOracle(evaluator=evaluator, metric_cache_loader=loader, evaluator_version="official_navsim_pdm_score")


def move_sample(sample: RealNavsimSample, device: torch.device) -> RealNavsimSample:
    sample.observation = type(sample.observation)(
        **{
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in sample.observation.__dict__.items()
        }
    )
    sample.targets = type(sample.targets)(
        **{
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in sample.targets.__dict__.items()
        }
    )
    return sample


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--candidates", type=int, default=64)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--frame-index", type=int, default=755)
    parser.add_argument("--output", default="artifacts/real_two_stage_smoke.json")
    parser.add_argument("--log", default=DEFAULT_LOG)
    parser.add_argument("--sensor-root", default=DEFAULT_SENSOR)
    parser.add_argument("--multigt-root", default=DEFAULT_MULTIGT)
    parser.add_argument("--cache-root", default=DEFAULT_CACHE)
    parser.add_argument("--dino-checkpoint", default=DEFAULT_DINO)
    args = parser.parse_args()

    if args.candidates != 64:
        raise ValueError("v6 online DrivoR training oracle requires exactly 64 current candidates")
    torch.manual_seed(17)
    np.random.seed(17)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample = load_real_navsim_sample(
        args.log,
        args.sensor_root,
        args.multigt_root,
        frame_index=args.frame_index,
        cache_root=args.cache_root,
    )
    targets = build_targets_with_teacher(sample, args.dino_checkpoint, device, target_dim=256)
    sample.targets = targets
    sample = move_sample(sample, device)
    model = DirectedPairDiT(
        d_model=256,
        heads=8,
        ffn_dim=1024,
        depth=args.depth,
        dino_checkpoint=args.dino_checkpoint,
        dino_model_name="vit_large_patch16_dinov3.lvd1689m",
        dino_lora_rank=16,
        dino_lora_alpha=16.0,
    ).to(device)
    scorer = ScratchDrivoRScorer(
        d_model=256,
        heads=1,
        ffn_dim=1024,
        decoder_layers=4,
    ).to(device)
    groups = parameter_groups(model)
    norms_before = {
        "eta": parameter_norm(groups["eta"]),
        "phi": parameter_norm(groups["phi"]),
        "theta": parameter_norm(groups["theta"]),
        "psi": parameter_norm(scorer.parameters()),
    }
    stage1_optimizer, _ = build_optimizers(model, scorer, stage=1, lr=2e-4, scorer_lr=2e-4)
    modes = ImmutableModeBuffer().subset(args.candidates).to(device)
    initializer = StructuredSourceInitializer(jitter_std=0.02)
    oracle = build_official_drivor_oracle(sample, args.cache_root)
    stage1_logs = []
    stage2_logs = []
    start = time.time()
    for step in range(args.steps):
        result = stage1_step(sample.observation, sample.targets, model, stage1_optimizer, modes, initializer)
        stage1_logs.append({"step": step, "loss_future": float(result.loss_future), "matched": len(result.assignment[0].as_pairs())})
    generator_optimizer, scorer_optimizer = build_optimizers(
        model, scorer, stage=2, lr=2e-4, scorer_lr=2e-4
    )
    for step in range(args.steps):
        result = stage2_step(
            sample.observation,
            sample.targets,
            [sample.scene_id],
            model,
            scorer,
            oracle,
            generator_optimizer,
            scorer_optimizer,
            modes,
            initializer,
            batch_id=f"real-smoke-{step}",
        )
        stage2_logs.append(
            {
                "step": step,
                "loss_action": float(result.loss_action),
                "loss_future": float(result.loss_future),
                "loss_score": float(result.loss_score),
                "oracle_failures": len(result.oracle.failure_diagnostics),
                "oracle_final_mean": float(result.oracle.final_score.nanmean()),
                "matched": len(result.assignment[0].as_pairs()),
            }
        )
    with torch.no_grad():
        source = StructuredSourceInitializer(jitter_std=0.0).clean(sample.observation.ego_history, modes.expand_for_batch(1, device))
        outputs = model.forward_pairs(sample.observation, source, modes.expand_for_batch(1, device))
        logits = scorer(outputs.scene_context, outputs.trajectory, outputs.predicted_register)
        selected, utility = scorer.select(logits)
    norms_after = {
        "eta": parameter_norm(groups["eta"]),
        "phi": parameter_norm(groups["phi"]),
        "theta": parameter_norm(groups["theta"]),
        "psi": parameter_norm(scorer.parameters()),
    }
    result = {
        "device": str(device),
        "environment": "/mnt/zhaozc_workspace/envs/driov_dpo",
        "python": "3.10+cu11.8",
        "real_navsim": True,
        "scene_id": sample.scene_id,
        "log_name": sample.log_name,
        "frame_index": sample.frame_index,
        "candidate_count": args.candidates,
        "steps_per_stage": args.steps,
        "depth": args.depth,
        "teacher_checkpoint": args.dino_checkpoint,
        "metric_cache": str(sample.cache_path),
        "scorer_contract": scorer.feature_audit,
        "vision_runtime_audit": {
            "rollout_tensor_before_dinov3": list(sample.observation.rgb_history.shape),
            "source_image_hw": sample.metadata["source_image_hw"],
            "loaded_image_hw": list(sample.observation.rgb_history.shape[-2:]),
            "cameras": int(sample.observation.rgb_history.shape[2]),
            "history_frames": int(sample.observation.rgb_history.shape[1] - 1),
            "current_frame_layout": model.eta.student.token_layout(32),
            "history_frame_layout": model.eta.student.token_layout(8),
            "final_scene_tokens": 66,
        },
        "stage1": stage1_logs,
        "stage2": stage2_logs,
        "finite_losses": all(np.isfinite(list(item.values())[1:]).all() for item in stage1_logs + stage2_logs),
        "parameter_norms_before": norms_before,
        "parameter_norms_after": norms_after,
        "parameter_updates": {key: norms_after[key] != norms_before[key] for key in norms_before},
        "score_shape": list(logits.shape),
        "selected_candidate": int(selected.item()),
        "selected_utility": float(utility[0, selected.item()]),
        "selection_is_finite": bool(torch.isfinite(utility).all()),
        "elapsed_s": time.time() - start,
        "sample_metadata": sample.metadata,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
