"""Full-navtrain SPLIT training for NVS v6 (DDP, single-node 16-GPU).

Two stages only, matching the proven WAN sibling: first train the trajectory
(generator), then train the scorer.  The register/future branch (phi) is NOT a
training target -- it is frozen for the whole run.  phi still executes forward
inside ``_directed_pass`` to supply *detached* action-attention memory, but it
receives no gradient and is excluded from the optimizer.

Phase A (Stage 1) - trajectory generation.  Train eta + theta with Action
(residual-velocity MSE against the multi-GT trajectory via Hungarian matching).
NO oracle, NO scorer, phi frozen.  The generator learns to produce 64 candidate
trajectories per scene.

Freeze + Cache build - freeze every generator parameter, run the frozen model
once over the whole dataset (rank-sharded), persist per scene: scene_context
K/V, 64 proposals, DrivoR oracle labels (official get_sub_score), and the
padded multi-GT trajectory.  The expensive PDM oracle runs EXACTLY ONCE per
scene.

Phase B (Stage 2) - scorer-only.  Train psi (ScratchDrivoRScorer, random init)
against the stationary cached context/proposals/labels.  No generator, no
oracle, no image IO.  Matches the scorer_only_fixed_proposals approach verified
in the two-scene overfit experiment (PID 1663817): generator frozen, candidates
computed once, only DrivoRLoss.score_loss updates psi.  Because the cache also
stores GT, Phase B reports selected/oracle-best GT-ADE alongside the score
loss.

Usage:
    python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=16 \
        scripts/run_navtrain_v6_twophase_ddp.py [OPTIONS]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from navsim.agents.drivoR.preference_module import mine_pair

from nvs_v6.data import Observation, PairTargets
from nvs_v6.losses import (
    assert_disjoint_parameter_groups,
    hungarian_xy,
    paired_residual_velocity_mse,
)
from nvs_v6.model import DirectedPairDiT, parameter_groups
from nvs_v6.modes import ImmutableModeBuffer, StructuredSourceInitializer
from nvs_v6.oracle import HEAD_INDICES, DrivoRBatchMetricCacheOracle, OracleLabels
from nvs_v6.real_data import load_real_navsim_sample
from nvs_v6.scorer import ScratchDrivoRScorer
from nvs_v6.selection import aggregate_score_logits
from nvs_v6.stages import _set_generator_trainability, _set_scorer_trainability


# --------------------------------------------------------------------------- #
# Data roots
# --------------------------------------------------------------------------- #
def _env(name: str, default: str) -> str:
    return os.environ.get(name) or default


DEFAULT_MULTIGT = _env("NVS_MULTIGT_ROOT", "/mnt/zhaozc_workspace/project/drivezero_goal_gen/outputs/navsim_multigt_0p5s_8step_16gpu")
DEFAULT_SENSOR = _env("NVS_SENSOR_ROOT", "/mnt/data/navsim/trainval_all/trainval_sensor_blobs/trainval")
DEFAULT_LOG_ROOT = _env("NVS_LOG_ROOT", "/mnt/data/navsim/trainval_navsim_logs/trainval")
DEFAULT_CACHE = _env("NVS_CACHE_ROOT", "/mnt/zhaozc_workspace/official_DrivoR_clean/data_runtime/official_metric_cache_navtrain_20261005/metric_cache")
DEFAULT_DINO = _env("NVS_DINO_CHECKPOINT", "/mnt/LLM_weight/timm/vit_large_patch16_dinov3.lvd1689m/model.safetensors")
DEFAULT_OUTPUT = _env("NVS_OUTPUT_DIR", "artifacts/navtrain_v6_twophase_ddp")
ORACLE_VERSION = "drivor_get_sub_score_navtrain_20261005"


# --------------------------------------------------------------------------- #
# Distributed helpers
# --------------------------------------------------------------------------- #
@dataclass
class Dist:
    rank: int
    world: int
    local_rank: int
    device: torch.device
    enabled: bool

    @property
    def main(self) -> bool:
        return self.rank == 0


def setup_distributed() -> Dist:
    has_env = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/PPU device required")
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    enabled = False
    if has_env and world > 0:
        dist.init_process_group(backend="nccl", timeout=timedelta(hours=3))
        enabled = True
    return Dist(rank=rank, world=world if has_env else 1, local_rank=local_rank, device=device, enabled=enabled)


def all_reduce_sum(value: torch.Tensor, dctx: Dist) -> torch.Tensor:
    if dctx.enabled:
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


def all_reduce_min(value: torch.Tensor, dctx: Dist) -> torch.Tensor:
    if dctx.enabled:
        dist.all_reduce(value, op=dist.ReduceOp.MIN)
    return value


def log0(dctx: Dist, payload: dict[str, Any]) -> None:
    if dctx.main:
        print(json.dumps(payload), flush=True)


def barrier(dctx: Dist) -> None:
    if dctx.enabled:
        dist.barrier()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Scene index (same logic as wan twophase)
# --------------------------------------------------------------------------- #
def _read_sidecar(path: Path, meta: dict[str, str]) -> dict | None:
    try:
        if not path.exists():
            return None
        with path.open() as handle:
            blob = json.load(handle)
        if not isinstance(blob, dict) or blob.get("_meta") != meta:
            return None
        return blob.get("index")
    except Exception:
        return None


def _write_sidecar(path: Path, meta: dict[str, str], index: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
        with tmp.open("w") as handle:
            json.dump({"_meta": meta, "index": index}, handle)
        os.replace(tmp, path)
    except Exception:
        pass


def load_token_cache_index(cache_root: str, index_dir: Path, meta: dict[str, str]) -> dict[str, tuple[str, Path]]:
    sidecar = index_dir / "_v6_token_cache_index.json"
    cached = _read_sidecar(sidecar, meta)
    if cached:
        return {token: (log_name, Path(cache_path)) for token, (log_name, cache_path) in cached.items()}
    index: dict[str, tuple[str, Path]] = {}
    nofuture = Path(cache_root) / "_nofuture_index.json"
    raw = None
    try:
        if nofuture.exists():
            with nofuture.open() as handle:
                raw = json.load(handle)
    except Exception:
        raw = None
    if isinstance(raw, dict) and raw:
        for token, value in raw.items():
            try:
                log_name, cache_path = value
                index[token] = (log_name, Path(cache_path))
            except Exception:
                continue
    else:
        root = Path(cache_root)
        for log_dir in root.iterdir():
            unknown = log_dir / "unknown"
            if not unknown.is_dir():
                continue
            for token_dir in unknown.iterdir():
                cache_path = token_dir / "metric_cache.pkl"
                if cache_path.exists():
                    index[token_dir.name] = (log_dir.name, cache_path)
    _write_sidecar(sidecar, meta, {token: [log_name, str(cp)] for token, (log_name, cp) in index.items()})
    return index


def build_frame_index(
    token_cache_index: dict[str, tuple[str, Path]],
    log_root: str,
    index_dir: Path,
    meta: dict[str, str],
    limit_tokens: int | None,
    write_sidecar: bool,
) -> dict[str, int]:
    sidecar = index_dir / "_v6_frame_index.json"
    cached = _read_sidecar(sidecar, meta)
    if cached:
        return {token: int(index) for token, index in cached.items()}
    by_log: dict[str, list[str]] = {}
    for token, (log_name, _cache) in token_cache_index.items():
        by_log.setdefault(log_name, []).append(token)
    frame_index: dict[str, int] = {}
    truncated = False
    for log_name in sorted(by_log):
        log_path = Path(log_root) / f"{log_name}.pkl"
        if not log_path.exists():
            continue
        try:
            with log_path.open("rb") as handle:
                frames = pickle.load(handle)
        except Exception:
            continue
        total = len(frames)
        token_to_index = {str(frame.get("token")): index for index, frame in enumerate(frames)}
        for token in by_log[log_name]:
            index = token_to_index.get(token)
            if index is not None and index >= 4 and index + 8 < total:
                frame_index[token] = index
        if limit_tokens is not None and len(frame_index) >= limit_tokens:
            truncated = True
            break
    if write_sidecar and not truncated:
        _write_sidecar(sidecar, meta, frame_index)
    return frame_index


@dataclass
class Candidate:
    token: str
    log_name: str
    cache_path: str
    frame_index: int


def build_scene_index(
    multigt_root: str, cache_root: str, log_root: str,
    index_dir: Path, limit_scenes: int | None,
) -> list[Candidate]:
    meta = {"multigt": multigt_root, "cache": cache_root, "log": log_root}
    token_cache_index = load_token_cache_index(cache_root, index_dir, meta)
    multigt_navtrain = Path(multigt_root) / "navtrain"
    multigt_tokens = set(os.listdir(multigt_navtrain))
    limit_tokens = (limit_scenes * 4 + 64) if limit_scenes else None
    frame_index = build_frame_index(
        token_cache_index, log_root, index_dir, meta,
        limit_tokens=limit_tokens, write_sidecar=not limit_scenes,
    )
    resolved = sorted(set(frame_index) & multigt_tokens & set(token_cache_index))
    candidates: list[Candidate] = []
    for token in resolved:
        log_name, cache_path = token_cache_index[token]
        candidates.append(Candidate(token=token, log_name=log_name, cache_path=str(cache_path), frame_index=frame_index[token]))
        if limit_scenes and len(candidates) >= limit_scenes:
            break
    return candidates


# --------------------------------------------------------------------------- #
# Dataset / collate
# --------------------------------------------------------------------------- #
class NavtrainSceneDataset(Dataset):
    def __init__(self, candidates: list[Candidate], sensor_root: str,
                 multigt_root: str, cache_root: str, log_root: str, max_retry: int = 12) -> None:
        self.candidates = candidates
        self.sensor_root = sensor_root
        self.multigt_root = multigt_root
        self.cache_root = cache_root
        self.log_root = log_root
        self.max_retry = max_retry

    def __len__(self) -> int:
        return len(self.candidates)

    def __getitem__(self, index: int) -> dict[str, Any]:
        total = len(self.candidates)
        last_error: Exception | None = None
        for attempt in range(self.max_retry):
            candidate = self.candidates[(index + attempt) % total]
            try:
                sample = load_real_navsim_sample(
                    Path(self.log_root) / f"{candidate.log_name}.pkl",
                    self.sensor_root, self.multigt_root,
                    frame_index=candidate.frame_index, cache_root=self.cache_root,
                )
            except Exception as error:
                last_error = error
                continue
            obs, tgt = sample.observation, sample.targets
            return {
                "rgb_history": obs.rgb_history[0],
                "history_valid": obs.history_valid[0],
                "ego_history": obs.ego_history[0],
                "route_command": obs.route_command[0],
                "relative_ego_poses": obs.relative_ego_poses[0],
                "gt_trajectory": tgt.gt_trajectory[0],
                "gt_encoded": tgt.gt_encoded[0],
                "gt_valid": tgt.gt_valid[0],
                "future_register": tgt.future_register[0],
                "future_chunk_valid": tgt.future_chunk_valid[0],
                "future_quality": tgt.future_quality[0],
                "scene_id": candidate.token,
            }
        raise RuntimeError(f"could not load any scene near index {index}: {last_error!r}")


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    size = len(batch)
    max_gt = max(int(item["gt_valid"].shape[0]) for item in batch)

    def pad_gt(key: str, fill: Any) -> torch.Tensor:
        proto = batch[0][key]
        out = torch.full((size, max_gt, *proto.shape[1:]), fill, dtype=proto.dtype)
        for row, item in enumerate(batch):
            value = item[key]
            out[row, : value.shape[0]] = value
        return out

    return {
        "rgb_history": torch.stack([item["rgb_history"] for item in batch]),
        "history_valid": torch.stack([item["history_valid"] for item in batch]),
        "ego_history": torch.stack([item["ego_history"] for item in batch]),
        "route_command": torch.stack([item["route_command"] for item in batch]),
        "relative_ego_poses": torch.stack([item["relative_ego_poses"] for item in batch]),
        "gt_trajectory": pad_gt("gt_trajectory", 0.0),
        "gt_encoded": pad_gt("gt_encoded", 0.0),
        "gt_valid": pad_gt("gt_valid", False),
        "future_register": pad_gt("future_register", 0.0),
        "future_chunk_valid": pad_gt("future_chunk_valid", False),
        "future_quality": pad_gt("future_quality", 0.0),
        "scene_ids": [item["scene_id"] for item in batch],
    }


def batch_to_device(batch: dict[str, Any], device: torch.device) -> tuple[Observation, PairTargets, list[str]]:
    observation = Observation(
        rgb_history=batch["rgb_history"].to(device, non_blocking=True),
        history_valid=batch["history_valid"].to(device, non_blocking=True),
        ego_history=batch["ego_history"].to(device, non_blocking=True),
        route_command=batch["route_command"].to(device, non_blocking=True),
        relative_ego_poses=batch["relative_ego_poses"].to(device, non_blocking=True),
    )
    targets = PairTargets(
        gt_trajectory=batch["gt_trajectory"].to(device, non_blocking=True),
        gt_encoded=batch["gt_encoded"].to(device, non_blocking=True),
        gt_valid=batch["gt_valid"].to(device, non_blocking=True),
        future_register=batch["future_register"].to(device, non_blocking=True),
        future_chunk_valid=batch["future_chunk_valid"].to(device, non_blocking=True),
        future_quality=batch["future_quality"].to(device, non_blocking=True),
        pair_ids=[[{"gt_id": str(i)} for i in range(batch["gt_valid"].shape[1])]] * len(batch["scene_ids"]),
    )
    return observation, targets, batch["scene_ids"]


# --------------------------------------------------------------------------- #
# DDP wrappers
# --------------------------------------------------------------------------- #
class GeneratorModule(nn.Module):
    def __init__(self, model: DirectedPairDiT) -> None:
        super().__init__()
        self.model = model

    def forward(self, observation: Observation, source: torch.Tensor, modes: torch.Tensor):
        return self.model.forward_pairs(observation, source, modes)


class ScorerModule(nn.Module):
    def __init__(self, scorer: ScratchDrivoRScorer) -> None:
        super().__init__()
        self.scorer = scorer

    def forward(self, scene: torch.Tensor, trajectory: torch.Tensor) -> torch.Tensor:
        return self.scorer(scene, trajectory, None)


# --------------------------------------------------------------------------- #
# Optimizers / schedule
# --------------------------------------------------------------------------- #
def freeze_phi(model: DirectedPairDiT) -> None:
    for parameter in model.phi.parameters():
        parameter.requires_grad_(False)
    model.phi_time_queries.requires_grad_(False)


def build_generator_optimizer(model: DirectedPairDiT, lr: float, weight_decay: float):
    """Stage 1 trains the trajectory only (eta + theta).  The register/future
    branch (phi) is frozen and unused: it still runs forward as *detached*
    action-attention memory inside ``_directed_pass``, but receives no gradient
    and is excluded from the optimizer."""
    _set_generator_trainability(model, stage=2)
    freeze_phi(model)
    groups = parameter_groups(model)
    if groups["phi"]:
        raise AssertionError("phi/register branch must be frozen (register unused)")
    gen_params = list(groups["eta"]) + list(groups["theta"])
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    if {id(p) for p in gen_params} != expected:
        raise AssertionError("generator optimizer must contain every trainable generator parameter exactly once")
    return torch.optim.AdamW(gen_params, lr=lr, weight_decay=weight_decay), gen_params


def build_scorer_optimizer(model: DirectedPairDiT, scorer: ScratchDrivoRScorer, lr: float, weight_decay: float):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    _set_scorer_trainability(scorer, True)
    scorer_params = list(scorer.parameters())
    assert_disjoint_parameter_groups(scorer_params, [p for p in model.parameters() if p.requires_grad])
    return torch.optim.AdamW(scorer_params, lr=lr, weight_decay=weight_decay), scorer_params


def schedule_factor(step: int, total: int, warmup: int) -> float:
    if warmup > 0 and step <= warmup:
        return max(step, 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))


def make_labels(raw7: torch.Tensor, final_valid: torch.Tensor) -> OracleLabels:
    named = raw7[..., list(HEAD_INDICES)]
    return OracleLabels(
        raw7=raw7, named_six=named, head_valid=torch.isfinite(named),
        final_score=raw7[..., 6], final_valid=final_valid,
        scene_hashes=[], trajectory_hashes=[],
        evaluator_version=ORACLE_VERSION, failure_diagnostics=[],
    )


# --------------------------------------------------------------------------- #
# Phase-B cache dataset
# --------------------------------------------------------------------------- #
class CacheShardDataset(Dataset):
    def __init__(self, scene: torch.Tensor, trajectory: torch.Tensor,
                 raw7: torch.Tensor, final_valid: torch.Tensor,
                 gt_trajectory: torch.Tensor | None = None,
                 gt_valid: torch.Tensor | None = None,
                 gt_single: torch.Tensor | None = None,
                 gt_ok: torch.Tensor | None = None) -> None:
        self.scene = scene
        self.trajectory = trajectory
        self.raw7 = raw7
        self.final_valid = final_valid
        self.gt_trajectory = gt_trajectory
        self.gt_valid = gt_valid
        self.gt_single = gt_single
        self.gt_ok = gt_ok

    def __len__(self) -> int:
        return int(self.scene.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {"scene": self.scene[index], "trajectory": self.trajectory[index],
                "raw7": self.raw7[index], "final_valid": self.final_valid[index]}
        if self.gt_trajectory is not None:
            item["gt_trajectory"] = self.gt_trajectory[index]
            item["gt_valid"] = self.gt_valid[index]
        if self.gt_single is not None:
            item["gt_single"] = self.gt_single[index]
            item["gt_ok"] = self.gt_ok[index]
        return item


def cache_collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.stack([item[key] for item in batch]) for key in batch[0]}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phase-a-epochs", type=int, default=50, help="Stage 1: trajectory-only generator (eta+theta, phi frozen)")
    p.add_argument("--phase-b-epochs", type=int, default=50, help="Stage 2: scorer-only on cached oracle labels")
    p.add_argument("--batch-size", type=int, default=4, help="per-GPU batch (4 x 16 = global 64)")
    p.add_argument("--cache-batch-size", type=int, default=0, help="cache-build batch (defaults to --batch-size)")
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4, help="generator LR")
    p.add_argument("--scorer-lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--lambda-action", type=float, default=1.0)
    p.add_argument("--lambda-score", type=float, default=1.0)
    p.add_argument("--lambda-rank", type=float, default=0.1,
                   help="weight of the DrivoR-style scorer ranking (margin) loss; 0 disables")
    p.add_argument("--unsafe-threshold", type=float, default=0.2,
                   help="oracle final_score below this marks the rejected (unsafe) slot for the rank loss")
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--candidates", type=int, default=64)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--max-steps", type=int, default=0, help="cap optimizer-steps PER PHASE (0=full)")
    p.add_argument("--limit-scenes", type=int, default=0)
    p.add_argument("--find-unused", action="store_true")
    p.add_argument("--use-all-native-tokens", action="store_true",
                   help="output all 201 DINOv3 native tokens per frame (5x201+2=1007 scene tokens)")
    p.add_argument("--resume-cache", action="store_true",
                   help="skip Phase A + cache build; load generator.pt + cache shards, run Phase B only")
    p.add_argument("--multigt-root", default=DEFAULT_MULTIGT)
    p.add_argument("--sensor-root", default=DEFAULT_SENSOR)
    p.add_argument("--log-root", default=DEFAULT_LOG_ROOT)
    p.add_argument("--cache-root", default=DEFAULT_CACHE)
    p.add_argument("--dino-checkpoint", default=DEFAULT_DINO)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    p.add_argument("--index-dir", default="")
    p.add_argument("--cache-dir", default="")
    return p.parse_args()


def build_model(args, device) -> DirectedPairDiT:
    return DirectedPairDiT(
        d_model=256, heads=8, ffn_dim=1024, depth=args.depth,
        dino_checkpoint=args.dino_checkpoint,
        dino_model_name="vit_large_patch16_dinov3.lvd1689m",
        dino_lora_rank=16, dino_lora_alpha=16.0,
        use_all_native_tokens=args.use_all_native_tokens,
    ).to(device)


# --------------------------------------------------------------------------- #
# Phase A (Stage 1): trajectory-only generator (eta+theta, phi frozen)
# --------------------------------------------------------------------------- #
def run_phase_a(args, dctx: Dist, gen_net: nn.Module, model: DirectedPairDiT,
                loader, sampler, opt, gen_params, modes, initializer, total_steps, accum) -> list[dict]:
    device = dctx.device
    history: list[dict] = []
    opt_step = 0
    micro = 0
    start = time.time()
    stop = False
    gen_net.train()
    for epoch in range(args.phase_a_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            observation, targets, _ = batch_to_device(batch, device)
            size = observation.rgb_history.shape[0]
            mode_codes = modes.expand_for_batch(size, device)
            source = initializer.clean(observation.ego_history, mode_codes)
            factor = schedule_factor(opt_step + 1, total_steps, args.warmup)
            for group in opt.param_groups:
                group["lr"] = args.lr * factor
            outputs = gen_net(observation, source, mode_codes)
            with torch.no_grad():
                assignment = hungarian_xy(outputs.trajectory.detach(), targets.gt_trajectory, targets.gt_valid)
            loss_action = paired_residual_velocity_mse(outputs.endpoint, source, targets.gt_encoded, assignment)
            is_boundary = ((micro + 1) % accum) == 0
            ctx = gen_net.no_sync() if (dctx.enabled and accum > 1 and not is_boundary) else nullcontext()
            with ctx:
                (args.lambda_action * loss_action / accum).backward()
            micro += 1
            if not is_boundary:
                continue
            torch.nn.utils.clip_grad_norm_(gen_params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            opt_step += 1
            if opt_step % args.log_every == 0 or opt_step == total_steps:
                ade = matched_ade(outputs.trajectory.detach(), targets, assignment)
                metric = torch.tensor([ade[0], ade[1]], device=device)
                metric = all_reduce_sum(metric, dctx)
                record = {"event": "step", "phase": "A", "epoch": epoch, "opt_step": opt_step,
                          "total": total_steps, "lr": round(args.lr * factor, 8),
                          "loss_action": round(float(loss_action.detach()), 6),
                          "eval_matched_gt_ade_m": round(float(metric[0]) / max(float(metric[1]), 1.0), 4),
                          "sec_per_step": round((time.time() - start) / opt_step, 3)}
                history.append(record)
                log0(dctx, record)
            if args.max_steps > 0 and opt_step >= args.max_steps:
                stop = True
                break
        if stop:
            break
    log0(dctx, {"event": "phase_a_done", "opt_step": opt_step, "elapsed_s": round(time.time() - start, 2)})
    return history


@torch.no_grad()
def matched_ade(trajectory: torch.Tensor, targets: PairTargets, assignment: list[Any]) -> tuple[float, float]:
    total = 0.0
    count = 0
    for row in range(trajectory.shape[0]):
        for gt_index, candidate_index in assignment[row].as_pairs():
            ade = (trajectory[row, candidate_index, :, :2] - targets.gt_trajectory[row, gt_index, :, :2]).norm(dim=-1).mean()
            total += float(ade)
            count += 1
    return total, float(count)


def _pad_gt_to_fixed(gt_trajectory: torch.Tensor, gt_valid: torch.Tensor, k: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad/truncate Multi-GT to a fixed K so per-scene rows stack across batches.

    The Multi-GT contract caps K at 4; padding rows carry ``gt_valid=False`` so
    they never contribute to the ADE below.
    """
    batch, current_k = gt_valid.shape
    if current_k == k:
        return gt_trajectory, gt_valid
    if current_k > k:
        return gt_trajectory[:, :k], gt_valid[:, :k]
    pad_n = k - current_k
    traj_pad = gt_trajectory.new_zeros(batch, pad_n, *gt_trajectory.shape[2:])
    valid_pad = gt_valid.new_zeros(batch, pad_n)
    return torch.cat([gt_trajectory, traj_pad], dim=1), torch.cat([gt_valid, valid_pad], dim=1)


def _min_gt_ade(trajectory: torch.Tensor, index: torch.Tensor, gt: torch.Tensor, gt_valid: torch.Tensor) -> torch.Tensor:
    """Mean xy distance (over 8 steps) from each picked proposal to its nearest valid GT.

    ``trajectory`` is [B,M,8,3], ``index`` [B] selects one proposal per scene,
    ``gt`` is [B,K,8,3].  Returns [B]; scenes with no valid GT yield ``inf``.
    """
    steps, dims = trajectory.shape[2], trajectory.shape[3]
    pick = trajectory.gather(1, index[:, None, None, None].expand(-1, 1, steps, dims)).squeeze(1)
    per_gt = (pick[:, None, :, :2] - gt[..., :2]).norm(dim=-1).mean(dim=-1)
    return per_gt.masked_fill(~gt_valid, float("inf")).min(dim=1).values


def load_navtrain_gt(scene_ids: list[str], multigt_root: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Read each scene's first valid navtrain Multi-GT from local disk for official mining.

    Byte-for-byte the GT source used by the live scorer-rank job so ``mine_pair``
    sees the identical chosen/rejected slots: ``idx = argmax(mask)`` (first valid
    GT), else 0; ``ok`` is True whenever the scene has at least one GT row.
    """
    root = Path(multigt_root) / "navtrain"
    gts: list[np.ndarray] = []
    oks: list[bool] = []
    for sid in scene_ids:
        gt_path = root / sid / "gt_trajectories.npy"
        mask_path = root / sid / "gt_valid_mask.npy"
        gt = np.zeros((8, 3), dtype=np.float32)
        ok = False
        if gt_path.exists():
            arr = np.load(gt_path)
            if arr.ndim == 3 and arr.shape[0] > 0:
                idx = 0
                if mask_path.exists():
                    mask = np.load(mask_path).astype(bool).reshape(-1)
                    if mask.any():
                        idx = int(np.argmax(mask))
                gt = arr[idx].astype(np.float32)
                ok = True
        gts.append(gt)
        oks.append(ok)
    return torch.from_numpy(np.stack(gts)), torch.tensor(oks, dtype=torch.bool)


# --------------------------------------------------------------------------- #
# Cache build
# --------------------------------------------------------------------------- #
@torch.no_grad()
def run_cache_build(args, dctx: Dist, model: DirectedPairDiT, dataset: NavtrainSceneDataset,
                    oracle: DrivoRBatchMetricCacheOracle, modes: ImmutableModeBuffer,
                    initializer: StructuredSourceInitializer, cache_dir: Path) -> dict[str, Any]:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    cache_batch = args.cache_batch_size if args.cache_batch_size > 0 else args.batch_size
    sampler = DistributedSampler(dataset, num_replicas=dctx.world, rank=dctx.rank, shuffle=False, drop_last=True) if dctx.enabled else None
    loader = DataLoader(dataset, batch_size=cache_batch, sampler=sampler, shuffle=False,
                        num_workers=args.workers, collate_fn=collate, drop_last=True,
                        pin_memory=False, persistent_workers=args.workers > 0,
                        prefetch_factor=2 if args.workers > 0 else None)
    scenes: list[torch.Tensor] = []
    trajectories: list[torch.Tensor] = []
    raw7s: list[torch.Tensor] = []
    valids: list[torch.Tensor] = []
    gts: list[torch.Tensor] = []
    gt_valids: list[torch.Tensor] = []
    ids: list[str] = []
    skipped = 0
    start = time.time()
    for step, batch in enumerate(loader):
        observation, targets, scene_ids = batch_to_device(batch, dctx.device)
        size = observation.rgb_history.shape[0]
        mode_codes = modes.expand_for_batch(size, dctx.device)
        source = initializer.clean(observation.ego_history, mode_codes)
        outputs = model.forward_pairs(observation, source, mode_codes)
        trajectory = outputs.trajectory.detach()
        try:
            labels = oracle.evaluate(scene_ids, trajectory, batch_id=f"cache{step}")
        except Exception:
            labels = None
        if labels is None:
            skipped += size
            continue
        scene_cpu = outputs.scene_context.detach().to("cpu", torch.float32)
        traj_cpu = trajectory.to("cpu", torch.float32)
        raw7_cpu = labels.raw7.to("cpu", torch.float32)
        valid_cpu = labels.final_valid.to("cpu")
        gt_fixed, gt_valid_fixed = _pad_gt_to_fixed(targets.gt_trajectory, targets.gt_valid)
        gt_cpu = gt_fixed.to("cpu", torch.float32)
        gt_valid_cpu = gt_valid_fixed.to("cpu")
        for row, scene_id in enumerate(scene_ids):
            row_ok = bool(valid_cpu[row].all()) and not any(f.get("scene_id") == scene_id for f in labels.failure_diagnostics)
            if not row_ok:
                skipped += 1
                continue
            scenes.append(scene_cpu[row])
            trajectories.append(traj_cpu[row])
            raw7s.append(raw7_cpu[row])
            valids.append(valid_cpu[row])
            gts.append(gt_cpu[row])
            gt_valids.append(gt_valid_cpu[row])
            ids.append(scene_id)
        if (step + 1) % args.log_every == 0:
            log0(dctx, {"event": "cache_progress", "step": step + 1, "cached": len(ids), "skipped": skipped,
                        "sec_per_step": round((time.time() - start) / (step + 1), 3)})
    if not ids:
        raise RuntimeError(f"rank {dctx.rank} cached zero scenes (skipped={skipped})")
    shard = {"scene": torch.stack(scenes), "trajectory": torch.stack(trajectories),
             "raw7": torch.stack(raw7s), "final_valid": torch.stack(valids),
             "gt_trajectory": torch.stack(gts), "gt_valid": torch.stack(gt_valids),
             "scene_ids": ids, "skipped": skipped}
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(shard, cache_dir / f"shard_{dctx.rank}.pt")
    return {"cached": len(ids), "skipped": skipped, "elapsed_s": round(time.time() - start, 2)}


# --------------------------------------------------------------------------- #
# Phase B: scorer-only (matches PID 1663817 scorer_only_fixed_proposals)
# --------------------------------------------------------------------------- #
def phase_b(args, dctx: Dist, model: DirectedPairDiT, shard: dict[str, Any],
            common_n: int, output_dir: Path) -> list[dict]:
    device = dctx.device
    has_gt = "gt_trajectory" in shard and "gt_valid" in shard
    use_rank = args.lambda_rank > 0.0
    scorer = ScratchDrivoRScorer(d_model=256, heads=1, ffn_dim=1024, decoder_layers=4,
                                 scene_tokens=model.scene_tokens).to(device)
    opt_scorer, scorer_params = build_scorer_optimizer(model, scorer, args.scorer_lr, args.weight_decay)
    scorer_net: nn.Module = ScorerModule(scorer)
    if dctx.enabled:
        scorer_net = DDP(scorer_net, device_ids=[dctx.local_rank], output_device=dctx.local_rank,
                         find_unused_parameters=args.find_unused, broadcast_buffers=False)
    # Official DrivoR pair mining needs ONE GT per scene, loaded exactly like the
    # live scorer-rank job (first valid Multi-GT row + a has-GT mask).
    if use_rank:
        gt_single, gt_ok = load_navtrain_gt(shard["scene_ids"][:common_n], args.multigt_root)
    else:
        gt_single = gt_ok = None
    cache_dataset = CacheShardDataset(shard["scene"][:common_n], shard["trajectory"][:common_n],
                                      shard["raw7"][:common_n], shard["final_valid"][:common_n],
                                      shard["gt_trajectory"][:common_n] if has_gt else None,
                                      shard["gt_valid"][:common_n] if has_gt else None,
                                      gt_single, gt_ok)
    loader = DataLoader(cache_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
                        num_workers=0, collate_fn=cache_collate)
    steps_per_epoch = len(loader)
    total_b = steps_per_epoch * args.phase_b_epochs
    if args.max_steps > 0:
        total_b = min(total_b, args.max_steps)
    log0(dctx, {"event": "phase_b_start", "trainable_scorer": sum(p.numel() for p in scorer_params),
                "generator_frozen": not any(p.requires_grad for p in model.parameters()),
                "training_mode": "scorer_only_fixed_proposals",
                "lambda_rank": args.lambda_rank, "unsafe_threshold": args.unsafe_threshold,
                "rank_pair_source": "drivor_mine_pair" if use_rank else "none",
                "gt_ade_reporting": has_gt,
                "steps_per_epoch": steps_per_epoch, "total_opt_steps": total_b})
    history: list[dict] = []
    opt_step = 0
    epoch = 0
    start = time.time()
    stop = False
    scorer_net.train()
    for epoch in range(args.phase_b_epochs):
        for batch in loader:
            scene = batch["scene"].to(device, non_blocking=True)
            trajectory = batch["trajectory"].to(device, non_blocking=True)
            raw7 = batch["raw7"].to(device, non_blocking=True)
            final_valid = batch["final_valid"].to(device, non_blocking=True)
            gt = batch["gt_trajectory"].to(device, non_blocking=True) if has_gt else None
            gt_valid = batch["gt_valid"].to(device, non_blocking=True) if has_gt else None
            mine_gt = batch["gt_single"].to(device, non_blocking=True) if use_rank else None
            gt_ok = batch["gt_ok"].to(device, non_blocking=True) if use_rank else None
            factor = schedule_factor(opt_step + 1, total_b, args.warmup)
            for group in opt_scorer.param_groups:
                group["lr"] = args.scorer_lr * factor
            logits = scorer_net(scene, trajectory)
            labels = make_labels(raw7, final_valid)
            loss_score = scorer.source_loss(logits, labels)
            if use_rank:
                # Official DrivoR preference_module rank loss, IDENTICAL to the live
                # scorer-rank job: mine_pair picks the chosen/rejected slots from the
                # 64 proposals + one GT + oracle PDMS, then push the scorer utility of
                # the chosen slot above the rejected slot.  Clamp logits so aggregate's
                # log(sigmoid) stays finite (avoids -inf/NaN).
                utility = aggregate_score_logits(logits.clamp(-30.0, 30.0))
                final_score = labels.final_score.detach()
                scene_ok = labels.final_valid.all(dim=1) & gt_ok
                with torch.no_grad():
                    chosen, rejected, pair_valid = mine_pair(
                        trajectory.detach(), mine_gt, final_score, args.unsafe_threshold)
                pair_ok = (scene_ok & pair_valid).to(utility.dtype)
                margin = F.softplus(-(utility.gather(1, chosen[:, None]).squeeze(1)
                                      - utility.gather(1, rejected[:, None]).squeeze(1)))
                margin = torch.nan_to_num(margin, nan=0.0, posinf=0.0, neginf=0.0)
                rank_loss = (margin * pair_ok).sum() / pair_ok.sum().clamp_min(1.0)
                loss_score = loss_score + args.lambda_rank * rank_loss
            else:
                rank_loss = loss_score * 0.0
            (args.lambda_score * loss_score).backward()
            torch.nn.utils.clip_grad_norm_(scorer_params, 1.0)
            opt_scorer.step()
            opt_scorer.zero_grad(set_to_none=True)
            opt_step += 1
            if opt_step % args.log_every == 0 or opt_step == total_b:
                with torch.no_grad():
                    selected, _utility = scorer.select(logits)
                    final_score = raw7[..., 6]
                    oracle_best = final_score.argmax(dim=1)
                    best = final_score.max(dim=1).values
                    chosen = final_score.gather(1, selected.unsqueeze(1)).squeeze(1)
                    match = (selected == oracle_best).to(chosen.dtype)
                    n_scene = torch.tensor(float(final_score.shape[0]), device=device)
                    zero = torch.zeros((), device=device)
                    if has_gt:
                        # PID 1663817 stage2 reports ADE against GT next to the oracle
                        # PDMS: distance from the scorer's pick (and the oracle-best
                        # pick) to the nearest valid Multi-GT trajectory.
                        sel_ade = _min_gt_ade(trajectory, selected, gt, gt_valid)
                        best_ade = _min_gt_ade(trajectory, oracle_best, gt, gt_valid)
                        ade_ok = torch.isfinite(sel_ade)
                        sel_ade_sum = torch.where(ade_ok, sel_ade, zero).sum()
                        best_ade_sum = torch.where(ade_ok, best_ade, zero).sum()
                        ade_n = ade_ok.to(sel_ade.dtype).sum()
                    else:
                        sel_ade_sum = best_ade_sum = ade_n = zero
                    metric = torch.stack([chosen.sum(), best.sum(), match.sum(), n_scene,
                                          sel_ade_sum, best_ade_sum, ade_n])
                metric = all_reduce_sum(metric, dctx)
                denom = max(float(metric[3]), 1.0)
                record = {"event": "step", "phase": "B", "epoch": epoch, "opt_step": opt_step,
                          "total": total_b, "lr": round(args.scorer_lr * factor, 8),
                          "loss_score": round(float(loss_score.detach()), 6),
                          "loss_rank": round(float(rank_loss.detach()), 6),
                          "eval_selected_pdms": round(float(metric[0]) / denom, 4),
                          "eval_oracle_best_pdms": round(float(metric[1]) / denom, 4),
                          "eval_oracle_match_rate": round(float(metric[2]) / denom, 4),
                          "sec_per_step": round((time.time() - start) / opt_step, 4)}
                if has_gt:
                    ade_denom = max(float(metric[6]), 1.0)
                    record["eval_selected_gt_ade_m"] = round(float(metric[4]) / ade_denom, 4)
                    record["eval_oracle_best_gt_ade_m"] = round(float(metric[5]) / ade_denom, 4)
                history.append(record)
                log0(dctx, record)
            if args.save_every > 0 and opt_step % args.save_every == 0:
                save_final(output_dir, model, scorer, opt_scorer, epoch, opt_step, args, dctx)
            if args.max_steps > 0 and opt_step >= args.max_steps:
                stop = True
                break
        if stop:
            break
    save_final(output_dir, model, scorer, opt_scorer, epoch, opt_step, args, dctx)
    log0(dctx, {"event": "phase_b_done", "opt_step": opt_step, "elapsed_s": round(time.time() - start, 2),
                "last": history[-1] if history else None})
    return history


def save_final(output_dir: Path, model, scorer, opt_scorer, epoch, opt_step, args, dctx: Dist) -> None:
    if not dctx.main:
        return
    torch.save({"model": model.state_dict(), "scorer": scorer.state_dict(),
                "opt_scorer": opt_scorer.state_dict(), "epoch": epoch, "opt_step": opt_step,
                "config": vars(args)}, output_dir / "checkpoint.pt")


def synchronize_shard_length(dctx: Dist, local_n: int) -> int:
    tensor = torch.tensor([local_n], dtype=torch.long, device=dctx.device)
    return int(all_reduce_min(tensor, dctx).item())


def finalize(dctx: Dist) -> None:
    barrier(dctx)
    if dctx.enabled:
        dist.destroy_process_group()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    args = parse_args()
    if args.candidates != 64:
        raise ValueError("DrivoR online oracle requires exactly 64 candidates")
    dctx = setup_distributed()
    device = dctx.device
    seed_everything(args.seed)

    output_dir = Path(args.output_dir)
    index_dir = Path(args.index_dir) if args.index_dir else output_dir
    cache_dir = Path(args.cache_dir) if args.cache_dir else output_dir / "phase_b_cache"
    if dctx.main:
        output_dir.mkdir(parents=True, exist_ok=True)
        index_dir.mkdir(parents=True, exist_ok=True)
    barrier(dctx)

    model = build_model(args, device)
    modes = ImmutableModeBuffer().subset(args.candidates).to(device)
    initializer = StructuredSourceInitializer(jitter_std=0.0)
    log0(dctx, {"event": "model_built", "scene_tokens": model.scene_tokens,
                "use_all_native_tokens": args.use_all_native_tokens,
                "trainable_total": sum(p.numel() for p in model.parameters() if p.requires_grad)})

    if args.resume_cache:
        gen_ckpt = output_dir / "generator.pt"
        if not gen_ckpt.exists():
            raise RuntimeError(f"--resume-cache needs {gen_ckpt}")
        model.load_state_dict(torch.load(gen_ckpt, map_location=device)["model"])
        if dctx.world == 1:
            shard_paths = sorted(cache_dir.glob("shard_*.pt"))
            if not shard_paths:
                raise RuntimeError(f"--resume-cache needs shards under {cache_dir}")
            shards = [torch.load(p, map_location="cpu") for p in shard_paths]
            shard = {"scene": torch.cat([s["scene"] for s in shards], dim=0),
                     "trajectory": torch.cat([s["trajectory"] for s in shards], dim=0),
                     "raw7": torch.cat([s["raw7"] for s in shards], dim=0),
                     "final_valid": torch.cat([s["final_valid"] for s in shards], dim=0),
                     "scene_ids": [sid for s in shards for sid in s["scene_ids"]],
                     "skipped": sum(int(s["skipped"]) for s in shards)}
            if all("gt_trajectory" in s and "gt_valid" in s for s in shards):
                shard["gt_trajectory"] = torch.cat([s["gt_trajectory"] for s in shards], dim=0)
                shard["gt_valid"] = torch.cat([s["gt_valid"] for s in shards], dim=0)
        else:
            shard_path = cache_dir / f"shard_{dctx.rank}.pt"
            if not shard_path.exists():
                raise RuntimeError(f"--resume-cache needs {shard_path}")
            shard = torch.load(shard_path, map_location="cpu")
        common_n = synchronize_shard_length(dctx, int(shard["scene"].shape[0]))
        phase_b(args, dctx, model, shard, common_n, output_dir)
        finalize(dctx)
        return 0

    # ---- scene index ---- #
    index_start = time.time()
    limit_scenes = args.limit_scenes if args.limit_scenes > 0 else None
    if dctx.main:
        candidates = build_scene_index(args.multigt_root, args.cache_root, args.log_root, index_dir, limit_scenes)
    barrier(dctx)
    if not dctx.main:
        candidates = build_scene_index(args.multigt_root, args.cache_root, args.log_root, index_dir, limit_scenes)
    if len(candidates) < args.batch_size * dctx.world:
        raise RuntimeError(f"only {len(candidates)} scenes; need >= {args.batch_size * dctx.world}")
    dataset = NavtrainSceneDataset(candidates, args.sensor_root, args.multigt_root, args.cache_root, args.log_root)
    log0(dctx, {"event": "index_built", "scenes": len(candidates), "elapsed_s": round(time.time() - index_start, 2)})

    accum = max(1, args.grad_accum)

    # ---- Phase A (Stage 1): trajectory-only generator (eta+theta, phi frozen) ---- #
    opt_a, params_a = build_generator_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    gen_net: nn.Module = GeneratorModule(model)
    if dctx.enabled:
        gen_net = DDP(gen_net, device_ids=[dctx.local_rank], output_device=dctx.local_rank,
                      find_unused_parameters=args.find_unused, broadcast_buffers=False)
    sampler_a = DistributedSampler(dataset, num_replicas=dctx.world, rank=dctx.rank, shuffle=True, drop_last=True) if dctx.enabled else None
    loader_a = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler_a, shuffle=sampler_a is None,
                          num_workers=args.workers, collate_fn=collate, drop_last=True, pin_memory=False,
                          persistent_workers=args.workers > 0, prefetch_factor=2 if args.workers > 0 else None)
    total_a = max(1, len(loader_a) // accum) * args.phase_a_epochs
    if args.max_steps > 0:
        total_a = min(total_a, args.max_steps)
    log0(dctx, {"event": "phase_a_setup", "trainable": sum(p.numel() for p in params_a),
                "phi_frozen": True, "scorer_active": False, "total_opt_steps": total_a})
    history_a = run_phase_a(args, dctx, gen_net, model, loader_a, sampler_a, opt_a, params_a,
                            modes, initializer, total_a, accum)

    # ---- Freeze + persist generator ---- #
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if dctx.main:
        torch.save({"model": model.state_dict(), "config": vars(args)}, output_dir / "generator.pt")
    barrier(dctx)

    # ---- Cache build ---- #
    cache_paths = {c.token: c.cache_path for c in candidates}
    oracle = DrivoRBatchMetricCacheOracle(cache_paths=cache_paths, evaluator_version=ORACLE_VERSION)
    log0(dctx, {"event": "cache_build_start", "cache_dir": str(cache_dir)})
    build_stats = run_cache_build(args, dctx, model, dataset, oracle, modes, initializer, cache_dir)
    shard = torch.load(cache_dir / f"shard_{dctx.rank}.pt", map_location="cpu")
    barrier(dctx)
    log0(dctx, {"event": "cache_build_done", "rank0_cached": build_stats["cached"],
                "rank0_skipped": build_stats["skipped"], "elapsed_s": build_stats["elapsed_s"]})

    # ---- Phase B: scorer-only ---- #
    common_n = synchronize_shard_length(dctx, int(shard["scene"].shape[0]))
    log0(dctx, {"event": "phase_b_setup", "common_shard_scenes": common_n})
    history_b = phase_b(args, dctx, model, shard, common_n, output_dir)

    if dctx.main:
        (output_dir / "history.json").write_text(
            json.dumps({"phase_a": history_a, "phase_b": history_b}, indent=2) + "\n")
    finalize(dctx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
