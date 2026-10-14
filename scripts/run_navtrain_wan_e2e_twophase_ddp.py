"""Full-navtrain STAGED (two-phase) training for the Wan ActionDiT e2e variant (DDP).

This is the "先分阶段" stage of the e2e protocol: Phase A trains the trajectory
generator, Phase B trains the scorer on a frozen cache.  It is a SEPARATE,
self-contained copy; it only imports and calls ``nvs_v6_wan_e2e`` (the
DrivoR-exact end-to-end scorer package: Scene K/V is NOT detached) and never
edits any package file.  Pass ``--use-all-native-tokens`` for the full 1007-token
Scene K/V.  The final "联合微调" stage is ``run_navtrain_wan_e2e_ddp.py --resume
<this-output>/checkpoint.pt --use-all-native-tokens``, where the un-detached
score loss also shapes perception (eta).

Strategy (the "distributed / split" Stage-2 the request asks for)
-----------------------------------------------------------------
Phase A - trajectory generation only.  Train ``eta`` + ``theta`` (Wan ActionDiT)
with the Multi-GT residual-velocity loss.  ``phi`` (register/future branch) is
frozen; ``lambda_future = 0``.  NO oracle and NO scorer run here, so Phase A is
cheap and data-bound.

Freeze - every generator parameter is frozen after Phase A.

Cache build - run the frozen generator once over the whole dataset (rank-sharded)
and persist, per scene: the Scene K/V context, the 64 proposals, and the DrivoR
oracle labels (official ``get_sub_score``).  The expensive PDM oracle therefore
runs EXACTLY ONCE per scene instead of every step for every epoch.  Only fully
valid scenes are cached, so Phase B labels never trip ``source_loss``'s guard.

Phase B - score only.  Train ``psi`` (``ScratchDrivoRScorer``, ``scratch_random``)
against the stationary cached context/proposals/labels.  No generator, no oracle,
no image/log IO during Phase B, so it is very fast.

Distributed: torchrun single node (``--nproc_per_node=16``).  Phase A wraps the
generator in DDP; Phase B wraps the scorer in DDP; each rank trains on its own
cache shard (data-parallel).  Prints every ``--log-every`` optimizer steps
(default 10): step, loss, LR and eval progress, all-reduced across ranks.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import time
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

from nvs_v6_wan_e2e.data import Observation
from nvs_v6_wan_e2e.losses import (
    assert_disjoint_parameter_groups,
    hungarian_xy,
    paired_residual_velocity_mse,
)
from nvs_v6_wan_e2e.model import DirectedPairDiT, parameter_groups
from nvs_v6_wan_e2e.modes import ImmutableModeBuffer, StructuredSourceInitializer
from nvs_v6_wan_e2e.oracle import HEAD_INDICES, DrivoRBatchMetricCacheOracle, OracleLabels
from nvs_v6_wan_e2e.real_data import load_real_navsim_sample
from nvs_v6_wan_e2e.scorer import ScratchDrivoRScorer
from nvs_v6_wan_e2e.selection import aggregate_score_logits
from navsim.agents.drivoR.preference_module import mine_pair
from nvs_v6_wan_e2e.stages import _set_generator_trainability


# --------------------------------------------------------------------------- #
# Data roots: CLI- and env-overridable so the same script runs anywhere.
# --------------------------------------------------------------------------- #
def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value else default


DEFAULT_MULTIGT = _env("NVS_MULTIGT_ROOT", "/mnt/zhaozc_workspace/project/drivezero_goal_gen/outputs/navsim_multigt_0p5s_8step_16gpu")
DEFAULT_SENSOR = _env("NVS_SENSOR_ROOT", "/mnt/data/navsim/trainval_all/trainval_sensor_blobs/trainval")
DEFAULT_LOG_ROOT = _env("NVS_LOG_ROOT", "/mnt/data/navsim/trainval_navsim_logs/trainval")
DEFAULT_CACHE = _env("NVS_CACHE_ROOT", "/mnt/zhaozc_workspace/official_DrivoR_clean/data_runtime/official_metric_cache_navtrain_20261005/metric_cache")
DEFAULT_DINO = _env("NVS_DINO_CHECKPOINT", "/mnt/LLM_weight/timm/vit_large_patch16_dinov3.lvd1689m/model.safetensors")
DEFAULT_OUTPUT = _env("NVS_OUTPUT_DIR", "artifacts/navtrain_wan_e2e_twophase_ddp")
ORACLE_VERSION = "drivor_get_sub_score_navtrain_20261005"


# --------------------------------------------------------------------------- #
# Distributed helpers.
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
        raise RuntimeError("the full-navtrain Wan run requires the configured CUDA/PPU device")
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


# --------------------------------------------------------------------------- #
# Scene index (identical logic to the joint trainer; copied, not shared).
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
    sidecar = index_dir / "_wan_token_cache_index.json"
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
    sidecar = index_dir / "_wan_frame_index.json"
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
    multigt_root: str,
    cache_root: str,
    log_root: str,
    index_dir: Path,
    limit_scenes: int | None,
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
# Scene dataset / collate (real navtrain streaming).
# --------------------------------------------------------------------------- #
class NavtrainSceneDataset(Dataset):
    def __init__(
        self,
        candidates: list[Candidate],
        sensor_root: str,
        multigt_root: str,
        cache_root: str,
        log_root: str,
        max_retry: int = 12,
    ) -> None:
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
                    self.sensor_root,
                    self.multigt_root,
                    frame_index=candidate.frame_index,
                    cache_root=self.cache_root,
                )
            except Exception as error:
                last_error = error
                continue
            observation, targets = sample.observation, sample.targets
            return {
                "rgb_history": observation.rgb_history[0],
                "history_valid": observation.history_valid[0],
                "ego_history": observation.ego_history[0],
                "route_command": observation.route_command[0],
                "relative_ego_poses": observation.relative_ego_poses[0],
                "gt_trajectory": targets.gt_trajectory[0],
                "gt_encoded": targets.gt_encoded[0],
                "gt_valid": targets.gt_valid[0],
                "scene_id": candidate.token,
            }
        raise RuntimeError(f"could not load any scene near index {index}: {last_error!r}")


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    size = len(batch)
    max_gt = max(int(item["gt_valid"].shape[0]) for item in batch)

    def pad_ground_truth(key: str, fill: Any) -> torch.Tensor:
        prototype = batch[0][key]
        out = torch.full((size, max_gt, *prototype.shape[1:]), fill, dtype=prototype.dtype)
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
        "gt_trajectory": pad_ground_truth("gt_trajectory", 0.0),
        "gt_encoded": pad_ground_truth("gt_encoded", 0.0),
        "gt_valid": pad_ground_truth("gt_valid", False),
        "scene_ids": [item["scene_id"] for item in batch],
    }


def batch_to_device(batch: dict[str, Any], device: torch.device) -> tuple[Observation, dict[str, torch.Tensor], list[str]]:
    observation = Observation(
        rgb_history=batch["rgb_history"].to(device, non_blocking=True),
        history_valid=batch["history_valid"].to(device, non_blocking=True),
        ego_history=batch["ego_history"].to(device, non_blocking=True),
        route_command=batch["route_command"].to(device, non_blocking=True),
        relative_ego_poses=batch["relative_ego_poses"].to(device, non_blocking=True),
    )
    targets = {
        "gt_trajectory": batch["gt_trajectory"].to(device, non_blocking=True),
        "gt_encoded": batch["gt_encoded"].to(device, non_blocking=True),
        "gt_valid": batch["gt_valid"].to(device, non_blocking=True),
    }
    return observation, targets, batch["scene_ids"]


# --------------------------------------------------------------------------- #
# DDP wrappers.
# --------------------------------------------------------------------------- #
class GeneratorModule(nn.Module):
    """Phase-A generator forward: endpoint (grad) + trajectory (detached, for ADE)."""

    def __init__(self, model: DirectedPairDiT) -> None:
        super().__init__()
        self.model = model

    def forward(self, observation: Observation, source: torch.Tensor, modes: torch.Tensor):
        outputs = self.model.forward_pairs(observation, source, modes)
        return outputs.endpoint, outputs.trajectory.detach()


class ScorerModule(nn.Module):
    """Phase-B scorer forward on cached (scene, trajectory)."""

    def __init__(self, scorer: ScratchDrivoRScorer) -> None:
        super().__init__()
        self.scorer = scorer

    def forward(self, scene: torch.Tensor, trajectory: torch.Tensor) -> torch.Tensor:
        return self.scorer(scene, trajectory, None)


# --------------------------------------------------------------------------- #
# Optimizers / schedule.
# --------------------------------------------------------------------------- #
def freeze_phi(model: DirectedPairDiT) -> None:
    for parameter in model.phi.parameters():
        parameter.requires_grad_(False)
    model.phi_time_queries.requires_grad_(False)


def build_generator_optimizer(model: DirectedPairDiT, lr: float, weight_decay: float):
    _set_generator_trainability(model, stage=2)
    freeze_phi(model)
    groups = parameter_groups(model)
    if groups["phi"]:
        raise AssertionError("phi/register branch must be frozen when future-target=none")
    generator_parameters = list(groups["eta"]) + list(groups["theta"])
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    if {id(p) for p in generator_parameters} != expected:
        raise AssertionError("generator optimizer must contain every trainable generator parameter exactly once")
    return torch.optim.AdamW(generator_parameters, lr=lr, weight_decay=weight_decay), generator_parameters


def build_scorer_optimizer(model: DirectedPairDiT, scorer: ScratchDrivoRScorer, lr: float, weight_decay: float):
    for parameter in model.parameters():  # generator fully frozen for Phase B
        parameter.requires_grad_(False)
    for parameter in scorer.parameters():
        parameter.requires_grad_(True)
    scorer_parameters = list(scorer.parameters())
    assert_disjoint_parameter_groups(scorer_parameters, [p for p in model.parameters() if p.requires_grad])
    return torch.optim.AdamW(scorer_parameters, lr=lr, weight_decay=weight_decay), scorer_parameters


def schedule_factor(step: int, total: int, warmup: int) -> float:
    if warmup > 0 and step <= warmup:
        return max(step, 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))


def make_labels(raw7: torch.Tensor, final_valid: torch.Tensor) -> OracleLabels:
    """Rebuild an OracleLabels from cached raw7 so scorer.source_loss can run."""
    named = raw7[..., list(HEAD_INDICES)]
    return OracleLabels(
        raw7=raw7,
        named_six=named,
        head_valid=torch.isfinite(named),
        final_score=raw7[..., 6],
        final_valid=final_valid,
        scene_hashes=[],
        trajectory_hashes=[],
        evaluator_version=ORACLE_VERSION,
        failure_diagnostics=[],
    )


# --------------------------------------------------------------------------- #
# Phase-B in-memory cache dataset.
# --------------------------------------------------------------------------- #
class CacheShardDataset(Dataset):
    def __init__(self, scene: torch.Tensor, trajectory: torch.Tensor, raw7: torch.Tensor,
                 final_valid: torch.Tensor, gt: torch.Tensor, gt_ok: torch.Tensor) -> None:
        self.scene = scene
        self.trajectory = trajectory
        self.raw7 = raw7
        self.final_valid = final_valid
        self.gt = gt
        self.gt_ok = gt_ok

    def __len__(self) -> int:
        return int(self.scene.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "scene": self.scene[index],
            "trajectory": self.trajectory[index],
            "raw7": self.raw7[index],
            "final_valid": self.final_valid[index],
            "gt": self.gt[index],
            "gt_ok": self.gt_ok[index],
        }


def cache_collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.stack([item[key] for item in batch]) for key in batch[0]}


def load_navtrain_gt(scene_ids: list[str], multigt_root: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Read each scene's first valid navtrain Multi-GT from local disk for official mining."""
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
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase-a-epochs", type=int, default=50, help="generator-only epochs")
    parser.add_argument("--phase-b-epochs", type=int, default=50, help="scorer-only epochs on the cache")
    parser.add_argument("--batch-size", type=int, default=12, help="per-GPU batch (12 x 16 x 1 => global 192)")
    parser.add_argument("--cache-batch-size", type=int, default=0, help="cache-build batch (defaults to --batch-size)")
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--workers", type=int, default=8, help="DataLoader workers per rank (Phase A / cache build)")
    parser.add_argument("--lr", type=float, default=2e-4, help="generator (eta+theta) LR")
    parser.add_argument("--scorer-lr", type=float, default=5e-4, help="scorer (psi) LR")
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup", type=int, default=500, help="linear LR warmup optimizer-steps per phase")
    parser.add_argument("--lambda-action", type=float, default=1.0)
    parser.add_argument("--lambda-score", type=float, default=1.0)
    parser.add_argument("--lambda-rank", type=float, default=0.1,
                        help="weight of the DrivoR-style scorer ranking (margin) loss; 0 disables")
    parser.add_argument("--unsafe-threshold", type=float, default=0.2,
                        help="oracle PDMS below this counts as an unsafe rejected slot")
    parser.add_argument("--future-target", default="none", choices=["none"], help="only 'none' is supported (lambda_future=0, phi frozen)")
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--candidates", type=int, default=64)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=500, help="Phase-B checkpoint cadence in optimizer-steps")
    parser.add_argument("--max-steps", type=int, default=0, help="cap optimizer-steps PER PHASE (0 = full)")
    parser.add_argument("--limit-scenes", type=int, default=0, help="cap the scene index (0 = full navtrain)")
    parser.add_argument("--find-unused", action="store_true", help="DDP find_unused_parameters=True (safer, slower)")
    parser.add_argument("--use-all-native-tokens", action="store_true",
                        help="output all 201 DINOv3 native tokens per frame (5x201+2=1007 scene tokens)")
    parser.add_argument("--resume-cache", action="store_true", help="skip Phase A + build; load cache-dir shards + generator.pt")
    parser.add_argument("--multigt-root", default=DEFAULT_MULTIGT)
    parser.add_argument("--sensor-root", default=DEFAULT_SENSOR)
    parser.add_argument("--log-root", default=DEFAULT_LOG_ROOT)
    parser.add_argument("--cache-root", default=DEFAULT_CACHE)
    parser.add_argument("--dino-checkpoint", default=DEFAULT_DINO)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--index-dir", default="", help="sidecar index dir (defaults to output-dir)")
    parser.add_argument("--cache-dir", default="", help="phase-B shard dir (defaults to output-dir/phase_b_cache)")
    return parser.parse_args()


def build_model(args, device) -> DirectedPairDiT:
    return DirectedPairDiT(
        d_model=256, heads=8, ffn_dim=1024, depth=args.depth,
        dino_checkpoint=args.dino_checkpoint,
        dino_model_name="vit_large_patch16_dinov3.lvd1689m",
        dino_lora_rank=16, dino_lora_alpha=16.0,
        use_all_native_tokens=args.use_all_native_tokens,
    ).to(device)


@torch.no_grad()
def run_cache_build(args, dctx: Dist, model: DirectedPairDiT, dataset: NavtrainSceneDataset,
                    oracle: DrivoRBatchMetricCacheOracle, modes: ImmutableModeBuffer,
                    initializer: StructuredSourceInitializer, cache_dir: Path) -> dict[str, Any]:
    """Frozen-generator pass: persist scene K/V, proposals and oracle labels per rank."""
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    cache_batch = args.cache_batch_size if args.cache_batch_size > 0 else args.batch_size
    sampler = DistributedSampler(dataset, num_replicas=dctx.world, rank=dctx.rank, shuffle=False, drop_last=True) if dctx.enabled else None
    loader = DataLoader(
        dataset, batch_size=cache_batch, sampler=sampler, shuffle=False,
        num_workers=args.workers, collate_fn=collate, drop_last=True,
        pin_memory=False, persistent_workers=args.workers > 0,
        prefetch_factor=2 if args.workers > 0 else None,
    )
    scenes: list[torch.Tensor] = []
    trajectories: list[torch.Tensor] = []
    raw7s: list[torch.Tensor] = []
    valids: list[torch.Tensor] = []
    ids: list[str] = []
    skipped = 0
    start = time.time()
    for step, batch in enumerate(loader):
        observation, _targets, scene_ids = batch_to_device(batch, dctx.device)
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
        for row, scene_id in enumerate(scene_ids):
            row_ok = bool(valid_cpu[row].all()) and not any(f.get("scene_id") == scene_id for f in labels.failure_diagnostics)
            if not row_ok:
                skipped += 1
                continue
            scenes.append(scene_cpu[row])
            trajectories.append(traj_cpu[row])
            raw7s.append(raw7_cpu[row])
            valids.append(valid_cpu[row])
            ids.append(scene_id)
        if (step + 1) % args.log_every == 0:
            log0(dctx, {"event": "cache_progress", "step": step + 1, "cached": len(ids), "skipped": skipped,
                        "sec_per_step": round((time.time() - start) / (step + 1), 3)})
    if not ids:
        raise RuntimeError(f"rank {dctx.rank} cached zero scenes (skipped={skipped}); cannot run Phase B")
    shard = {
        "scene": torch.stack(scenes),
        "trajectory": torch.stack(trajectories),
        "raw7": torch.stack(raw7s),
        "final_valid": torch.stack(valids),
        "scene_ids": ids,
        "skipped": skipped,
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(shard, cache_dir / f"shard_{dctx.rank}.pt")
    return {"cached": len(ids), "skipped": skipped, "elapsed_s": round(time.time() - start, 2)}


def main() -> int:
    args = parse_args()
    if args.candidates != 64:
        raise ValueError("the DrivoR online oracle requires exactly 64 candidates")
    if args.future_target != "none":
        raise ValueError("only --future-target none is supported by this two-phase trainer")

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

    if args.resume_cache:
        gen_ckpt = output_dir / "generator.pt"
        if not gen_ckpt.exists():
            raise RuntimeError(f"--resume-cache needs {gen_ckpt}")
        model.load_state_dict(torch.load(gen_ckpt, map_location=device)["model"])
        if dctx.world == 1:
            # Single-GPU replay of a multi-rank cache: concat every rank shard so the
            # scorer still sees the full navtrain cache, not just rank 0's slice.
            shard_paths = sorted(cache_dir.glob("shard_*.pt"))
            if not shard_paths:
                raise RuntimeError(f"--resume-cache needs shards under {cache_dir}")
            shards = [torch.load(p, map_location="cpu") for p in shard_paths]
            shard = {
                "scene": torch.cat([s["scene"] for s in shards], dim=0),
                "trajectory": torch.cat([s["trajectory"] for s in shards], dim=0),
                "raw7": torch.cat([s["raw7"] for s in shards], dim=0),
                "final_valid": torch.cat([s["final_valid"] for s in shards], dim=0),
                "scene_ids": [sid for s in shards for sid in s["scene_ids"]],
                "skipped": sum(int(s["skipped"]) for s in shards),
            }
            log0(dctx, {"event": "resumed_cache", "shards": len(shards), "scenes": int(shard["scene"].shape[0])})
        else:
            shard_path = cache_dir / f"shard_{dctx.rank}.pt"
            if not shard_path.exists():
                raise RuntimeError(f"--resume-cache needs {shard_path}")
            log0(dctx, {"event": "resumed_cache", "shard": str(shard_path)})
            shard = torch.load(shard_path, map_location="cpu")
        if "gt" not in shard:
            gt, gt_ok = load_navtrain_gt(shard["scene_ids"], args.multigt_root)
            shard["gt"], shard["gt_ok"] = gt, gt_ok
            log0(dctx, {"event": "gt_attached", "scenes": int(gt.shape[0]),
                        "gt_ok_rate": round(float(gt_ok.float().mean()), 4)})
        common_n = synchronize_shard_length(dctx, int(shard["scene"].shape[0]))
        phase_b(args, dctx, model, shard, common_n, output_dir)
        finalize(dctx)
        return 0

    # ---- scene index + dataset -------------------------------------------- #
    # Rank 0 builds and persists the sidecars first, so 16 ranks never race to
    # walk every NAVSIM log; the barrier then lets the other ranks read them.
    index_start = time.time()
    limit_scenes = args.limit_scenes if args.limit_scenes > 0 else None
    if dctx.main:
        candidates = build_scene_index(args.multigt_root, args.cache_root, args.log_root, index_dir, limit_scenes)
    barrier(dctx)
    if not dctx.main:
        candidates = build_scene_index(args.multigt_root, args.cache_root, args.log_root, index_dir, limit_scenes)
    if len(candidates) < args.batch_size * dctx.world:
        raise RuntimeError(f"only {len(candidates)} scenes resolved; need >= batch_size*world ({args.batch_size * dctx.world})")
    dataset = NavtrainSceneDataset(candidates, args.sensor_root, args.multigt_root, args.cache_root, args.log_root)
    log0(dctx, {"event": "index_built", "scenes": len(candidates), "elapsed_s": round(time.time() - index_start, 2)})

    # ---- Phase A: generator only ------------------------------------------ #
    opt_generator, generator_parameters = build_generator_optimizer(model, args.lr, args.weight_decay)
    gen_net: nn.Module = GeneratorModule(model)
    if dctx.enabled:
        gen_net = DDP(gen_net, device_ids=[dctx.local_rank], output_device=dctx.local_rank,
                      find_unused_parameters=args.find_unused, broadcast_buffers=False)
    log0(dctx, {"event": "phase_a_setup", "trainable_generator": sum(p.numel() for p in generator_parameters),
                "phi_frozen": sum(p.numel() for p in model.phi.parameters()) + model.phi_time_queries.numel(),
                "lambda_future": 0.0, "scorer_contract_pending": True})

    sampler_a = DistributedSampler(dataset, num_replicas=dctx.world, rank=dctx.rank, shuffle=True, drop_last=True) if dctx.enabled else None
    loader_a = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler_a, shuffle=sampler_a is None,
                          num_workers=args.workers, collate_fn=collate, drop_last=True, pin_memory=False,
                          persistent_workers=args.workers > 0, prefetch_factor=2 if args.workers > 0 else None)
    accum = max(1, args.grad_accum)
    steps_per_epoch = len(loader_a)
    opt_steps_per_epoch = max(1, steps_per_epoch // accum)
    total_a = opt_steps_per_epoch * args.phase_a_epochs
    if args.max_steps > 0:
        total_a = min(total_a, args.max_steps)

    history_a = run_phase_a(args, dctx, gen_net, model, loader_a, sampler_a, opt_generator, generator_parameters,
                            modes, initializer, total_a, accum)

    # freeze + persist the trained generator
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if dctx.main:
        torch.save({"model": model.state_dict(), "config": vars(args)}, output_dir / "generator.pt")
    barrier(dctx)

    # ---- Cache build: frozen generator + oracle once per scene ------------- #
    cache_paths = {candidate.token: candidate.cache_path for candidate in candidates}
    oracle = DrivoRBatchMetricCacheOracle(cache_paths=cache_paths, evaluator_version=ORACLE_VERSION)
    log0(dctx, {"event": "cache_build_start", "cache_dir": str(cache_dir)})
    build_stats = run_cache_build(args, dctx, model, dataset, oracle, modes, initializer, cache_dir)
    shard = torch.load(cache_dir / f"shard_{dctx.rank}.pt", map_location="cpu")
    if "gt" not in shard:
        gt, gt_ok = load_navtrain_gt(shard["scene_ids"], args.multigt_root)
        shard["gt"], shard["gt_ok"] = gt, gt_ok
        log0(dctx, {"event": "gt_attached", "scenes": int(gt.shape[0]),
                    "gt_ok_rate": round(float(gt_ok.float().mean()), 4)})
    barrier(dctx)
    log0(dctx, {"event": "cache_build_done", "rank0_cached": build_stats["cached"], "rank0_skipped": build_stats["skipped"],
                "elapsed_s": build_stats["elapsed_s"]})

    common_n = synchronize_shard_length(dctx, int(shard["scene"].shape[0]))
    log0(dctx, {"event": "phase_b_setup", "common_shard_scenes": common_n})
    history_b = phase_b(args, dctx, model, shard, common_n, output_dir)

    if dctx.main:
        (output_dir / "history.json").write_text(json.dumps({"phase_a": history_a, "phase_b": history_b}, indent=2) + "\n")
    finalize(dctx)
    return 0


def synchronize_shard_length(dctx: Dist, local_n: int) -> int:
    tensor = torch.tensor([local_n], dtype=torch.long, device=dctx.device)
    tensor = all_reduce_min(tensor, dctx)
    return int(tensor.item())


def run_phase_a(args, dctx: Dist, gen_net: nn.Module, model: DirectedPairDiT, loader, sampler,
                opt_generator, generator_parameters, modes, initializer, total_a, accum) -> list[dict[str, Any]]:
    device = dctx.device
    history: list[dict[str, Any]] = []
    opt_step = 0
    micro = 0
    start = time.time()
    stop = False
    gen_net.train()
    for epoch in range(args.phase_a_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            observation, targets, _scene_ids = batch_to_device(batch, device)
            size = observation.rgb_history.shape[0]
            mode_codes = modes.expand_for_batch(size, device)
            source = initializer.clean(observation.ego_history, mode_codes)
            factor = schedule_factor(opt_step + 1, total_a, args.warmup)
            for group in opt_generator.param_groups:
                group["lr"] = args.lr * factor
            endpoint, trajectory = gen_net(observation, source, mode_codes)
            with torch.no_grad():
                assignment = hungarian_xy(trajectory, targets["gt_trajectory"], targets["gt_valid"])
            loss_action = paired_residual_velocity_mse(endpoint, source, targets["gt_encoded"], assignment)
            is_boundary = ((micro + 1) % accum) == 0
            ctx = gen_net.no_sync() if (dctx.enabled and accum > 1 and not is_boundary) else _null()
            with ctx:
                (args.lambda_action * loss_action / accum).backward()
            micro += 1
            if not is_boundary:
                continue
            torch.nn.utils.clip_grad_norm_(generator_parameters, 1.0)
            opt_generator.step()
            opt_generator.zero_grad(set_to_none=True)
            opt_step += 1
            if opt_step % args.log_every == 0 or opt_step == total_a:
                ade = matched_ade(trajectory, targets, assignment)
                metric = torch.tensor([ade[0], ade[1]], device=device)
                metric = all_reduce_sum(metric, dctx)
                record = {"event": "step", "phase": "A", "epoch": epoch, "opt_step": opt_step, "total_opt_steps": total_a,
                          "lr": round(args.lr * factor, 8), "loss_action": round(float(loss_action.detach()), 6),
                          "eval_matched_gt_ade_m": round(float(metric[0]) / max(float(metric[1]), 1.0), 4),
                          "sec_per_step": round((time.time() - start) / opt_step, 3)}
                history.append(record)
                log0(dctx, record)
            if args.max_steps > 0 and opt_step >= args.max_steps:
                stop = True
                break
        if stop:
            break
    log0(dctx, {"event": "phase_a_done", "opt_step": opt_step, "elapsed_s": round(time.time() - start, 2),
                "last_loss_action": history[-1]["loss_action"] if history else None})
    return history


@torch.no_grad()
def matched_ade(trajectory: torch.Tensor, targets: dict[str, torch.Tensor], assignment: list[Any]) -> tuple[float, float]:
    total = 0.0
    count = 0
    for row in range(trajectory.shape[0]):
        for gt_index, candidate_index in assignment[row].as_pairs():
            ade = (trajectory[row, candidate_index, :, :2] - targets["gt_trajectory"][row, gt_index, :, :2]).norm(dim=-1).mean()
            total += float(ade)
            count += 1
    return total, float(count)


def phase_b(args, dctx: Dist, model: DirectedPairDiT, shard: dict[str, Any], common_n: int, output_dir: Path) -> list[dict[str, Any]]:
    device = dctx.device
    scorer = ScratchDrivoRScorer(
        d_model=256, heads=1, ffn_dim=1024, decoder_layers=4, scene_tokens=model.scene_tokens
    ).to(device)
    opt_scorer, scorer_parameters = build_scorer_optimizer(model, scorer, args.scorer_lr, args.weight_decay)
    scorer_net: nn.Module = ScorerModule(scorer)
    if dctx.enabled:
        scorer_net = DDP(scorer_net, device_ids=[dctx.local_rank], output_device=dctx.local_rank,
                         find_unused_parameters=args.find_unused, broadcast_buffers=False)

    cache_dataset = CacheShardDataset(shard["scene"][:common_n], shard["trajectory"][:common_n],
                                      shard["raw7"][:common_n], shard["final_valid"][:common_n],
                                      shard["gt"][:common_n], shard["gt_ok"][:common_n])
    loader = DataLoader(cache_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
                        num_workers=0, collate_fn=cache_collate)
    steps_per_epoch = len(loader)
    total_b = steps_per_epoch * args.phase_b_epochs
    if args.max_steps > 0:
        total_b = min(total_b, args.max_steps)
    log0(dctx, {"event": "phase_b_start", "trainable_scorer": sum(p.numel() for p in scorer_parameters),
                "generator_frozen": not any(p.requires_grad for p in model.parameters()),
                "steps_per_epoch": steps_per_epoch, "total_opt_steps": total_b})

    history: list[dict[str, Any]] = []
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
            gt = batch["gt"].to(device, non_blocking=True)
            gt_ok = batch["gt_ok"].to(device, non_blocking=True)
            factor = schedule_factor(opt_step + 1, total_b, args.warmup)
            for group in opt_scorer.param_groups:
                group["lr"] = args.scorer_lr * factor
            logits = scorer_net(scene, trajectory)
            labels = make_labels(raw7, final_valid)
            loss_score = scorer.source_loss(logits, labels)
            if args.lambda_rank > 0.0:
                # DrivoR preference_module rank loss: push the scorer utility of the
                # oracle-best (chosen) slot above an unsafe / worst (rejected) slot.
                # Clamp logits so aggregate's log(sigmoid) stays finite (avoids -inf/NaN).
                utility = aggregate_score_logits(logits.clamp(-30.0, 30.0))
                final_score = labels.final_score.detach()
                scene_ok = labels.final_valid.all(dim=1) & gt_ok
                with torch.no_grad():
                    chosen, rejected, pair_valid = mine_pair(
                        trajectory.detach(), gt, final_score, args.unsafe_threshold)
                pair_ok = (scene_ok & pair_valid).to(utility.dtype)
                margin = F.softplus(-(utility.gather(1, chosen[:, None]).squeeze(1)
                                      - utility.gather(1, rejected[:, None]).squeeze(1)))
                margin = torch.nan_to_num(margin, nan=0.0, posinf=0.0, neginf=0.0)
                rank_loss = (margin * pair_ok).sum() / pair_ok.sum().clamp_min(1.0)
                loss_score = loss_score + args.lambda_rank * rank_loss
            else:
                rank_loss = loss_score * 0.0
            (args.lambda_score * loss_score).backward()
            torch.nn.utils.clip_grad_norm_(scorer_parameters, 1.0)
            opt_scorer.step()
            opt_scorer.zero_grad(set_to_none=True)
            opt_step += 1
            if opt_step % args.log_every == 0 or opt_step == total_b:
                with torch.no_grad():
                    selected, _utility = scorer.select(logits)
                    final_score = raw7[..., 6]
                    best = final_score.max(dim=1).values
                    chosen = final_score.gather(1, selected.unsqueeze(1)).squeeze(1)
                    match = (selected == final_score.argmax(dim=1)).to(chosen.dtype)
                    metric = torch.tensor([chosen.sum(), best.sum(), match.sum(), float(final_score.shape[0])], device=device)
                metric = all_reduce_sum(metric, dctx)
                denom = max(float(metric[3]), 1.0)
                record = {"event": "step", "phase": "B", "epoch": epoch, "opt_step": opt_step, "total_opt_steps": total_b,
                          "lr": round(args.scorer_lr * factor, 8), "loss_score": round(float(loss_score.detach()), 6),
                          "loss_rank": round(float(rank_loss.detach()), 6),
                          "eval_selected_pdms": round(float(metric[0]) / denom, 4),
                          "eval_oracle_best_pdms": round(float(metric[1]) / denom, 4),
                          "eval_oracle_match_rate": round(float(metric[2]) / denom, 4),
                          "sec_per_step": round((time.time() - start) / opt_step, 4)}
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


def finalize(dctx: Dist) -> None:
    barrier(dctx)
    if dctx.enabled:
        dist.destroy_process_group()


def _null():
    from contextlib import nullcontext
    return nullcontext()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    raise SystemExit(main())
