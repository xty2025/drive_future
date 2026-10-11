"""Full-navtrain JOINT Stage-2 training with the Wan ActionDiT variant (DDP).

This launcher is strictly additive: it only imports and calls ``nvs_v6_wan`` and
never edits any package file, so the architecture under test is untouched.

Architecture / supervision
--------------------------
* Generator = ``nvs_v6_wan.DirectedPairDiT`` (Wan ActionDiT: per-candidate
  self-attention with QK-RMSNorm + 1D RoPE, cross-attention over
  ``[Scene(66) | sg(Register)(4)]``, six-parameter adaLN modulation and a
  ``WanActionHead`` single-step residual-endpoint readout).  Every Wan module is
  randomly initialized; no Wan or DrivoR weights are ever loaded.
* ``--future-target none`` (the only supported mode here): ``lambda_future = 0``.
  The joint loss is ``lambda_action * paired_residual_velocity_mse +
  lambda_score * scorer.source_loss``.  The register/future branch ``phi`` is
  still executed (its four tokens are read DETACHED by the action branch) but it
  receives no gradient, so it is frozen and excluded from the optimizer.
* ``eta`` + ``theta`` are trained by the action loss; ``psi``
  (``ScratchDrivoRScorer``, ``scratch_random``) is trained by the DrivoR score
  loss against the streaming online oracle ``DrivoRBatchMetricCacheOracle``
  (official ``get_sub_score``, lazily evaluated per scene, never pre-hashed).

Distributed
-----------
One DDP wrapper around ``(model + scorer)`` so a single backward reduces every
trainable parameter.  Launch with torchrun on a single node, e.g.
``torchrun --standalone --nnodes=1 --nproc_per_node=16``.  ``DistributedSampler``
and ``DataLoader`` both use ``drop_last=True`` to keep ranks in lockstep; the
process-group timeout is long enough to absorb per-rank oracle/data skew.  Every
``--log-every`` optimizer steps (default 10) rank 0 prints step, losses, LR and
eval progress (selected vs oracle-best PDMS, oracle-match rate, matched-GT ADE),
all-reduced across ranks.

This file is intentionally a SEPARATE copy from the two-phase trainer
(``run_navtrain_wan_twophase_ddp.py``); the two are not merged on purpose.
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
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from nvs_v6_wan.data import Observation
from nvs_v6_wan.losses import (
    assert_disjoint_parameter_groups,
    hungarian_xy,
    paired_residual_velocity_mse,
)
from nvs_v6_wan.model import DirectedPairDiT, parameter_groups
from nvs_v6_wan.modes import ImmutableModeBuffer, StructuredSourceInitializer
from nvs_v6_wan.oracle import DrivoRBatchMetricCacheOracle
from nvs_v6_wan.real_data import load_real_navsim_sample
from nvs_v6_wan.scorer import ScratchDrivoRScorer
from nvs_v6_wan.stages import _set_generator_trainability


# --------------------------------------------------------------------------- #
# Data roots: every path is CLI- and env-overridable so the same script runs on
# a laptop and on an Alibaba PAI-DLC node without edits.
# --------------------------------------------------------------------------- #
def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value else default


DEFAULT_MULTIGT = _env("NVS_MULTIGT_ROOT", "/mnt/zhaozc_workspace/project/drivezero_goal_gen/outputs/navsim_multigt_0p5s_8step_16gpu")
DEFAULT_SENSOR = _env("NVS_SENSOR_ROOT", "/mnt/data/navsim/trainval_all/trainval_sensor_blobs/trainval")
DEFAULT_LOG_ROOT = _env("NVS_LOG_ROOT", "/mnt/data/navsim/trainval_navsim_logs/trainval")
DEFAULT_CACHE = _env("NVS_CACHE_ROOT", "/mnt/zhaozc_workspace/official_DrivoR_clean/data_runtime/official_metric_cache_navtrain_20261005/metric_cache")
DEFAULT_DINO = _env("NVS_DINO_CHECKPOINT", "/mnt/LLM_weight/timm/vit_large_patch16_dinov3.lvd1689m/model.safetensors")
DEFAULT_OUTPUT = _env("NVS_OUTPUT_DIR", "artifacts/navtrain_wan_joint_ddp")
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
    """Initialize torchrun/nccl process group; fall back to single-process CPU/GPU."""
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
        # A long timeout absorbs per-rank oracle (CPU PDM sim) and data-loading
        # skew so a slow rank never trips a spurious NCCL watchdog abort.
        dist.init_process_group(backend="nccl", timeout=timedelta(hours=3))
        enabled = True
    return Dist(rank=rank, world=world if has_env else 1, local_rank=local_rank, device=device, enabled=enabled)


def all_reduce_sum(value: torch.Tensor, dctx: Dist) -> torch.Tensor:
    if dctx.enabled:
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


def log0(dctx: Dist, payload: dict[str, Any]) -> None:
    if dctx.main:
        print(json.dumps(payload), flush=True)


# --------------------------------------------------------------------------- #
# Scene index: token -> (log_name, cache_path, frame_index), memoized to sidecars.
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
    """token -> (log_name, metric_cache_path), reusing any prebuilt sidecar."""
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
    """token -> frame_index (only frames leaving v6 history/future context).

    Walks each referenced NAVSIM log exactly once.  When ``limit_tokens`` is set
    the walk stops early and the (partial) result is NOT persisted, so a later
    full run always rebuilds a complete index.
    """
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
        token_cache_index,
        log_root,
        index_dir,
        meta,
        limit_tokens=limit_tokens,
        write_sidecar=not limit_scenes,
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
# Dataset / collate.
# --------------------------------------------------------------------------- #
class NavtrainSceneDataset(Dataset):
    """Streams real navtrain scenes; retries forward on a rare bad scene.

    A failed load must never change the per-rank batch count (that would desync
    DDP), so on failure the next candidate is tried until one loads.
    """

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
            except Exception as error:  # keep DDP in lockstep despite one bad scene
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
# DDP wrapper: one module so a single backward reduces eta + theta + psi.
# --------------------------------------------------------------------------- #
class JointStage2Module(nn.Module):
    def __init__(self, model: DirectedPairDiT, scorer: ScratchDrivoRScorer) -> None:
        super().__init__()
        self.model = model
        self.scorer = scorer

    def forward(self, observation: Observation, source: torch.Tensor, modes: torch.Tensor):
        outputs = self.model.forward_pairs(observation, source, modes)
        logits = self.scorer(outputs.scene_context, outputs.trajectory, None)
        # trajectory is returned detached: it feeds only the no-grad oracle and
        # the Hungarian assignment, never the loss, so DDP traces endpoint+logits.
        return outputs.endpoint, outputs.trajectory.detach(), logits


# --------------------------------------------------------------------------- #
# Optimizers / schedule (custom: phi frozen, so stages.build_optimizers is unusable).
# --------------------------------------------------------------------------- #
def build_optimizers(model: DirectedPairDiT, scorer: ScratchDrivoRScorer, lr: float, scorer_lr: float, weight_decay: float):
    _set_generator_trainability(model, stage=2)  # freeze DINO base, keep LoRA + NVS scene/theta trainable
    for parameter in model.phi.parameters():  # future/register branch: detached context, no supervision
        parameter.requires_grad_(False)
    model.phi_time_queries.requires_grad_(False)
    for parameter in scorer.parameters():
        parameter.requires_grad_(True)

    groups = parameter_groups(model)  # phi excluded because it is frozen
    if groups["phi"]:
        raise AssertionError("phi/register branch must be frozen when future-target=none")
    generator_parameters = list(groups["eta"]) + list(groups["theta"])
    scorer_parameters = list(scorer.parameters())
    assert_disjoint_parameter_groups(generator_parameters, scorer_parameters)
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    if {id(p) for p in generator_parameters} != expected:
        raise AssertionError("generator optimizer must contain every trainable generator parameter exactly once")
    opt_generator = torch.optim.AdamW(generator_parameters, lr=lr, weight_decay=weight_decay)
    opt_scorer = torch.optim.AdamW(scorer_parameters, lr=scorer_lr, weight_decay=weight_decay)
    return opt_generator, opt_scorer, generator_parameters, scorer_parameters


def schedule_factor(step: int, total: int, warmup: int) -> float:
    """Linear warmup then cosine decay to 5% of base LR."""
    if warmup > 0 and step <= warmup:
        return max(step, 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))


# --------------------------------------------------------------------------- #
# Eval metrics from the labels/logits the training step already produced (free).
# --------------------------------------------------------------------------- #
@torch.no_grad()
def batch_eval_metrics(
    scorer: ScratchDrivoRScorer,
    logits: torch.Tensor,
    trajectory: torch.Tensor,
    targets: dict[str, torch.Tensor],
    assignment: list[Any],
    labels: Any,
    labels_ok: bool,
    device: torch.device,
) -> torch.Tensor:
    """Return [sel_pdms, oracle_best, match, ade, ade_n, score_ok] sums for this batch."""
    out = torch.zeros(6, device=device)
    ade_sum = 0.0
    ade_n = 0
    for row in range(trajectory.shape[0]):
        for gt_index, candidate_index in assignment[row].as_pairs():
            ade = (trajectory[row, candidate_index, :, :2] - targets["gt_trajectory"][row, gt_index, :, :2]).norm(dim=-1).mean()
            ade_sum += float(ade)
            ade_n += 1
    if ade_n:
        out[3] = ade_sum
        out[4] = float(ade_n)
    if labels_ok:
        selected, _utility = scorer.select(logits)
        final_score = labels.final_score.to(device)
        best = final_score.max(dim=1).values
        chosen = final_score.gather(1, selected.to(device).unsqueeze(1)).squeeze(1)
        match = (selected.to(device) == final_score.argmax(dim=1)).to(chosen.dtype)
        out[0] = chosen.sum()
        out[1] = best.sum()
        out[2] = match.sum()
        out[5] = float(final_score.shape[0])
    return out


# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=12, help="per-GPU batch (12 x 16 x 1 => global 192)")
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers per rank")
    parser.add_argument("--lr", type=float, default=2e-4, help="generator (eta+theta) LR")
    parser.add_argument("--scorer-lr", type=float, default=5e-4, help="scorer (psi) LR")
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup", type=int, default=500, help="linear LR warmup optimizer-steps")
    parser.add_argument("--lambda-action", type=float, default=1.0)
    parser.add_argument("--lambda-score", type=float, default=1.0)
    parser.add_argument("--future-target", default="none", choices=["none"], help="only 'none' is supported (lambda_future=0, phi frozen)")
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--candidates", type=int, default=64)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--log-every", type=int, default=10, help="print step/loss/eval this many optimizer-steps")
    parser.add_argument("--save-every", type=int, default=500, help="checkpoint this many optimizer-steps")
    parser.add_argument("--max-steps", type=int, default=0, help="stop after this many optimizer-steps (0 = full run)")
    parser.add_argument("--limit-scenes", type=int, default=0, help="cap the scene index (0 = full navtrain)")
    parser.add_argument("--find-unused", action="store_true", help="DDP find_unused_parameters=True (safer, slower)")
    parser.add_argument("--multigt-root", default=DEFAULT_MULTIGT)
    parser.add_argument("--sensor-root", default=DEFAULT_SENSOR)
    parser.add_argument("--log-root", default=DEFAULT_LOG_ROOT)
    parser.add_argument("--cache-root", default=DEFAULT_CACHE)
    parser.add_argument("--dino-checkpoint", default=DEFAULT_DINO)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--index-dir", default="", help="sidecar index dir (defaults to output-dir)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.candidates != 64:
        raise ValueError("the DrivoR online oracle requires exactly 64 candidates")
    if args.future_target != "none":
        raise ValueError("only --future-target none is supported by this joint trainer")

    dctx = setup_distributed()
    device = dctx.device
    seed_everything(args.seed)

    output_dir = Path(args.output_dir)
    index_dir = Path(args.index_dir) if args.index_dir else output_dir
    if dctx.main:
        output_dir.mkdir(parents=True, exist_ok=True)
        index_dir.mkdir(parents=True, exist_ok=True)
    if dctx.enabled:
        dist.barrier()

    # ---- scene index (deterministic; identical on every rank) -------------- #
    # Rank 0 builds and persists the sidecars first, so 16 ranks never race to
    # walk every NAVSIM log; the barrier then lets the other ranks read them.
    index_start = time.time()
    limit_scenes = args.limit_scenes if args.limit_scenes > 0 else None
    if dctx.main:
        candidates = build_scene_index(args.multigt_root, args.cache_root, args.log_root, index_dir, limit_scenes)
    if dctx.enabled:
        dist.barrier()
    if not dctx.main:
        candidates = build_scene_index(args.multigt_root, args.cache_root, args.log_root, index_dir, limit_scenes)
    if len(candidates) < args.batch_size * dctx.world:
        raise RuntimeError(f"only {len(candidates)} scenes resolved; need >= batch_size*world ({args.batch_size * dctx.world})")
    log0(dctx, {"event": "index_built", "scenes": len(candidates), "elapsed_s": round(time.time() - index_start, 2)})

    dataset = NavtrainSceneDataset(candidates, args.sensor_root, args.multigt_root, args.cache_root, args.log_root)
    sampler = DistributedSampler(dataset, num_replicas=dctx.world, rank=dctx.rank, shuffle=True, drop_last=True) if dctx.enabled else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=args.workers,
        collate_fn=collate,
        drop_last=True,
        pin_memory=False,
        persistent_workers=args.workers > 0,
        prefetch_factor=2 if args.workers > 0 else None,
    )

    # ---- model / scorer / oracle ------------------------------------------ #
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
    scorer = ScratchDrivoRScorer(d_model=256, heads=1, ffn_dim=1024, decoder_layers=4).to(device)

    opt_generator, opt_scorer, generator_parameters, scorer_parameters = build_optimizers(
        model, scorer, lr=args.lr, scorer_lr=args.scorer_lr, weight_decay=args.weight_decay
    )

    net = JointStage2Module(model, scorer).to(device)
    if dctx.enabled:
        net = DDP(net, device_ids=[dctx.local_rank], output_device=dctx.local_rank, find_unused_parameters=args.find_unused, broadcast_buffers=False)

    cache_paths = {candidate.token: candidate.cache_path for candidate in candidates}
    oracle = DrivoRBatchMetricCacheOracle(cache_paths=cache_paths, evaluator_version=ORACLE_VERSION)
    modes = ImmutableModeBuffer().subset(args.candidates).to(device)
    initializer = StructuredSourceInitializer(jitter_std=0.0)

    steps_per_epoch = len(loader)
    accum = max(1, args.grad_accum)
    opt_steps_per_epoch = max(1, steps_per_epoch // accum)
    total_opt_steps = opt_steps_per_epoch * args.epochs
    if args.max_steps > 0:
        total_opt_steps = min(total_opt_steps, args.max_steps)

    trainable = {
        "generator_eta_theta": sum(p.numel() for p in generator_parameters),
        "scorer_psi": sum(p.numel() for p in scorer_parameters),
        "phi_frozen": sum(p.numel() for p in model.phi.parameters()) + model.phi_time_queries.numel(),
    }
    log0(dctx, {
        "event": "setup",
        "world": dctx.world,
        "per_gpu_batch": args.batch_size,
        "global_batch": args.batch_size * dctx.world,
        "epochs": args.epochs,
        "steps_per_epoch": steps_per_epoch,
        "opt_steps_per_epoch": opt_steps_per_epoch,
        "total_opt_steps": total_opt_steps,
        "trainable": trainable,
        "scorer_contract": scorer.feature_audit,
        "lambda_future": 0.0,
    })

    # ---- training loop ----------------------------------------------------- #
    run_start = time.time()
    opt_step = 0
    micro = 0
    oracle_failures = 0
    history: list[dict[str, Any]] = []
    stop = False
    net.train()
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        epoch_start = time.time()
        for batch in loader:
            observation, targets, scene_ids = batch_to_device(batch, device)
            size = observation.rgb_history.shape[0]
            mode_codes = modes.expand_for_batch(size, device)
            source = initializer.clean(observation.ego_history, mode_codes)

            factor = schedule_factor(opt_step + 1, total_opt_steps, args.warmup)
            for group in opt_generator.param_groups:
                group["lr"] = args.lr * factor
            for group in opt_scorer.param_groups:
                group["lr"] = args.scorer_lr * factor

            endpoint, trajectory, logits = net(observation, source, mode_codes)

            with torch.no_grad():
                try:
                    labels = oracle.evaluate(scene_ids, trajectory, batch_id=f"e{epoch}s{opt_step}")
                    labels_ok = (not labels.failure_diagnostics) and bool(labels.final_valid.all())
                except Exception:
                    labels, labels_ok = None, False
                assignment = hungarian_xy(trajectory, targets["gt_trajectory"], targets["gt_valid"])

            loss_action = paired_residual_velocity_mse(endpoint, source, targets["gt_encoded"], assignment)
            if labels_ok:
                loss_score = scorer.source_loss(logits, labels)
            else:
                # Keep psi in the autograd graph (zero grad) so DDP never sees an
                # unused parameter and ranks stay in lockstep on a failed oracle.
                oracle_failures += 1
                loss_score = (logits * 0.0).sum()
            loss = args.lambda_action * loss_action + args.lambda_score * loss_score

            is_boundary = ((micro + 1) % accum) == 0
            sync_ctx = net.no_sync() if (dctx.enabled and accum > 1 and not is_boundary) else nullcontext()
            with sync_ctx:
                (loss / accum).backward()
            micro += 1

            if not is_boundary:
                continue

            torch.nn.utils.clip_grad_norm_(generator_parameters, 1.0)
            torch.nn.utils.clip_grad_norm_(scorer_parameters, 1.0)
            opt_generator.step()
            opt_scorer.step()
            opt_generator.zero_grad(set_to_none=True)
            opt_scorer.zero_grad(set_to_none=True)
            opt_step += 1

            if opt_step % args.log_every == 0 or opt_step == total_opt_steps:
                metrics = batch_eval_metrics(scorer, logits, trajectory, targets, assignment, labels, labels_ok, device)
                metrics = all_reduce_sum(metrics, dctx)
                denom_scene = max(float(metrics[5].item()), 1.0)
                denom_ade = max(float(metrics[4].item()), 1.0)
                record = {
                    "event": "step",
                    "epoch": epoch,
                    "opt_step": opt_step,
                    "total_opt_steps": total_opt_steps,
                    "lr": round(args.lr * factor, 8),
                    "loss_action": round(float(loss_action.detach()), 6),
                    "loss_score": round(float(loss_score.detach()), 6),
                    "loss_total": round(float(loss.detach()), 6),
                    "eval_selected_pdms": round(float(metrics[0].item()) / denom_scene, 4) if labels_ok else None,
                    "eval_oracle_best_pdms": round(float(metrics[1].item()) / denom_scene, 4) if labels_ok else None,
                    "eval_oracle_match_rate": round(float(metrics[2].item()) / denom_scene, 4) if labels_ok else None,
                    "eval_matched_gt_ade_m": round(float(metrics[3].item()) / denom_ade, 4),
                    "oracle_failures": oracle_failures,
                    "sec_per_step": round((time.time() - run_start) / opt_step, 3),
                }
                history.append(record)
                log0(dctx, record)

            if args.save_every > 0 and (opt_step % args.save_every == 0 or opt_step == total_opt_steps):
                save_checkpoint(net, model, scorer, opt_generator, opt_scorer, epoch, opt_step, args, output_dir, dctx)

            if args.max_steps > 0 and opt_step >= args.max_steps:
                stop = True
                break
        if stop:
            break
        log0(dctx, {"event": "epoch_done", "epoch": epoch, "opt_step": opt_step, "epoch_sec": round(time.time() - epoch_start, 2)})

    save_checkpoint(net, model, scorer, opt_generator, opt_scorer, epoch, opt_step, args, output_dir, dctx)
    if dctx.main:
        (output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    log0(dctx, {
        "event": "done",
        "opt_step": opt_step,
        "oracle_failures": oracle_failures,
        "elapsed_s": round(time.time() - run_start, 2),
        "last": history[-1] if history else None,
    })
    if dctx.enabled:
        dist.barrier()
        dist.destroy_process_group()
    return 0


def save_checkpoint(net, model, scorer, opt_generator, opt_scorer, epoch, opt_step, args, output_dir: Path, dctx: Dist) -> None:
    if not dctx.main:
        return
    payload = {
        "model": model.state_dict(),
        "scorer": scorer.state_dict(),
        "opt_generator": opt_generator.state_dict(),
        "opt_scorer": opt_scorer.state_dict(),
        "epoch": epoch,
        "opt_step": opt_step,
        "config": {key: value for key, value in vars(args).items()},
    }
    torch.save(payload, output_dir / "checkpoint.pt")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    raise SystemExit(main())
