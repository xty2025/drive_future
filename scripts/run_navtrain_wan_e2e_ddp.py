"""Full-navtrain JOINT Stage-2 training with the Wan ActionDiT variant (DDP).

This is the single supported training entry point for the ``nvs_v6_wan_e2e``
lineage.  It imports and calls the package without editing it.

Architecture / supervision
--------------------------
* Generator = ``nvs_v6_wan_e2e.DirectedPairDiT``.  ``theta`` is the Wan2.2
  ActionDiT at ``hidden_dim=256``: per-candidate self-attention with QK-RMSNorm +
  1D RoPE, cross-attention over ``[Scene | sg(Register)]``, six-parameter adaLN
  modulation and a modulated velocity head.  Its backbone is loaded from
  ``--wan-backbone`` (``checkpoints/ActionDiT_Wan22_256hdim_3outdim.pt``, built by
  ``scripts/preprocess_wan_action_dit.py``) with ``strict=True``; only
  ``action_encoder``/``head``/the NVS condition projections stay random.
  ``--heads`` must be 4, the payload's ``num_heads``.
* ``theta`` is a rectified-flow denoiser (see ``nvs_v6_wan_e2e.flow``).  The 64
  anchors are the ``sigma = 1`` endpoint and each candidate's matched GT is the
  ``sigma = 0`` endpoint, so ``x_sigma = (1-sigma)*gt + sigma*anchor`` and the
  single-forward inverse ``x0 = x_sigma - sigma*v`` returns a metric trajectory
  with no rollout at train or eval time (``action_sampling_loops: 0``).  One
  random ``t`` is drawn per scene per step, exactly as SimWAM samples it.
* Trajectory loss is DrivoR's own objective, generalized to Multi-GT:
  full-weight Hungarian L1 minADE on the matched pairs + ``diversity_loss`` + a
  small dense nearest-GT pull that gives all 64 candidates gradient.  The
  previous hard-Hungarian L2 supervised exactly one candidate per scene and
  matched on ``[:, :2]`` only, so heading was never supervised -- that is why
  only the matched few looked good.
* ``--lambda-velocity`` adds SimWAM's ``v_loss``: MSE on the flow velocity in
  ``norm_odo`` units, bell-weighted by ``t``.  Normalized units matter because the
  metric L1 term weights 1 m of ``x`` the same as 1 rad (57 deg) of heading.
* Future Register loss (``--lambda-future``) is dense over the 64 candidates via
  ``dense_future_l1`` and needs real Teacher targets: pass ``--future-target
  cache --target-cache-root`` pointing at a ``build_future_target_cache.py``
  output, which also unfreezes phi.  Without it ``--lambda-future`` must be 0 and
  the branch stays frozen, so it can never be silently trained toward all-zero
  placeholders.
* ``eta`` + ``theta`` (+ ``phi`` when the future cache is present) are trained by
  the trajectory/velocity/future losses; ``psi`` (``ScratchDrivoRScorer``) by the
  DrivoR score loss against the streaming online oracle
  ``DrivoRBatchMetricCacheOracle``.  The score gradient path is DrivoR's verbatim:
  proposals enter the scorer DETACHED while the Scene context stays ATTACHED.
* ``--stage1-checkpoint`` restores eta+phi+theta from ``run_stage1_multiview.py``
  with ``strict=True`` and leaves the scorer at scratch init, which is the
  Stage-1-then-Stage-2 order the future representation is trained for.

Loss weights follow DrivoR's and SimWAM's published ratios (see the ``setup``
event): action 1.0 / score 1.0 / rank 0.1 / velocity 1.0 / future 0.1.

Distributed
-----------
One DDP wrapper around ``(model + scorer)`` so a single backward reduces every
trainable parameter.  Launch with torchrun on a single node, e.g.
``torchrun --standalone --nnodes=1 --nproc_per_node=16``.  ``DistributedSampler``
and ``DataLoader`` both use ``drop_last=True`` to keep ranks in lockstep; the
process-group timeout is long enough to absorb per-rank oracle/data skew.  Every
``--log-every`` optimizer steps rank 0 prints step, losses, LR and eval progress
(selected vs oracle-best PDMS, oracle-match rate, matched-GT ADE and the all-64
nearest-GT ADE that measures overall set quality), all-reduced across ranks.
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

from nvs_v6_wan_e2e.data import Observation, PairTargets
from nvs_v6_wan_e2e.flow import INFERENCE_SIGMA
from nvs_v6_wan_e2e.losses import (
    assert_disjoint_parameter_groups,
    dense_future_l1,
    flow_velocity_loss,
    gather_flow_data,
    hungarian_xy,
    multigt_trajectory_loss,
    nearest_gt_index,
)
from nvs_v6_wan_e2e.model import DirectedPairDiT, parameter_groups
from nvs_v6_wan_e2e.modes import ImmutableModeBuffer, StructuredSourceInitializer
from nvs_v6_wan_e2e.oracle import DrivoRBatchMetricCacheOracle
from nvs_v6_wan_e2e.real_data import load_real_navsim_sample
from nvs_v6_wan_e2e.scorer import ScratchDrivoRScorer
from nvs_v6_wan_e2e.selection import aggregate_score_logits
from nvs_v6_wan_e2e.stages import _set_generator_trainability
from nvs_v6_wan_e2e.target_cache import compressor_path, read_scene_payload, scene_target_path
from nvs_v6_wan_e2e.vision import (
    DRIVOR_LORA_RANK,
    DRIVOR_NUM_SCENE_TOKENS,
    DRIVOR_OFFICIAL_IMAGE_HW,
)
from navsim.agents.drivoR.preference_module import mine_pair


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
DEFAULT_WAN_BACKBONE = _env("NVS_WAN_BACKBONE", "checkpoints/ActionDiT_Wan22_256hdim_3outdim.pt")
DEFAULT_OUTPUT = _env("NVS_OUTPUT_DIR", "artifacts/navtrain_wan_e2e_ddp")
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
        target_cache_root: str | None = None,
        target_dim: int = 256,
        max_retry: int = 12,
    ) -> None:
        self.candidates = candidates
        self.sensor_root = sensor_root
        self.multigt_root = multigt_root
        self.cache_root = cache_root
        self.log_root = log_root
        self.target_cache_root = target_cache_root
        self.target_dim = target_dim
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
            future = self._cached_future(candidate.token, targets.gt_trajectory.shape[0])
            return {
                "rgb_history": observation.rgb_history[0],
                "history_valid": observation.history_valid[0],
                "ego_history": observation.ego_history[0],
                "route_command": observation.route_command[0],
                "relative_ego_poses": observation.relative_ego_poses[0],
                "gt_trajectory": targets.gt_trajectory[0],
                "gt_valid": targets.gt_valid[0],
                "future_register": future["future_register"],
                "future_chunk_valid": future["future_chunk_valid"],
                "future_quality": future["future_quality"],
                "scene_id": candidate.token,
            }
        raise RuntimeError(f"could not load any scene near index {index}: {last_error!r}")

    def _cached_future(self, scene_id: str, gt_count: int) -> dict[str, torch.Tensor]:
        """Real Teacher Future Register targets, or honest invalid placeholders.

        ``load_real_navsim_sample`` cannot synthesize future views, so without a
        target cache it reports ``future_chunk_valid=False`` and the future loss
        becomes an exact zero instead of training the register toward all-zero
        targets.  With ``--target-cache-root`` the cached Teacher compression
        (8 future frames -> 4 chunks of 2 frames -> PCA 256) is read back.
        """
        if self.target_cache_root is None:
            return {
                "future_register": torch.zeros(gt_count, 4, self.target_dim),
                "future_chunk_valid": torch.zeros(gt_count, 4, dtype=torch.bool),
                "future_quality": torch.zeros(gt_count, 4),
            }
        payload = read_scene_payload(scene_target_path(self.target_cache_root, scene_id))
        register = payload["future_register"]
        if register.shape[0] != gt_count:
            raise ValueError(
                f"scene {scene_id}: cached future_register has {register.shape[0]} GT rows, "
                f"the Multi-GT artifact has {gt_count}"
            )
        if register.shape[-1] != self.target_dim:
            raise ValueError(
                f"scene {scene_id}: cached future_register is {register.shape[-1]}-d, "
                f"the model was built with --target-dim {self.target_dim}"
            )
        return {
            "future_register": register,
            "future_chunk_valid": payload["future_chunk_valid"],
            "future_quality": payload["future_quality"],
        }


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
        "gt_valid": pad_ground_truth("gt_valid", False),
        "future_register": pad_ground_truth("future_register", 0.0),
        "future_chunk_valid": pad_ground_truth("future_chunk_valid", False),
        "future_quality": pad_ground_truth("future_quality", 0.0),
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
        "gt_valid": batch["gt_valid"].to(device, non_blocking=True),
        "future_register": batch["future_register"].to(device, non_blocking=True),
        "future_chunk_valid": batch["future_chunk_valid"].to(device, non_blocking=True),
        "future_quality": batch["future_quality"].to(device, non_blocking=True),
    }
    return observation, targets, batch["scene_ids"]


def pair_targets(targets: dict[str, torch.Tensor]) -> PairTargets:
    """View the batch dict as the ``PairTargets`` the future losses expect."""
    return PairTargets(
        gt_trajectory=targets["gt_trajectory"],
        gt_valid=targets["gt_valid"],
        future_register=targets["future_register"],
        future_chunk_valid=targets["future_chunk_valid"],
        future_quality=targets["future_quality"],
        pair_ids=None,
    )


# --------------------------------------------------------------------------- #
# DDP wrapper: one module so a single backward reduces eta + theta + psi.
# --------------------------------------------------------------------------- #
class JointStage2Module(nn.Module):
    def __init__(self, model: DirectedPairDiT, scorer: ScratchDrivoRScorer) -> None:
        super().__init__()
        self.model = model
        self.scorer = scorer

    def forward(
        self,
        observation: Observation,
        source: torch.Tensor,
        modes: torch.Tensor,
        flow_data: torch.Tensor,
        timestep: torch.Tensor,
    ):
        """One rectified-flow forward: noisy action in, metric trajectory out.

        ``flow_data`` (the per-candidate ``sigma = 0`` endpoint) and ``timestep``
        are computed outside because neither touches a parameter, which keeps DDP's
        forward a pure function of the parameters it must reduce.
        """
        flow = self.model.forward_flow(observation, source, modes, flow_data, timestep)
        outputs = flow.pairs
        logits = self.scorer(outputs.scene_context, outputs.trajectory, outputs.predicted_register)
        # endpoint/velocity/register carry the generator graph; logits carry psi
        # plus eta through the scorer's attached Scene K/V.  velocity_target and the
        # detached endpoint are parameter-free: they feed the v_loss target, the
        # no-grad oracle, the Hungarian assignment and the metrics.
        return (outputs.endpoint, flow.velocity, flow.velocity_target, outputs.predicted_register,
                outputs.endpoint.detach(), logits)


def first_valid_gt(gt_trajectory: torch.Tensor, gt_valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce multi-GT [B,K,8,3] to each scene's first valid GT [B,8,3] for mine_pair.

    Mirrors load_navtrain_gt's single-GT choice so the Joint rank loss mines the
    same chosen/rejected slots as Phase B.  gt_ok is False for all-invalid scenes.
    """
    idx = gt_valid.to(torch.long).argmax(dim=1)
    rows = torch.arange(gt_trajectory.shape[0], device=gt_trajectory.device)
    gt = gt_trajectory[rows, idx]
    gt_ok = gt_valid.any(dim=1)
    return gt, gt_ok


# --------------------------------------------------------------------------- #
# Optimizers / schedule (custom: phi is frozen unless real future targets exist).
# --------------------------------------------------------------------------- #
def build_optimizers(
    model: DirectedPairDiT,
    scorer: ScratchDrivoRScorer,
    lr: float,
    scorer_lr: float,
    weight_decay: float,
    train_phi: bool = False,
):
    _set_generator_trainability(model, stage=2)  # freeze DINO base, keep LoRA + NVS scene/theta trainable
    if not train_phi:
        # No Teacher target cache: the register would be supervised by nothing (or
        # by all-zero placeholders), so the branch must not receive gradient at all.
        for parameter in model.phi.parameters():
            parameter.requires_grad_(False)
        model.phi_time_queries.requires_grad_(False)
    # 方案A（对齐 DrivoR 感知容量）：Joint 阶段冻结 scene_blocks（8 层 1007-token
    # self-attn）。DrivoR 的 backbone 只训 LoRA、base 冻结，所以敢用 final_score_weight
    # =1.0；我们额外训了大容量的 scene_blocks，未 detach 的 source_loss 会把它冲离
    # Phase A 学到的 GT 表征（最早 sel 跌到 0.85 的根因）。冻结 scene_blocks 后可训
    # 感知容量回到 LoRA 级别，source 才能安全提回 1.0 让 scorer utility 满强度训练。
    # 冻结后 parameter_groups 自动跳过它们（requires_grad=False），下面的断言仍成立。
    # Stage2 从 Stage1 的 checkpoint 续训，冻结 scene_blocks 也正是「保留 Stage1 学到
    # 的未来表征、只学生成与打分」的实现方式。
    # 回退：删掉这个 for 循环即恢复 scene_blocks 可训（详见 AGENTS.md「方案A」节）。
    for name, parameter in model.named_parameters():
        if name.startswith("scene_blocks."):
            parameter.requires_grad_(False)
    for parameter in scorer.parameters():
        parameter.requires_grad_(True)

    groups = parameter_groups(model)
    if not train_phi and groups["phi"]:
        raise AssertionError("phi/register branch must be frozen when no future target cache is present")
    if train_phi and not groups["phi"]:
        raise AssertionError("--lambda-future > 0 needs a trainable phi branch")
    generator_parameters = list(groups["eta"]) + list(groups["theta"]) + list(groups["phi"])
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
    """Return [sel, oracle_best, match, ade, ade_n, score_ok, near_ade, near_n] sums.

    ``ade`` is the Hungarian-pair ADE: it measures the few candidates that won the
    matching, so it stays low even when the other 60 are nonsense.  ``near_ade``
    is every supervised candidate against its own nearest valid GT, which is the
    number that actually reports overall set quality -- the user's complaint was
    exactly that only the matched few looked good.
    """
    out = torch.zeros(8, device=device)
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
    nearest, supervised = nearest_gt_index(trajectory, targets["gt_trajectory"], targets["gt_valid"])
    if bool(supervised.any()):
        batch, candidates = nearest.shape
        rows = torch.arange(batch, device=nearest.device).view(batch, 1).expand(batch, candidates)
        gt_xy = targets["gt_trajectory"][rows, nearest][:, :, :, :2]
        per_candidate = (trajectory[..., :2] - gt_xy).norm(dim=-1).mean(dim=-1)
        masked = per_candidate[supervised]
        out[6] = masked.sum()
        out[7] = float(masked.numel())
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
    parser.add_argument("--lambda-action", type=float, default=1.0,
                        help="DrivoR trajectory_weight: Hungarian L1 minADE + dense + diversity")
    parser.add_argument("--lambda-velocity", type=float, default=1.0,
                        help="SimWAM v_loss weight: MSE on the rectified-flow velocity, bell-weighted by t")
    parser.add_argument("--lambda-score", type=float, default=1.0,
                        help="DrivoR final_score_weight: 6-head BCE against the online PDMS oracle")
    parser.add_argument("--lambda-rank", type=float, default=0.1,
                        help="weight of the DrivoR-style scorer ranking (margin) loss; 0 disables")
    parser.add_argument("--lambda-future", type=float, default=0.1,
                        help="dense Future Register L1; only applied with --future-target cache")
    parser.add_argument("--dense-weight", type=float, default=0.1,
                        help="DrivoR dense nearest-GT pull on the 60 unmatched candidates")
    parser.add_argument("--diversity-weight", type=float, default=1.0,
                        help="DrivoR anti-collapse spread penalty over the 64 candidates")
    parser.add_argument("--dense-gate-m", type=float, default=8.0,
                        help="candidates farther than this from every GT get no dense pull")
    parser.add_argument("--unsafe-threshold", type=float, default=0.2,
                        help="PDMS below this counts as an unsafe rejected slot for rank mining")
    parser.add_argument("--future-target", default="none", choices=["none", "cache"],
                        help="'cache' reads real Teacher Future Register targets from --target-cache-root "
                             "and unfreezes phi; 'none' forces lambda_future=0 and keeps phi frozen")
    parser.add_argument("--target-cache-root", default="",
                        help="build_future_target_cache.py output holding compressor.pt and targets/")
    parser.add_argument("--target-dim", type=int, default=256, help="Future Register width (must match Stage 1)")
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--heads", type=int, default=4,
                        help="must equal the Wan payload's num_heads (4 at hidden_dim=256)")
    parser.add_argument("--wan-backbone", default=DEFAULT_WAN_BACKBONE,
                        help="ActionDiT_Wan22 payload from scripts/preprocess_wan_action_dit.py")
    parser.add_argument("--stage1-checkpoint", default="",
                        help="stage1_checkpoint.pt to restore eta+phi from; the scorer stays at scratch init")
    parser.add_argument("--candidates", type=int, default=64)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--log-every", type=int, default=10, help="print step/loss/eval this many optimizer-steps")
    parser.add_argument("--save-every", type=int, default=500, help="checkpoint this many optimizer-steps")
    parser.add_argument("--max-steps", type=int, default=0, help="stop after this many optimizer-steps (0 = full run)")
    parser.add_argument("--limit-scenes", type=int, default=0, help="cap the scene index (0 = full navtrain)")
    parser.add_argument("--find-unused", action="store_true", help="DDP find_unused_parameters=True (safer, slower)")
    parser.add_argument("--grad-checkpointing", action="store_true",
                        help="checkpoint the DINOv3 student forward to cut activation memory")
    parser.add_argument("--multigt-root", default=DEFAULT_MULTIGT)
    parser.add_argument("--sensor-root", default=DEFAULT_SENSOR)
    parser.add_argument("--log-root", default=DEFAULT_LOG_ROOT)
    parser.add_argument("--cache-root", default=DEFAULT_CACHE)
    parser.add_argument("--dino-checkpoint", default=DEFAULT_DINO)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--index-dir", default="", help="sidecar index dir (defaults to output-dir)")
    parser.add_argument("--resume", default="",
                        help="checkpoint.pt to load model+scorer weights from (joint finetune after the "
                             "staged two-phase); optimizers/LR schedule start fresh")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.candidates != 64:
        raise ValueError("the DrivoR online oracle requires exactly 64 candidates")

    # ---- future supervision: real Teacher targets or nothing at all --------- #
    args.target_cache_root = args.target_cache_root.strip()
    if args.future_target == "cache":
        if not args.target_cache_root:
            raise ValueError("--future-target cache requires --target-cache-root")
        if not compressor_path(args.target_cache_root).exists():
            raise ValueError(
                f"{compressor_path(args.target_cache_root)} is missing; run "
                f"scripts/build_future_target_cache.py --target-cache-root {args.target_cache_root} first"
            )
        train_phi = args.lambda_future > 0.0
    else:
        if args.target_cache_root:
            raise ValueError("--target-cache-root needs --future-target cache")
        if args.lambda_future > 0.0:
            raise ValueError(
                "--lambda-future > 0 needs --future-target cache; without real Teacher targets the "
                "register would be supervised by all-zero placeholders"
            )
        train_phi = False
    # A zero weight means the term is never built, so phi stays out of the graph
    # and out of the DDP reducer rather than receiving an identically-zero gradient.
    lambda_future = args.lambda_future if train_phi else 0.0

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

    dataset = NavtrainSceneDataset(
        candidates,
        args.sensor_root,
        args.multigt_root,
        args.cache_root,
        args.log_root,
        target_cache_root=args.target_cache_root or None,
        target_dim=args.target_dim,
    )
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
    # Every hyperparameter here must equal Stage 1's, or --stage1-checkpoint cannot
    # load with strict=True: LoRA rank sets the adapter shapes, target_dim the width,
    # heads the Wan payload's num_heads (4 at hidden_dim=256, since Wan's own
    # convention is num_heads * attn_head_dim == hidden_dim).
    model = DirectedPairDiT(
        d_model=args.target_dim,
        heads=args.heads,
        ffn_dim=4 * args.target_dim,
        depth=args.depth,
        num_scene_tokens=DRIVOR_NUM_SCENE_TOKENS,
        dino_checkpoint=args.dino_checkpoint,
        dino_image_hw=DRIVOR_OFFICIAL_IMAGE_HW,
        dino_lora_rank=DRIVOR_LORA_RANK,
        dino_grad_checkpointing=args.grad_checkpointing,
        wan_backbone_path=args.wan_backbone if Path(args.wan_backbone).exists() else None,
    ).to(device)
    scorer = ScratchDrivoRScorer(
        d_model=args.target_dim, heads=1, ffn_dim=4 * args.target_dim,
        decoder_layers=4, scene_tokens=model.scene_tokens,
    ).to(device)
    if args.wan_backbone and not Path(args.wan_backbone).exists():
        log0(dctx, {"event": "warn", "message": f"Wan backbone {args.wan_backbone} not found; theta starts random"})

    if args.stage1_checkpoint:
        # weights_only=True: the Stage 1 payload is tensors plus a scene-id list, so
        # nothing here needs to unpickle arbitrary Python objects.
        stage1 = torch.load(args.stage1_checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(stage1["model"], strict=True)
        log0(dctx, {"event": "stage1_restored", "checkpoint": str(args.stage1_checkpoint),
                    "scene_ids": len(stage1.get("scene_ids", [])),
                    "note": "eta+phi+theta restored; the scorer stays at scratch init"})
        del stage1

    if args.resume:
        resume_ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(resume_ckpt["model"], strict=True)
        scorer.load_state_dict(resume_ckpt["scorer"], strict=True)
        if dctx.main:
            print(json.dumps({"event": "resumed", "checkpoint": str(args.resume),
                              "prior_opt_step": int(resume_ckpt.get("opt_step", -1)),
                              "prior_epoch": int(resume_ckpt.get("epoch", -1))}), flush=True)
        del resume_ckpt

    opt_generator, opt_scorer, generator_parameters, scorer_parameters = build_optimizers(
        model, scorer, lr=args.lr, scorer_lr=args.scorer_lr,
        weight_decay=args.weight_decay, train_phi=train_phi,
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
        "generator_eta_theta_phi": sum(p.numel() for p in generator_parameters),
        "scorer_psi": sum(p.numel() for p in scorer_parameters),
        "phi_trainable": train_phi,
        "phi_parameters": sum(p.numel() for p in model.phi.parameters()) + model.phi_time_queries.numel(),
        "scene_blocks_frozen": sum(p.numel() for n, p in model.named_parameters() if n.startswith("scene_blocks.")),
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
        "wan_backbone": model.wan_backbone_info,
        # Loss weights, with the reference each ratio comes from:
        #   lambda_action / lambda_score = DrivoR trajectory_weight / final_score_weight = 1.0 / 1.0
        #   lambda_rank                  = DrivoR preference_weight * preference_rank_weight = 1.0 * 0.1
        #   lambda_velocity              = SimWAM v_loss (and FastWAM's action loss) at 1.0
        #   dense/diversity              = DrivoR's own trajectory objective internals
        #   lambda_future                = FastWAM ema_joint's auxiliary-term scale (0.1)
        "loss_weights": {
            "lambda_action": args.lambda_action,
            "lambda_velocity": args.lambda_velocity,
            "lambda_score": args.lambda_score,
            "lambda_rank": args.lambda_rank,
            "lambda_future": lambda_future,
            "dense_weight": args.dense_weight,
            "diversity_weight": args.diversity_weight,
            "dense_gate_m": args.dense_gate_m,
        },
        "flow": {
            "num_train_timesteps": model.flow.num_train_timesteps,
            "shift": model.flow.scheduler.shift,
            "inference_sigma": INFERENCE_SIGMA,
            "action_sampling_loops": 0,
        },
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
            # One random flow timestep per scene per step, exactly as SimWAM samples
            # it: no step1->step2->... rollout, a single forward returns a trajectory.
            timestep = model.flow.sample_timestep((size,), device, source.dtype)
            flow_data = gather_flow_data(source, targets["gt_trajectory"], targets["gt_valid"])

            factor = schedule_factor(opt_step + 1, total_opt_steps, args.warmup)
            for group in opt_generator.param_groups:
                group["lr"] = args.lr * factor
            for group in opt_scorer.param_groups:
                group["lr"] = args.scorer_lr * factor

            endpoint, velocity, velocity_target, register, trajectory, logits = net(
                observation, source, mode_codes, flow_data, timestep
            )

            with torch.no_grad():
                try:
                    labels = oracle.evaluate(scene_ids, trajectory, batch_id=f"e{epoch}s{opt_step}")
                    labels_ok = (not labels.failure_diagnostics) and bool(labels.final_valid.all())
                except Exception:
                    labels, labels_ok = None, False
                assignment = hungarian_xy(trajectory, targets["gt_trajectory"], targets["gt_valid"])

            # DrivoR's trajectory objective generalized to Multi-GT: the Hungarian
            # pairs get full-weight L1 minADE, the remaining candidates get a small
            # dense nearest-GT pull plus the anti-collapse diversity penalty.
            trajectory_terms = multigt_trajectory_loss(
                endpoint,
                targets["gt_trajectory"],
                targets["gt_valid"],
                dense_weight=args.dense_weight,
                diversity_weight=args.diversity_weight,
                dense_gate_m=args.dense_gate_m,
                assignment=assignment,
            )
            loss_action = trajectory_terms["loss_trajectory"]
            # SimWAM's v_loss on the rectified-flow velocity in normalized units, so
            # metres and radians contribute comparably and the bell weight by t applies.
            loss_velocity = flow_velocity_loss(velocity, velocity_target, model.flow.weight(timestep))
            if lambda_future > 0.0:
                loss_future = dense_future_l1(register, pair_targets(targets), endpoint)
            else:
                loss_future = loss_action * 0.0
            if labels_ok:
                source_loss = scorer.source_loss(logits, labels)
                if args.lambda_rank > 0.0:
                    # DrivoR preference_module rank loss: push the scorer utility of the
                    # oracle-best (chosen) slot above an unsafe / worst (rejected) slot.
                    # Weighted INDEPENDENTLY of lambda_score so its effective coefficient
                    # equals DrivoR's preference_weight(1.0)*preference_rank_weight(0.1)
                    # = 0.1 (drivoR.yaml).  Nesting it under lambda_score diluted it 5x
                    # (0.2*0.1=0.02), which is why sel_pdms stalled below Phase B's 0.978.
                    # NOTE: oracle.evaluate returns CPU labels (get_sub_score runs on CPU
                    # numpy), unlike Phase B whose labels come from the .to(device) cache.
                    # Align every label-derived tensor to logits' device before mixing
                    # with the CUDA gt/utility/trajectory tensors.
                    device = logits.device
                    gt, gt_ok = first_valid_gt(targets["gt_trajectory"], targets["gt_valid"])
                    gt = gt.to(device)
                    gt_ok = gt_ok.to(device)
                    utility = aggregate_score_logits(logits.clamp(-30.0, 30.0))
                    final_score = labels.final_score.detach().to(device)
                    scene_ok = labels.final_valid.all(dim=1).to(device) & gt_ok
                    with torch.no_grad():
                        chosen, rejected, pair_valid = mine_pair(
                            trajectory.detach(), gt, final_score, args.unsafe_threshold)
                    pair_ok = (scene_ok & pair_valid).to(utility.dtype)
                    margin = F.softplus(-(utility.gather(1, chosen[:, None]).squeeze(1)
                                          - utility.gather(1, rejected[:, None]).squeeze(1)))
                    margin = torch.nan_to_num(margin, nan=0.0, posinf=0.0, neginf=0.0)
                    rank_loss = (margin * pair_ok).sum() / pair_ok.sum().clamp_min(1.0)
                else:
                    rank_loss = source_loss * 0.0
            else:
                # Keep psi in the autograd graph (zero grad) so DDP never sees an
                # unused parameter and ranks stay in lockstep on a failed oracle.
                oracle_failures += 1
                source_loss = (logits * 0.0).sum()
                rank_loss = source_loss * 0.0
            loss = (args.lambda_action * loss_action
                    + args.lambda_velocity * loss_velocity
                    + args.lambda_score * source_loss
                    + args.lambda_rank * rank_loss
                    + lambda_future * loss_future)

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
                denom_near = max(float(metrics[7].item()), 1.0)
                record = {
                    "event": "step",
                    "epoch": epoch,
                    "opt_step": opt_step,
                    "total_opt_steps": total_opt_steps,
                    "lr": round(args.lr * factor, 8),
                    "loss_action": round(float(loss_action.detach()), 6),
                    "loss_velocity": round(float(loss_velocity.detach()), 6),
                    "loss_future": round(float(loss_future.detach()), 6),
                    "loss_score": round(float(source_loss.detach()), 6),
                    "loss_rank": round(float(rank_loss.detach()), 6),
                    "loss_total": round(float(loss.detach()), 6),
                    "matched_ade_m": round(float(trajectory_terms["matched_ade"].detach()), 4),
                    "dense_ade_m": round(float(trajectory_terms["dense_ade"].detach()), 4),
                    "diversity": round(float(trajectory_terms["diversity"].detach()), 4),
                    "flow_sigma_mean": round(float((timestep.detach() / model.flow.num_train_timesteps).mean()), 4),
                    "eval_selected_pdms": round(float(metrics[0].item()) / denom_scene, 4) if labels_ok else None,
                    "eval_oracle_best_pdms": round(float(metrics[1].item()) / denom_scene, 4) if labels_ok else None,
                    "eval_oracle_match_rate": round(float(metrics[2].item()) / denom_scene, 4) if labels_ok else None,
                    "eval_matched_gt_ade_m": round(float(metrics[3].item()) / denom_ade, 4),
                    "eval_all64_nearest_gt_ade_m": round(float(metrics[6].item()) / denom_near, 4),
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
