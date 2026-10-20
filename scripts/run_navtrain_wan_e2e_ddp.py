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
* ``--negative-root`` + ``--lambda-neg`` add DrivoR's ``rd_loss`` (repulsive
  distance), the ONLY way DrivoR consumes an external ``negative_trajectory``:
  offline-mined ``poor_negative`` trajectories (the drivezero_goal_gen posneg
  corpus, same ``[N,8,3]`` metric layout as the Multi-GT artifacts) repel the
  oracle-chosen generator proposal in trajectory space,
  ``rd = -||chosen - negative||.clamp_max(--rd-margin)``.  Because ``chosen``
  indexes the ATTACHED endpoint, the gradient shapes ``theta`` (the generator
  learns to steer away from the mined bad trajectory), NOT the scorer -- the
  scorer still only ranks the model's own proposals via ``--lambda-rank``.  rd
  needs the oracle to pick ``chosen``, so a step whose oracle fails carries no
  negative signal (an honest zero), exactly as in DrivoR.  With the flags absent
  the positives-only trainer is bit-identical: no rd term, same loss expression,
  same autograd graph, same RNG stream.
* ``--lambda-neg-match`` (EXPERIMENT, NVS-only -- NOT a DrivoR mechanism, default
  0 = off) additionally regresses some proposals toward the mined bad trajectories
  via the same Hungarian + metric-L1 machinery used for the good GTs (target
  detached, gradient to ``theta``).  Motivation: the generated trajectory is the Q
  the scorer attends over ``scene_kv`` with, so letting the proposal set span good
  AND bad gives the oracle-supervised scorer the contrast it needs to learn WHY a
  trajectory is bad in a given scene (e.g. a left turn the scene forbids when the
  only GT goes straight).  It complements ``rd_loss`` rather than replacing it: rd
  repels the oracle-CHOSEN good slot, neg-match pulls OTHER slots toward the bad
  ones, and a slot dragged toward a bad trajectory scores low so it is never the
  oracle-best -- the two objectives land on different slots.  With the flag at 0
  the term is absent and the rd-only trainer is unchanged.
* ``--lambda-neg-score`` (EXPERIMENT, NVS-only -- NOT a DrivoR mechanism, default
  0 = off) is the score-part-only negative: psi scores each mined poor_negative in
  a SECOND scorer pass over the same attached ``scene_context``, and a DrivoR-style
  utility margin pushes the oracle-chosen good proposal ABOVE every valid negative
  (``softplus(neg_utility - chosen_utility)``).  Because the scorer detaches the
  trajectory internally, the gradient shapes ``psi`` (and eta through the attached
  Scene K/V) but NEVER theta's trajectory generation -- so unlike ``rd_loss`` and
  ``--lambda-neg-match`` it does NOT touch the Hungarian/GT-matching front.  Set
  ``--lambda-neg 0 --lambda-neg-match 0 --lambda-neg-score w`` to train the scorer
  on negatives with no trajectory-space negative term at all.  With the flag at 0
  the forward's second scorer pass never runs and the trainer is bit-identical.
* Future Register loss (``--lambda-future``) is dense over the 64 candidates via
  ``dense_future_l1`` and needs real Teacher targets: pass ``--future-target
  cache --target-cache-root`` pointing at a ``build_future_target_cache.py``
  output.  Without it ``--lambda-future`` must be 0 and the future branch is
  frozen, so it can never be silently trained toward all-zero placeholders.
* ``--negative-target-cache-root`` is the future-branch twin of Stage 1's flag of the
  same name: a second ``build_future_target_cache.py`` output holding the mined
  ``poor_negative`` future register targets, built over the negative future-video
  corpus with the SAME ``standardizer.pt``.  Its rows are appended after the positive
  GT rows in the dataset, so ``gather_flow_data``, the Hungarian match,
  ``multigt_trajectory_loss`` AND ``dense_future_l1`` all span positives and negatives:
  each candidate is supervised toward the future of whichever GT (good or bad) it is
  nearest to, which teaches the model what future a bad trajectory CAUSES rather than
  only repelling it in the score head.  This is the "negatives as matched GT" front
  that beats the score-only ``--lambda-neg-score``.  Requires ``--future-target cache``;
  empty (default) keeps the positives-only trainer bit-identical.
* ``eta`` + ``theta`` are trained by the trajectory/velocity/future losses, and
  ``theta`` additionally by ``rd_loss`` when negatives are enabled; ``psi``
  (``ScratchDrivoRScorer``) by the DrivoR score loss against the streaming online
  oracle ``DrivoRBatchMetricCacheOracle``.  The score gradient path is DrivoR's
  verbatim: proposals enter the scorer DETACHED while the Scene context stays
  ATTACHED.
* ``--stage1-checkpoint`` restores eta+theta from ``run_stage1_multiview.py``
  with ``strict=True`` and leaves the scorer at scratch init, which is the
  Stage-1-then-Stage-2 order the future representation is trained for.  Restoring
  one also keeps the future query tokens in the DiT sequence even when this run has
  no targets of its own: Stage 1 shaped the trunk around 16 tokens, and dropping 8
  of them would evaluate a model on a sequence it never saw.

Loss weights follow DrivoR's and SimWAM's published ratios (see the ``setup``
event): action 1.0 / score 1.0 / rank 0.1 / velocity 1.0 / future 0.1, plus
rd (``--lambda-neg``) 0.1 when negatives are enabled.

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

from nvs_v6_wan_e2e.config import EXPECTED_REGISTER_TARGET_DIM
from nvs_v6_wan_e2e.data import Observation, PairTargets, REGISTER_TOKENS
from nvs_v6_wan_e2e.flow import INFERENCE_SIGMA
from nvs_v6_wan_e2e.losses import (
    assert_disjoint_parameter_groups,
    dense_future_l1,
    flow_velocity_loss,
    gather_flow_data,
    hungarian_matched_l1,
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
from nvs_v6_wan_e2e.stages import _set_generator_trainability, schedule_factor
from nvs_v6_wan_e2e.target_cache import load_standardizer, read_scene_payload, scene_target_path, standardizer_path
from nvs_v6_wan_e2e.teacher import TargetStandardizer
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

    #: Row-axis tensors a mined-negative payload contributes when merged into its scene,
    #: identical to Stage 1's ``target_cache.Stage1SceneDataset.NEGATIVE_ROW_KEYS``.  The
    #: observation side is deliberately absent: a negative row is the same scene at the same
    #: ``t0``, so the positive payload's history is the only history, and carrying a second
    #: copy would only create a chance to disagree.
    NEGATIVE_ROW_KEYS = (
        "gt_trajectory",
        "gt_valid",
        "future_register",
        "future_chunk_valid",
        "future_quality",
    )

    def __init__(
        self,
        candidates: list[Candidate],
        sensor_root: str,
        multigt_root: str,
        cache_root: str,
        log_root: str,
        target_cache_root: str | None = None,
        register_target_dim: int = EXPECTED_REGISTER_TARGET_DIM,
        target_fingerprint: str | None = None,
        negative_root: str | None = None,
        negative_target_cache_root: str | None = None,
        max_retry: int = 12,
    ) -> None:
        self.candidates = candidates
        self.sensor_root = sensor_root
        self.multigt_root = multigt_root
        self.cache_root = cache_root
        self.log_root = log_root
        self.target_cache_root = target_cache_root
        self.register_target_dim = register_target_dim
        self.target_fingerprint = target_fingerprint
        self.negative_root = negative_root
        self.negative_target_cache_root = negative_target_cache_root
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
            negative = self._load_negative(candidate.token)
            item = {
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
                "neg_trajectory": negative["neg_trajectory"],
                "neg_valid": negative["neg_valid"],
                "neg_score": negative["neg_score"],
                "scene_id": candidate.token,
            }
            # Append the mined poor_negative FUTURE rows AFTER the positive GT rows, exactly
            # as Stage 1's target_cache does, so gather_flow_data / hungarian_xy /
            # multigt_trajectory_loss / dense_future_l1 all span positives AND negatives:
            # every candidate is supervised toward the future of whichever GT (good or bad) it
            # is nearest to, which is what teaches the model what future a bad trajectory
            # causes.  Positives stay first, so row k of a positives-only batch is still row k
            # here and a pos-only and a pos+neg run stay diffable row by row.  The count rides
            # in metadata only -- never in a tensor -- so the model cannot see how many of its
            # GT rows were mined negatives.
            negative_rows = self._load_negative_rows(candidate.token)
            if negative_rows is not None:
                for key in self.NEGATIVE_ROW_KEYS:
                    item[key] = torch.cat([item[key], negative_rows[key]], dim=0)
                item["negative_future_rows"] = int(negative_rows["gt_valid"].shape[0])
            elif self.negative_target_cache_root is not None:
                item["negative_future_rows"] = 0
            return item
        raise RuntimeError(f"could not load any scene near index {index}: {last_error!r}")

    def _load_negative(self, scene_id: str) -> dict[str, torch.Tensor]:
        """Mined ``poor_negative`` trajectories for this scene, or honest empties.

        The negative root uses the Multi-GT per-token layout directly under the
        root (``<root>/<token>/gt_trajectories.npy`` [N,8,3] in metres at
        +0.5..+4.0 s, plus ``gt_valid_mask.npy`` / ``gt_scores.npy``), so no
        rescaling and no split subdirectory.  A missing, empty or unreadable
        entry means "this scene has no mined negatives" and must never fail the
        scene: the preference loss masks on ``neg_valid`` anyway, and ~25% of
        the navtrain corpus legitimately produced zero negatives.
        """
        empty = {
            "neg_trajectory": torch.zeros(0, 8, 3),
            "neg_valid": torch.zeros(0, dtype=torch.bool),
            "neg_score": torch.zeros(0),
        }
        if self.negative_root is None:
            return empty
        scene_dir = Path(self.negative_root) / scene_id
        try:
            trajectory = torch.from_numpy(
                np.load(scene_dir / "gt_trajectories.npy").astype(np.float32)
            ).reshape(-1, 8, 3)
            if trajectory.shape[0] == 0:
                return empty
            valid = torch.isfinite(trajectory).all(dim=(1, 2))
            mask_path = scene_dir / "gt_valid_mask.npy"
            if mask_path.exists():
                valid = valid & torch.from_numpy(np.load(mask_path).astype(bool)).reshape(-1)
            scores = torch.zeros(trajectory.shape[0])
            score_path = scene_dir / "gt_scores.npy"
            if score_path.exists():
                scores = torch.from_numpy(np.load(score_path).astype(np.float32)).reshape(-1)
        except Exception:
            return empty
        return {"neg_trajectory": trajectory, "neg_valid": valid, "neg_score": scores}

    def _cached_future(self, scene_id: str, gt_count: int) -> dict[str, torch.Tensor]:
        """Real Teacher Future Register targets, or honest invalid placeholders.

        ``load_real_navsim_sample`` cannot synthesize future views, so without a
        target cache it reports ``future_chunk_valid=False`` and the future loss
        becomes an exact zero instead of training the register toward all-zero
        targets.  With ``--target-cache-root`` the cached query extraction (eight
        future frames, four DINOv3 prefix-query readouts each, standardized per
        dimension) is read back.
        """
        if self.target_cache_root is None:
            return {
                "future_register": torch.zeros(gt_count, REGISTER_TOKENS, self.register_target_dim),
                "future_chunk_valid": torch.zeros(gt_count, REGISTER_TOKENS, dtype=torch.bool),
                "future_quality": torch.zeros(gt_count, REGISTER_TOKENS),
            }
        payload = read_scene_payload(scene_target_path(self.target_cache_root, scene_id))
        # The width check below cannot catch a cache whose two halves were built in
        # different target spaces: temperature and head count change the tensors, not
        # their shape.  The fingerprint is what distinguishes them.
        if self.target_fingerprint is not None and payload["target_fingerprint"] != self.target_fingerprint:
            raise ValueError(
                f"scene {scene_id} was extracted with target space {payload['target_fingerprint']}, "
                f"but the cache standardizer is {self.target_fingerprint}"
            )
        register = payload["future_register"]
        if register.shape[0] != gt_count:
            raise ValueError(
                f"scene {scene_id}: cached future_register has {register.shape[0]} GT rows, "
                f"the Multi-GT artifact has {gt_count}"
            )
        if register.shape[-1] != self.register_target_dim:
            raise ValueError(
                f"scene {scene_id}: cached future_register is {register.shape[-1]}-d, "
                f"the model was built with --register-target-dim {self.register_target_dim}"
            )
        return {
            "future_register": register,
            "future_chunk_valid": payload["future_chunk_valid"],
            "future_quality": payload["future_quality"],
        }

    def _load_negative_rows(self, scene_id: str) -> dict[str, torch.Tensor] | None:
        """Mined ``poor_negative`` FUTURE rows for this scene, or ``None`` when it has none.

        Symmetric to the positive ``_cached_future``: the negative Stage-0 cache is built by
        ``build_future_target_cache.py`` over the negative future-video corpus with the SAME
        ``standardizer.pt`` as the positive cache, so its ``future_register`` lives in the
        identical target space and its rows can be appended after the positive ones.  This is
        the Stage-2 twin of ``target_cache.Stage1SceneDataset.load_negative_rows``.

        ``None`` is the normal case, not a failure: a positive scene with no mined negative
        counterpart must not be dropped.  A *present but mismatched* ``target_fingerprint`` is
        the error -- concatenating rows standardized into two spaces would supervise one
        ``future_head`` against a mixture no single scaling describes, and every number would
        still look finite.
        """
        if self.negative_target_cache_root is None:
            return None
        path = scene_target_path(self.negative_target_cache_root, scene_id)
        if not path.exists():
            return None
        payload = read_scene_payload(path)
        if self.target_fingerprint is not None and payload["target_fingerprint"] != self.target_fingerprint:
            raise ValueError(
                f"negative scene {scene_id} was extracted with target space {payload['target_fingerprint']}, "
                f"but the positive cache standardizer is {self.target_fingerprint}; rebuild the negative "
                f"cache re-using the positive cache's standardizer.pt"
            )
        register = payload["future_register"]
        if register.shape[-1] != self.register_target_dim:
            raise ValueError(
                f"negative scene {scene_id}: cached future_register is {register.shape[-1]}-d, "
                f"the model was built with --register-target-dim {self.register_target_dim}"
            )
        return {
            "gt_trajectory": payload["gt_trajectory"],
            "gt_valid": payload["gt_valid"],
            "future_register": register,
            "future_chunk_valid": payload["future_chunk_valid"],
            "future_quality": payload["future_quality"],
        }


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    size = len(batch)
    max_gt = max(int(item["gt_valid"].shape[0]) for item in batch)

    def pad_rows(key: str, count: int, fill: Any) -> torch.Tensor:
        # prototype from batch[0] is safe even when row 0 is empty: the trailing
        # dims ([8,3] / scalar) are constant across scenes, only N varies.
        prototype = batch[0][key]
        out = torch.full((size, count, *prototype.shape[1:]), fill, dtype=prototype.dtype)
        for row, item in enumerate(batch):
            value = item[key]
            out[row, : value.shape[0]] = value
        return out

    # Negative keys are optional so datasets that predate the mined-negative
    # hook (the evaluators' NavtestSceneDataset) keep collating unchanged: a
    # missing key is exactly "this scene has no negatives", the honest empty.
    has_neg = all("neg_valid" in item for item in batch)
    max_neg = max(int(item["neg_valid"].shape[0]) for item in batch) if has_neg else 0
    if has_neg:
        neg_trajectory = pad_rows("neg_trajectory", max_neg, 0.0)
        neg_valid = pad_rows("neg_valid", max_neg, False)
        neg_score = pad_rows("neg_score", max_neg, 0.0)
    else:
        neg_trajectory = torch.zeros(size, 0, 8, 3)
        neg_valid = torch.zeros(size, 0, dtype=torch.bool)
        neg_score = torch.zeros(size, 0)

    result = {
        "rgb_history": torch.stack([item["rgb_history"] for item in batch]),
        "history_valid": torch.stack([item["history_valid"] for item in batch]),
        "ego_history": torch.stack([item["ego_history"] for item in batch]),
        "route_command": torch.stack([item["route_command"] for item in batch]),
        "relative_ego_poses": torch.stack([item["relative_ego_poses"] for item in batch]),
        "gt_trajectory": pad_rows("gt_trajectory", max_gt, 0.0),
        "gt_valid": pad_rows("gt_valid", max_gt, False),
        "future_register": pad_rows("future_register", max_gt, 0.0),
        "future_chunk_valid": pad_rows("future_chunk_valid", max_gt, False),
        "future_quality": pad_rows("future_quality", max_gt, 0.0),
        "neg_trajectory": neg_trajectory,
        "neg_valid": neg_valid,
        "neg_score": neg_score,
        "scene_ids": [item["scene_id"] for item in batch],
    }
    if any("negative_future_rows" in item for item in batch):
        result["negative_future_rows"] = sum(int(item.get("negative_future_rows", 0)) for item in batch)
    return result


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
        "neg_trajectory": batch["neg_trajectory"].to(device, non_blocking=True),
        "neg_valid": batch["neg_valid"].to(device, non_blocking=True),
        "neg_score": batch["neg_score"].to(device, non_blocking=True),
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
        neg_trajectory: torch.Tensor | None = None,
    ):
        """One rectified-flow forward: noisy action in, metric trajectory out.

        ``flow_data`` (the per-candidate ``sigma = 0`` endpoint) and ``timestep``
        are computed outside because neither touches a parameter, which keeps DDP's
        forward a pure function of the parameters it must reduce.

        ``neg_trajectory`` is ``None`` on every existing path, so the forward graph
        and the returned 6 values are unchanged (the 7th, ``neg_logits``, is None).
        It is set only by ``--lambda-neg-score``: the mined poor_negatives are scored
        by ``psi`` in a SECOND scorer pass over the SAME attached ``scene_context``.
        The scorer detaches the trajectory internally, so this shapes ``psi`` (and
        DrivoR's attached Scene K/V -> eta) but never theta's trajectory generation
        -- the negative enters the SCORE head only, not the Hungarian/GT-matching
        front that ``rd_loss``/``--lambda-neg-match`` use.  DrivoR's own
        ``rd_loss`` repels the oracle-chosen proposal in trajectory space and is
        applied to the returned ATTACHED ``endpoint`` in the training loop.
        """
        flow = self.model.forward_flow(observation, source, modes, flow_data, timestep)
        outputs = flow.pairs
        logits = self.scorer(outputs.scene_context, outputs.trajectory, outputs.predicted_register)
        neg_logits = None
        if neg_trajectory is not None:
            neg_logits = self.scorer(outputs.scene_context, neg_trajectory, outputs.predicted_register)
        # endpoint/velocity/register carry the generator graph; logits carry psi
        # plus eta through the scorer's attached Scene K/V.  velocity_target and the
        # detached endpoint are parameter-free: they feed the v_loss target, the
        # no-grad oracle, the Hungarian assignment and the metrics.
        return (outputs.endpoint, flow.velocity, flow.velocity_target, outputs.predicted_register,
                outputs.endpoint.detach(), logits, neg_logits)

    @torch.no_grad()
    def inference(self, observation: Observation, source: torch.Tensor, modes: torch.Tensor):
        """The deployed single forward: the anchor prior denoised once at sigma = 1.

        Evaluation has no GT, so ``sigma = 0`` endpoint is set to the anchor itself.
        ``add_noise`` then returns the anchor unchanged and ``recover`` subtracts the
        full predicted velocity -- algebraically the same expression the training
        forward uses, which is why eval cannot drift from the trained path.
        ``action_sampling_loops`` stays 0: no rollout here either.
        """
        timestep = torch.full(
            (source.shape[0],), INFERENCE_SIGMA * self.model.flow.num_train_timesteps,
            device=source.device, dtype=source.dtype,
        )
        flow = self.model.forward_flow(observation, source, modes, source, timestep)
        outputs = flow.pairs
        logits = self.scorer(outputs.scene_context, outputs.trajectory, outputs.predicted_register)
        return outputs.endpoint, outputs.trajectory, logits


def build_generator_and_scorer(
    *,
    target_dim: int = 256,
    register_target_dim: int = EXPECTED_REGISTER_TARGET_DIM,
    heads: int = 4,
    depth: int = 8,
    dino_checkpoint: str = DEFAULT_DINO,
    wan_backbone: str = DEFAULT_WAN_BACKBONE,
    grad_checkpointing: bool = False,
    vision_bf16: bool = False,
    device: torch.device | None = None,
    predict_future: bool = True,
) -> tuple[DirectedPairDiT, ScratchDrivoRScorer]:
    """The one construction path shared by training and both evaluators.

    Every hyperparameter here must equal Stage 1's, or ``--stage1-checkpoint`` and a
    Stage 2 ``--checkpoint`` cannot load with ``strict=True``: the LoRA rank sets the
    adapter shapes, ``target_dim`` the width, and ``heads`` the Wan payload's
    ``num_heads`` (4 at ``hidden_dim=256``, since Wan's own convention is
    ``num_heads * attn_head_dim == hidden_dim``).  ``dino_image_hw`` is DrivoR's
    official 672x1148 so the patch16 grid -- and therefore the position embeddings --
    match the checkpoint the Student was tuned against.

    ``register_target_dim`` is the future head's *output* width and is deliberately a
    separate argument: the query-extracted target is the Teacher's own feature width
    (1024 at ``vit_large_patch16``), not ``d_model``.  Folding the two into
    ``target_dim`` would widen the whole DiT to 1024 and then fail the Wan payload
    check above, since ``attn_head_dim`` would have to become 256 for
    ``num_heads * attn_head_dim == hidden_dim`` to hold.

    ``predict_future=False`` leaves the future query tokens out of the DiT sequence
    entirely, reproducing Wan's own 8-token action stream.  That is only correct for
    a run that neither has Teacher targets nor restores a Stage 1 checkpoint: eight
    unsupervised tokens at Wan's condition RMS would perturb a pretrained action
    stream, but eight tokens Stage 1 trained are part of the trunk and must stay.
    The parameters are allocated either way, so a Stage 1 checkpoint loads with
    ``strict=True`` regardless.
    """
    model = DirectedPairDiT(
        d_model=target_dim,
        heads=heads,
        ffn_dim=4 * target_dim,
        # DirectedPairDiT raises unless freq_dim == d_model and defaults freq_dim to
        # 256, so without this the joint stage cannot be built at any other width --
        # including the width of a Stage 1 checkpoint it is supposed to restore.
        freq_dim=target_dim,
        depth=depth,
        register_target_dim=register_target_dim,
        num_scene_tokens=DRIVOR_NUM_SCENE_TOKENS,
        dino_checkpoint=dino_checkpoint,
        dino_image_hw=DRIVOR_OFFICIAL_IMAGE_HW,
        dino_lora_rank=DRIVOR_LORA_RANK,
        dino_grad_checkpointing=grad_checkpointing,
        dino_vision_bf16=vision_bf16,
        wan_backbone_path=wan_backbone if Path(wan_backbone).exists() else None,
        predict_future=predict_future,
    )
    scorer = ScratchDrivoRScorer(
        d_model=target_dim, heads=1, ffn_dim=4 * target_dim,
        decoder_layers=4, scene_tokens=model.scene_tokens,
    )
    if device is not None:
        model, scorer = model.to(device), scorer.to(device)
    return model, scorer


def load_joint_checkpoint(
    checkpoint: str | Path,
    *,
    device: torch.device,
    depth: int = 0,
    heads: int = 0,
    target_dim: int = 0,
    register_target_dim: int = 0,
    dino_checkpoint: str = "",
    wan_backbone: str = "",
    grad_checkpointing: bool = False,
) -> tuple[JointStage2Module, DirectedPairDiT, ScratchDrivoRScorer, dict[str, Any]]:
    """Restore a Stage 2 checkpoint at the geometry it was trained with.

    A falsy override means "trust the checkpoint": its ``config`` is the trainer's
    own argparse namespace, so depth/heads/target_dim/dino_checkpoint are exactly
    the values that produced the state_dict.  Guessing them instead makes
    ``strict=True`` fail on the LoRA and Wan shapes -- or worse, silently builds a
    different scene-token layout.  ``weights_only=True`` because the payload is
    tensors plus ints plus that namespace; nothing here needs arbitrary unpickling.

    ``register_target_dim`` resolves in the same order, except that its last resort is
    the width of ``theta.future_head.1.weight`` in the payload rather than ``target_dim``:
    the target became a query extraction at the Teacher's own feature width, so the future
    head no longer emits ``d_model`` and the two must not be conflated.
    """
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}

    def resolved(key: str, override: Any, fallback: Any) -> Any:
        return override if override not in (0, "", None) else config.get(key, fallback)

    resolved_depth = int(resolved("depth", depth, 8))
    resolved_heads = int(resolved("heads", heads, 4))
    resolved_dim = int(resolved("target_dim", target_dim, 256))
    state = ckpt.get("model", {}) if isinstance(ckpt, dict) else {}
    head_weight = state.get("theta.future_head.1.weight")
    # The future head's output width *is* the register width, and it is present in every
    # payload that has a future branch, so read it off the weights.  Falling back to
    # ``target_dim`` instead is only right for a pre-split checkpoint -- where the two were
    # the same number anyway -- and for a post-split one whose config lost the key it
    # silently builds a d_model-wide head, which then fails as a bare ``strict=True``
    # "size mismatch" that never mentions the target space.
    head_fallback = int(head_weight.shape[0]) if head_weight is not None else resolved_dim
    resolved_register_dim = int(resolved("register_target_dim", register_target_dim, head_fallback))
    resolved_dino = str(resolved("dino_checkpoint", dino_checkpoint, DEFAULT_DINO))
    resolved_wan = str(resolved("wan_backbone", wan_backbone, DEFAULT_WAN_BACKBONE))
    # The trained sequence length is a property of the run, not of the weights: a
    # checkpoint trained without the future tokens must be evaluated without them
    # too, or the action stream attends to eight tokens it never saw.  It is saved
    # explicitly; the fallback reproduces main()'s rule for older checkpoints from
    # the trainer's own vars(args).
    resolved_future = bool(
        ckpt.get(
            "predict_future",
            bool(config.get("stage1_checkpoint"))
            or (config.get("future_target", "none") == "cache" and float(config.get("lambda_future", 0.0)) > 0.0),
        )
    )
    model, scorer = build_generator_and_scorer(
        target_dim=resolved_dim, register_target_dim=resolved_register_dim,
        heads=resolved_heads, depth=resolved_depth,
        dino_checkpoint=resolved_dino, wan_backbone=resolved_wan,
        grad_checkpointing=grad_checkpointing, device=device,
        predict_future=resolved_future,
    )
    # ``trajectory_gates`` was added to NvsActionDiT after the currently-running
    # (gate-free) checkpoints were saved, so those payloads lack the keys.  The gate
    # is applied ONLY when ``future is not None`` (see NvsActionDiT.forward); on this
    # predict_future=False eval/resume path ``future`` is None, so the gates never
    # touch the forward and tolerating their absence is bit-identical to loading the
    # pre-gates model.  Guard it: any missing key outside trajectory_gates, or any
    # unexpected key, is a real mismatch and must still hard-fail.
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    bad_missing = [k for k in missing if not k.startswith("theta.trajectory_gates.")]
    if bad_missing or unexpected:
        raise RuntimeError(
            "checkpoint/model mismatch beyond inert trajectory_gates: "
            f"missing={bad_missing} unexpected={list(unexpected)}"
        )
    scorer.load_state_dict(ckpt["scorer"], strict=True)
    model.eval()
    scorer.eval()
    meta = {
        "checkpoint": str(checkpoint),
        "opt_step": int(ckpt.get("opt_step", -1)),
        "epoch": int(ckpt.get("epoch", -1)),
        "depth": resolved_depth,
        "heads": resolved_heads,
        "target_dim": resolved_dim,
        "register_target_dim": resolved_register_dim,
        "dino_checkpoint": resolved_dino,
        "wan_backbone": resolved_wan,
        "predict_future": resolved_future,
        "scene_tokens": int(model.scene_tokens),
        "dit_sequence_tokens": int(
            model.scene_tokens + model.action_tokens + (model.register_tokens if resolved_future else 0)
        ),
        "self_attention_mask": "single_segment_block",
        "segment_visibility": [list(row) for row in model.theta.SEGMENT_VISIBILITY],
        "model_tensors": len(ckpt["model"]),
        "scorer_tensors": len(ckpt["scorer"]),
    }
    del ckpt
    net = JointStage2Module(model, scorer).to(device).eval()
    return net, model, scorer, meta


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
# Optimizers / schedule (custom: the future branch only trains with real targets).
# --------------------------------------------------------------------------- #
def build_optimizers(
    model: DirectedPairDiT,
    scorer: ScratchDrivoRScorer,
    lr: float,
    scorer_lr: float,
    weight_decay: float,
    train_future: bool = False,
):
    _set_generator_trainability(model, stage=2)  # freeze DINO base, keep LoRA + NVS scene/theta trainable
    if not train_future:
        # No future loss this run: with ``lambda_future == 0`` the register segment
        # receives no gradient at all, so freezing it is what keeps those parameters
        # out of the DDP reducer, which only expects gradients from parameters with
        # ``requires_grad=True``.  It also stops a Stage 1-restored register from being
        # reshaped by the action loss, which never reads it anyway.
        # ``register_type_embedding`` joins ``future_queries``/``future_head`` here: it
        # is the register segment's type embedding, added to the DiT sequence only when
        # ``predict_future`` is on, and read by no loss once ``lambda_future == 0`` (the
        # scorer drops ``predicted_register``).  Leaving it trainable makes DDP wait on a
        # gradient that never arrives -- "parameters not used in producing loss" -- in
        # BOTH the no-future run (register tokens absent) and a Stage-1-restored run with
        # ``--future-target none`` (register computed but unsupervised).  Freezing, not
        # removing, keeps the forward sequence length identical for the action stream.
        model.theta.future_queries.requires_grad_(False)
        model.theta.register_type_embedding.requires_grad_(False)
        for parameter in model.theta.future_head.parameters():
            parameter.requires_grad_(False)
        # The trajectory gates write only the register slice, so with no future loss and
        # (in the no-future case) no register segment at all they would be trainable
        # parameters that produce no gradient -- exactly what DDP's reducer rejects.
        for parameter in model.theta.trajectory_gates.parameters():
            parameter.requires_grad_(False)
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
    future_parameters = [
        model.theta.future_queries,
        model.theta.register_type_embedding,
        *model.theta.future_head.parameters(),
    ]
    future_ids = {id(parameter) for parameter in future_parameters}
    trainable_ids = {id(parameter) for group in groups.values() for parameter in group}
    if not train_future and future_ids & trainable_ids:
        raise AssertionError("the future/register branch must be frozen when lambda_future is 0")
    if train_future and not future_ids <= trainable_ids:
        raise AssertionError("--lambda-future > 0 needs a trainable future/register branch")
    generator_parameters = list(groups["eta"]) + list(groups["theta"])
    scorer_parameters = list(scorer.parameters())
    assert_disjoint_parameter_groups(generator_parameters, scorer_parameters)
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    if {id(p) for p in generator_parameters} != expected:
        raise AssertionError("generator optimizer must contain every trainable generator parameter exactly once")
    opt_generator = torch.optim.AdamW(generator_parameters, lr=lr, weight_decay=weight_decay)
    opt_scorer = torch.optim.AdamW(scorer_parameters, lr=scorer_lr, weight_decay=weight_decay)
    return opt_generator, opt_scorer, generator_parameters, scorer_parameters


# --------------------------------------------------------------------------- #
# Eval metrics from the labels/logits the training step already produced (free).
# --------------------------------------------------------------------------- #
@torch.no_grad()
def nearest_gt_ade(
    trajectory: torch.Tensor, gt_trajectory: torch.Tensor, gt_valid: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``[B,M]`` ADE of every candidate against its own nearest valid GT, plus the mask.

    Shared by ``batch_eval_metrics`` and the evaluators' per-scene records so the
    batch sum and the per-scene numbers cannot drift apart.
    """
    nearest, supervised = nearest_gt_index(trajectory, gt_trajectory, gt_valid)
    batch, candidates = nearest.shape
    rows = torch.arange(batch, device=nearest.device).view(batch, 1).expand(batch, candidates)
    gt_xy = gt_trajectory[rows, nearest][:, :, :, :2]
    return (trajectory[..., :2] - gt_xy).norm(dim=-1).mean(dim=-1), supervised


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

    ``ade`` is the Hungarian-pair ADE: it measures only the few candidates that won
    the matching, so it stays low even when the other 60 are nonsense.  ``near_ade``
    is every supervised candidate against its own nearest valid GT, which is the
    number that reports overall set quality.  Reporting the matched pair alone would
    hide a collapsed candidate set.
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
    nearest_ade, supervised = nearest_gt_ade(trajectory, targets["gt_trajectory"], targets["gt_valid"])
    if bool(supervised.any()):
        masked = nearest_ade[supervised]
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
    parser.add_argument("--negative-root", default="",
                        help="root of mined poor_negative scenes laid out as "
                             "<root>/<token>/gt_trajectories.npy (+ gt_valid_mask.npy, gt_scores.npy), "
                             "e.g. drivezero_goal_gen/outputs/navtrain_poor4_multigt_dense_0p1s_f64_16gpu; "
                             "empty disables the negative preference loss")
    parser.add_argument("--negative-target-cache-root", default="",
                        help="a second Stage-0 cache of mined poor_negative FUTURE REGISTER targets, built by "
                             "build_future_target_cache.py over the negative future-video corpus with the SAME "
                             "standardizer.pt as --target-cache-root. Its rows are appended after the positive "
                             "GT rows, so the action matching AND dense_future_l1 both span positives and "
                             "negatives: each candidate is supervised toward the future of whichever GT (good or "
                             "bad) it is nearest to, teaching the model what future a bad trajectory causes -- the "
                             "future-branch twin of Stage 1's --negative-target-cache-root. Requires "
                             "--future-target cache. Empty (default) keeps the positives-only trainer bit-identical.")
    parser.add_argument("--lambda-neg", type=float, default=0.0,
                        help="weight of DrivoR's rd_loss: repels the oracle-chosen generator "
                             "proposal from each mined poor_negative in trajectory space, so the "
                             "gradient shapes theta (the generator), not the scorer.  0.1 matches "
                             "DrivoR's preference_weight(1.0)*preference_rd_weight(0.1).  0 (default) "
                             "keeps the positives-only trainer bit-identical: no rd term, same graph")
    parser.add_argument("--rd-margin", type=float, default=1.0,
                        help="DrivoR preference_rd_margin: the repulsive-distance clamp in metres.  "
                             "A chosen/negative pair farther apart than this contributes a constant "
                             "-rd_margin (zero gradient), so rd only pushes apart nearby negatives")
    parser.add_argument("--lambda-neg-match", type=float, default=0.0,
                        help="EXPERIMENT (default 0 = off, trainer unchanged): additionally regress "
                             "some proposals toward the mined bad trajectories (Hungarian-matched, "
                             "full-weight metric L1, target detached -> gradient to theta), so the "
                             "trajectory-Q the scorer attends with spans good AND bad and the "
                             "oracle-supervised scorer learns WHY a bad trajectory is bad in this "
                             "scene.  Complements rd_loss (which repels the oracle-chosen good slot); "
                             "the two land on different slots.  Does not touch the future branch")
    parser.add_argument("--lambda-neg-score", type=float, default=0.0,
                        help="EXPERIMENT (default 0 = off, trainer unchanged): score the mined "
                             "poor_negatives with psi ONLY -- a second scorer pass over the same "
                             "attached scene_context, then a DrivoR-style utility margin pushing the "
                             "oracle-chosen proposal's score ABOVE every valid negative's score "
                             "(softplus(neg_utility - chosen_utility)).  The scorer detaches the "
                             "trajectory internally, so the gradient shapes psi (+ eta via the "
                             "attached Scene K/V) and NEVER theta's trajectory generation: the "
                             "negative enters the SCORE head, not the Hungarian/GT-matching front "
                             "that --lambda-neg (rd) and --lambda-neg-match use.  Needs the oracle to "
                             "pick the chosen slot, so a failed-oracle step carries no signal (an "
                             "honest zero).  Does not touch the future branch")
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
    parser.add_argument("--oracle-workers", type=int, default=0,
                        help="parallel get_sub_score workers per rank for the online PDMS oracle "
                             "(DrivoR scores its per-scene calls in parallel via ray worker_map; the "
                             "sequential loop idles the accelerator for the whole CPU oracle every "
                             "step). 0 = exact sequential path (bit-identical, default). Set to "
                             "~min(12, cores_per_node/world_size) to restore DrivoR-style throughput; "
                             "scores are bit-identical either way and any pool failure falls back to "
                             "sequential.")
    parser.add_argument("--phase-timing", action="store_true",
                        help="DIAGNOSTIC ONLY (default off): cuda.synchronize around each phase "
                             "(data/forward/oracle/loss/backward/opt) and log cumulative phase_*_ms "
                             "on rank 0. The syncs add overhead, so never enable for production.")
    parser.add_argument("--vision-bf16", action="store_true",
                        help="TRUNK-ONLY bf16 autocast on the frozen DINOv3 (default off = exact fp32 "
                             "path). Casts just the O(tokens^2) attention over ~3k patch tokens x 5 "
                             "frames that dominates the joint backward; the scene output returns to "
                             "fp32, so theta/psi/trajectory/losses are numerically unchanged. A "
                             "narrower alternative to --precision bf16, which also covers the trunk.")
    # Stage 2 mirrors Stage 1's --precision flag verbatim (run_stage1_cache_ddp.py): the SAME
    # autocast_ctx() around net(), the SAME fp32 loss on .float() outputs. Stage 1 defaults to
    # bf16 ("3.97x faster than fp32, peak mem 29.69 -> 19.72") and DrivoR trains 16-mixed, so
    # whole-net bf16 is the established way; the joint trainer had been left on fp32, which is
    # why its backward through the frozen ViT-L trunk cost ~45s/step (71% of the step).
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32",
                        help="autocast dtype for the whole net() forward (vision trunk + DiT theta + "
                             "scorer psi), matching Stage 1's --precision and DrivoR's 16-mixed. The "
                             "metric trajectory, Hungarian match, losses and oracle stay fp32: outputs "
                             "are upcast to .float() before anything downstream, so numerics are "
                             "unchanged (matched_ade / oracle_best_pdms identical to fp32). Default "
                             "fp32 = exact pre-flag op sequence; bf16 is the production setting. "
                             "Supersedes --vision-bf16 (covers the trunk and everything after it).")
    parser.add_argument("--future-target", default="none", choices=["none", "cache"],
                        help="'cache' reads real Teacher Future Register targets from --target-cache-root "
                             "and unfreezes the future branch; 'none' forces lambda_future=0 and freezes it")
    parser.add_argument("--target-cache-root", default="",
                        help="build_future_target_cache.py output holding standardizer.pt and targets/")
    parser.add_argument("--target-dim", type=int, default=256,
                        help="d_model: the DiT/Scene hidden width, and therefore the Wan payload's "
                             "hidden_dim (must match Stage 1)")
    parser.add_argument("--register-target-dim", type=int, default=0,
                        help="Future Register width, i.e. the future head's output. 0 derives it: from "
                             "the cache standardizer, else from --stage1-checkpoint, else from "
                             "--target-dim (the pre-query-extraction layout). The query-extracted "
                             "target is the Teacher's feature_dim, 1024, not d_model")
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--heads", type=int, default=4,
                        help="must equal the Wan payload's num_heads (4 at hidden_dim=256)")
    parser.add_argument("--wan-backbone", default=DEFAULT_WAN_BACKBONE,
                        help="ActionDiT_Wan22 payload from scripts/preprocess_wan_action_dit.py")
    parser.add_argument("--stage1-checkpoint", default="",
                        help="stage1_checkpoint.pt to restore eta+theta from; the scorer stays at scratch "
                             "init. Restoring one also puts the future query tokens in the DiT sequence, "
                             "since that is the sequence length Stage 1 trained")
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
    cache_standardizer: TargetStandardizer | None = None
    if args.future_target == "cache":
        if not args.target_cache_root:
            raise ValueError("--future-target cache requires --target-cache-root")
        if not standardizer_path(args.target_cache_root).exists():
            raise ValueError(
                f"{standardizer_path(args.target_cache_root)} is missing; run "
                f"scripts/build_future_target_cache.py --target-cache-root {args.target_cache_root} first"
            )
        # Loaded before the model is built rather than lazily per batch: its
        # feature_dim is the register target's width and its fingerprint is the
        # target-space identity every cached payload is checked against.  Both are
        # unknowable from a payload alone, since a cache built at another query
        # temperature holds different tensors of exactly the same shape.
        cache_standardizer, _ = load_standardizer(args.target_cache_root)
        train_future = args.lambda_future > 0.0
    else:
        if args.target_cache_root:
            raise ValueError("--target-cache-root needs --future-target cache")
        if args.lambda_future > 0.0:
            raise ValueError(
                "--lambda-future > 0 needs --future-target cache; without real Teacher targets the "
                "register would be supervised by all-zero placeholders"
            )
        train_future = False
    # A zero weight means the term is never built, so the future branch receives no
    # gradient and is frozen below.
    lambda_future = args.lambda_future if train_future else 0.0
    # ---- mined negatives (scorer preference only; orthogonal to the future branch) #
    # Disabled unless BOTH a root and a positive weight are given.  When disabled
    # the training-step arithmetic, the DDP-observed graph and the RNG stream are
    # bit-identical to the positives-only trainer: no extra scorer pass is built,
    # and the loss sum below does not even include a zero term.  Negatives never
    # touch the generator trajectory loss, the future cache or predict_future.
    args.negative_root = args.negative_root.strip()
    if args.negative_root and not Path(args.negative_root).is_dir():
        raise ValueError(f"--negative-root {args.negative_root} is not a directory")
    if args.negative_root and args.lambda_neg <= 0.0 and args.lambda_neg_match <= 0.0 and args.lambda_neg_score <= 0.0:
        raise ValueError(
            "--negative-root needs --lambda-neg > 0 (e.g. 0.1, DrivoR's preference_rd_weight) "
            "or --lambda-neg-match > 0 or --lambda-neg-score > 0; loading negatives no loss reads "
            "would silently waste the dataloader work"
        )
    if args.lambda_neg > 0.0 and not args.negative_root:
        raise ValueError("--lambda-neg > 0 needs --negative-root pointing at the mined poor_negative corpus")
    if args.lambda_neg_match > 0.0 and not args.negative_root:
        raise ValueError("--lambda-neg-match > 0 needs --negative-root pointing at the mined poor_negative corpus")
    if args.lambda_neg_score > 0.0 and not args.negative_root:
        raise ValueError("--lambda-neg-score > 0 needs --negative-root pointing at the mined poor_negative corpus")
    neg_enabled = bool(args.negative_root)
    # Score-part-only negatives: psi scores the mined poor_negatives (a utility margin
    # against the oracle-chosen proposal), with rd_loss and neg-match left at 0 so the
    # trajectory/GT-matching front is untouched.  Gated here so the forward's second
    # scorer pass and the loss term only exist when this run actually asks for them.
    neg_score_enabled = neg_enabled and args.lambda_neg_score > 0.0
    # ---- mined-negative FUTURE cache (the future-branch twin of Stage 1) ---------- #
    # Symmetric to the positive --target-cache-root: a second Stage-0 cache holding the
    # poor_negative future register targets, built over the negative future-video corpus
    # with the SAME standardizer.pt.  Its rows are appended after the positive GT rows in
    # the dataset, so gather_flow_data / hungarian_xy / multigt_trajectory_loss AND
    # dense_future_l1 all span positives and negatives -- the model predicts each bad
    # trajectory's OWN future instead of only being repelled by it in the score head.
    # Disabled unless a root is given; when disabled the trainer is bit-identical.
    args.negative_target_cache_root = args.negative_target_cache_root.strip()
    negative_future_enabled = bool(args.negative_target_cache_root)
    if negative_future_enabled:
        if args.future_target != "cache":
            raise ValueError(
                "--negative-target-cache-root needs --future-target cache: negatives supply future "
                "register rows, which only exist when the positive future branch is on"
            )
        if not standardizer_path(args.negative_target_cache_root).exists():
            raise ValueError(
                f"{standardizer_path(args.negative_target_cache_root)} is missing; build the negative "
                f"cache with build_future_target_cache.py re-using the positive cache's standardizer.pt"
            )
        # Checked once here rather than per scene: __getitem__ retries the next token on any
        # exception, so a target-space mismatch found per scene would be swallowed into
        # "could not load any scene near index" instead of naming the actual problem -- two
        # caches standardized into different spaces.
        negative_standardizer, _ = load_standardizer(args.negative_target_cache_root)
        if cache_standardizer is not None and negative_standardizer.fingerprint() != cache_standardizer.fingerprint():
            raise ValueError(
                f"negative cache {args.negative_target_cache_root} is standardized as "
                f"{negative_standardizer.fingerprint()} but the positive cache is "
                f"{cache_standardizer.fingerprint()}; rebuild the negative cache re-using the "
                f"positive cache's standardizer.pt so both live in one target space"
            )
    # Whether the future tokens sit in the sequence is a *different* question from
    # whether they train.  Restoring a Stage 1 checkpoint means the trunk was shaped
    # around 16 tokens and the future branch carries Stage 1's whole product, which
    # the action stream reads through self-attention even when Stage 2 has no future
    # targets of its own.  Dropping the tokens there would evaluate a model against
    # a sequence length it was never trained on.  Freezing (not removing) them is
    # also what keeps `future_head` out of the DDP reducer: a parameter with
    # `requires_grad=False` is never expected to produce a gradient.
    predict_future = train_future or bool(args.stage1_checkpoint)

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

    # ---- register target width --------------------------------------------- #
    # Resolved before the model is built, since it sizes the future head, and before
    # the Dataset is built, since that is what checks every cached payload against it.
    # An explicit --register-target-dim must *agree* with the other sources rather than
    # override them: a wrong width surfaces much later as a bare "size mismatch" from
    # strict=True, with nothing pointing back here.
    #
    # weights_only=True: the Stage 1 payload is tensors plus a scene-id list plus ints,
    # so nothing here needs to unpickle arbitrary Python objects.  Loaded this early
    # only for the width it records; the weights are applied below as before.
    stage1_payload = (
        torch.load(args.stage1_checkpoint, map_location="cpu", weights_only=True)
        if args.stage1_checkpoint
        else None
    )
    width_sources: dict[str, int] = {}
    if args.register_target_dim:
        width_sources["--register-target-dim"] = int(args.register_target_dim)
    if cache_standardizer is not None:
        width_sources["cache_standardizer"] = int(cache_standardizer.feature_dim)
    if stage1_payload is not None and "register_target_dim" in stage1_payload:
        width_sources["stage1_checkpoint"] = int(stage1_payload["register_target_dim"])
    if len(set(width_sources.values())) > 1:
        raise ValueError(
            f"conflicting Future Register widths {width_sources}: the query-extracted target is the "
            f"Teacher's own feature width and is recorded by both the cache and Stage 1, so a "
            f"disagreement means the cache and the Stage 1 checkpoint were built against different "
            f"targets. Rebuild one of them."
        )
    # The empty case is the pre-split layout, where the future head emitted d_model;
    # that fallback is what keeps an older Stage 1 checkpoint loadable at all.
    register_target_dim = next(iter(width_sources.values()), int(args.target_dim))
    # Written back so save_checkpoint's vars(args) records the resolved width rather
    # than a 0, which is what load_joint_checkpoint reads from a Stage 2 checkpoint.
    args.register_target_dim = register_target_dim
    log0(dctx, {"event": "register_width_resolved", "register_target_dim": register_target_dim,
                "d_model": args.target_dim, "sources": width_sources})

    dataset = NavtrainSceneDataset(
        candidates,
        args.sensor_root,
        args.multigt_root,
        args.cache_root,
        args.log_root,
        target_cache_root=args.target_cache_root or None,
        register_target_dim=register_target_dim,
        target_fingerprint=cache_standardizer.fingerprint() if cache_standardizer is not None else None,
        negative_root=args.negative_root or None,
        negative_target_cache_root=args.negative_target_cache_root or None,
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
    model, scorer = build_generator_and_scorer(
        target_dim=args.target_dim,
        register_target_dim=register_target_dim,
        heads=args.heads,
        depth=args.depth,
        dino_checkpoint=args.dino_checkpoint,
        wan_backbone=args.wan_backbone,
        grad_checkpointing=args.grad_checkpointing,
        vision_bf16=args.vision_bf16,
        device=device,
        predict_future=predict_future,
    )
    if args.wan_backbone and not Path(args.wan_backbone).exists():
        log0(dctx, {"event": "warn", "message": f"Wan backbone {args.wan_backbone} not found; theta starts random"})

    if stage1_payload is not None:
        model.load_state_dict(stage1_payload["model"], strict=True)
        log0(dctx, {"event": "stage1_restored", "checkpoint": str(args.stage1_checkpoint),
                    "scene_ids": len(stage1_payload.get("scene_ids", [])),
                    "register_target_dim": int(stage1_payload.get("register_target_dim", args.target_dim)),
                    "future_trainable": train_future,
                    "note": "eta+theta restored; the scorer stays at scratch init"})
        del stage1_payload

    if args.resume:
        # weights_only=True: the payload is state_dicts plus ints plus this script's
        # own argparse namespace, so nothing here needs arbitrary unpickling.
        resume_ckpt = torch.load(args.resume, map_location="cpu", weights_only=True)
        model.load_state_dict(resume_ckpt["model"], strict=True)
        scorer.load_state_dict(resume_ckpt["scorer"], strict=True)
        if dctx.main:
            print(json.dumps({"event": "resumed", "checkpoint": str(args.resume),
                              "prior_opt_step": int(resume_ckpt.get("opt_step", -1)),
                              "prior_epoch": int(resume_ckpt.get("epoch", -1))}), flush=True)
        del resume_ckpt

    opt_generator, opt_scorer, generator_parameters, scorer_parameters = build_optimizers(
        model, scorer, lr=args.lr, scorer_lr=args.scorer_lr,
        weight_decay=args.weight_decay, train_future=train_future,
    )

    net = JointStage2Module(model, scorer).to(device)
    if dctx.enabled:
        net = DDP(net, device_ids=[dctx.local_rank], output_device=dctx.local_rank, find_unused_parameters=args.find_unused, broadcast_buffers=False)

    cache_paths = {candidate.token: candidate.cache_path for candidate in candidates}
    oracle = DrivoRBatchMetricCacheOracle(cache_paths=cache_paths, evaluator_version=ORACLE_VERSION, parallel_workers=args.oracle_workers)
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
        "future_trainable": train_future,
        "future_in_sequence": model.predict_future,
        "future_parameters": model.theta.future_queries.numel() + sum(p.numel() for p in model.theta.future_head.parameters()),
        # [scene | action | register]: Scene is prepended into self-attention, FastWAM's
        # layout, and is additionally still the cross-attention context.
        "dit_token_order": ["scene", "action", "register"],
        "dit_sequence_tokens": model.scene_tokens + model.action_tokens + (model.register_tokens if model.predict_future else 0),
        # One mask for every stage and entry point.  This trainer reaches the DiT
        # through ``forward_flow``/``forward_pairs``; both run the same table Stage 1
        # ran, in which the Register reads the Action and the Action never reads back.
        "self_attention_mask": "single_segment_block",
        "segment_visibility": [list(row) for row in model.theta.SEGMENT_VISIBILITY],
        "action_reads_register": bool(model.theta.SEGMENT_VISIBILITY[1][2]),
        "register_reads_action": bool(model.theta.SEGMENT_VISIBILITY[2][1]),
        "trainable_parameters_by_group": {
            key: sum(parameter.numel() for parameter in value)
            for key, value in parameter_groups(model).items()
        },
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
        #   lambda_neg (rd)              = DrivoR preference_weight * preference_rd_weight = 1.0 * 0.1
        #   lambda_neg_match (experiment)= NVS-only, NOT DrivoR: regresses proposals toward bad
        #                                  trajectories so the scorer sees them as Q over scene_kv
        #   lambda_velocity              = SimWAM v_loss (and FastWAM's action loss) at 1.0
        #   dense/diversity              = DrivoR's own trajectory objective internals
        #   lambda_future                = FastWAM ema_joint's auxiliary-term scale (0.1)
        "loss_weights": {
            "lambda_action": args.lambda_action,
            "lambda_velocity": args.lambda_velocity,
            "lambda_score": args.lambda_score,
            "lambda_rank": args.lambda_rank,
            "lambda_future": lambda_future,
            "lambda_neg": args.lambda_neg if neg_enabled else 0.0,
            "rd_margin": args.rd_margin if neg_enabled else None,
            "lambda_neg_match": args.lambda_neg_match if neg_enabled else 0.0,
            "lambda_neg_score": args.lambda_neg_score if neg_enabled else 0.0,
            "negative_root": args.negative_root or None,
            "negative_target_cache_root": args.negative_target_cache_root or None,
            "dense_weight": args.dense_weight,
            "diversity_weight": args.diversity_weight,
            "dense_gate_m": args.dense_gate_m,
        },
        "oracle_workers": args.oracle_workers,
        "vision_bf16": bool(args.vision_bf16),
        "precision": args.precision,
        "precision_scope": "whole_net_forward; outputs upcast to fp32 for match/loss/oracle",
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
    # --phase-timing (diagnostic, default OFF): cumulative wall-clock per phase on
    # rank 0.  Every region is fenced by a cuda sync ONLY when enabled, so the
    # production path (flag off) runs no extra sync and its loss/grad math is
    # untouched -- the phase_acc writes are pure side-channel bookkeeping.
    phase_timing = bool(args.phase_timing)
    precision_bf16 = args.precision == "bf16"
    phase_acc = {"data": 0.0, "fwd": 0.0, "oracle": 0.0, "loss": 0.0, "bwd": 0.0, "opt": 0.0}
    phase_iters = 0
    _t_prev_end = time.perf_counter()

    def _phase_sync() -> None:
        if phase_timing and torch.cuda.is_available():
            torch.cuda.synchronize()

    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        epoch_start = time.time()
        for batch in loader:
            _t_data0 = _t_prev_end
            if phase_timing:
                phase_iters += 1
            observation, targets, scene_ids = batch_to_device(batch, device)
            _phase_sync(); _t_data1 = time.perf_counter()
            if phase_timing:
                phase_acc["data"] += _t_data1 - _t_data0
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

            # --precision bf16: Stage 1's autocast_ctx() and DrivoR's Lightning
            # `precision: 16-mixed` are the SAME thing -- the WHOLE forward runs under
            # autocast.  bf16 (scaler-free, fp32 exponent range) wraps net(), then the
            # small outputs are upcast back to fp32 so the metric trajectory, Hungarian
            # match, losses and the oracle all stay fp32-precise while the heavy
            # attention/matmul ran in bf16.  When --precision fp32 we enter NO context
            # (not enabled=False), so --vision-bf16's narrower trunk autocast, if any,
            # still applies untouched.
            amp_ctx = torch.autocast(device.type, dtype=torch.bfloat16) if precision_bf16 else nullcontext()
            with amp_ctx:
                endpoint, velocity, velocity_target, register, trajectory, logits, neg_logits = net(
                    observation, source, mode_codes, flow_data, timestep,
                    targets["neg_trajectory"] if neg_score_enabled else None,
                )
            if precision_bf16:
                endpoint = endpoint.float()
                velocity = velocity.float()
                trajectory = trajectory.float()
                logits = logits.float()
                if velocity_target is not None:
                    velocity_target = velocity_target.float()
                if register is not None:
                    register = register.float()
                if neg_logits is not None:
                    neg_logits = neg_logits.float()
            _phase_sync(); _t_fwd1 = time.perf_counter()
            if phase_timing:
                phase_acc["fwd"] += _t_fwd1 - _t_data1

            with torch.no_grad():
                try:
                    labels = oracle.evaluate(scene_ids, trajectory, batch_id=f"e{epoch}s{opt_step}")
                    labels_ok = (not labels.failure_diagnostics) and bool(labels.final_valid.all())
                except Exception:
                    labels, labels_ok = None, False
                assignment = hungarian_xy(trajectory, targets["gt_trajectory"], targets["gt_valid"])
            _phase_sync(); _t_orc1 = time.perf_counter()
            if phase_timing:
                phase_acc["oracle"] += _t_orc1 - _t_fwd1

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
            rd_loss = None
            neg_match_loss = None
            neg_score_loss = None
            if labels_ok:
                source_loss = scorer.source_loss(logits, labels)
                # NOTE: oracle.evaluate returns CPU labels (get_sub_score runs on CPU
                # numpy), unlike Phase B whose labels come from the .to(device) cache.
                # Align every label-derived tensor to logits' device before mixing with
                # the CUDA gt/utility/trajectory tensors.  chosen/rejected/pair_valid are
                # mined ONCE and shared by the rank and rd objectives below, exactly as
                # DrivoR's preference_losses mines a single pair that both terms consume.
                device = logits.device
                gt, gt_ok = first_valid_gt(targets["gt_trajectory"], targets["gt_valid"])
                gt = gt.to(device)
                gt_ok = gt_ok.to(device)
                final_score = labels.final_score.detach().to(device)
                scene_ok = labels.final_valid.all(dim=1).to(device) & gt_ok
                with torch.no_grad():
                    chosen, rejected, pair_valid = mine_pair(
                        trajectory.detach(), gt, final_score, args.unsafe_threshold)
                if args.lambda_rank > 0.0 or neg_score_enabled:
                    # Shared by the rank margin and the score-part negative margin.  When
                    # only --lambda-rank is on this is the identical expression/value as
                    # before, so the existing paths stay bit-identical.
                    utility = aggregate_score_logits(logits.clamp(-30.0, 30.0))
                else:
                    utility = None
                if args.lambda_rank > 0.0:
                    # DrivoR preference_module rank loss: push the scorer utility of the
                    # oracle-best (chosen) slot above an unsafe / worst (rejected) slot.
                    # Weighted INDEPENDENTLY of lambda_score so its effective coefficient
                    # equals DrivoR's preference_weight(1.0)*preference_rank_weight(0.1)
                    # = 0.1 (drivoR.yaml).  Nesting it under lambda_score diluted it 5x
                    # (0.2*0.1=0.02), which is why sel_pdms stalled below Phase B's 0.978.
                    pair_ok = (scene_ok & pair_valid).to(utility.dtype)
                    margin = F.softplus(-(utility.gather(1, chosen[:, None]).squeeze(1)
                                          - utility.gather(1, rejected[:, None]).squeeze(1)))
                    margin = torch.nan_to_num(margin, nan=0.0, posinf=0.0, neginf=0.0)
                    rank_loss = (margin * pair_ok).sum() / pair_ok.sum().clamp_min(1.0)
                else:
                    rank_loss = source_loss * 0.0
                if neg_enabled and args.lambda_neg > 0.0:
                    # DrivoR preference_module rd_loss -- the ONLY place DrivoR feeds an
                    # external negative_trajectory.  It repels the oracle-chosen proposal
                    # from the mined poor_negative in trajectory space:
                    #     rd = -||chosen_traj - negative||.clamp_max(rd_margin)
                    # ``chosen_traj`` indexes the ATTACHED endpoint (not the detached
                    # trajectory), so the repulsion gradient shapes THETA (the generator),
                    # never the scorer.  That is the faithful DrivoR behavior and the exact
                    # opposite of a scorer-utility margin on the negative.  DrivoR scores a
                    # single negative [B,8,3]; this corpus mines N per scene, so the same
                    # norm/mean/clamp broadcasts over [B,N,8,3] and is masked down to the
                    # valid mined negatives of scenes whose oracle pair is valid.  Gated on
                    # lambda_neg > 0 so the score-part-only run (--lambda-neg 0) never even
                    # computes this GT-matching-front term: rd_loss stays None, the loss line
                    # adds an honest zero, and theta's gradient comes only from loss_action.
                    chosen_traj = endpoint[torch.arange(endpoint.shape[0], device=device), chosen]
                    neg = targets["neg_trajectory"]
                    distance = torch.linalg.norm(chosen_traj[:, None] - neg, dim=-1).mean(-1)
                    rd_per = -distance.clamp_max(args.rd_margin)
                    rd_ok = (targets["neg_valid"] & scene_ok[:, None] & pair_valid[:, None]).to(rd_per.dtype)
                    rd_loss = (rd_per * rd_ok).sum() / rd_ok.sum().clamp_min(1.0)
                if neg_score_enabled:
                    # Score-part-only negatives (the point of --lambda-neg-score): psi has
                    # already scored each mined poor_negative over the SAME attached
                    # scene_context in forward's second scorer pass (neg_logits).  A
                    # DrivoR-style utility margin pushes the oracle-chosen good proposal
                    # ABOVE every valid negative: softplus(neg_utility - chosen_utility),
                    # minimized when chosen >> negative.  The scorer detaches the trajectory
                    # internally, so the gradient shapes psi (+ eta through the attached
                    # Scene K/V) and NEVER theta's trajectory generation -- the exact
                    # opposite of rd_loss/neg-match, which both land on the endpoint.  Like
                    # rd, it needs the oracle-chosen slot, so an invalid pair contributes
                    # an honest zero.
                    neg_utility = aggregate_score_logits(neg_logits.clamp(-30.0, 30.0))
                    chosen_utility = utility.gather(1, chosen[:, None]).squeeze(1)
                    neg_margin = F.softplus(neg_utility - chosen_utility[:, None])
                    neg_margin = torch.nan_to_num(neg_margin, nan=0.0, posinf=0.0, neginf=0.0)
                    neg_score_ok = (targets["neg_valid"] & scene_ok[:, None] & pair_valid[:, None]).to(neg_margin.dtype)
                    neg_score_loss = (neg_margin * neg_score_ok).sum() / neg_score_ok.sum().clamp_min(1.0)
            else:
                # Keep psi in the autograd graph (zero grad) so DDP never sees an
                # unused parameter and ranks stay in lockstep on a failed oracle.
                # rd also needs the oracle-chosen slot, so a failed oracle yields no
                # negative signal this step -- identical to DrivoR, whose rd_loss lives
                # inside the same preference_losses call that mines the chosen pair.
                oracle_failures += 1
                source_loss = (logits * 0.0).sum()
                rank_loss = source_loss * 0.0
            if neg_enabled and args.lambda_neg_match > 0.0:
                # EXPERIMENT (gated, default off): also regress some proposals toward the
                # mined bad trajectories, so the trajectory-Q the scorer attends with spans
                # good AND bad and the oracle-supervised scorer learns WHY a bad trajectory
                # is bad in this scene's context (e.g. a left turn the scene forbids when the
                # only GT goes straight).  Reuses the good-GT machinery verbatim: Hungarian
                # matches each valid negative to a distinct proposal slot, full-weight metric
                # L1 (no rescale) pulls that slot toward the bad trajectory, target detached
                # so the gradient shapes theta.  Distinct from rd_loss, which repels the
                # oracle-CHOSEN (good) slot -- a slot dragged toward a bad trajectory scores
                # low and is never the oracle-best, so the two objectives land on different
                # slots.  Independent of labels_ok (pure regression) and of the future branch.
                neg_assignment = hungarian_xy(endpoint.detach(), targets["neg_trajectory"], targets["neg_valid"])
                neg_match_loss = hungarian_matched_l1(endpoint, targets["neg_trajectory"], neg_assignment)
            loss = (args.lambda_action * loss_action
                    + args.lambda_velocity * loss_velocity
                    + args.lambda_score * source_loss
                    + args.lambda_rank * rank_loss
                    + lambda_future * loss_future)
            if neg_enabled:
                # Separate statement on purpose: with negatives disabled the sum
                # above is the exact positives-only expression, same terms, same
                # order, same autograd graph.  A step whose oracle failed (rd_loss
                # None) contributes an honest zero on theta's graph instead of
                # changing shape, so DDP's reducer still sees every parameter.
                loss = loss + args.lambda_neg * (rd_loss if rd_loss is not None else loss_action * 0.0)
                if args.lambda_neg_match > 0.0:
                    # Same separate-statement discipline: with lambda_neg_match=0 this
                    # block never runs, so the rd-only trainer stays bit-identical.  A
                    # scene with no valid negative yields hungarian_matched_l1's graph-
                    # preserving zero, keeping theta in the reducer.
                    loss = loss + args.lambda_neg_match * (neg_match_loss if neg_match_loss is not None else loss_action * 0.0)
                if neg_score_enabled:
                    # Same discipline: with lambda_neg_score=0 this never runs, so the
                    # rd-only and positives-only trainers stay bit-identical.  A step whose
                    # oracle failed (neg_score_loss None) contributes an honest zero on
                    # theta's graph, keeping every parameter in the DDP reducer.
                    loss = loss + args.lambda_neg_score * (neg_score_loss if neg_score_loss is not None else loss_action * 0.0)

            _phase_sync(); _t_loss1 = time.perf_counter()
            if phase_timing:
                phase_acc["loss"] += _t_loss1 - _t_orc1
            is_boundary = ((micro + 1) % accum) == 0
            sync_ctx = net.no_sync() if (dctx.enabled and accum > 1 and not is_boundary) else nullcontext()
            with sync_ctx:
                (loss / accum).backward()
            micro += 1
            _phase_sync(); _t_bwd1 = time.perf_counter()
            if phase_timing:
                phase_acc["bwd"] += _t_bwd1 - _t_loss1

            if not is_boundary:
                _t_prev_end = time.perf_counter()
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
                neg_per_scene = None
                neg_scene_frac = None
                neg_future_rows = None
                if neg_enabled:
                    # Coverage of the mined corpus inside the current global batch:
                    # ~25% of navtrain scenes legitimately carry zero negatives, so
                    # a silent drop in these two numbers means the loader, not the
                    # loss, stopped finding them.  Collective on every rank, exactly
                    # like the metrics reduce above.
                    neg_cov = torch.zeros(2, device=device)
                    neg_valid = targets["neg_valid"]
                    if neg_valid.numel():
                        neg_cov[0] = neg_valid.sum()
                        neg_cov[1] = (neg_valid.sum(dim=1) > 0).sum()
                    neg_cov = all_reduce_sum(neg_cov, dctx)
                    denom_neg = float(size * dctx.world)
                    neg_per_scene = round(float(neg_cov[0]) / denom_neg, 3)
                    neg_scene_frac = round(float(neg_cov[1]) / denom_neg, 4)
                if negative_future_enabled:
                    # How many poor_negative FUTURE rows were appended to the positive GT rows
                    # across the global batch this step.  A silent zero means the negative future
                    # cache stopped matching these tokens, so gather_flow_data / the Hungarian
                    # match / dense_future_l1 are quietly positives-only again.  Reduced on every
                    # rank exactly like the coverage above.
                    neg_future_cov = torch.tensor([float(batch.get("negative_future_rows", 0))], device=device)
                    neg_future_cov = all_reduce_sum(neg_future_cov, dctx)
                    neg_future_rows = int(neg_future_cov.item())
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
                    "rd_loss": round(float(rd_loss.detach()), 6) if rd_loss is not None else None,
                    "neg_match_loss": round(float(neg_match_loss.detach()), 6) if neg_match_loss is not None else None,
                    "neg_score_loss": round(float(neg_score_loss.detach()), 6) if neg_score_loss is not None else None,
                    "neg_per_scene": neg_per_scene,
                    "neg_scene_frac": neg_scene_frac,
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
                if negative_future_enabled:
                    record["negative_future_rows"] = neg_future_rows
                if phase_timing and phase_iters:
                    record.update({f"phase_{k}_ms": round(phase_acc[k] / phase_iters * 1000.0, 1)
                                   for k in phase_acc})
                    record["phase_sum_ms"] = round(sum(phase_acc.values()) / phase_iters * 1000.0, 1)
                history.append(record)
                log0(dctx, record)

            if args.save_every > 0 and (opt_step % args.save_every == 0 or opt_step == total_opt_steps):
                save_checkpoint(net, model, scorer, opt_generator, opt_scorer, epoch, opt_step, args, output_dir, dctx)

            _phase_sync(); _t_prev_end = time.perf_counter()
            if phase_timing:
                phase_acc["opt"] += _t_prev_end - _t_bwd1

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
        # The trained sequence length is a property of these weights, not of the
        # command line, so it is recorded from the model itself.
        "predict_future": bool(model.predict_future),
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
