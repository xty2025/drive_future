"""No-future-register ablation on navtrain.

This script is strictly additive: it only *imports and calls* ``nvs_v6`` and never
edits any package file, so the original architecture is untouched.  The training
path removes the future/register branch entirely:

* Generator forward is replicated here so the Action branch reads only
  ``[Scene, Action]`` (the detached register memory is never concatenated into the
  ActionBlock key/value).  ``phi``/register parameters are frozen, excluded from
  the optimizer, and are never called, so they cannot change.
* Supervision is Multi-GT trajectory only (``paired_residual_velocity_mse``) plus
  the DrivoR score loss on the scratch scorer.  There is no Stage-1 future-register
  pretraining and no ``paired_future_l1`` term.
* Data is navtrain: observations come from the real NAVSIM logs/sensors, Multi-GT
  trajectories from the DriveZero navtrain artifact, and score labels from the
  DrivoR-reproduction metric cache via ``NativeTrainOracle`` (official
  ``get_sub_score``).

Convergence is a two-phase schedule.  Phase A trains the generator (Multi-GT
residual-velocity loss) and the scorer jointly.  Phase B freezes the generator so
the 64 proposals, their scene context and their oracle labels become stationary,
caches them once, and trains the scorer alone until its selected PDMS reaches each
scene's best proposal.  Freezing for Phase B mirrors DrivoR, whose scorer learns
against a fixed proposal set; it removes the moving-target effect that keeps a
jointly-trained scorer from locking onto a ranking.  The scratch scorer never loads
DrivoR weights and ``phi`` stays byte-frozen throughout.
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
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

from nvs_v6.data import Observation, PairTargets, decode_trajectory
from nvs_v6.losses import (
    assert_disjoint_parameter_groups,
    hungarian_xy,
    paired_residual_velocity_mse,
)
from nvs_v6.model import DirectedPairDiT, parameter_groups
from nvs_v6.modes import ImmutableModeBuffer, StructuredSourceInitializer
from nvs_v6.native_drivor import NativeTrainOracle
from nvs_v6.real_data import RealNavsimSample, load_real_navsim_sample
from nvs_v6.scorer import ScratchDrivoRScorer
from nvs_v6.stages import _set_generator_trainability


DEFAULT_MULTIGT = "/mnt/zhaozc_workspace/project/drivezero_goal_gen/outputs/navsim_multigt_0p5s_8step_16gpu"
DEFAULT_SENSOR = "/mnt/data/navsim/trainval_all/trainval_sensor_blobs/trainval"
DEFAULT_LOG_ROOT = "/mnt/data/navsim/trainval_navsim_logs/trainval"
# DrivoR-reproduction online cache used by NativeTrainOracle (official get_sub_score).
DEFAULT_CACHE = "/mnt/zhaozc_workspace/official_DrivoR_clean/data_runtime/official_metric_cache_navtrain_20261005/metric_cache"
DEFAULT_DINO = "/mnt/LLM_weight/timm/vit_large_patch16_dinov3.lvd1689m/model.safetensors"


# --------------------------------------------------------------------------- #
# Register-free generator forward (reuses nvs_v6 weights; never touches phi).
# --------------------------------------------------------------------------- #
def _action_block_scene_action_only(block: torch.nn.Module, action: torch.Tensor, scene: torch.Tensor) -> torch.Tensor:
    """Replicate ``nvs_v6.model.ActionBlock.forward`` with register memory removed.

    The only difference from the package ActionBlock is that the detached register
    memory is *not* concatenated into ``key_value``; the Action branch attends over
    ``[Scene, Action]`` only.  All weights (``norm_attn``/``attn``/``norm_ffn``/
    ``ffn``) are the block's own parameters.
    """
    b, m, na, d = action.shape
    scene_expanded = scene.unsqueeze(1).expand(-1, m, -1, -1).reshape(b * m, scene.shape[1], d)
    action_flat = action.reshape(b * m, na, d)
    query = block.norm_attn(action_flat)
    key_value = torch.cat((scene_expanded, action_flat), dim=1)
    action_flat = action_flat + block.attn(query, key_value, key_value, need_weights=False)[0]
    action_flat = action_flat + block.ffn(block.norm_ffn(action_flat))
    return action_flat.reshape(b, m, na, d)


def forward_no_future(model: DirectedPairDiT, observation: Observation, source: torch.Tensor, modes: torch.Tensor):
    """Generator forward with the future/register branch fully removed.

    Returns ``(scene_context, endpoint, trajectory)``.  ``model.phi`` and the
    register tokens are never referenced.
    """
    clean_ego = model.eta.encode_clean_ego(observation)
    scene = model.eta(observation, clean_ego=clean_ego)
    action = model._initialize_action(source, modes, clean_ego)
    for scene_block, action_block in zip(model.scene_blocks, model.theta["blocks"]):
        scene = scene_block(scene)
        action = _action_block_scene_action_only(action_block, action, scene)
    endpoint = source + model.theta["trajectory_head"](action)
    source_yaw = torch.atan2(source[..., 2], source[..., 3])
    trajectory = decode_trajectory(endpoint, model.xy_scale, source_yaw=source_yaw)
    return scene, endpoint, trajectory


# --------------------------------------------------------------------------- #
# navtrain scene resolution: token -> cache/log -> frame_index -> real sample.
# --------------------------------------------------------------------------- #
def _load_cache_index_sidecar(sidecar: Path) -> dict[str, tuple[str, Path]] | None:
    """Load a memoized cache index, or ``None`` if it is absent/stale/corrupt."""
    if not sidecar.exists():
        return None
    try:
        raw = json.loads(sidecar.read_text())
    except Exception:
        return None
    if not isinstance(raw, dict) or not raw:
        return None
    index: dict[str, tuple[str, Path]] = {}
    for token, value in raw.items():
        try:
            log_name, cache_path = value
        except Exception:
            return None
        index[token] = (log_name, Path(cache_path))
    # Spot-check a few entries; if the cache moved or was pruned, rebuild.
    for token in list(index)[:5]:
        if not index[token][1].exists():
            return None
    return index


def _write_cache_index_sidecar(sidecar: Path, index: dict[str, tuple[str, Path]]) -> None:
    try:
        payload = {token: [log_name, str(cache_path)] for token, (log_name, cache_path) in index.items()}
        sidecar.write_text(json.dumps(payload))
    except Exception:
        pass


def build_cache_index(cache_root: str | Path) -> dict[str, tuple[str, Path]]:
    """One walk of the metric cache: ``token -> (log_name, metric_cache_path)``.

    The walk over the full navtrain metric cache is slow (~90-140s), so the
    result is memoized to a JSON sidecar beside the cache root.  The sidecar is
    validated (non-empty, spot-checked paths exist) before reuse; any problem
    transparently falls back to a fresh walk, so it is only ever a cache.
    """
    root = Path(cache_root)
    sidecar = root / "_nofuture_index.json"
    cached = _load_cache_index_sidecar(sidecar)
    if cached is not None:
        return cached
    index: dict[str, tuple[str, Path]] = {}
    for log_dir in root.iterdir():
        if not log_dir.is_dir():
            continue
        unknown = log_dir / "unknown"
        if not unknown.is_dir():
            continue
        for token_dir in unknown.iterdir():
            cache_path = token_dir / "metric_cache.pkl"
            if cache_path.exists():
                index[token_dir.name] = (log_dir.name, cache_path)
    _write_cache_index_sidecar(sidecar, index)
    return index


@dataclass
class SceneCandidate:
    token: str
    log_name: str
    log_path: Path
    frame_index: int
    cache_path: Path
    multigt_dir: Path


class _LogFrameResolver:
    """Resolve ``token -> frame_index`` inside a NAVSIM log, caching loaded logs."""

    def __init__(self, log_root: str | Path) -> None:
        self.log_root = Path(log_root)
        self._frames: dict[str, Any] = {}

    def _load(self, log_name: str):
        if log_name in self._frames:
            return self._frames[log_name]
        path = self.log_root / f"{log_name}.pkl"
        if not path.exists():
            self._frames[log_name] = None
            return None
        with path.open("rb") as handle:
            frames = pickle.load(handle)
        self._frames[log_name] = frames
        return frames

    def resolve(self, log_name: str, token: str) -> tuple[int, int] | None:
        frames = self._load(log_name)
        if frames is None:
            return None
        for index, frame in enumerate(frames):
            if str(frame.get("token")) == token:
                return index, len(frames)
        return None


def iter_scene_candidates(
    multigt_root: str | Path,
    cache_root: str | Path,
    log_root: str | Path,
    count: int,
    seed: int,
    scan_limit: int,
) -> Iterator[SceneCandidate]:
    """Yield navtrain tokens that have cache + log + in-range frame + Multi-GT."""
    cache_index = build_cache_index(cache_root)
    multigt_navtrain = Path(multigt_root) / "navtrain"
    tokens = sorted(os.listdir(multigt_navtrain))
    random.Random(seed).shuffle(tokens)
    resolver = _LogFrameResolver(log_root)
    emitted = 0
    scanned = 0
    for token in tokens:
        if emitted >= count or scanned >= scan_limit:
            return
        scanned += 1
        hit = cache_index.get(token)
        if hit is None:
            continue
        log_name, cache_path = hit
        multigt_dir = multigt_navtrain / token
        if not (multigt_dir / "gt_trajectories.npy").exists():
            continue
        resolved = resolver.resolve(log_name, token)
        if resolved is None:
            continue
        frame_index, total = resolved
        if frame_index < 4 or frame_index + 8 >= total:
            continue
        emitted += 1
        yield SceneCandidate(
            token=token,
            log_name=log_name,
            log_path=Path(log_root) / f"{log_name}.pkl",
            frame_index=frame_index,
            cache_path=cache_path,
            multigt_dir=multigt_dir,
        )


def load_samples(
    candidates: list[SceneCandidate],
    sensor_root: str,
    multigt_root: str,
    cache_root: str,
    needed: int,
) -> tuple[list[RealNavsimSample], list[dict[str, str]]]:
    samples: list[RealNavsimSample] = []
    skipped: list[dict[str, str]] = []
    for candidate in candidates:
        if len(samples) >= needed:
            break
        try:
            sample = load_real_navsim_sample(
                candidate.log_path,
                sensor_root,
                multigt_root,
                frame_index=candidate.frame_index,
                cache_root=cache_root,
            )
        except Exception as error:  # keep the smoke robust to a single bad scene
            skipped.append({"token": candidate.token, "error": repr(error)})
            continue
        samples.append(sample)
    return samples, skipped


# --------------------------------------------------------------------------- #
# Batch collation / device move.
# --------------------------------------------------------------------------- #
def collate(samples: list[RealNavsimSample]) -> tuple[Observation, PairTargets]:
    observation = Observation(
        rgb_history=torch.cat([s.observation.rgb_history for s in samples], dim=0),
        history_valid=torch.cat([s.observation.history_valid for s in samples], dim=0),
        ego_history=torch.cat([s.observation.ego_history for s in samples], dim=0),
        route_command=torch.cat([s.observation.route_command for s in samples], dim=0),
        relative_ego_poses=torch.cat([s.observation.relative_ego_poses for s in samples], dim=0),
        camera_metadata=[s.observation.camera_metadata for s in samples],
    )
    max_gt = max(s.targets.gt_trajectory.shape[1] for s in samples)

    def pad(value: torch.Tensor, fill: Any) -> torch.Tensor:
        padded = value.new_full((1, max_gt, *value.shape[2:]), fill)
        padded[:, : value.shape[1]] = value
        return padded

    targets = PairTargets(
        gt_trajectory=torch.cat([pad(s.targets.gt_trajectory, 0) for s in samples], dim=0),
        gt_encoded=torch.cat([pad(s.targets.gt_encoded, 0) for s in samples], dim=0),
        gt_valid=torch.cat([pad(s.targets.gt_valid, False) for s in samples], dim=0),
        future_register=torch.cat([pad(s.targets.future_register, 0) for s in samples], dim=0),
        future_chunk_valid=torch.cat([pad(s.targets.future_chunk_valid, False) for s in samples], dim=0),
        future_quality=torch.cat([pad(s.targets.future_quality, 0) for s in samples], dim=0),
        pair_ids=[s.targets.pair_ids[0] for s in samples],
    )
    return observation, targets


def move_to_device(observation: Observation, targets: PairTargets, device: torch.device) -> tuple[Observation, PairTargets]:
    observation = Observation(
        **{key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in observation.__dict__.items()}
    )
    targets = PairTargets(
        **{key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in targets.__dict__.items()}
    )
    return observation, targets


# --------------------------------------------------------------------------- #
# Optimizers: train eta + theta (generator) and psi (scorer); freeze phi.
# --------------------------------------------------------------------------- #
def build_no_future_optimizers(model: DirectedPairDiT, scorer: ScratchDrivoRScorer, lr: float, scorer_lr: float):
    _set_generator_trainability(model, stage=2)  # freeze DINO base, keep LoRA + NVS scene/theta trainable
    # Freeze the entire future/register branch; it is never called by forward_no_future.
    for parameter in model.phi.parameters():
        parameter.requires_grad_(False)
    model.phi_time_queries.requires_grad_(False)
    for parameter in scorer.parameters():
        parameter.requires_grad_(True)

    groups = parameter_groups(model)  # phi excluded because it is frozen
    if groups["phi"]:
        raise AssertionError("phi/register branch must be frozen in the no-future experiment")
    generator_parameters = list(groups["eta"]) + list(groups["theta"])
    scorer_parameters = list(scorer.parameters())
    assert_disjoint_parameter_groups(generator_parameters, scorer_parameters)
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    if {id(p) for p in generator_parameters} != expected:
        raise AssertionError("generator optimizer must contain every trainable generator parameter exactly once")
    if len(generator_parameters) != len({id(p) for p in generator_parameters}):
        raise AssertionError("duplicate parameters in generator optimizer")
    return torch.optim.AdamW(generator_parameters, lr=lr), torch.optim.AdamW(scorer_parameters, lr=scorer_lr)


def _norm(parameters) -> float:
    values = [p.detach().float().norm() for p in parameters]
    return float(torch.stack(values).norm()) if values else 0.0


def collect_groups(model: DirectedPairDiT, scorer: ScratchDrivoRScorer) -> dict[str, list]:
    eta = [p for n, p in model.named_parameters() if n.startswith("eta.") or n.startswith("scene_blocks.")]
    theta = [p for n, p in model.named_parameters() if n.startswith("theta.") or n == "theta_time_embedding"]
    phi = [p for n, p in model.named_parameters() if n.startswith("phi.") or n == "phi_time_queries"]
    return {"eta": eta, "theta": theta, "phi": phi, "psi": list(scorer.parameters())}


def snapshot(groups: dict[str, list]) -> dict[str, float]:
    return {key: _norm(value) for key, value in groups.items()}


# --------------------------------------------------------------------------- #
# Training steps and schedules.
# --------------------------------------------------------------------------- #
def _nonzero_grad(params) -> int:
    return sum(1 for p in params if p.grad is not None and float(p.grad.detach().abs().sum()) > 0.0)


def apply_schedule(optimizer: torch.optim.Optimizer, base_lr: float, step: int, total: int, warmup: int) -> float:
    """Linear warmup then cosine decay to 5% of base LR (DrivoR uses a warmup+decay schedule)."""
    if warmup > 0 and step <= warmup:
        factor = step / warmup
    else:
        progress = (step - warmup) / max(1, total - warmup)
        factor = 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))
    for group in optimizer.param_groups:
        group["lr"] = base_lr * factor
    return base_lr * factor


def generator_step(
    observation: Observation,
    targets: PairTargets,
    model: DirectedPairDiT,
    opt_generator: torch.optim.Optimizer,
    modes: ImmutableModeBuffer,
    initializer: StructuredSourceInitializer,
    lambda_action: float,
) -> dict[str, Any]:
    """Phase A: Multi-GT residual-velocity supervision only.

    The DrivoR score gradient is detached from the generator, so training the
    generator on the action loss alone is equivalent to the joint step for eta/theta
    while avoiding the expensive per-step PDM oracle.  ``phi`` is frozen and never
    called, so it receives no gradient.
    """
    model.train()
    opt_generator.zero_grad(set_to_none=True)
    mode_codes = modes.expand_for_batch(observation.rgb_history.shape[0], observation.rgb_history.device)
    source = initializer.clean(observation.ego_history, mode_codes)
    _, endpoint, trajectory = forward_no_future(model, observation, source, mode_codes)
    with torch.no_grad():
        assignment = hungarian_xy(trajectory.detach(), targets.gt_trajectory, targets.gt_valid)
    loss_action = paired_residual_velocity_mse(endpoint, source, targets.gt_encoded, assignment)
    (lambda_action * loss_action).backward()
    named = list(model.named_parameters())
    grad = {
        "eta": _nonzero_grad([p for n, p in named if n.startswith("eta.") or n.startswith("scene_blocks.")]),
        "theta": _nonzero_grad([p for n, p in named if n.startswith("theta.") or n == "theta_time_embedding"]),
        "phi": _nonzero_grad([p for n, p in named if n.startswith("phi.") or n == "phi_time_queries"]),
    }
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
    opt_generator.step()
    return {
        "loss_action": float(loss_action.detach()),
        "matched": [len(a.as_pairs()) for a in assignment],
        "grad": grad,
    }


@torch.no_grad()
def generator_eval(
    observation: Observation,
    targets: PairTargets,
    model: DirectedPairDiT,
    modes: ImmutableModeBuffer,
    initializer: StructuredSourceInitializer,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Matched-GT ADE per scene (generator quality only; no scorer, no oracle)."""
    model.eval()
    mode_codes = modes.expand_for_batch(observation.rgb_history.shape[0], device)
    source = initializer.clean(observation.ego_history, mode_codes)
    _, _, trajectory = forward_no_future(model, observation, source, mode_codes)
    assignment = hungarian_xy(trajectory, targets.gt_trajectory, targets.gt_valid)
    rows = []
    for index in range(len(assignment)):
        pairs = assignment[index].as_pairs()
        ades = [
            float((trajectory[index, cand, :, :2] - targets.gt_trajectory[index, gt, :, :2]).norm(dim=-1).mean())
            for gt, cand in pairs
        ]
        rows.append({
            "matched_gt": len(pairs),
            "mean_ade_m": float(np.mean(ades)) if ades else float("nan"),
            "max_ade_m": float(np.max(ades)) if ades else float("nan"),
        })
    return rows


@torch.no_grad()
def final_eval(
    observation: Observation,
    targets: PairTargets,
    scene_ids: list[str],
    model: DirectedPairDiT,
    scorer: ScratchDrivoRScorer,
    oracle: NativeTrainOracle,
    modes: ImmutableModeBuffer,
    initializer: StructuredSourceInitializer,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    scorer.eval()
    mode_codes = modes.expand_for_batch(observation.rgb_history.shape[0], device)
    source = initializer.clean(observation.ego_history, mode_codes)
    scene, endpoint, trajectory = forward_no_future(model, observation, source, mode_codes)
    logits = scorer(scene, trajectory, None)
    selected, utility = scorer.select(logits)
    labels = oracle.evaluate(scene_ids, trajectory, batch_id="final-eval")
    assignment = hungarian_xy(trajectory, targets.gt_trajectory, targets.gt_valid)
    rows = []
    for index, scene_id in enumerate(scene_ids):
        sel = int(selected[index])
        matched = assignment[index].as_pairs()
        ades = []
        for gt_index, candidate_index in matched:
            ade = float((trajectory[index, candidate_index, :, :2] - targets.gt_trajectory[index, gt_index, :, :2]).norm(dim=-1).mean())
            ades.append(ade)
        rows.append(
            {
                "scene_id": scene_id,
                "matched_gt": len(matched),
                "selected_candidate": sel,
                "selected_pdms": float(labels.final_score[index, sel]),
                "oracle_best_pdms": float(labels.final_score[index].max()),
                "matches_oracle": bool(sel == int(labels.final_score[index].argmax())),
                "matched_gt_mean_ade_m": float(np.mean(ades)) if ades else float("nan"),
                "matched_gt_max_ade_m": float(np.max(ades)) if ades else float("nan"),
            }
        )
    return {
        "score_shape": list(logits.shape),
        "selection_is_finite": bool(torch.isfinite(utility).all()),
        "scenes": rows,
    }


# --------------------------------------------------------------------------- #
# Phase B: frozen generator -> stationary scorer targets (DrivoR trains its
# scorer against a fixed proposal set; freezing the generator reproduces that
# stationarity so the ranking head can actually converge on two scenes).
# --------------------------------------------------------------------------- #
def freeze_generator(model: DirectedPairDiT) -> None:
    """Freeze every generator parameter (eta/theta/phi) for the scorer-only phase."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)


@torch.no_grad()
def cache_scorer_inputs(
    observation: Observation,
    scene_ids: list[str],
    model: DirectedPairDiT,
    oracle: NativeTrainOracle,
    modes: ImmutableModeBuffer,
    initializer: StructuredSourceInitializer,
    device: torch.device,
    batch_id: str = "phase-b-cache",
) -> tuple[torch.Tensor, torch.Tensor, Any]:
    """Run the frozen generator once and cache scene context, proposals and labels."""
    model.eval()
    mode_codes = modes.expand_for_batch(observation.rgb_history.shape[0], device)
    source = initializer.clean(observation.ego_history, mode_codes)
    scene, _, trajectory = forward_no_future(model, observation, source, mode_codes)
    labels = oracle.evaluate(scene_ids, trajectory, batch_id=batch_id)
    if labels.failure_diagnostics or not labels.final_valid.all():
        raise RuntimeError("do not train the scratch scorer on failed oracle evaluations")
    return scene.detach(), trajectory.detach(), labels


def scorer_only_step(
    scene_cache: torch.Tensor,
    trajectory_cache: torch.Tensor,
    labels_cache: Any,
    scorer: ScratchDrivoRScorer,
    opt_scorer: torch.optim.Optimizer,
    lambda_score: float,
) -> float:
    """One Phase-B step: train psi only on stationary cached inputs/labels."""
    scorer.train()
    opt_scorer.zero_grad(set_to_none=True)
    logits = scorer(scene_cache, trajectory_cache, None, detach_inputs=True)
    loss_score = scorer.source_loss(logits, labels_cache)
    (lambda_score * loss_score).backward()
    torch.nn.utils.clip_grad_norm_([p for p in scorer.parameters() if p.requires_grad], 1.0)
    opt_scorer.step()
    return float(loss_score.detach())


def eval_selection(ev: dict[str, Any]) -> dict[str, Any]:
    """Reduce a final_eval result to the convergence signals we care about."""
    scenes = ev["scenes"]
    return {
        "selected_pdms": [round(s["selected_pdms"], 4) for s in scenes],
        "oracle_best_pdms": [round(s["oracle_best_pdms"], 4) for s in scenes],
        "matches_oracle": [s["matches_oracle"] for s in scenes],
        "mean_ade_m": [round(s["matched_gt_mean_ade_m"], 3) for s in scenes],
        "max_ade_m": [round(s["matched_gt_max_ade_m"], 3) for s in scenes],
    }


def selection_converged(ev: dict[str, Any], tol: float) -> bool:
    """Converged when every scene's selected PDMS is within tol of its best proposal."""
    return all(
        s["selected_pdms"] >= s["oracle_best_pdms"] - tol
        for s in ev["scenes"]
    )


# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gen-steps", type=int, default=1500, help="Phase A: generator (Multi-GT) steps")
    parser.add_argument("--scorer-steps", type=int, default=500, help="Phase B: frozen-generator scorer-only steps")
    parser.add_argument("--eval-every", type=int, default=25, help="evaluate ADE / selection this often in each phase")
    parser.add_argument("--log-every", type=int, default=50, help="print a raw step line this often")
    parser.add_argument("--warmup", type=int, default=50, help="linear LR warmup steps per phase")
    parser.add_argument("--pdms-tol", type=float, default=0.02, help="Phase B early-stop: selected within tol of oracle best")
    parser.add_argument("--scenes", type=int, default=2, help="navtrain scenes held fixed across the run")
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--candidates", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4, help="generator (eta+theta) LR; proven-stable v6 value")
    parser.add_argument("--scorer-lr", type=float, default=5e-4, help="scorer (psi) LR; larger than generator per request")
    parser.add_argument("--lambda-action", type=float, default=1.0)
    parser.add_argument("--lambda-score", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--scan-limit", type=int, default=50000)
    parser.add_argument("--multigt-root", default=DEFAULT_MULTIGT)
    parser.add_argument("--sensor-root", default=DEFAULT_SENSOR)
    parser.add_argument("--log-root", default=DEFAULT_LOG_ROOT)
    parser.add_argument("--cache-root", default=DEFAULT_CACHE)
    parser.add_argument("--dino-checkpoint", default=DEFAULT_DINO)
    parser.add_argument("--output-dir", default="artifacts/navtrain_nofuture_overfit")
    parser.add_argument("--dry-run", action="store_true", help="resolve + load scenes only, then exit")
    args = parser.parse_args()

    if args.candidates != 64:
        raise ValueError("the DrivoR online oracle requires exactly 64 candidates")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("the real navtrain no-future run requires the configured CUDA/PPU device")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- resolve navtrain scenes ------------------------------------------ #
    resolve_start = time.time()
    candidates = list(
        iter_scene_candidates(
            args.multigt_root, args.cache_root, args.log_root,
            count=args.scenes * 3 + 2, seed=args.seed, scan_limit=args.scan_limit,
        )
    )
    if not candidates:
        raise RuntimeError("no navtrain scenes resolved with cache + log + multigt")
    samples, skipped = load_samples(candidates, args.sensor_root, args.multigt_root, args.cache_root, args.scenes)
    if len(samples) < args.scenes:
        raise RuntimeError(f"only resolved {len(samples)}/{args.scenes} navtrain scenes; skipped={skipped}")
    scene_ids = [s.scene_id for s in samples]
    print(json.dumps({
        "event": "scenes_resolved",
        "scene_ids": scene_ids,
        "log_names": [s.log_name for s in samples],
        "frame_indices": [s.frame_index for s in samples],
        "gt_counts": [int(s.targets.gt_valid.shape[1]) for s in samples],
        "gt_valid": [int(s.targets.gt_valid[0].sum()) for s in samples],
        "skipped": skipped,
        "resolve_elapsed_s": round(time.time() - resolve_start, 2),
    }), flush=True)

    observation, targets = collate(samples)
    observation, targets = move_to_device(observation, targets, device)
    if args.dry_run:
        print(json.dumps({
            "event": "dry_run_complete",
            "rgb_history": list(observation.rgb_history.shape),
            "gt_trajectory": list(targets.gt_trajectory.shape),
            "gt_valid": targets.gt_valid[0].tolist(),
        }), flush=True)
        return 0

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
    oracle = NativeTrainOracle(samples, cache_root=args.cache_root)
    modes = ImmutableModeBuffer().subset(args.candidates).to(device)
    initializer = StructuredSourceInitializer(jitter_std=0.0)

    opt_generator, opt_scorer = build_no_future_optimizers(model, scorer, lr=args.lr, scorer_lr=args.scorer_lr)
    groups = collect_groups(model, scorer)
    norms_before = snapshot(groups)
    trainable = {
        "generator_eta_theta": sum(p.numel() for p in groups["eta"] + groups["theta"] if p.requires_grad),
        "scorer_psi": sum(p.numel() for p in groups["psi"] if p.requires_grad),
        "phi_frozen": sum(p.numel() for p in groups["phi"]),
        "phi_requires_grad": any(p.requires_grad for p in groups["phi"]),
    }
    print(json.dumps({"event": "setup", "trainable": trainable, "norms_before": norms_before,
                      "lr": args.lr, "scorer_lr": args.scorer_lr}), flush=True)

    # ---- Phase A: generator only, Multi-GT residual-velocity supervision --- #
    start = time.time()
    phase_a_steps: list[dict[str, Any]] = []
    phase_a_evals: list[dict[str, Any]] = []
    for step in range(args.gen_steps):
        cur_lr = apply_schedule(opt_generator, args.lr, step + 1, args.gen_steps, args.warmup)
        record = generator_step(observation, targets, model, opt_generator, modes, initializer, args.lambda_action)
        record["step"] = step + 1
        record["lr"] = cur_lr
        phase_a_steps.append(record)
        if (step + 1) % args.log_every == 0 or (step + 1) == args.gen_steps:
            print(json.dumps({
                "phase": "A", "step": step + 1,
                "loss_action": round(record["loss_action"], 6),
                "lr": round(cur_lr, 7),
                "grad_eta_nonzero": record["grad"]["eta"],
                "grad_theta_nonzero": record["grad"]["theta"],
                "grad_phi_nonzero": record["grad"]["phi"],
            }), flush=True)
        if (step + 1) % args.eval_every == 0 or (step + 1) == args.gen_steps:
            ade = generator_eval(observation, targets, model, modes, initializer, device)
            sel = {"mean_ade_m": [round(r["mean_ade_m"], 3) for r in ade], "max_ade_m": [round(r["max_ade_m"], 3) for r in ade]}
            phase_a_evals.append({"step": step + 1, **sel})
            print(json.dumps({"phase": "A", "eval_step": step + 1, **sel}), flush=True)

    # ---- Phase B: freeze generator -> stationary labels -> train scorer ---- #
    freeze_generator(model)
    generator_frozen = not any(p.requires_grad for p in model.parameters())
    scene_cache, trajectory_cache, labels_cache = cache_scorer_inputs(
        observation, scene_ids, model, oracle, modes, initializer, device
    )
    print(json.dumps({
        "event": "phase_b_start",
        "generator_frozen": generator_frozen,
        "scorer_trainable_params": sum(p.numel() for p in scorer.parameters() if p.requires_grad),
        "cached_oracle_best": [round(float(labels_cache.final_score[b].max()), 4) for b in range(len(scene_ids))],
    }), flush=True)

    phase_b_steps: list[dict[str, Any]] = []
    phase_b_evals: list[dict[str, Any]] = []
    converged_at: int | None = None
    for step in range(args.scorer_steps):
        cur_lr = apply_schedule(opt_scorer, args.scorer_lr, step + 1, args.scorer_steps, args.warmup)
        loss_score = scorer_only_step(scene_cache, trajectory_cache, labels_cache, scorer, opt_scorer, args.lambda_score)
        phase_b_steps.append({"step": step + 1, "loss_score": loss_score, "lr": cur_lr})
        is_eval = (step + 1) % args.eval_every == 0 or (step + 1) == args.scorer_steps
        if is_eval:
            print(json.dumps({"phase": "B", "step": step + 1, "loss_score": round(loss_score, 6), "lr": round(cur_lr, 7)}), flush=True)
            ev = final_eval(observation, targets, scene_ids, model, scorer, oracle, modes, initializer, device)
            sel = eval_selection(ev)
            phase_b_evals.append({"step": step + 1, **sel})
            print(json.dumps({"phase": "B", "eval_step": step + 1, **sel}), flush=True)
            if selection_converged(ev, args.pdms_tol):
                converged_at = step + 1
                break

    norms_after = snapshot(groups)
    evaluation = final_eval(observation, targets, scene_ids, model, scorer, oracle, modes, initializer, device)
    sel_final = eval_selection(evaluation)
    finite = (
        all(np.isfinite(r["loss_action"]) for r in phase_a_steps)
        and all(np.isfinite(r["loss_score"]) for r in phase_b_steps)
    )
    result = {
        "experiment": "navtrain_no_future_register_two_phase_overfit",
        "package": "nvs_v6 (unmodified; register/future branch never called)",
        "device": str(device),
        "seed": args.seed,
        "gen_steps": args.gen_steps,
        "scorer_steps": args.scorer_steps,
        "scorer_steps_run": len(phase_b_steps),
        "scorer_converged_at_step": converged_at,
        "eval_every": args.eval_every,
        "pdms_tol": args.pdms_tol,
        "scenes": scene_ids,
        "log_names": [s.log_name for s in samples],
        "frame_indices": [s.frame_index for s in samples],
        "depth": args.depth,
        "candidates": args.candidates,
        "lr": args.lr,
        "scorer_lr": args.scorer_lr,
        "lambda_action": args.lambda_action,
        "lambda_score": args.lambda_score,
        "cache_root": args.cache_root,
        "multigt_root": args.multigt_root,
        "sensor_root": args.sensor_root,
        "dino_checkpoint": args.dino_checkpoint,
        "scorer_contract": scorer.feature_audit,
        "oracle_evaluator": oracle.evaluate.__qualname__,
        "trainable": trainable,
        "future_register_used": False,
        "stage1_used": False,
        "phase_a_steps": phase_a_steps,
        "phase_a_evals": phase_a_evals,
        "phase_b_steps": phase_b_steps,
        "phase_b_evals": phase_b_evals,
        "finite_losses": bool(finite),
        "norms_before": norms_before,
        "norms_after": norms_after,
        "parameter_updates": {key: norms_after[key] != norms_before[key] for key in norms_before},
        "phi_frozen_unchanged": bool(abs(norms_after["phi"] - norms_before["phi"]) < 1e-9),
        "generator_frozen_phase_b": generator_frozen,
        "evaluation": evaluation,
        "final_selection": sel_final,
        "elapsed_s": round(time.time() - start, 2),
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    torch.save({"model": model.state_dict(), "scorer": scorer.state_dict()}, output_dir / "checkpoint.pt")

    ade_curve = (
        [[e["step"], e["mean_ade_m"]] for e in phase_a_evals]
        + [[args.gen_steps + e["step"], e["mean_ade_m"]] for e in phase_b_evals]
    )
    print(json.dumps({
        "event": "summary",
        "finite_losses": result["finite_losses"],
        "phi_frozen_unchanged": result["phi_frozen_unchanged"],
        "parameter_updates": result["parameter_updates"],
        "generator_frozen_phase_b": generator_frozen,
        "scorer_converged_at_step": converged_at,
        "phase_a_loss_action_first": phase_a_steps[0]["loss_action"] if phase_a_steps else None,
        "phase_a_loss_action_last": phase_a_steps[-1]["loss_action"] if phase_a_steps else None,
        "phase_a_mean_ade_first": phase_a_evals[0]["mean_ade_m"] if phase_a_evals else None,
        "phase_a_mean_ade_last": phase_a_evals[-1]["mean_ade_m"] if phase_a_evals else None,
        "phase_b_loss_score_last": phase_b_steps[-1]["loss_score"] if phase_b_steps else None,
        "mean_ade_curve": ade_curve,
        "selected_pdms_curve": [[e["step"], e["selected_pdms"]] for e in phase_b_evals],
        "oracle_best_curve": [[e["step"], e["oracle_best_pdms"]] for e in phase_b_evals],
        "final_selection": sel_final,
        "elapsed_s": result["elapsed_s"],
    }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
