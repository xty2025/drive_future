"""Offline PDMS evaluation + NAVSIM ego-BEV visualization for the JOINT run.

Strictly additive: this script only IMPORTS ``nvs_v6_wan`` and reuses the joint
trainer's own helpers (``build_scene_index`` / ``NavtrainSceneDataset`` /
``collate`` / ``batch_to_device`` / ``JointStage2Module`` / ``batch_eval_metrics``
/ ``hungarian_xy`` / ``DrivoRBatchMetricCacheOracle``) so the offline numbers are
produced by the exact same code path as the in-training eval.  It never edits any
package file.

What it reports for a checkpoint trained on ``navtrain``:
  * selected PDMS   -- official DrivoR ``get_sub_score`` of the candidate the
                       learned scorer picks (``scorer.select``).
  * oracle-best PDMS-- best official score among the 64 candidates (upper bound).
  * oracle-match    -- how often the scorer picks the exact argmax candidate.
  * matched-GT ADE  -- mean L2 (m) between each Hungarian-matched candidate and
                       its official Multi-GT trajectory.

For the first ``--bev-scenes`` scenes it renders a top-down ego-BEV (x forward,
y left, meters) overlaying the ego history, all 64 faint candidates (colored by
official PDMS), the scorer-SELECTED trajectory, the ORACLE-BEST trajectory, and
every valid official GT trajectory.

HONESTY: ``navtrain`` is the TRAINING split, so this is IN-DISTRIBUTION eval on
scenes the model has seen -- it is NOT a held-out or leaderboard number, and the
checkpoint is early.  Do not read these as full-NAVSIM metrics.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# Reuse the joint trainer's exact data + eval path (module import is side-effect
# free: its main() is guarded by ``if __name__ == "__main__"``).
import run_navtrain_wan_joint_ddp as trainer  # noqa: E402
from nvs_v6_wan.data import (  # noqa: E402
    Observation,
    drivor_observable_ego_feature,
    drivor_route_command,
    encode_trajectory,
)
from nvs_v6_wan.real_data import _frame_image, _relative_pose  # noqa: E402


# navtest has no Multi-GT; its official metric cache lives under a nested
# ``metric_cache`` dir and its GT is the logged human future (8 ego poses).
NAVTEST_CACHE = "/mnt/zhaozc_workspace/official_DrivoR_clean/data_runtime/official_metric_cache_navtest_20261006/metric_cache"
NAVTEST_LOG_ROOT = "/mnt/data/navsim/test_navsim_logs/test"
NAVTEST_SENSOR_ROOT = "/mnt/data/navsim/test_sensor_blobs/test"
ORACLE_VERSION_NAVTEST = "drivor_get_sub_score_navtest_20261006"


# --------------------------------------------------------------------------- #
# BEV rendering (self-contained matplotlib; ego-relative meter frame).
# --------------------------------------------------------------------------- #
def _prepend_origin(traj: np.ndarray) -> np.ndarray:
    """Prepend the ego origin (0,0,0) so each path starts at the vehicle."""
    return np.vstack([np.zeros((1, traj.shape[-1]), dtype=traj.dtype), traj])


def draw_bev_eval(
    scene_id: str,
    candidates: np.ndarray,       # [64, 8, 3] meters, ego frame (x fwd, y left)
    final_score: np.ndarray,      # [64] official PDMS (may contain NaN)
    selected_idx: int,
    oracle_idx: int,
    gt_paths: np.ndarray,         # [K, 8, 3]
    gt_valid: np.ndarray,         # [K] bool
    ego_history: np.ndarray,      # [5, 3] relative poses, ends at origin
    out_path: Path,
    sel_pdms: float,
    best_pdms: float,
    ade_row: float,
    opt_step: int,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 8), dpi=150)

    finite = final_score[np.isfinite(final_score)]
    norm = plt.Normalize(float(finite.min()), float(finite.max())) if finite.size else plt.Normalize(0.0, 1.0)
    cmap = plt.cm.viridis

    # faint 64 candidates, colored by official PDMS
    for i, cand in enumerate(candidates):
        path = _prepend_origin(cand)
        color = cmap(norm(final_score[i])) if np.isfinite(final_score[i]) else "lightgray"
        ax.plot(path[:, 0], path[:, 1], color=color, alpha=0.30, lw=0.7, zorder=2)

    # official Multi-GT trajectories (black dashed)
    first_gt = True
    for j in range(gt_paths.shape[0]):
        if not bool(gt_valid[j]):
            continue
        path = _prepend_origin(gt_paths[j])
        ax.plot(path[:, 0], path[:, 1], color="black", ls="--", lw=1.7, alpha=0.8,
                zorder=5, label="official GT" if first_gt else None)
        ax.scatter(path[:, 0], path[:, 1], color="black", s=12, alpha=0.6, zorder=6)
        first_gt = False

    # ego history trail (blue), ends at the current origin
    ax.plot(ego_history[:, 0], ego_history[:, 1], color="#1f77b4", lw=2.0,
            marker="o", ms=3.5, alpha=0.85, zorder=7, label="ego history")

    # oracle-best candidate (orange dotted)
    oracle_path = _prepend_origin(candidates[oracle_idx])
    ax.plot(oracle_path[:, 0], oracle_path[:, 1], color="#ff9f00", lw=2.4, ls=":",
            zorder=8, label=f"oracle-best #{oracle_idx} (PDMS {best_pdms:.3f})")

    # scorer-SELECTED candidate (green)
    sel_path = _prepend_origin(candidates[selected_idx])
    ax.plot(sel_path[:, 0], sel_path[:, 1], color="#00a651", lw=3.0,
            zorder=9, label=f"selected #{selected_idx} (PDMS {sel_pdms:.3f})")
    ax.scatter(sel_path[:, 0], sel_path[:, 1], color="#00a651", s=26, zorder=10)

    # ego "now"
    ax.scatter(0.0, 0.0, marker="^", s=110, c="#e31a1c", zorder=11, label="ego (now)")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("candidate official PDMS", fontsize=8)

    ax.set(xlabel="x forward (m)", ylabel="y left (m)",
           title=f"{scene_id}  |  opt_step={opt_step}  selected PDMS={sel_pdms:.3f}  "
                 f"oracle-best={best_pdms:.3f}  matched-GT ADE={ade_row:.3f} m")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# navtest path: no Multi-GT exists, so the official GT is the logged human
# future (8 ego poses) and the scene index comes from the navtest metric cache
# intersected with the trainval log frame index (no multigt intersection).
# --------------------------------------------------------------------------- #
def load_navtest_sample(log_path: Path, sensor_root: str, frame_index: int,
                        camera: str = "CAM_F0", image_size: int = 224) -> dict:
    """Mirror ``load_real_navsim_sample``'s Observation exactly, GT from the log."""
    with log_path.open("rb") as handle:
        frames = pickle.load(handle)
    if frame_index < 4 or frame_index + 8 >= len(frames):
        raise ValueError(f"frame_index={frame_index} does not leave history/future context")
    frame = frames[frame_index]
    frame_token = str(frame["token"])
    context_indices = list(range(frame_index - 4, frame_index + 1))
    future_indices = list(range(frame_index + 1, frame_index + 9))
    reference = np.asarray(frame["ego2global"], dtype=np.float64)

    rgb = torch.stack([_frame_image(frames[i], Path(sensor_root), camera, image_size) for i in context_indices])
    relative_poses = torch.from_numpy(
        np.stack([_relative_pose(reference, np.asarray(frames[i]["ego2global"], dtype=np.float64)) for i in context_indices])
    )
    ego_features = []
    for index, pose in zip(context_indices, relative_poses):
        dynamic = np.asarray(frames[index]["ego_dynamic_state"], dtype=np.float32).reshape(-1)
        ego_features.append(drivor_observable_ego_feature(pose, torch.from_numpy(dynamic)))
    ego_history = torch.stack(ego_features)
    route = drivor_route_command(frame["driving_command"])

    gt = torch.from_numpy(
        np.stack([_relative_pose(reference, np.asarray(frames[i]["ego2global"], dtype=np.float64)) for i in future_indices])
    ).to(torch.float32)  # [8,3] official human future in the current ego frame
    trajectories = gt.unsqueeze(0)  # [1,8,3]
    return {
        "rgb_history": rgb.unsqueeze(1),                 # [5,1,3,H,W]
        "history_valid": torch.ones(5, 1, dtype=torch.bool),
        "ego_history": ego_history,                       # [5,F]
        "route_command": route,
        "relative_ego_poses": relative_poses,             # [5,3]
        "gt_trajectory": trajectories,                    # [1,8,3]
        "gt_encoded": encode_trajectory(trajectories),    # [1,8,4]
        "gt_valid": torch.ones(1, dtype=torch.bool),
        "scene_id": frame_token,
    }


class NavtestSceneDataset(torch.utils.data.Dataset):
    def __init__(self, candidates, sensor_root: str, log_root: str, max_retry: int = 12) -> None:
        self.candidates = candidates
        self.sensor_root = sensor_root
        self.log_root = log_root
        self.max_retry = max_retry

    def __len__(self) -> int:
        return len(self.candidates)

    def __getitem__(self, index: int) -> dict:
        total = len(self.candidates)
        last_error: Exception | None = None
        for attempt in range(self.max_retry):
            candidate = self.candidates[(index + attempt) % total]
            try:
                return load_navtest_sample(
                    Path(self.log_root) / f"{candidate.log_name}.pkl",
                    self.sensor_root, candidate.frame_index,
                )
            except Exception as error:
                last_error = error
                continue
        raise RuntimeError(f"could not load any navtest scene near index {index}: {last_error!r}")


def build_navtest_index(cache_root: str, log_root: str, index_dir: Path, limit_scenes: int) -> list:
    """token -> Candidate from the navtest metric cache ∩ trainval log frames."""
    meta = {"multigt": "", "cache": cache_root, "log": log_root}
    token_cache_index = trainer.load_token_cache_index(cache_root, index_dir, meta)
    frame_index = trainer.build_frame_index(token_cache_index, log_root, index_dir, meta,
                                            limit_tokens=None, write_sidecar=True)
    resolved = sorted(set(frame_index) & set(token_cache_index))
    candidates = []
    for token in resolved:
        log_name, cache_path = token_cache_index[token]
        candidates.append(trainer.Candidate(token=token, log_name=log_name, cache_path=str(cache_path),
                                            frame_index=frame_index[token]))
        if limit_scenes and len(candidates) >= limit_scenes:
            break
    return candidates


# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default="/tmp/nvs_eval_snap/joint_latest.pt")
    parser.add_argument("--split", default="navtrain", choices=["navtrain", "navtest"],
                        help="navtrain = in-distribution training split; navtest = held-out official test split")
    # roots default to the SAME env-resolved constants the joint run used, so the
    # sidecar scene index in --index-dir is reused verbatim (no log re-walk).
    parser.add_argument("--multigt-root", default=trainer.DEFAULT_MULTIGT)
    parser.add_argument("--sensor-root", default="", help="empty = resolve per --split")
    parser.add_argument("--log-root", default="", help="empty = resolve per --split")
    parser.add_argument("--cache-root", default="", help="empty = resolve per --split")
    parser.add_argument("--dino-checkpoint", default="")
    parser.add_argument("--index-dir", default="", help="empty = resolve per --split (navtest uses its own)")
    parser.add_argument("--output-dir", default="", help="empty = resolve per --split")
    parser.add_argument("--max-scenes", type=int, default=128, help="deterministic navtrain subset size")
    parser.add_argument("--bev-scenes", type=int, default=8, help="how many scenes to render as BEV")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--depth", type=int, default=0, help="0 = read depth from checkpoint config")
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    is_navtest = args.split == "navtest"
    cache_root = args.cache_root or (NAVTEST_CACHE if is_navtest else trainer.DEFAULT_CACHE)
    sensor_root = args.sensor_root or (NAVTEST_SENSOR_ROOT if is_navtest else trainer.DEFAULT_SENSOR)
    log_root = args.log_root or (NAVTEST_LOG_ROOT if is_navtest else trainer.DEFAULT_LOG_ROOT)
    index_dir = args.index_dir or str(
        REPO_ROOT / ("artifacts/navtest_wan_joint_eval_index" if is_navtest else "artifacts/navtrain_wan_joint_ddp")
    )
    output_dir = Path(args.output_dir or str(
        REPO_ROOT / ("artifacts/navtrain_wan_joint_eval_navtest" if is_navtest else "artifacts/navtrain_wan_joint_eval")
    ))
    bev_dir = output_dir / "bev"
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(index_dir).mkdir(parents=True, exist_ok=True)

    # ---- load checkpoint --------------------------------------------------- #
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    depth = args.depth or int(config.get("depth", 8))
    dino = args.dino_checkpoint or config.get("dino_checkpoint", trainer.DEFAULT_DINO)
    opt_step = int(ckpt.get("opt_step", -1))
    epoch = int(ckpt.get("epoch", -1))
    print(json.dumps({"event": "ckpt_loaded", "checkpoint": args.checkpoint,
                      "opt_step": opt_step, "epoch": epoch, "depth": depth}), flush=True)

    model = trainer.DirectedPairDiT(
        d_model=256, heads=8, ffn_dim=1024, depth=depth,
        dino_checkpoint=dino, dino_model_name="vit_large_patch16_dinov3.lvd1689m",
        dino_lora_rank=16, dino_lora_alpha=16.0,
    )
    model.load_state_dict(ckpt["model"], strict=True)
    scorer = trainer.ScratchDrivoRScorer(d_model=256, heads=1, ffn_dim=1024, decoder_layers=4)
    scorer.load_state_dict(ckpt["scorer"], strict=True)
    model.to(device).eval()
    scorer.to(device).eval()
    net = trainer.JointStage2Module(model, scorer).to(device).eval()

    # ---- deterministic scene subset (reuse sidecar index when present) ----- #
    index_start = time.time()
    if is_navtest:
        candidates = build_navtest_index(cache_root, log_root, Path(index_dir), args.max_scenes)
    else:
        candidates = trainer.build_scene_index(
            args.multigt_root, cache_root, log_root, Path(index_dir), args.max_scenes
        )
    if not candidates:
        raise RuntimeError(f"no {args.split} scenes resolved; check roots / index-dir")
    print(json.dumps({"event": "index_ready", "split": args.split, "scenes": len(candidates),
                      "elapsed_s": round(time.time() - index_start, 2)}), flush=True)

    cache_paths = {c.token: c.cache_path for c in candidates}
    oracle_version = ORACLE_VERSION_NAVTEST if is_navtest else trainer.ORACLE_VERSION
    oracle = trainer.DrivoRBatchMetricCacheOracle(cache_paths=cache_paths, evaluator_version=oracle_version)
    if is_navtest:
        dataset = NavtestSceneDataset(candidates, sensor_root, log_root)
    else:
        dataset = trainer.NavtrainSceneDataset(candidates, sensor_root, args.multigt_root, cache_root, log_root)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                        collate_fn=trainer.collate, drop_last=False,
                        persistent_workers=args.workers > 0, prefetch_factor=2 if args.workers > 0 else None)

    modes = trainer.ImmutableModeBuffer().subset(64).to(device)
    initializer = trainer.StructuredSourceInitializer(jitter_std=0.0)

    acc = torch.zeros(6)  # [sel_pdms, oracle_best, match, ade_sum, ade_n, score_ok]
    per_scene: list[dict] = []
    bev_done = 0
    oracle_failures = 0
    batch_i = 0
    run_start = time.time()

    for batch in loader:
        observation, targets, scene_ids = trainer.batch_to_device(batch, device)
        size = observation.rgb_history.shape[0]
        mode_codes = modes.expand_for_batch(size, device)
        source = initializer.clean(observation.ego_history, mode_codes)
        with torch.no_grad():
            endpoint, trajectory, logits = net(observation, source, mode_codes)
            try:
                labels = oracle.evaluate(scene_ids, trajectory, batch_id=f"eval{batch_i}")
                labels_ok = (not labels.failure_diagnostics) and bool(labels.final_valid.all())
            except Exception as error:  # a single un-simulatable scene must not kill the eval
                labels, labels_ok = None, False
                print(json.dumps({"event": "oracle_error", "batch": batch_i, "error": repr(error)}), flush=True)
            assignment = trainer.hungarian_xy(trajectory, targets["gt_trajectory"], targets["gt_valid"])
            metrics = trainer.batch_eval_metrics(scorer, logits, trajectory, targets, assignment, labels, labels_ok, device)
            selected, _utility = scorer.select(logits)

        acc += metrics.cpu()
        if not labels_ok:
            oracle_failures += 1

        # ---- per-scene records + BEV --------------------------------------- #
        traj_np = trajectory.detach().cpu().numpy()
        gt_np = targets["gt_trajectory"].cpu().numpy()
        gtvalid_np = targets["gt_valid"].cpu().numpy()
        ego_hist_np = observation.relative_ego_poses.cpu().numpy()
        sel_np = selected.cpu().numpy()
        fs_np = labels.final_score.cpu().numpy() if labels_ok else None
        fvalid_np = labels.final_valid.cpu().numpy() if labels_ok else None

        for row in range(size):
            token = scene_ids[row]
            ade_pairs = [(traj_np[row, c, :, :2] - gt_np[row, g, :, :2]) for g, c in assignment[row].as_pairs()]
            ade_row = float(np.mean([np.linalg.norm(d, axis=-1).mean() for d in ade_pairs])) if ade_pairs else float("nan")
            rec: dict = {"scene_id": token, "n_valid_gt": int(gtvalid_np[row].sum()),
                         "matched_gt_ade_m": round(ade_row, 4)}
            if fs_np is not None:
                sel_idx = int(sel_np[row])
                oracle_idx = int(np.argmax(fs_np[row]))
                rec.update({
                    "selected_index": sel_idx,
                    "selected_pdms": round(float(fs_np[row, sel_idx]), 4),
                    "oracle_best_pdms": round(float(fs_np[row].max()), 4),
                    "oracle_match": bool(sel_idx == oracle_idx),
                })
            per_scene.append(rec)

            if (bev_done < args.bev_scenes and fs_np is not None
                    and fvalid_np is not None and bool(fvalid_np[row].all())):
                sel_idx = int(sel_np[row])
                oracle_idx = int(np.argmax(fs_np[row]))
                draw_bev_eval(
                    scene_id=token,
                    candidates=traj_np[row],
                    final_score=fs_np[row],
                    selected_idx=sel_idx,
                    oracle_idx=oracle_idx,
                    gt_paths=gt_np[row],
                    gt_valid=gtvalid_np[row],
                    ego_history=ego_hist_np[row],
                    out_path=bev_dir / f"{token}_bev.png",
                    sel_pdms=float(fs_np[row, sel_idx]),
                    best_pdms=float(fs_np[row].max()),
                    ade_row=ade_row,
                    opt_step=opt_step,
                )
                bev_done += 1
        batch_i += 1
        print(json.dumps({"event": "batch", "batch": batch_i, "scenes": int(acc[5].item()),
                          "elapsed_s": round(time.time() - run_start, 1)}), flush=True)

    denom_scene = max(float(acc[5].item()), 1.0)
    denom_ade = max(float(acc[4].item()), 1.0)
    summary = {
        "checkpoint": args.checkpoint,
        "opt_step": opt_step,
        "epoch": epoch,
        "split": ("navtest (HELD-OUT official test split)" if is_navtest
                  else "navtrain (IN-DISTRIBUTION training split, not held-out)"),
        "scenes_evaluated": int(denom_scene),
        "scenes_indexed": len(candidates),
        "selected_pdms": round(float(acc[0].item()) / denom_scene, 4),
        "oracle_best_pdms": round(float(acc[1].item()) / denom_scene, 4),
        "oracle_match_rate": round(float(acc[2].item()) / denom_scene, 4),
        "matched_gt_ade_m": round(float(acc[3].item()) / denom_ade, 4),
        "oracle_failures": oracle_failures,
        "bev_rendered": bev_done,
        "elapsed_s": round(time.time() - run_start, 1),
    }
    (output_dir / "result.json").write_text(json.dumps({"summary": summary, "per_scene": per_scene}, indent=2) + "\n")
    print(json.dumps({"event": "summary", **summary}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
