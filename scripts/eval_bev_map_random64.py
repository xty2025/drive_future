"""Random-64-scene official-PDMS BEV evaluation rendered ON the NAVSIM map.

One checkpoint per invocation; run it twice (joint / two-phase) with the SAME
``--scenes-json`` so both checkpoints are scored on identical scenes.  The
scene list is a uniform random sample (fixed ``--seed``) of the resolved
navtrain index -- NOT a best-PDMS or hand pick.

The BEV is drawn over the road environment (lanes, intersections, walkways,
carpark areas, crosswalks, lane baselines) using the vendored NAVSIM helper
``navsim.visualization.bev.add_map_to_bev_ax`` backed by the nuPlan gpkg maps
(``NUPLAN_MAPS_ROOT``).  That helper's axis convention is
horizontal = y left, vertical = x forward (ego-relative meters); every
trajectory overlay here uses the same convention so map and paths agree.

Per scene the figure shows: all 64 candidates faint, viridis-colored by their
official PDMS (colorbar); the scorer-SELECTED trajectory (green); the
ORACLE-BEST candidate (orange dotted); every valid official Multi-GT (black
dashed); the ego history (blue) ending at the red ego-now triangle.

Strictly additive: imports ``run_navtrain_wan_joint_ddp`` and the vendored
navsim/nuplan map+viz modules; edits nothing.  ``navtrain`` is the TRAINING
split, so numbers are in-distribution, not held-out.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("NUPLAN_MAPS_ROOT", "/mnt/data/navsim/maps")

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

import run_navtrain_wan_joint_ddp as trainer  # noqa: E402

from navsim.visualization.bev import add_map_to_bev_ax  # noqa: E402
from navsim.visualization.config import LIGHT_GREY, MAP_LAYER_CONFIG, NEW_TAB_10  # noqa: E402
from nuplan.common.actor_state.state_representation import StateSE2  # noqa: E402
from nuplan.common.maps.abstract_map import SemanticMapLayer  # noqa: E402
from nuplan.common.maps.nuplan_map.map_factory import get_maps_api  # noqa: E402

MAP_VERSION = "nuplan-maps-v1.0"


# --------------------------------------------------------------------------- #
# BEV over map (navsim axis convention: horizontal=y left, vertical=x forward)
# --------------------------------------------------------------------------- #
def _prepend_origin(traj: np.ndarray) -> np.ndarray:
    return np.vstack([np.zeros((1, traj.shape[-1]), dtype=traj.dtype), traj])


def draw_bev_map_eval(
    scene_id: str,
    map_name: str,
    map_api,
    ego_global: StateSE2,
    candidates: np.ndarray,       # [64, 8, 3] ego frame (x fwd, y left)
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
    tag: str,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 8), dpi=150)

    add_map_to_bev_ax(ax, map_api, ego_global)

    finite = final_score[np.isfinite(final_score)]
    norm = plt.Normalize(float(finite.min()), float(finite.max())) if finite.size else plt.Normalize(0.0, 1.0)
    cmap = plt.cm.viridis

    for i, cand in enumerate(candidates):
        path = _prepend_origin(cand)
        color = cmap(norm(final_score[i])) if np.isfinite(final_score[i]) else "lightgray"
        ax.plot(path[:, 1], path[:, 0], color=color, alpha=0.30, lw=0.7, zorder=2)

    first_gt = True
    for j in range(gt_paths.shape[0]):
        if not bool(gt_valid[j]):
            continue
        path = _prepend_origin(gt_paths[j])
        ax.plot(path[:, 1], path[:, 0], color="black", ls="--", lw=1.7, alpha=0.8,
                zorder=5, label="official GT" if first_gt else None)
        ax.scatter(path[:, 1], path[:, 0], color="black", s=12, alpha=0.6, zorder=6)
        first_gt = False

    ax.plot(ego_history[:, 1], ego_history[:, 0], color="#1f77b4", lw=2.0,
            marker="o", ms=3.5, alpha=0.85, zorder=7, label="ego history")

    oracle_path = _prepend_origin(candidates[oracle_idx])
    ax.plot(oracle_path[:, 1], oracle_path[:, 0], color="#ff9f00", lw=2.4, ls=":",
            zorder=8, label=f"oracle-best #{oracle_idx} (PDMS {best_pdms:.3f})")

    sel_path = _prepend_origin(candidates[selected_idx])
    ax.plot(sel_path[:, 1], sel_path[:, 0], color="#00a651", lw=3.0,
            zorder=9, label=f"selected #{selected_idx} (PDMS {sel_pdms:.3f})")
    ax.scatter(sel_path[:, 1], sel_path[:, 0], color="#00a651", s=26, zorder=10)

    ax.scatter(0.0, 0.0, marker="^", s=110, c="#e31a1c", zorder=11, label="ego (now)")

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    map_handles = [
        Patch(facecolor=LIGHT_GREY, label="lane / intersection"),
        Patch(facecolor=MAP_LAYER_CONFIG[SemanticMapLayer.WALKWAYS]["fill_color"], label="walkway"),
        Patch(facecolor=MAP_LAYER_CONFIG[SemanticMapLayer.CARPARK_AREA]["fill_color"], label="carpark"),
        Patch(facecolor=NEW_TAB_10[6], alpha=0.3, label="crosswalk"),
        Line2D([0], [0], color=MAP_LAYER_CONFIG[SemanticMapLayer.BASELINE_PATHS]["line_color"],
               ls="--", lw=1.0, label="lane baseline"),
    ]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=map_handles + handles, labels=[h.get_label() for h in map_handles] + labels,
              fontsize=7, loc="best")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("candidate official PDMS", fontsize=8)

    ax.set(xlabel="y left (m)", ylabel="x forward (m)",
           title=f"{scene_id}  |  {tag}  opt_step={opt_step}  map={map_name}\n"
                 f"selected PDMS={sel_pdms:.3f}  oracle-best={best_pdms:.3f}  matched-GT ADE={ade_row:.3f} m")
    ax.set_xlim(-45.0, 45.0)
    ax.set_ylim(-25.0, 65.0)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tag", required=True, help="output subdir name, e.g. joint / twophase")
    parser.add_argument("--scenes-json", required=True, help="shared random scene list (created by --create-scenes)")
    parser.add_argument("--create-scenes", action="store_true",
                        help="if --scenes-json is missing, uniformly sample --n-scenes with --seed and write it")
    parser.add_argument("--n-scenes", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--multigt-root", default=trainer.DEFAULT_MULTIGT)
    parser.add_argument("--sensor-root", default=trainer.DEFAULT_SENSOR)
    parser.add_argument("--log-root", default=trainer.DEFAULT_LOG_ROOT)
    parser.add_argument("--cache-root", default=trainer.DEFAULT_CACHE)
    parser.add_argument("--index-dir", default=str(REPO_ROOT / "artifacts/navtrain_wan_joint_ddp"))
    parser.add_argument("--maps-root", default=os.environ.get("NUPLAN_MAPS_ROOT", "/mnt/data/navsim/maps"))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--dino-checkpoint", default="")
    parser.add_argument("--depth", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=6)
    return parser.parse_args()


def resolve_scenes(args) -> list[dict]:
    path = Path(args.scenes_json)
    if path.exists():
        with path.open() as handle:
            records = json.load(handle)
        print(json.dumps({"event": "scenes_loaded", "path": str(path), "scenes": len(records)}), flush=True)
        return records
    if not args.create_scenes:
        raise RuntimeError(f"--scenes-json {path} missing; rerun with --create-scenes")
    full = trainer.build_scene_index(args.multigt_root, args.cache_root, args.log_root,
                                     Path(args.index_dir), 0)
    if len(full) < args.n_scenes:
        raise RuntimeError(f"only {len(full)} scenes resolved, need {args.n_scenes}")
    picked = random.Random(args.seed).sample(full, args.n_scenes)
    records = [{"token": c.token, "log_name": c.log_name, "cache_path": c.cache_path,
                "frame_index": c.frame_index} for c in picked]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(records, handle, indent=1)
    print(json.dumps({"event": "scenes_sampled", "path": str(path), "scenes": len(records),
                      "pool": len(full), "seed": args.seed}), flush=True)
    return records


def main() -> int:
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    records = resolve_scenes(args)
    candidates = [trainer.Candidate(token=r["token"], log_name=r["log_name"],
                                    cache_path=r["cache_path"], frame_index=r["frame_index"])
                  for r in records]

    output_dir = Path(args.output_dir or str(REPO_ROOT / "artifacts/bev_map_random64" / args.tag))
    bev_dir = output_dir / "bev"
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    depth = args.depth or int(config.get("depth", 8))
    dino = args.dino_checkpoint or config.get("dino_checkpoint", trainer.DEFAULT_DINO)
    opt_step = int(ckpt.get("opt_step", -1))
    epoch = int(ckpt.get("epoch", -1))
    print(json.dumps({"event": "ckpt_loaded", "tag": args.tag, "checkpoint": args.checkpoint,
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

    cache_paths = {c.token: c.cache_path for c in candidates}
    oracle = trainer.DrivoRBatchMetricCacheOracle(cache_paths=cache_paths,
                                                  evaluator_version=trainer.ORACLE_VERSION)
    dataset = trainer.NavtrainSceneDataset(candidates, args.sensor_root, args.multigt_root,
                                           args.cache_root, args.log_root)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                        collate_fn=trainer.collate, drop_last=False,
                        persistent_workers=args.workers > 0, prefetch_factor=2 if args.workers > 0 else None)

    modes = trainer.ImmutableModeBuffer().subset(64).to(device)
    initializer = trainer.StructuredSourceInitializer(jitter_std=0.0)

    map_apis: dict[str, object] = {}
    log_frames: dict[str, list] = {}

    def scene_map_context(log_name: str, frame_index: int):
        if log_name not in log_frames:
            with (Path(args.log_root) / f"{log_name}.pkl").open("rb") as handle:
                log_frames[log_name] = pickle.load(handle)
        frame = log_frames[log_name][frame_index]
        map_name = str(frame["map_location"])
        ego2global = np.asarray(frame["ego2global"], dtype=np.float64)
        translation = ego2global[:3, 3]
        rotation = ego2global[:3, :3]
        yaw = float(math.atan2(rotation[1, 0], rotation[0, 0]))
        if map_name not in map_apis:
            map_apis[map_name] = get_maps_api(args.maps_root, MAP_VERSION, map_name)
        return map_name, map_apis[map_name], StateSE2(float(translation[0]), float(translation[1]), yaw)

    per_scene: list[dict] = []
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
            except Exception as error:
                labels, labels_ok = None, False
                print(json.dumps({"event": "oracle_error", "batch": batch_i, "error": repr(error)}), flush=True)
            assignment = trainer.hungarian_xy(trajectory, targets["gt_trajectory"], targets["gt_valid"])
            selected, _utility = scorer.select(logits)

        traj_np = trajectory.detach().cpu().numpy()
        gt_np = targets["gt_trajectory"].cpu().numpy()
        gtvalid_np = targets["gt_valid"].cpu().numpy()
        ego_hist_np = observation.relative_ego_poses.cpu().numpy()
        sel_np = selected.cpu().numpy()
        fs_np = labels.final_score.cpu().numpy() if labels_ok else None
        fvalid_np = labels.final_valid.cpu().numpy() if labels_ok else None

        for row in range(size):
            token = scene_ids[row]
            meta = next(r for r in records if r["token"] == token)
            ade_pairs = [(traj_np[row, c, :, :2] - gt_np[row, g, :, :2]) for g, c in assignment[row].as_pairs()]
            ade_row = float(np.mean([np.linalg.norm(d, axis=-1).mean() for d in ade_pairs])) if ade_pairs else float("nan")
            rec: dict = {"scene_id": token, "map_name": "", "n_valid_gt": int(gtvalid_np[row].sum()),
                         "matched_gt_ade_m": round(ade_row, 4)}
            if fs_np is not None and fvalid_np is not None:
                sel_idx = int(sel_np[row])
                oracle_idx = int(np.argmax(fs_np[row]))
                fin = fs_np[row][np.isfinite(fs_np[row])]
                rec.update({
                    "selected_index": sel_idx,
                    "selected_pdms": round(float(fs_np[row, sel_idx]), 4),
                    "oracle_best_pdms": round(float(fs_np[row].max()), 4),
                    "oracle_match": bool(sel_idx == oracle_idx),
                    "noscorer_mean64_pdms": round(float(fin.mean()), 4) if fin.size else None,
                    "noscorer_cand0_pdms": round(float(fs_np[row, 0]), 4) if np.isfinite(fs_np[row, 0]) else None,
                })
            else:
                oracle_failures += 1
            per_scene.append(rec)

            if fs_np is not None and fvalid_np is not None and bool(fvalid_np[row].all()):
                map_name, map_api, ego_global = scene_map_context(meta["log_name"], meta["frame_index"])
                rec["map_name"] = map_name
                draw_bev_map_eval(
                    scene_id=token,
                    map_name=map_name,
                    map_api=map_api,
                    ego_global=ego_global,
                    candidates=traj_np[row],
                    final_score=fs_np[row],
                    selected_idx=int(sel_np[row]),
                    oracle_idx=int(np.argmax(fs_np[row])),
                    gt_paths=gt_np[row],
                    gt_valid=gtvalid_np[row],
                    ego_history=ego_hist_np[row],
                    out_path=bev_dir / f"{token}_bev.png",
                    sel_pdms=rec.get("selected_pdms", float("nan")),
                    best_pdms=rec.get("oracle_best_pdms", float("nan")),
                    ade_row=ade_row,
                    opt_step=opt_step,
                    tag=args.tag,
                )
        print(json.dumps({"event": "batch", "tag": args.tag, "batch": batch_i,
                          "scenes": len(per_scene), "elapsed_s": round(time.time() - run_start, 1)}), flush=True)
        batch_i += 1

    scored = [r for r in per_scene if "selected_pdms" in r]
    ades = [r["matched_gt_ade_m"] for r in per_scene if np.isfinite(r["matched_gt_ade_m"])]
    summary = {
        "tag": args.tag,
        "checkpoint": args.checkpoint,
        "opt_step": opt_step,
        "epoch": epoch,
        "split": "navtrain (IN-DISTRIBUTION training split, not held-out)",
        "scenes_json": str(args.scenes_json),
        "scenes_evaluated": len(per_scene),
        "bev_rendered": len(list(bev_dir.glob("*_bev.png"))),
        "selected_pdms": round(float(np.mean([r["selected_pdms"] for r in scored])), 4) if scored else None,
        "oracle_best_pdms": round(float(np.mean([r["oracle_best_pdms"] for r in scored])), 4) if scored else None,
        "oracle_match_rate": round(float(np.mean([r["oracle_match"] for r in scored])), 4) if scored else None,
        "matched_gt_ade_m": round(float(np.mean(ades)), 4) if ades else None,
        "noscorer_mean64_pdms": round(float(np.mean([r["noscorer_mean64_pdms"] for r in scored])), 4) if scored else None,
        "noscorer_cand0_pdms": round(float(np.mean([r["noscorer_cand0_pdms"] for r in scored])), 4) if scored else None,
        "oracle_failures": oracle_failures,
        "elapsed_s": round(time.time() - run_start, 1),
    }
    with (output_dir / "result.json").open("w") as handle:
        json.dump({"summary": summary, "per_scene": per_scene}, handle, indent=1)
    print(json.dumps({"event": "summary", **summary}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
