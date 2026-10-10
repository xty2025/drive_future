"""Visualize the saved two-scene 64-candidate DrivoR comparison.

The candidate tensors are already in NavSim ego coordinates (x forward, y left).
This deliberately uses the exact saved tensors and selections from the rerank
audit instead of running another model pass.
"""

from pathlib import Path
import json

import cv2
import matplotlib.pyplot as plt
import numpy as np


ARTIFACT = Path(__file__).resolve().parents[1] / "artifacts/two_scene_overfit_10epoch_drivor_native_cache"
ROLLOUT = Path("/mnt/zhaozc_workspace/project/multitra2egoview/cosmos3/batch_native10/rollouts/single_gpu_test_gtfallback_v2")
SCENES = ["8f4a85d43ef351f9", "b00d83c03bbf5b47"]
OWN_SELECTED = {"8f4a85d43ef351f9": 7, "b00d83c03bbf5b47": 5}
DRIVOR_SELECTED = {"8f4a85d43ef351f9": 30, "b00d83c03bbf5b47": 16}


def load_gt(scene_id: str) -> list[np.ndarray]:
    paths = []
    for index in range(4):
        pose = np.load(ROLLOUT / scene_id / f"trajectory_{index}" / "ego_pose.npy")
        # 40-frame rollout at 0.1s; candidates/DrivoR are 8 points at 0.5s.
        paths.append(pose[5::5][:8])
    return paths


def limits(candidates: np.ndarray, gt_paths: list[np.ndarray]) -> tuple[float, float, float, float]:
    all_xy = np.concatenate([candidates[:, :, :2], *[p[:, :2][None] for p in gt_paths]], axis=1)
    x0, x1 = all_xy[..., 0].min(), all_xy[..., 0].max()
    y0, y1 = all_xy[..., 1].min(), all_xy[..., 1].max()
    pad = 0.08 * max(x1 - x0, y1 - y0, 1.0)
    return x0 - pad, x1 + pad, y0 - pad, y1 + pad


def draw_bev(scene_id: str, candidates: np.ndarray, official: np.ndarray, own_index: int, drivor_index: int, gt_paths: list[np.ndarray], output: Path) -> dict:
    oracle_index = int(np.nanargmax(official[:, 6]))
    fig, ax = plt.subplots(figsize=(9, 8), dpi=180)
    pdms = official[:, 6]
    norm = plt.Normalize(float(np.nanmin(pdms)), float(np.nanmax(pdms)))
    for i, candidate in enumerate(candidates):
        candidate_plot = np.vstack([np.zeros((1, 3), dtype=candidate.dtype), candidate])
        ax.plot(candidate_plot[:, 0], candidate_plot[:, 1], color=plt.cm.viridis(norm(pdms[i])), alpha=0.35, lw=0.8)
    for i, gt in enumerate(gt_paths):
        ax.plot(gt[:, 0], gt[:, 1], color="black", lw=1.4, ls="--", alpha=0.65, label="GT" if i == 0 else None)
    own_plot = np.vstack([np.zeros((1, 3)), candidates[own_index]])
    drivor_plot = np.vstack([np.zeros((1, 3)), candidates[drivor_index]])
    oracle_plot = np.vstack([np.zeros((1, 3)), candidates[oracle_index]])
    ax.plot(own_plot[:, 0], own_plot[:, 1], color="#00a651", lw=3.0, label=f"own scorer #{own_index} (PDM {pdms[own_index]:.3f})", zorder=6)
    ax.plot(drivor_plot[:, 0], drivor_plot[:, 1], color="#e31a1c", lw=3.0, label=f"DrivoR scorer #{drivor_index} (PDM {pdms[drivor_index]:.3f})", zorder=7)
    ax.plot(oracle_plot[:, 0], oracle_plot[:, 1], color="#ff9f00", lw=2.5, ls=":", label=f"oracle #{oracle_index} (PDM {pdms[oracle_index]:.3f})", zorder=8)
    ax.scatter(0, 0, marker="x", s=70, c="black", label="ego", zorder=9)
    ax.set(xlabel="x forward (m)", ylabel="y left (m)", title=f"{scene_id}: 64 NavSim ego-BEV candidates")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    def trajectory_metrics(pred: np.ndarray) -> tuple[float, float, float]:
        errors = [np.linalg.norm(pred[:, :2] - gt[:, :2], axis=-1) for gt in gt_paths]
        ade = min(float(error.mean()) for error in errors)
        fde = min(float(error[-1]) for error in errors)
        gt = gt_paths[int(np.argmin([error.mean() for error in errors]))][:, :2].reshape(-1)
        flat = pred[:, :2].reshape(-1)
        cosine = float(np.dot(flat, gt) / (np.linalg.norm(flat) * np.linalg.norm(gt) + 1e-8))
        return ade, fde, cosine

    metrics = {}
    for name, index in (("own", own_index), ("drivor", drivor_index), ("oracle", oracle_index)):
        ade, fde, cosine = trajectory_metrics(candidates[index])
        metrics.update({f"{name}_ade_m": ade, f"{name}_fde_m": fde, f"{name}_cosine": cosine})
    return {"scene_id": scene_id, "own_index": own_index, "drivor_index": drivor_index, "oracle_index": oracle_index, "own_pdms": float(pdms[own_index]), "drivor_pdms": float(pdms[drivor_index]), "oracle_pdms": float(pdms[oracle_index]), **metrics, "image": str(output)}


def read_future_frames(scene_id: str) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(ROLLOUT / scene_id / "trajectory_0" / "future.mp4"))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for frame_index in np.linspace(0, max(total - 1, 0), 4).astype(int):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = capture.read()
        if ok:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    return frames


def draw_future(scene_id: str, candidates: np.ndarray, own_index: int, drivor_index: int, oracle_index: int, gt_paths: list[np.ndarray], output: Path) -> None:
    frames = read_future_frames(scene_id)
    fig = plt.figure(figsize=(13, 7), dpi=160)
    grid = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.25])
    for col, frame in enumerate(frames):
        ax = fig.add_subplot(grid[0, col])
        ax.imshow(frame)
        ax.set_title(f"GT future frame {col + 1}", fontsize=8)
        ax.axis("off")
    ax = fig.add_subplot(grid[1, :])
    for i, gt in enumerate(gt_paths):
        ax.plot(gt[:, 0], gt[:, 1], "k--", alpha=0.55, lw=1.3, label="GT" if i == 0 else None)
    own_plot = np.vstack([np.zeros((1, 3)), candidates[own_index]])
    drivor_plot = np.vstack([np.zeros((1, 3)), candidates[drivor_index]])
    oracle_plot = np.vstack([np.zeros((1, 3)), candidates[oracle_index]])
    ax.plot(own_plot[:, 0], own_plot[:, 1], color="#00a651", lw=2.8, label=f"own prediction #{own_index}")
    ax.plot(drivor_plot[:, 0], drivor_plot[:, 1], color="#e31a1c", lw=2.8, label=f"DrivoR prediction #{drivor_index}")
    ax.plot(oracle_plot[:, 0], oracle_plot[:, 1], color="#ff9f00", lw=2.5, ls=":", label=f"oracle candidate #{oracle_index}")
    ax.scatter(0, 0, c="black", marker="x", s=55, label="ego")
    ax.set(xlabel="x forward (m)", ylabel="y left (m)", title=f"{scene_id}: future GT (black dashed) vs predicted trajectories")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=4, loc="best")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    out = ARTIFACT / "visualizations"
    out.mkdir(parents=True, exist_ok=True)
    candidates = np.load(ARTIFACT / "generated_trajectories.npy")
    official = np.load(ARTIFACT / "official_raw7_scores.npy")
    manifest = []
    for scene_index, scene_id in enumerate(SCENES):
        gt_paths = load_gt(scene_id)
        oracle_index = int(np.nanargmax(official[scene_index, :, 6]))
        bev = draw_bev(scene_id, candidates[scene_index], official[scene_index], OWN_SELECTED[scene_id], DRIVOR_SELECTED[scene_id], gt_paths, out / f"{scene_id}_64_candidates_bev_origin_aligned.png")
        draw_future(scene_id, candidates[scene_index], OWN_SELECTED[scene_id], DRIVOR_SELECTED[scene_id], oracle_index, gt_paths, out / f"{scene_id}_future_gt_vs_predictions_origin_aligned.png")
        manifest.append(bev)
    (out / "selection_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
