"""How much of a future frame is already fixed by the scene, before any trajectory.

The register target's variance decomposes 79.3% background / 20.7% trajectory in
feature space (measured on 400 cached scenes), but that number cannot be *seen*, and
"the Teacher's feature variance" is not the same claim as "the rendered future is
mostly the scene continuing".  This measures the same split directly on the pixels the
Teacher read, and draws it.

Per horizon it reports

    background_share = var_across_scenes / (var_across_scenes + var_across_gt_within_scene)

where the within-scene term is the variance over that scene's own GT trajectories --
exactly the part reading the trajectory can win, and exactly what ``residual_*_l1``
isolates in feature space.  Both terms are computed on per-scene GT means so that the
scene term is not inflated by trajectory spread.

Usage::

    python scripts/probe_future_background.py --scenes 12 --out artifacts/future_background_probe
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from nvs_v6_wan_e2e.multiview_data import FUTURE_POSE_INDICES

HORIZONS_S = [(index + 1) * 0.5 for index in range(len(FUTURE_POSE_INDICES))]


def read_frames(path: Path, indices: tuple[int, ...]) -> np.ndarray:
    """The same frames the loader hands the Teacher, as float32 ``[F,H,W,3]`` in [0,1]."""
    capture = cv2.VideoCapture(str(path))
    wanted = dict((index, slot) for slot, index in enumerate(indices))
    out = np.zeros((len(indices), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), 3), dtype=np.float32)
    position = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if position in wanted:
            out[wanted[position]] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)[..., ::-1].copy() / 255.0
        position += 1
    capture.release()
    return out


def montage(rows: list[list[np.ndarray]], gap: int = 4) -> np.ndarray:
    """Tile images with a white gutter; rows are padded to the widest one."""
    width = max(len(row) for row in rows)
    padded = [row + [np.ones_like(row[0])] * (width - len(row)) for row in rows]
    heights = [max(image.shape[0] for image in row) for row in padded]
    strips = []
    for row, height in zip(padded, heights):
        scaled = [image if image.shape[0] == height
                  else cv2.resize(image, (int(image.shape[1] * height / image.shape[0]), height))
                  for image in row]
        widths = [image.shape[1] for image in scaled]
        strip = np.ones((height, sum(widths) + gap * (len(scaled) - 1), 3), dtype=np.float32)
        cursor = 0
        for image, image_width in zip(scaled, widths):
            strip[:, cursor : cursor + image_width] = image
            cursor += image_width + gap
        strips.append(strip)
    strip_widths = [strip.shape[1] for strip in strips]
    canvas = np.ones((sum(heights) + gap * (len(strips) - 1), max(strip_widths), 3), dtype=np.float32)
    cursor = 0
    for strip, strip_width in zip(strips, strip_widths):
        canvas[cursor : cursor + strip.shape[0], :strip_width] = strip
        cursor += strip.shape[0] + gap
    return canvas


def label(image: np.ndarray, text: str) -> np.ndarray:
    out = (image * 255).astype(np.uint8)
    cv2.rectangle(out, (0, 0), (out.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(out, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out.astype(np.float32) / 255.0


def heatmap(std: np.ndarray) -> np.ndarray:
    """Per-pixel across-GT std -> jet, scaled to its own max so frames are comparable in shape."""
    normalized = std / max(float(std.max()), 1e-6)
    return cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)[..., ::-1].copy() / 255.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--multiview-root", default="/mnt/data/multi_future_view/"
                        "navtrain_full_allcosmos_cuda_v1_gpu64/navtrain_full_allcosmos_cuda_v1_gpu64")
    parser.add_argument("--scenes", type=int, default=12)
    parser.add_argument("--max-scan", type=int, default=4000, help="scene directories to inspect for readiness")
    parser.add_argument("--out", default="artifacts/future_background_probe")
    parser.add_argument("--show-horizons", type=float, nargs="*", default=[0.5, 2.0, 4.0])
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(args.multiview_root)
    candidates = sorted(path.name for path in root.iterdir() if path.is_dir())[: args.max_scan]
    picked = []
    for scene_id in candidates:
        scene_dir = root / scene_id
        if not (scene_dir / "completion.json").exists() or not (scene_dir / "initial_front.jpg").exists():
            continue
        trajectories = sorted(scene_dir.glob("trajectory_*"))
        if len(trajectories) < 2 or not all((path / "future.mp4").exists() for path in trajectories):
            continue
        picked.append((scene_id, trajectories))
        if len(picked) >= args.scenes:
            break
    if not picked:
        raise SystemExit(f"no ready scene with >=2 trajectories under {root}")
    print(f"[probe] {len(picked)} scenes, {len(FUTURE_POSE_INDICES)} horizons each")

    slots = [slot for slot, horizon in enumerate(HORIZONS_S) if horizon in args.show_horizons]
    stacks, scene_ids = [], []
    for scene_id, trajectories in picked:
        frames = [read_frames(path / "future.mp4", FUTURE_POSE_INDICES) for path in trajectories]
        stacks.append(np.stack(frames))  # [K, F, H, W, 3]
        scene_ids.append(scene_id)
    data = np.stack(stacks)  # [N, K, F, H, W, 3]
    print(f"[probe] video tensor {data.shape}, {data.nbytes / 1e6:.1f} MB")

    # Variance decomposition per horizon, on pixels.  ``within`` keeps the scene axis so
    # the per-scene spread can be inspected; ``between`` reduces H, W *and* the colour
    # channel, leaving one number per horizon -- dropping the channel axis is what makes
    # the two terms commensurable.
    scene_mean = data.mean(axis=1)                    # [N,F,H,W,3] the scene's average future
    within = ((data - scene_mean[:, None]) ** 2).mean(axis=(1, 3, 4, 5))   # [N,F] trajectory variance
    between = scene_mean.var(axis=0).mean(axis=(1, 2, 3))                  # [F]   scene variance
    per_scene_within = within.mean(axis=0)                                 # [F]
    share = between / np.maximum(between + per_scene_within, 1e-12)

    table = {
        "scenes": len(picked),
        "scene_ids": scene_ids,
        "horizons_s": HORIZONS_S,
        "across_scene_variance": between.tolist(),
        "within_scene_across_gt_variance": per_scene_within.tolist(),
        "background_share": share.tolist(),
        "trajectory_share": (1.0 - share).tolist(),
    }
    (out / "future_background_decomposition.json").write_text(json.dumps(table, indent=2))

    print("\n  horizon   across-scene var   within-scene(GT) var   background share   trajectory share")
    for index, horizon in enumerate(HORIZONS_S):
        print(f"   +{horizon:.1f}s      {between[index]:.6f}          {per_scene_within[index]:.6f}"
              f"            {share[index] * 100:5.2f}%           {(1 - share[index]) * 100:5.2f}%")
    print(f"\n  pooled over horizons: background {share.mean() * 100:.2f}%  trajectory {(1 - share.mean()) * 100:.2f}%")

    # Per-scene figure: one row per horizon, each row being the history frame (first row
    # only), that horizon's future for every GT, and the across-GT std heatmap.  Laying
    # it out per horizon rather than per panel keeps the heatmap column-aligned with the
    # frames it summarises.
    blank = np.zeros_like(stacks[0][0, 0])
    for position, (scene_id, trajectories) in enumerate(picked):
        stack = data[position]                                  # [K,F,H,W,3]
        history = cv2.imread(str(root / scene_id / "initial_front.jpg"))
        history = cv2.cvtColor(history, cv2.COLOR_BGR2RGB)[..., ::-1].copy() / 255.0
        history = cv2.resize(history, (stack.shape[3], stack.shape[2]))
        rows = []
        for row_index, slot in enumerate(slots):
            horizon = HORIZONS_S[slot]
            head = label(history, "history t=0") if row_index == 0 else label(blank, "")
            panels = [head]
            for gt_index in range(stack.shape[0]):
                panels.append(label(stack[gt_index, slot], f"GT{gt_index} +{horizon:.1f}s"))
            across_gt = np.sqrt(((stack[:, slot] - stack[:, slot].mean(0)) ** 2).mean(axis=(0, 3)))
            panels.append(label(heatmap(across_gt), f"across-GT std +{horizon:.1f}s (max {across_gt.max():.3f})"))
            rows.append(panels)
        canvas = montage(rows)
        cv2.imwrite(str(out / f"scene_{position:02d}_{scene_id}.png"),
                    cv2.cvtColor((canvas * 255).astype(np.uint8)[..., ::-1], cv2.COLOR_BGR2RGB))

    # Aggregate curve.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(1, 2, figsize=(11, 3.6))
        axes[0].plot(HORIZONS_S, share * 100, "o-", color="#c0392b", label="background (scene fixes it)")
        axes[0].plot(HORIZONS_S, (1 - share) * 100, "s-", color="#2471a3", label="trajectory (only the GT changes it)")
        axes[0].set_xlabel("horizon [s]")
        axes[0].set_ylabel("share of pixel variance [%]")
        axes[0].set_title(f"Future pixel variance, {len(picked)} scenes")
        axes[0].legend(fontsize=7)
        axes[0].grid(alpha=0.3)
        axes[1].semilogy(HORIZONS_S, between, "o-", color="#c0392b", label="across-scene")
        axes[1].semilogy(HORIZONS_S, per_scene_within, "s-", color="#2471a3", label="within-scene across GT")
        axes[1].set_xlabel("horizon [s]")
        axes[1].set_ylabel("variance [log]")
        axes[1].set_title("The two terms")
        axes[1].legend(fontsize=7)
        axes[1].grid(alpha=0.3, which="both")
        figure.tight_layout()
        figure.savefig(out / "background_share_vs_horizon.png", dpi=140)
        print(f"[probe] wrote {out / 'background_share_vs_horizon.png'}")
    except ImportError:
        print("[probe] matplotlib unavailable, skipped the aggregate figure")

    print(f"[probe] wrote {len(picked)} scene montages + decomposition JSON to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
