"""Composite per-scene BEV PNGs into one 8-column contact sheet for at-a-glance review.

Usage: python _make_bev_contact_sheet.py --bev-dir DIR --result-json JSON --out PNG --title TEXT
Annotates each thumbnail with the scorer-selected / oracle-best PDMS from result.json.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--bev-dir", required=True)
parser.add_argument("--result-json", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--title", default="64-scene ego-BEV contact sheet | candidates colored by official PDMS, green=scorer pick, orange=oracle-best, black=GT")
parser.add_argument("--max-tiles", type=int, default=64)
args = parser.parse_args()

bev = Path(args.bev_dir)
pngs = sorted(bev.glob("*_bev.png"))[: args.max_tiles]
if not pngs:
    raise SystemExit(f"no BEV pngs under {bev}")

recs = {}
result = Path(args.result_json)
if result.exists():
    data = json.loads(result.read_text())
    for r in data.get("per_scene", []):
        recs[r["scene_id"]] = r

n = len(pngs)
cols = 8
rows = (n + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.1, rows * 2.9))
axes = axes.flatten() if rows * cols > 1 else [axes]
for ax in axes:
    ax.axis("off")
for i, p in enumerate(pngs):
    token = p.name.replace("_bev.png", "")
    img = plt.imread(p)
    axes[i].imshow(img)
    r = recs.get(token, {})
    sel = r.get("selected_pdms"); best = r.get("oracle_best_pdms")
    title = token[:8]
    if r.get("map_name"):
        title += f"  {r['map_name'].replace('us-nv-', '').replace('us-pa-', '').replace('sg-', '').replace('us-ma-', '')}"
    if sel is not None:
        title += f"\nsel={sel:.3f} best={best:.3f}"
    axes[i].set_title(title, fontsize=6)
    axes[i].axis("off")
fig.suptitle(args.title, fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.98))
out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=110, bbox_inches="tight")
plt.close(fig)
print("tiles:", n, "->", out)
