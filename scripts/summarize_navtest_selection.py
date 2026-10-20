"""Selected-vs-pre-selection statistics over a FULL merged navtest result.json.

The per-scene records written by ``eval_navtest_official_pdms_e2e.py`` already carry
the scorer's pick (``selected_pdms_official``), the pre-selection pool summary
(``noscorer_mean64_official`` = mean over all 64 candidates, ``noscorer_cand0_official``
= the anchor/prior candidate #0), and the pool max (``oracle_best_pdms_official``).
This aggregates those into the same "what the scorer picked vs the pool before it
picked" framing the 64-scene viz uses, but over the whole held-out split, and draws
the distributions.  Read-only; needs no GPU, no checkpoint, no oracle.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--result", required=True, help="merged result.json from the official navtest eval")
    p.add_argument("--tag", required=True)
    p.add_argument("--output-dir", default="")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    result_path = Path(args.result)
    data = json.loads(result_path.read_text())
    summary = data.get("summary", {})
    per_scene = data.get("per_scene", [])

    sel, best, pmean, cand0, match, near_ade, matched_ade = [], [], [], [], [], [], []
    for r in per_scene:
        if r.get("selected_pdms_official") is None:
            continue
        sel.append(float(r["selected_pdms_official"]))
        best.append(float(r["oracle_best_pdms_official"]))
        pmean.append(float(r["noscorer_mean64_official"]))
        cand0.append(float(r["noscorer_cand0_official"]) if r.get("noscorer_cand0_official") is not None else np.nan)
        match.append(1.0 if r.get("oracle_match") else 0.0)
        if r.get("all64_nearest_gt_ade_m") is not None:
            near_ade.append(float(r["all64_nearest_gt_ade_m"]))
        if r.get("matched_gt_ade_m") is not None and np.isfinite(r.get("matched_gt_ade_m")):
            matched_ade.append(float(r["matched_gt_ade_m"]))

    sel = np.array(sel); best = np.array(best); pmean = np.array(pmean)
    cand0 = np.array(cand0); match = np.array(match)
    gain = sel - pmean
    regret = best - sel

    def _m(x):
        x = np.asarray(x, float); x = x[np.isfinite(x)]
        return float(x.mean()) if x.size else float("nan")

    stats = {
        "tag": args.tag,
        "result": str(result_path),
        "n_scored_scenes": int(sel.size),
        "preselection_pool_mean_pdms": round(_m(pmean), 4),
        "preselection_cand0_pdms": round(_m(cand0), 4),
        "selected_mean_pdms": round(_m(sel), 4),
        "oracle_best_mean_pdms": round(_m(best), 4),
        "selection_gain_over_pool_mean": round(_m(gain), 4),
        "selection_regret_vs_oracle": round(_m(regret), 4),
        "oracle_match_rate": round(_m(match), 4),
        "selected_beats_cand0_rate": round(float(np.nanmean(sel > cand0)), 4) if cand0.size else None,
        "matched_gt_ade_m": round(_m(matched_ade), 4),
        "all64_nearest_gt_ade_m": round(_m(near_ade), 4),
        "summary_selected_pdms": summary.get("selected_pdms"),
        "summary_oracle_best_pdms": summary.get("oracle_best_pdms"),
        "summary_oracle_match_rate": summary.get("oracle_match_rate"),
        "summary_noscorer_mean64_pdms": summary.get("noscorer_mean64_pdms"),
        "summary_noscorer_cand0_pdms": summary.get("noscorer_cand0_pdms"),
    }

    out_dir = Path(args.output_dir or str(result_path.parent / "selection_full"))
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(15, 9), dpi=110)
    gs = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.28)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.hist(sel[np.isfinite(sel)], bins=60, color="#00a651", edgecolor="none", alpha=0.85, label="selected")
    ax0.hist(pmean[np.isfinite(pmean)], bins=60, color="#9ecae1", edgecolor="none", alpha=0.7, label="pool mean")
    ax0.axvline(_m(sel), color="#006400", ls="-", lw=1.6)
    ax0.axvline(_m(pmean), color="#6a51a3", ls="--", lw=1.6)
    ax0.set(title="selected vs pre-selection pool mean", xlabel="official PDMS", ylabel="count")
    ax0.legend(fontsize=7)

    ax1 = fig.add_subplot(gs[0, 1])
    bp = ax1.boxplot([pmean[np.isfinite(pmean)], sel[np.isfinite(sel)], best[np.isfinite(best)],
                      cand0[np.isfinite(cand0)]],
                     tick_labels=["pool mean", "selected", "oracle-best", "cand0"], showfliers=False, patch_artist=True)
    for patch, c in zip(bp["boxes"], ["#9ecae1", "#00a651", "#ff9f00", "#bdbdbd"]):
        patch.set_facecolor(c)
    ax1.set(title="PDMS by selection stage", ylabel="official PDMS")

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.scatter(pmean, sel, s=6, c="#00a651", alpha=0.35)
    lo = float(np.nanmin(pmean)) if pmean.size else 0.0
    hi = float(np.nanmax(best)) if best.size else 1.0
    ax2.plot([lo, hi], [lo, hi], color="gray", ls="--", lw=1.0)
    ax2.set(title="per scene: pool mean (x) vs selected (y)", xlabel="pool mean PDMS", ylabel="selected PDMS")

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.hist(gain[np.isfinite(gain)], bins=60, color="#00a651", edgecolor="none")
    ax3.axvline(0, color="gray", ls="--", lw=1.0)
    ax3.axvline(_m(gain), color="black", ls="-", lw=1.6, label=f"mean {_m(gain):+.3f}")
    ax3.set(title="selection gain (selected - pool mean)", xlabel="PDMS delta", ylabel="count")
    ax3.legend(fontsize=7)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.hist(regret[np.isfinite(regret)], bins=60, color="#ff9f00", edgecolor="none")
    ax4.axvline(_m(regret), color="black", ls="-", lw=1.6, label=f"mean {_m(regret):.3f}")
    ax4.set(title="regret (oracle-best - selected)", xlabel="PDMS delta", ylabel="count")
    ax4.legend(fontsize=7)

    ax5 = fig.add_subplot(gs[1, 2])
    ax5.hist(match, bins=[-0.5, 0.5, 1.5], color="#6a51a3", edgecolor="none")
    ax5.set(title=f"oracle match (rate {_m(match):.3f})", xlabel="0 = miss, 1 = exact argmax", ylabel="count")
    ax5.set_xticks([0, 1])

    fig.suptitle(f"navtest FULL selection stats | {args.tag} | {int(sel.size)} scenes | "
                 f"selected {_m(sel):.3f} vs pool {_m(pmean):.3f} vs oracle {_m(best):.3f}", fontsize=12)
    fig.savefig(out_dir / "selection_stats_full.png", bbox_inches="tight")
    plt.close(fig)

    (out_dir / "selection_stats_full.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
