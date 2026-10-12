"""Plot training-loss curves for the JOINT run and the SEPARATE (two-phase) run.

Data sources (real logs on disk):
  joint    : artifacts/navtrain_wan_joint_ddp/train_20261011_112117.log   (JSON step lines)
  separate : artifacts/navtrain_wan_twophase_ddp/history.json             ({"phase_a":[...],"phase_b":[...]})
Outputs two PNGs under artifacts/.
"""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/mnt/zhaozc_workspace/project/NVS")
JOINT_LOG = ROOT / "artifacts/navtrain_wan_joint_ddp/train_20261011_112117.log"
TWOPHASE_HIST = ROOT / "artifacts/navtrain_wan_twophase_ddp/history.json"


def smooth(y, w=25):
    """Centered running mean whose window shrinks at the edges (no zero padding,
    so the curve does not fake a collapse at the boundaries)."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < w:
        return y
    cum = np.concatenate([[0.0], np.cumsum(y)])
    idx = np.arange(n)
    lo = np.clip(idx - w // 2, 0, n)
    hi = np.clip(idx + w // 2 + 1, 0, n)
    return (cum[hi] - cum[lo]) / (hi - lo)


def parse_joint():
    raw = JOINT_LOG.read_bytes().decode("utf-8", "ignore")
    recs = []
    for m in re.finditer(r'\{[^{}]*"event": "step"[^{}]*\}', raw):
        try:
            recs.append(json.loads(m.group(0)))
        except Exception:
            pass
    recs.sort(key=lambda r: r["opt_step"])
    return recs


def col(recs, key):
    return np.array([r.get(key, np.nan) for r in recs], dtype=float)


# ---------------- Joint ---------------- #
j = parse_joint()
j_step = col(j, "opt_step")
j_action, j_score, j_total = col(j, "loss_action"), col(j, "loss_score"), col(j, "loss_total")
j_pdms, j_oracle, j_ade = col(j, "eval_selected_pdms"), col(j, "eval_oracle_best_pdms"), col(j, "eval_matched_gt_ade_m")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5.4))
fig.suptitle(f"JOINT training (nvs_v6_wan) - single backward, score loss detached from perception  |  in progress: step {int(j_step[-1])}/{int(j[0]['total_opt_steps'])}", fontsize=12)
for arr, name, c in ((j_total, "loss_total", "#111111"), (j_score, "loss_score (PDMS, psi)", "#d62728"), (j_action, "loss_action (Multi-GT, eta+theta)", "#1f77b4")):
    a1.plot(j_step, arr, color=c, alpha=0.16, lw=0.7)
    a1.plot(j_step, smooth(arr), color=c, lw=1.9, label=name)
a1.set_yscale("log")
a1.set_xlabel("optimizer step"); a1.set_ylabel("loss (log scale)")
a1.set_title("Training loss"); a1.grid(True, which="both", alpha=0.3); a1.legend(fontsize=9)

a2.plot(j_step, j_oracle, color="#999999", alpha=0.25, lw=0.7)
a2.plot(j_step, smooth(j_oracle), color="#7f7f7f", lw=1.8, label="oracle-best PDMS")
a2.plot(j_step, j_pdms, color="#2ca02c", alpha=0.25, lw=0.7)
a2.plot(j_step, smooth(j_pdms), color="#2ca02c", lw=1.9, label="selected PDMS (scorer pick)")
a2.set_ylim(0, 1.02); a2.set_xlabel("optimizer step"); a2.set_ylabel("PDMS")
a2.set_title("Eval PDMS"); a2.grid(True, alpha=0.3); a2.legend(fontsize=9, loc="lower right")
a2b = a2.twinx()
a2b.plot(j_step, smooth(j_ade), color="#ff7f0e", lw=1.6, ls="--", label="matched-GT ADE (m)")
a2b.set_yscale("log"); a2b.set_ylabel("ADE (m, log)", color="#ff7f0e"); a2b.legend(fontsize=9, loc="center right")
fig.tight_layout(rect=(0, 0, 1, 0.95))
out_j = ROOT / "artifacts/loss_curve_joint.png"
fig.savefig(out_j, dpi=130); plt.close(fig)

# ---------------- Separate (two-phase) ---------------- #
h = json.loads(TWOPHASE_HIST.read_text())
pa, pb = h["phase_a"], h["phase_b"]
a_step = col(pa, "opt_step"); a_action = col(pa, "loss_action"); a_ade = col(pa, "eval_matched_gt_ade_m")
b_step = col(pb, "opt_step"); b_score = col(pb, "loss_score"); b_pdms = col(pb, "eval_selected_pdms"); b_oracle = col(pb, "eval_oracle_best_pdms")

fig, (c1, c2) = plt.subplots(1, 2, figsize=(15, 5.4))
fig.suptitle("SEPARATE two-phase training (nvs_v6_wan) - Phase A generator (Multi-GT), then Phase B scorer (oracle PDMS) on frozen cached proposals", fontsize=12)

c1.plot(a_step, a_action, color="#1f77b4", alpha=0.16, lw=0.7)
c1.plot(a_step, smooth(a_action), color="#1f77b4", lw=1.9, label="loss_action (Multi-GT Hungarian)")
c1.set_yscale("log"); c1.set_xlabel("Phase A optimizer step"); c1.set_ylabel("loss_action (log scale)")
c1.set_title("Phase A - generator (eta+theta), phi frozen"); c1.grid(True, which="both", alpha=0.3); c1.legend(fontsize=9)
c1b = c1.twinx()
c1b.plot(a_step, smooth(a_ade), color="#ff7f0e", lw=1.6, ls="--", label="matched-GT ADE (m)")
c1b.set_yscale("log"); c1b.set_ylabel("ADE (m, log)", color="#ff7f0e"); c1b.legend(fontsize=9, loc="upper right")

c2.plot(b_step, b_score, color="#d62728", alpha=0.16, lw=0.7)
c2.plot(b_step, smooth(b_score), color="#d62728", lw=1.9, label="loss_score (official DrivoRLoss)")
c2.set_yscale("log"); c2.set_xlabel("Phase B optimizer step"); c2.set_ylabel("loss_score (log scale)")
c2.set_title("Phase B - scorer (psi) only, generator frozen"); c2.grid(True, which="both", alpha=0.3); c2.legend(fontsize=9)
c2b = c2.twinx()
c2b.plot(b_step, smooth(b_oracle), color="#7f7f7f", lw=1.4, ls=":", label="oracle-best PDMS")
c2b.plot(b_step, smooth(b_pdms), color="#2ca02c", lw=1.8, label="selected PDMS")
c2b.set_ylim(0, 1.02); c2b.set_ylabel("PDMS"); c2b.legend(fontsize=9, loc="lower right")
fig.tight_layout(rect=(0, 0, 1, 0.95))
out_s = ROOT / "artifacts/loss_curve_twophase_separate.png"
fig.savefig(out_s, dpi=130); plt.close(fig)

print("JOINT   recs:", len(j), "step", int(j_step[0]), "->", int(j_step[-1]))
print("  loss_total %.4f -> %.4f | loss_score %.4f -> %.4f | loss_action %.5f -> %.6f" % (j_total[0], j_total[-1], j_score[0], j_score[-1], j_action[0], j_action[-1]))
print("  selected_pdms %.4f -> %.4f | ade %.3f -> %.4f" % (j_pdms[0], j_pdms[-1], j_ade[0], j_ade[-1]))
print("SEPARATE phase_a recs:", len(pa), "| phase_b recs:", len(pb))
print("  A loss_action %.5f -> %.6f | ade %.3f -> %.4f" % (a_action[0], a_action[-1], a_ade[0], a_ade[-1]))
print("  B loss_score  %.4f -> %.4f | selected_pdms %.4f -> %.4f" % (b_score[0], b_score[-1], b_pdms[0], b_pdms[-1]))
print("SAVED:", out_j)
print("SAVED:", out_s)
