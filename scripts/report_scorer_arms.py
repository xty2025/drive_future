"""Collect the scorer fine-tune arms into one comparison table.

Reads the merged navtest result of every arm plus the frozen ``joint@17500``
baseline, and reports each arm's selected PDMS with a PAIRED difference against
the baseline (same scenes, same 64 candidates, so the difference is attributable
to psi alone).  Also carries the navtrain pool-vs-selection diagnosis and the
three contrasts the experiments were designed to isolate:

    E1  - E1v   net effect of mining hard scenes instead of random ones
    E1s - E1    net effect of dropping pool-limited scenes from the hard set
    E2  - B0    net effect of training on the oracle top-10 + bottom-10 only

Prints a plain-text table and writes REPORT.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

BASELINE_DEFAULT = "artifacts/navtest_official_pdms/joint17500_full/result.json"
WORK_DEFAULT = "artifacts/scorer_hard_mining"
ARMS_DEFAULT = "B0 E1 E1v E1s E2"
CONTRASTS = (("E1", "E1v", "hard mining vs size-matched random"),
             ("E1s", "E1", "drop pool-limited scenes from the hard set"),
             ("E2", "B0", "top10+bottom10 candidates vs all 64"))
PERFECT_WHERE_SELECTABLE = 0.9718
PERFECT_EVERYWHERE = 0.9855


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", default=BASELINE_DEFAULT)
    parser.add_argument("--work-dir", default=WORK_DEFAULT)
    parser.add_argument("--arms", default=ARMS_DEFAULT, help="space-separated arm tags")
    parser.add_argument("--output", default="", help="default: <work-dir>/REPORT.json")
    parser.add_argument("--hard-threshold", type=float, default=0.80)
    return parser.parse_args()


def load_result(path: Path) -> dict[str, Any]:
    blob = json.loads(path.read_text())
    return blob


def arm_result_path(work_dir: Path, tag: str) -> Path | None:
    """The DLC eval stage writes <work>/eval/<tag>/result.json; accept <work>/arms/<tag>/ too."""
    for candidate in (work_dir / "eval" / tag / "result.json", work_dir / "arms" / tag / "result.json"):
        if candidate.exists():
            return candidate
    return None


def selected_by_scene(blob: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for record in blob.get("per_scene", []):
        if "selected_pdms_official" in record:
            out[record["scene_id"]] = float(record["selected_pdms_official"])
    return out


def paired(left: dict[str, float], right: dict[str, float]) -> dict[str, Any]:
    """Paired stats of (left - right) over the scenes both evaluated."""
    common = sorted(set(left) & set(right))
    if not common:
        return {"paired_scenes": 0}
    diff = np.array([left[s] - right[s] for s in common], dtype=np.float64)
    mean = float(diff.mean())
    std = float(diff.std(ddof=1)) if len(diff) > 1 else 0.0
    se = std / np.sqrt(len(diff)) if len(diff) else 0.0
    return {
        "paired_scenes": len(common),
        "mean_diff": round(mean, 5),
        "std_diff": round(std, 5),
        "se_diff": round(se, 5),
        "t_mean_over_se": round(mean / se, 2) if se > 0 else None,
        "wins": int((diff > 0).sum()),
        "losses": int((diff < 0).sum()),
        "ties": int((diff == 0).sum()),
    }


def main() -> int:
    args = parse_args()
    work_dir = Path(args.work_dir)
    baseline_path = Path(args.baseline)
    if not baseline_path.exists():
        raise RuntimeError(f"baseline result not found: {baseline_path}")
    baseline = load_result(baseline_path)
    base_summary = baseline["summary"]
    base_scores = selected_by_scene(baseline)

    diagnosis_path = work_dir / "diagnosis.json"
    diagnosis = json.loads(diagnosis_path.read_text()) if diagnosis_path.exists() else None

    arms: dict[str, dict[str, Any]] = {}
    for tag in args.arms.split():
        arm_dir = work_dir / "arms" / tag
        result_path = arm_result_path(work_dir, tag)
        if result_path is None:
            arms[tag] = {"status": "missing", "result": str(work_dir / "eval" / tag / "result.json")}
            continue
        blob = load_result(result_path)
        summary = blob["summary"]
        scores = selected_by_scene(blob)
        history_path = arm_dir / "history.json"
        heldout = None
        best_step = None
        initial_heldout = None
        if history_path.exists():
            arm_summary = json.loads(history_path.read_text()).get("summary", {})
            heldout = arm_summary.get("best_heldout_selected_pdms")
            best_step = arm_summary.get("best_step")
            initial = arm_summary.get("initial") or {}
            initial_heldout = initial.get("heldout_selected_pdms")
        entry = {
            "status": "ok",
            "result_path": str(result_path),
            "scenes_evaluated": summary.get("scenes_evaluated"),
            "selected_pdms": summary.get("selected_pdms"),
            "oracle_best_pdms": summary.get("oracle_best_pdms"),
            "oracle_match_rate": summary.get("oracle_match_rate"),
            "matched_gt_ade_m": summary.get("matched_gt_ade_m"),
            "noscorer_mean64_pdms": summary.get("noscorer_mean64_pdms"),
            "oracle_failures": summary.get("oracle_failures"),
            "heldout_selected_pdms_navtrain": heldout,
            "heldout_initial_pdms_navtrain": initial_heldout,
            "best_step": best_step,
            # best_step == 0 means no eval beat the starting psi, so this arm's
            # checkpoint_best.pt is the baseline scorer and its navtest number
            # must be read as "no effect", not as a trained result.
            "best_is_untouched_initial": bool(best_step == 0),
            "vs_baseline": paired(scores, base_scores),
        }
        if history_path.exists():
            entry["arm_summary"] = json.loads(history_path.read_text()).get("summary", {})
        arms[tag] = entry

    contrasts = {}
    for left, right, meaning in CONTRASTS:
        if left in arms and right in arms and arms[left].get("status") == "ok" and arms[right].get("status") == "ok":
            l_scores = selected_by_scene(load_result(arm_result_path(work_dir, left)))
            r_scores = selected_by_scene(load_result(arm_result_path(work_dir, right)))
            contrasts[f"{left}-{right}"] = {"meaning": meaning, **paired(l_scores, r_scores)}
        else:
            contrasts[f"{left}-{right}"] = {"meaning": meaning, "status": "missing arm"}

    report = {
        "baseline": {
            "path": str(baseline_path),
            "checkpoint": base_summary.get("checkpoint"),
            "opt_step": base_summary.get("opt_step"),
            "scenes_evaluated": base_summary.get("scenes_evaluated"),
            "selected_pdms": base_summary.get("selected_pdms"),
            "oracle_best_pdms": base_summary.get("oracle_best_pdms"),
            "oracle_match_rate": base_summary.get("oracle_match_rate"),
            "matched_gt_ade_m": base_summary.get("matched_gt_ade_m"),
            "noscorer_mean64_pdms": base_summary.get("noscorer_mean64_pdms"),
        },
        "reference_ceilings_on_navtest": {
            "perfect_selection_where_oracle_best_ge_0.95": PERFECT_WHERE_SELECTABLE,
            "perfect_selection_everywhere": PERFECT_EVERYWHERE,
            "note": "computed from the baseline navtest run; scorer training cannot exceed these",
        },
        "navtrain_diagnosis": diagnosis,
        "arms": arms,
        "contrasts": contrasts,
    }
    output = Path(args.output) if args.output else work_dir / "REPORT.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")

    lines: list[str] = []
    lines.append("")
    lines.append("=" * 128)
    lines.append("navtest official PDMS -- scorer fine-tune arms (same scenes, same 64 candidates, representation frozen)")
    lines.append("=" * 128)
    header = f"{'arm':<6}{'scenes':>8}{'selected':>10}{'d_vs_base':>11}{'t':>8}{'W/L/T':>18}{'oracle_best':>12}{'match':>8}{'ADE':>8}{'heldout':>9}{'best@step':>10}"
    lines.append(header)
    lines.append("-" * 128)
    lines.append(f"{'base':<6}{str(base_summary.get('scenes_evaluated')):>8}"
                 f"{_f(base_summary.get('selected_pdms')):>10}{'--':>11}{'--':>8}{'--':>18}"
                 f"{_f(base_summary.get('oracle_best_pdms')):>12}{_f(base_summary.get('oracle_match_rate')):>8}"
                 f"{_f(base_summary.get('matched_gt_ade_m')):>8}{'--':>9}{'--':>10}")
    for tag in args.arms.split():
        entry = arms[tag]
        if entry.get("status") != "ok":
            lines.append(f"{tag:<6}{'MISSING':>8}   ({entry.get('result')})")
            continue
        vs = entry["vs_baseline"]
        wlt = f"{vs.get('wins', 0)}/{vs.get('losses', 0)}/{vs.get('ties', 0)}"
        lines.append(f"{tag:<6}{str(entry['scenes_evaluated']):>8}{_f(entry['selected_pdms']):>10}"
                     f"{_f(vs.get('mean_diff'), 5):>11}{_f(vs.get('t_mean_over_se'), 2):>8}{wlt:>18}"
                     f"{_f(entry['oracle_best_pdms']):>12}{_f(entry['oracle_match_rate']):>8}"
                     f"{_f(entry['matched_gt_ade_m']):>8}{_f(entry['heldout_selected_pdms_navtrain']):>9}"
                     f"{('*' if entry['best_is_untouched_initial'] else '') + str(entry['best_step']):>10}")
    lines.append("-" * 128)
    lines.append("(* = no held-out eval beat the initial psi, so that arm's checkpoint_best.pt is the baseline scorer)")
    lines.append("contrasts (paired on navtest):")
    for key, value in contrasts.items():
        if value.get("status"):
            lines.append(f"  {key:<10} {value['meaning']} -- {value['status']}")
        else:
            lines.append(f"  {key:<10} {value['meaning']}: {value['mean_diff']:+.5f} "
                         f"(SE {value['se_diff']:.5f}, t={value['t_mean_over_se']}, "
                         f"W/L/T {value['wins']}/{value['losses']}/{value['ties']}, n={value['paired_scenes']})")
    if diagnosis:
        diag = diagnosis["diagnosis"]
        lines.append("-" * 128)
        lines.append("navtrain pool-vs-selection diagnosis (DrivoR label scale):")
        lines.append(f"  scenes {diag['scenes']}  selected {diag['mean_selected_pdms']}  "
                     f"oracle_best {diag['mean_oracle_best']}  gap {diag['total_gap']}")
        lines.append(f"  hard (selected<{args.hard_threshold}): {diag['hard']['scenes']} "
                     f"({diag['hard']['share_of_all']:.2%} of all, {diag['hard_share_of_total_gap']:.2%} of the gap)")
        lines.append(f"    selection-limited : {diag['hard_selection_limited']['scenes']:>6} "
                     f"({diag['hard_selection_limited']['share_of_hard']:.2%} of hard) "
                     f"recoverable {diag['hard_selection_limited']['recoverable_gap']:+.4f}")
        lines.append(f"    middle            : {diag['hard_middle']['scenes']:>6} "
                     f"({diag['hard_middle']['share_of_hard']:.2%} of hard) "
                     f"recoverable {diag['hard_middle']['recoverable_gap']:+.4f}")
        lines.append(f"    pool-limited      : {diag['hard_pool_limited']['scenes']:>6} "
                     f"({diag['hard_pool_limited']['share_of_hard']:.2%} of hard) "
                     f"recoverable {diag['hard_pool_limited']['recoverable_gap']:+.4f}")
        lines.append(f"  gap outside the hard set: {diag['non_hard_gap']:+.4f} "
                     f"(mining hard scenes alone cannot recover this)")
        lines.append(f"  suspect float32 map-artifact scenes (all 64 candidates DAC=0): "
                     f"{diag['suspect_map_artifact_scenes']} ({diag['suspect_map_artifact_share']:.2%})")
        lines.append(f"  ceilings: perfect where oracle_best>=0.95 -> {diag['ceiling_perfect_where_selectable']}, "
                     f"perfect everywhere -> {diag['ceiling_perfect_everywhere']}")
        lines.append("  lists: " + json.dumps(diagnosis["lists"]))
    lines.append("-" * 128)
    lines.append(f"navtest reference ceilings from the baseline: perfect selection where oracle_best>=0.95 -> "
                 f"{PERFECT_WHERE_SELECTABLE}, perfect everywhere -> {PERFECT_EVERYWHERE}")
    lines.append(f"report written to {output}")
    lines.append("=" * 128)
    print("\n".join(lines), flush=True)
    return 0


def _f(value: Any, digits: int = 4) -> str:
    if value is None:
        return "--"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
