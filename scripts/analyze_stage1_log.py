#!/usr/bin/env python
"""Read a finished Stage-1 DDP log and answer "did the future representation learn?".

The trainer's own ``verdict`` records are the evidence, but they are two JSON blobs per
run and the numbers that matter are ratios of each other.  This reduces one log to a
curve panel plus a table that states, for each baseline, whether the model beat it.

The three baselines are nested and that nesting is the whole reading:

* ``constant_target_l1`` -- predict zero.  Equal to ``target_abs_mean`` because the
  targets are standardized, so this is the "the head is still at its initialization"
  line.
* ``scene_mean_target_l1`` -- predict each scene's mean over its own GTs.  This is the
  best a **trajectory-blind** predictor can do, and it is far below the constant line,
  which is what makes "is it learning the scene?" and "is it learning the trajectory?"
  two separate questions.
* ``trajectory_blind_l1`` -- the model re-run with the trajectory input removed.  When
  it equals ``paired_future_l1`` the model is not reading the trajectory, however well
  it is doing on the two lines above.

Those three only exist at a verdict pass, roughly once per epoch.  The trainer also logs
two readings on every ``--log-every`` step -- ``mean_cos_future`` and
``mean_relative_l1_future``, the training-batch versions of the verdict's
``future_cosine_similarity`` and ``future_relative_l1`` -- which is a ~20x denser curve
and the only thing that shows a run stalling between two verdict passes.  They are
plotted against the held-out readings so a train/held-out split in the alignment is
visible directly; note that ``relative_l1`` crossing 1.0 means the register is worse
than emitting zeros, and that eval targets are standardized by train-fitted statistics,
so only the *shape* of the eval curve is comparable to the train one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

#: Verdict keys plotted against ``paired_future_l1``, in the order they should be read.
BASELINES = (
    ("constant_target_l1", "predict zero (= target |mean|)"),
    ("scene_mean_target_l1", "per-scene mean over GTs (trajectory-blind optimum)"),
    ("scene_blind_l1", "another scene's target"),
)

CONDITIONING = (
    ("trajectory_conditioning_fidelity", "trajectory conditioning fidelity"),
    ("trajectory_response_alignment", "trajectory response alignment"),
    ("register_std_across_candidates", "register std across GT candidates"),
    ("future_cosine_similarity", "future cosine similarity"),
)


def parse(log_path: Path) -> dict[str, list[dict[str, Any]]]:
    events: dict[str, list[dict[str, Any]]] = {}
    with log_path.open("rb") as handle:
        for raw in handle:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("{"):
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict) and isinstance(record.get("event"), str):
                events.setdefault(record["event"], []).append(record)
    return events


def plot(events: dict[str, list[dict[str, Any]]], path: Path) -> None:
    records = events["step"]
    steps = [record["opt_step"] for record in records]
    window = [record["mean_loss_future"] for record in records]
    reported = [record["all_rank_mean_loss_future"] for record in records]
    verdicts = events.get("verdict", [])
    vsteps = [record["opt_step"] for record in verdicts]

    figure, axes = plt.subplots(2, 3, figsize=(19.0, 8.6))

    axis = axes[0, 0]
    axis.plot(steps, window, "-o", ms=3, color="#1f77b4", label="train loss_future (rank-0 window mean)")
    axis.plot(steps, reported, "-x", ms=3, color="#aaaaaa", label="all_rank_mean_loss_future (as logged)")
    for record in verdicts:
        axis.axvline(record["opt_step"], color="#bbbbbb", ls=":", lw=0.8)
    axis.set_yscale("log")
    axis.set_xlabel("opt_step"), axis.set_ylabel("L1 (log)")
    axis.set_title("Stage-1 loss", fontsize=10)
    axis.legend(fontsize=7), axis.grid(alpha=0.25)

    axis = axes[0, 1]
    axis.plot(vsteps, [record["train"]["mean_paired_future_l1"] for record in verdicts], "-o",
              color="#d62728", lw=2, label="paired_future_l1 (the model)")
    axis.plot(vsteps, [record["eval"]["mean_paired_future_l1"] for record in verdicts], "--s",
              color="#ff7f0e", lw=2, label="paired_future_l1 (held-out scenes)")
    for key, label in BASELINES:
        axis.plot(vsteps, [record["train"][f"mean_{key}"] for record in verdicts], ":",
                  marker=".", label=label)
    axis.plot(vsteps, [record["train"]["mean_trajectory_blind_l1"] for record in verdicts], "-.^",
              color="#2ca02c", lw=2, label="trajectory_blind_l1 (trajectory input removed)")
    axis.set_xlabel("opt_step"), axis.set_ylabel("L1 against the standardized target")
    axis.set_title("Lower is better; the green and red lines coincide == no trajectory conditioning", fontsize=9)
    axis.legend(fontsize=6.5), axis.grid(alpha=0.25)

    axis = axes[0, 2]
    # Older logs predate the per-step alignment fields; the panel then stays empty
    # rather than raising, because this script is also how those runs get re-read.
    if any("mean_cos_future" in record for record in records):
        held_out = [record for record in verdicts if record.get("eval")]

        axis.plot(steps, [record.get("mean_cos_future") for record in records], "-o", ms=2.5,
                  color="#1f77b4", label="train cos_future (per-step window)")
        axis.plot(vsteps, [record["train"]["mean_future_cosine_similarity"] for record in verdicts], "^",
                  ms=6, color="#1f77b4", mfc="none", label="train cosine (verdict)")
        axis.plot([record["opt_step"] for record in held_out],
                  [record["eval"]["mean_future_cosine_similarity"] for record in held_out], "^",
                  ms=6, color="#ff7f0e", mfc="none", label="held-out cosine (verdict)")
        axis.axhline(0.0, color="#888888", lw=0.8, ls="--")
        axis.set_xlabel("opt_step"), axis.set_ylabel("cosine against the real future")
        axis.set_ylim(-0.2, 1.05)
        axis.set_title("Alignment: 1.0 = reads the future, 0 = orthogonal to it", fontsize=9)
        axis.legend(fontsize=6.5, loc="lower right"), axis.grid(alpha=0.25)

        twin = axis.twinx()
        twin.plot(steps, [record.get("mean_relative_l1_future") for record in records], "-s", ms=2.5,
                  color="#d62728", alpha=0.7, label="train relative_l1 (per-step)")
        twin.plot(vsteps, [record["train"]["mean_future_relative_l1"] for record in verdicts], "^",
                  ms=5, color="#d62728", mfc="none", label="train relative_l1 (verdict)")
        twin.plot([record["opt_step"] for record in held_out],
                  [record["eval"]["mean_future_relative_l1"] for record in held_out], "v",
                  ms=5, color="#ff7f0e", mfc="none", label="held-out relative_l1 (verdict)")
        # 1.0 is the all-zero predictor: below it the register beats emitting nothing.
        twin.axhline(1.0, color="#d62728", lw=1.0, ls=":")
        twin.set_ylabel("relative L1 (1.0 = predict zero)", color="#d62728")
        twin.tick_params(axis="y", labelcolor="#d62728")
        twin.legend(fontsize=6.5, loc="upper right")
    else:
        axis.set_title("no per-step alignment fields in this log (older trainer)", fontsize=9)
        axis.axis("off")

    axis = axes[1, 0]
    for key, label in CONDITIONING:
        axis.plot(vsteps, [record["train"][f"mean_{key}"] for record in verdicts], "-o", ms=4, label=label)
    axis.set_yscale("symlog", linthresh=1e-4)
    axis.set_xlabel("opt_step"), axis.set_ylabel("value (symlog)")
    axis.set_title("Trajectory-conditioning diagnostics", fontsize=10)
    axis.legend(fontsize=7), axis.grid(alpha=0.25)

    axis = axes[1, 1]
    axis.plot(steps, [record["lr"] for record in records], "-o", ms=3, color="#9467bd")
    axis.set_xlabel("opt_step"), axis.set_ylabel("lr", color="#9467bd")
    axis.tick_params(axis="y", labelcolor="#9467bd")
    axis.grid(alpha=0.25)
    twin = axis.twinx()
    twin.plot(steps, [record["sec_per_step"] for record in records], "-s", ms=3, color="#8c564b",
              label="sec_per_step (cumulative mean)")
    # The cumulative mean cannot show a slowdown: averaged over thousands of steps, a
    # 2x regression starting three quarters of the way in moves it by a few percent.
    # The window mean over just the steps since the last record is what exposes one,
    # which on a ~27 h run is the difference between noticing at hour 8 and at hour 27.
    if any(record.get("sec_per_step_window") is not None for record in records):
        twin.plot(steps, [record.get("sec_per_step_window") for record in records], "-", lw=1.0, ms=2,
                  color="#e377c2", label="sec_per_step_window (since the last record)")
    twin.set_ylabel("s/step", color="#8c564b")
    twin.tick_params(axis="y", labelcolor="#8c564b")
    twin.legend(fontsize=6.5, loc="upper right")
    axis.set_title("Schedule and throughput", fontsize=10)

    axis = axes[1, 2]
    axis.plot(steps, [record["supervised_gt_rows"] for record in records], "-o", ms=2.5, color="#2ca02c",
              label="supervised GT rows per micro-batch (rank 0)")
    expected = int(records[-1]["global_batch_scenes"])
    axis.axhline(expected, color="#888888", lw=0.9, ls="--",
                 label=f"global batch = {expected} scenes (1 row per scene is the floor)")
    for record in events.get("epoch_done", []):
        axis.axvline(record["opt_step"], color="#bbbbbb", ls=":", lw=0.8)
    axis.set_xlabel("opt_step"), axis.set_ylabel("GT rows with gradient")
    axis.set_title("Supervision density; dotted verticals are epoch boundaries", fontsize=9)
    axis.legend(fontsize=6.5, loc="lower right"), axis.grid(alpha=0.25)

    figure.suptitle(path.stem, fontsize=11)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
    figure.savefig(path, dpi=110)
    plt.close(figure)


def report(events: dict[str, list[dict[str, Any]]], log_path: Path) -> None:
    steps = events["step"]
    verdicts = events.get("verdict", [])
    schedule = events.get("schedule", [{}])[0]
    start = events.get("start", [{}])[0]
    split = events.get("split", [{}])[0]
    cache = events.get("standardizer_loaded", [{}])[0]

    total = max(int(schedule.get("total_opt_steps", 1)), 1)
    print(f"log        : {log_path}")
    print(f"world      : {start.get('world')} ranks")
    print(f"scenes     : train={split.get('train')} eval={split.get('eval')} rejected={split.get('rejected')}")
    print(f"steps      : {len(steps)} logged, last={steps[-1]['opt_step']} of {total} "
          f"({100.0 * float(steps[-1]['opt_step']) / total:.1f}%)")
    print(f"throughput : {steps[-1]['sec_per_step']} s/step cumulative, "
          f"{steps[-1].get('sec_per_step_window')} s/step over the last window, "
          f"global batch {steps[-1]['global_batch_scenes']} scenes "
          f"=> {float(steps[-1]['sec_per_step']) / max(int(steps[-1]['global_batch_scenes']), 1):.3f} s/scene")
    if "mean_cos_future" in steps[0]:
        cosines = [record["mean_cos_future"] for record in steps]
        relatives = [record["mean_relative_l1_future"] for record in steps]
        print(f"alignment  : cos_future {cosines[0]:+.4f} -> {cosines[-1]:+.4f} (peak {max(cosines):+.4f}) | "
              f"relative_l1 {relatives[0]:.4f} -> {relatives[-1]:.4f} (best {min(relatives):.4f})")
        print("             cosine 0 = orthogonal to the real future, 1 = reads it; "
              "relative_l1 1.0 = no better than emitting zeros")
    if cache:
        print(f"cache      : sep={cache.get('separation_ratio')} "
              f"pearson={cache.get('trajectory_distance_pearson')} "
              f"entropy_mean={cache.get('attention_entropy_mean')} "
              f"entropy_max={cache.get('attention_entropy_max')} "
              f"clamped={cache.get('clamped_components')}")

    if not verdicts:
        print("no verdict records in this log")
        return

    print()
    header = f"{'metric':<44}" + "".join(f"{v['opt_step']:>12}" for v in verdicts)
    print(header)
    print("-" * len(header))
    for side in ("train", "eval"):
        keys = ("paired_future_l1", "trajectory_blind_l1", "scene_blind_l1", "constant_target_l1",
                "scene_mean_target_l1", "target_spread_across_gt", "distance_to_mean_target_l1",
                "register_abs_mean", "register_std_across_candidates", "future_cosine_similarity",
                "future_relative_l1", "trajectory_conditioning_fidelity")
        print(f"[{side}]")
        for key in keys:
            row = f"  {key:<42}" + "".join(f"{v[side][f'mean_{key}']:>12.4f}" for v in verdicts)
            print(row)
        # ``collapse_toward_mean`` is a ratio, so the trainer files it under the verdict
        # block with a ``<side>_`` prefix instead of inside the per-side metric dict.
        row = f"  {'collapse_toward_mean':<42}" + "".join(
            f"{v['register_verdict'][f'{side}_collapse_toward_mean']:>12.4f}" for v in verdicts
        )
        print(row)
    print()
    for key, label in (("beats_scene_mean_baseline", "beats the trajectory-blind optimum"),
                       ("learned_trajectory_conditional_future", "learned a trajectory-conditional future"),
                       ("generalizes_beyond_fitted_scenes", "generalizes to held-out scenes")):
        # The first two are reported per split, the third once for the whole run.
        def verdict_of(record: dict[str, Any]) -> str:
            block = record["register_verdict"]
            if f"train_{key}" not in block:
                return str(block.get(key))
            return f"train={block[f'train_{key}']} eval={block[f'eval_{key}']}"

        print(f"  {label:<52}: {'  ->  '.join(verdict_of(v) for v in verdicts)}")

    last = verdicts[-1]
    paired, blind = last["train"]["mean_paired_future_l1"], last["train"]["mean_trajectory_blind_l1"]
    floor = last["train"]["mean_scene_mean_target_l1"]
    constant = last["train"]["mean_constant_target_l1"]
    print()
    print(f"  paired - trajectory_blind = {paired - blind:+.3e}  (the entire trajectory contribution)")
    print(f"  paired - scene_mean floor = {paired - floor:+.4f}   (how far above the trajectory-blind optimum)")
    print(f"  paired - predict-zero     = {paired - constant:+.4f}   (how far the head moved at all)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log", help="a Stage-1 DDP training log")
    parser.add_argument("--out", default=None, help="where to write the PNG; defaults to <log>.png")
    args = parser.parse_args()

    log_path = Path(args.log)
    events = parse(log_path)
    if "step" not in events:
        raise SystemExit(f"{log_path} has no parseable step records")
    out = Path(args.out) if args.out else log_path.with_suffix(".png")
    plot(events, out)
    report(events, log_path)
    print(f"\nfigure -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
