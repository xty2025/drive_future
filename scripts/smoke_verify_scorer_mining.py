"""Assertion harness for the scorer hard-mining pipeline (runs on DLC, small sample).

Every check exits non-zero with a one-line ``FAIL`` so the DLC driver aborts the
smoke stage instead of silently producing a bad 2.5 h run.  Checks are read-only:
they only open artifacts produced by the smoke stage.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("NUPLAN_MAPS_ROOT", "/mnt/data/navsim/maps")
os.environ.setdefault("NAVSIM_EXP_ROOT", "/mnt/zhaozc_workspace/official_DrivoR_clean/data_runtime")
os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402

SHARD_KEYS = ("scene", "trajectory", "raw7", "final_valid", "selected_index", "scene_ids", "skipped", "base_opt_step")
EXPECTED_SHAPES = {"scene": (66, 256), "trajectory": (64, 8, 3), "raw7": (64, 7), "final_valid": (64,)}
CANDIDATES = 64


def ok(msg: str) -> None:
    print(f"[SMOKE-OK] {msg}", flush=True)


def fail(msg: str) -> None:
    raise AssertionError(msg)


def check(cond: bool, msg: str) -> None:
    if not cond:
        fail(msg)
    ok(msg)


def load_shards(cache_dir: Path) -> list[tuple[Path, dict]]:
    paths = sorted(cache_dir.glob("shard_*.pt"))
    check(bool(paths), f"{cache_dir} holds shard_*.pt files ({len(paths)})")
    out = []
    for path in paths:
        out.append((path, torch.load(path, map_location="cpu", weights_only=True)))
    return out


def cmd_builder(args) -> int:
    cache_dir = Path(args.cache_dir)
    shards = load_shards(cache_dir)
    seen: set[str] = set()
    total = 0
    for path, payload in shards:
        missing = [k for k in SHARD_KEYS if k not in payload]
        check(not missing, f"{path.name}: all {len(SHARD_KEYS)} keys present")
        n = len(payload["scene_ids"])
        check(n > 0, f"{path.name}: {n} scenes")
        for key, tail in EXPECTED_SHAPES.items():
            shape = tuple(payload[key].shape)
            check(shape == (n,) + tail, f"{path.name}: {key} shape {shape} == {(n,) + tail}")
        check(payload["scene"].dtype == torch.float32, f"{path.name}: scene dtype float32")
        check(payload["trajectory"].dtype == torch.float32, f"{path.name}: trajectory dtype float32")
        check(payload["raw7"].dtype == torch.float32, f"{path.name}: raw7 dtype float32")
        check(payload["final_valid"].dtype == torch.bool, f"{path.name}: final_valid dtype bool")
        check(bool(torch.isfinite(payload["raw7"]).all()), f"{path.name}: raw7 all finite")
        check(bool(torch.isfinite(payload["scene"]).all()), f"{path.name}: scene all finite")
        check(bool(torch.isfinite(payload["trajectory"]).all()), f"{path.name}: trajectory all finite")
        check(bool(payload["final_valid"].all()), f"{path.name}: final_valid all True (failed rows dropped)")
        sel = payload["selected_index"]
        check(sel.dtype in (torch.int16, torch.int32, torch.int64), f"{path.name}: selected_index dtype {sel.dtype}")
        check(bool((sel >= 0).all() and (sel < CANDIDATES).all()), f"{path.name}: selected_index within [0,{CANDIDATES})")
        dup = [s for s in payload["scene_ids"] if s in seen]
        check(not dup, f"{path.name}: no scene_id repeated across shards (dups={dup[:3]})")
        seen.update(payload["scene_ids"])
        total += n
    check(total > 0, f"cache holds {total} scenes across {len(shards)} shards")
    if (cache_dir / "_builder_meta.json").exists():
        meta = json.loads((cache_dir / "_builder_meta.json").read_text())
        check(meta.get("base_opt_step", -1) == args.expect_opt_step or args.expect_opt_step == 0,
              f"_builder_meta base_opt_step={meta.get('base_opt_step')}")

    if args.oracle_scenes > 0:
        _check_oracle(cache_dir, shards, args)
    return 0


def _check_oracle(cache_dir: Path, shards: list[tuple[Path, dict]], args) -> None:
    import run_navtrain_wan_joint_ddp as trainer
    from nvs_v6_wan.oracle import DrivoRBatchMetricCacheOracle

    limit = args.limit_scenes if args.limit_scenes > 0 else None
    candidates = trainer.build_scene_index(args.multigt_root, args.cache_root, args.log_root,
                                           Path(args.index_dir), limit)
    cache_paths = {c.token: c.cache_path for c in candidates}
    oracle = DrivoRBatchMetricCacheOracle(cache_paths=cache_paths, evaluator_version=trainer.ORACLE_VERSION)

    picked = []
    for _path, payload in shards:
        for row, sid in enumerate(payload["scene_ids"]):
            if sid in cache_paths and len(picked) < args.oracle_scenes:
                picked.append((sid, payload["trajectory"][row], payload["raw7"][row]))
    check(len(picked) == args.oracle_scenes,
          f"found {len(picked)} scenes to re-score with DrivoRBatchMetricCacheOracle")

    ids = [p[0] for p in picked]
    traj = torch.stack([p[1] for p in picked])
    labels = oracle.evaluate(ids, traj, batch_id="smoke_recheck")
    check(not labels.failure_diagnostics, f"re-scored oracle reported no failures ({labels.failure_diagnostics[:2]})")
    worst = 0.0
    for i, (sid, _traj, stored) in enumerate(picked):
        diff = float((labels.raw7[i] - stored).abs().max())
        worst = max(worst, diff)
    check(worst == 0.0, f"cached raw7 == freshly re-scored raw7 for {ids} (max abs diff {worst:.3e})")


def cmd_shards(args) -> int:
    a = {sid: (payload["raw7"][i], payload["selected_index"][i])
         for _p, payload in load_shards(Path(args.a)) for i, sid in enumerate(payload["scene_ids"])}
    b = {sid: (payload["raw7"][i], payload["selected_index"][i])
         for _p, payload in load_shards(Path(args.b)) for i, sid in enumerate(payload["scene_ids"])}
    check(set(a) == set(b), f"nproc={args.nproc_a} and nproc={args.nproc_b} cover the identical {len(a)} scene_ids")
    worst_raw = 0.0
    sel_mismatch = 0
    for sid in a:
        worst_raw = max(worst_raw, float((a[sid][0] - b[sid][0]).abs().max()))
        sel_mismatch += int(a[sid][1] != b[sid][1])
    check(worst_raw == 0.0, f"raw7 identical across the two DDP partitions (max abs diff {worst_raw:.3e})")
    check(sel_mismatch == 0, f"selected_index identical across the two DDP partitions ({sel_mismatch} mismatches)")
    return 0


def cmd_mining(args) -> int:
    d = Path(args.dir)
    names = ["all", "hard_all", "hard_all_clean", "hard_selection", "hard_selection_clean",
             "hard_pool", "hard_middle", "random_matched", "random_matched_clean", "heldout"]
    lists: dict[str, set] = {}
    for name in names:
        path = d / f"scenes_{name}.json"
        check(path.exists(), f"scenes_{name}.json exists")
        blob = json.loads(path.read_text())
        scenes = blob["scenes"]
        check(blob["count"] == len(scenes), f"scenes_{name}.json count {blob['count']} == len {len(scenes)}")
        check(len(set(scenes)) == len(scenes), f"scenes_{name}.json has no duplicate scene_id")
        lists[name] = set(scenes)
    check((d / "diagnosis.json").exists(), "diagnosis.json exists")

    split_sum = len(lists["hard_selection"]) + len(lists["hard_middle"]) + len(lists["hard_pool"])
    check(split_sum == len(lists["hard_all"]),
          f"selection({len(lists['hard_selection'])})+middle({len(lists['hard_middle'])})"
          f"+pool({len(lists['hard_pool'])}) == hard_all({len(lists['hard_all'])})")
    check(not (lists["heldout"] & lists["hard_all"]), "heldout and hard_all are disjoint")
    check(not (lists["heldout"] & lists["random_matched"]), "heldout and random_matched are disjoint")
    check(not (lists["heldout"] & lists["all"]),
          "heldout is disjoint from scenes_all (scenes_all is the trainable pool)")
    check(lists["hard_all"] <= lists["all"], "hard_all is a subset of scenes_all")
    check(lists["random_matched"] <= lists["all"], "random_matched is a subset of scenes_all")
    check(len(lists["random_matched"]) == len(lists["hard_all"]),
          f"random_matched({len(lists['random_matched'])}) size-matches hard_all({len(lists['hard_all'])})")
    check(not (lists["random_matched"] & lists["heldout"]), "random_matched avoids the heldout scenes")
    check(not (lists["hard_selection"] & lists["hard_pool"]), "hard_selection and hard_pool are disjoint")
    check(lists["hard_selection_clean"] <= lists["hard_selection"], "hard_selection_clean subset of hard_selection")
    check(lists["hard_all_clean"] <= lists["hard_all"], "hard_all_clean subset of hard_all")

    diag = json.loads((d / "diagnosis.json").read_text())
    print(json.dumps({"event": "smoke_mining_summary",
                      "counts": {k: len(v) for k, v in lists.items()},
                      "diagnosis_keys": sorted(diag)}), flush=True)
    return 0


def load_checkpoint(path: Path) -> dict:
    for kwargs in ({"mmap": True, "weights_only": True}, {"weights_only": True}, {"weights_only": False}):
        try:
            return torch.load(path, map_location="cpu", **kwargs)
        except Exception:
            continue
    raise RuntimeError(f"cannot load checkpoint {path}")


def _scorer_from_base(base: dict):
    import run_navtrain_wan_joint_ddp as trainer
    scorer = trainer.ScratchDrivoRScorer(d_model=256, heads=1, ffn_dim=1024, decoder_layers=4)
    return trainer, scorer


def cmd_finetune(args) -> int:
    out = Path(args.dir)
    base = load_checkpoint(Path(args.base_checkpoint))
    hist_path = out / "history.json"
    check(hist_path.exists(), "history.json exists")
    hist = json.loads(hist_path.read_text())
    best_step = int(hist["summary"].get("best_step", -1))
    for kind in ("best", "last"):
        ckpt_path = out / f"checkpoint_{kind}.pt"
        check(ckpt_path.exists(), f"checkpoint_{kind}.pt exists")
        ckpt = load_checkpoint(ckpt_path)
        check(set(ckpt) >= {"model", "scorer", "opt_scorer", "config", "epoch", "opt_step"},
              f"checkpoint_{kind}.pt carries the joint-compatible keys {sorted(ckpt)}")
        check("opt_generator" not in ckpt, f"checkpoint_{kind}.pt drops the 2.4 GB opt_generator")

        _trainer, scorer = _scorer_from_base(base)
        scorer.load_state_dict(ckpt["scorer"], strict=True)
        ok(f"checkpoint_{kind}.pt['scorer'] loads into ScratchDrivoRScorer with strict=True")

        gen_diff = max(float((ckpt["model"][k].float() - base["model"][k].float()).abs().max())
                       for k in base["model"])
        check(gen_diff == 0.0, f"checkpoint_{kind}.pt['model'] bit-identical to the base (max abs diff {gen_diff:.3e})")
        check(set(ckpt["model"]) == set(base["model"]), f"checkpoint_{kind}.pt['model'] key set unchanged")

        scorer_diff = max(float((ckpt["scorer"][k].float() - base["scorer"][k].float()).abs().max())
                          for k in base["scorer"])
        # "last" always reflects the trained psi.  "best" is the heldout-best
        # snapshot, so with best_step == 0 nothing ever beat the initial eval and
        # the correct content IS the untouched base psi -- not a failure.
        if not args.expect_changed:
            check(scorer_diff == 0.0, f"zero-step export leaves psi untouched (max abs diff {scorer_diff:.3e})")
        elif kind == "last":
            check(scorer_diff > 0.0, f"checkpoint_last.pt psi actually moved (max abs diff {scorer_diff:.3e})")
        elif best_step == 0:
            check(scorer_diff == 0.0,
                  f"best_step==0 (no eval beat the initial) so checkpoint_best.pt is the base psi (diff {scorer_diff:.3e})")
        else:
            check(scorer_diff > 0.0,
                  f"checkpoint_best.pt psi moved at best_step={best_step} (max abs diff {scorer_diff:.3e})")
        check(int(ckpt["config"]["provenance"].get("finetune_opt_step", -1)) >= 0,
              f"checkpoint_{kind}.pt provenance.finetune_opt_step recorded")
        prov = ckpt["config"].get("provenance", {})
        check(bool(prov), f"checkpoint_{kind}.pt records config.provenance ({sorted(prov)[:6]}...)")

    losses = [r["loss_score"] for r in hist["history"]
              if r.get("event") == "step" and r.get("loss_score") is not None]
    evals = [r for r in hist["history"] if r.get("event") == "eval"]
    if args.expect_changed:
        check(bool(losses), f"history.json logged {len(losses)} step records")
        check(all(v == v and abs(v) < 1e6 for v in losses), "every logged loss is finite")
    check(bool(evals), f"history.json logged {len(evals)} held-out evals")
    check(all(r["heldout_selected_pdms"] == r["heldout_selected_pdms"] for r in evals),
          "heldout_selected_pdms finite at every eval")
    print(json.dumps({"event": "smoke_finetune_summary", "dir": str(out), "steps": len(losses),
                      "evals": len(evals), "best_step": best_step,
                      "final_heldout_selected_pdms": evals[-1]["heldout_selected_pdms"] if evals else None}),
                     flush=True)
    return 0


def cmd_eval_equal(args) -> int:
    a = json.loads(Path(args.a).read_text())
    b = json.loads(Path(args.b).read_text())
    ra = {r["scene_id"]: r for r in a["per_scene"]}
    rb = {r["scene_id"]: r for r in b["per_scene"]}
    check(set(ra) == set(rb), f"both runs cover the identical {len(ra)} navtest scenes")
    worst = 0.0
    idx_mismatch = 0
    for sid in ra:
        worst = max(worst, abs(ra[sid]["selected_pdms_official"] - rb[sid]["selected_pdms_official"]))
        idx_mismatch += int(ra[sid]["selected_index"] != rb[sid]["selected_index"])
    sa, sb = a["summary"]["selected_pdms"], b["summary"]["selected_pdms"]
    check(worst <= args.tol, f"per-scene selected PDMS agree within {args.tol} (max abs diff {worst:.3e})")
    check(idx_mismatch == 0, f"selected candidate index identical on every scene ({idx_mismatch} mismatches)")
    check(abs(sa - sb) <= args.tol, f"summary selected_pdms {sa} vs {sb} within {args.tol}")
    print(json.dumps({"event": "smoke_eval_equal", "selected_pdms_a": sa, "selected_pdms_b": sb,
                      "scenes": len(ra)}), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("builder", help="shard keys/shapes/finiteness + optional oracle re-score")
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--oracle-scenes", type=int, default=0)
    p.add_argument("--expect-opt-step", type=int, default=0)
    p.add_argument("--limit-scenes", type=int, default=0)
    p.add_argument("--index-dir", default=str(REPO_ROOT / "artifacts/navtrain_wan_joint_ddp"))
    p.add_argument("--multigt-root", default=os.environ.get("NVS_MULTIGT_ROOT", ""))
    p.add_argument("--cache-root", default=os.environ.get("NVS_CACHE_ROOT", ""))
    p.add_argument("--log-root", default=os.environ.get("NVS_LOG_ROOT", ""))
    p.set_defaults(func=cmd_builder)

    p = sub.add_parser("shards", help="two DDP partitions of the same scenes must agree")
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--nproc-a", default="1")
    p.add_argument("--nproc-b", default="2")
    p.set_defaults(func=cmd_shards)

    p = sub.add_parser("mining", help="scene-list invariants")
    p.add_argument("--dir", required=True)
    p.set_defaults(func=cmd_mining)

    p = sub.add_parser("finetune", help="merged checkpoint integrity + frozen generator")
    p.add_argument("--dir", required=True)
    p.add_argument("--base-checkpoint", required=True)
    p.add_argument("--expect-changed", action="store_true")
    p.set_defaults(func=cmd_finetune)

    p = sub.add_parser("eval-equal", help="two navtest result.json must agree per scene")
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--tol", type=float, default=1e-9)
    p.set_defaults(func=cmd_eval_equal)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as error:
        print(f"[SMOKE-FAIL] {error}", flush=True)
        raise SystemExit(4)
