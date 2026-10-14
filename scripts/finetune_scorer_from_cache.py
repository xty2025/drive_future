"""Scorer-only (psi) fine-tune on the frozen-generator cache -- 16-GPU DDP.

Continues the DrivoR scorer from ``joint@17500`` with the representation frozen.
The generator (eta + theta) and DINOv3 are **never instantiated here**: only
``ckpt['scorer']`` is read, so freezing the representation is structural rather
than a flag, and the exported checkpoint reuses the base ``model`` tensors
verbatim.

The loss is unchanged: ``ScratchDrivoRScorer.source_loss`` -> the official
``DrivoRLoss.score_loss`` utility term, exactly the ``lambda_score`` component
``run_navtrain_wan_joint_ddp.py`` used.  No rank loss (``lambda_rank = 0``), no
new objective, no DrivoR pretrained weights.

Two knobs implement the two experiments:

``--scene-list``      train on a mined subset (hard_all / hard_selection /
                      random_matched / all).  Held-out scenes are excluded from
                      every list by ``mine_hard_scenes.py``.
``--candidate-select`` ``all64`` (M=64) or ``top10bottom10``: per scene keep the
                      oracle top-10 and bottom-10 proposals, so M=20.  Training
                      uses M=20 but every evaluation and the final ``select``
                      still run on all 64 candidates, because at inference time
                      the oracle scores are unknown -- that is the real
                      deployment condition this arm measures.

Shard assignment: training rank ``r`` reads cache shard files ``[r::world]`` and
keeps only the rows whose scene_id is in the requested lists, so peak RSS stays
~1/world of the cache instead of replicating it.  Per-rank row counts differ, so
the epoch length is ``all_reduce_min`` of the per-rank step counts and the
held-out metrics are accumulated locally and all-reduced exactly once (never per
batch) to keep every rank in lockstep.

Launch with ``python -m torch.distributed.run --standalone --nnodes=1
--nproc_per_node=16``.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_navtrain_wan_joint_ddp as trainer  # noqa: E402
import run_navtrain_wan_scorer_rank_ddp as rank  # noqa: E402
from nvs_v6_wan.scorer import ScratchDrivoRScorer  # noqa: E402

DEFAULT_CACHE = "artifacts/scorer_hard_mining/cache"
DEFAULT_BASE = "artifacts/navtrain_wan_joint_ddp/checkpoint.pt"
DEFAULT_MINED = "artifacts/scorer_hard_mining"
PDMS = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", required=True, help="arm name; also the output subdir")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE)
    parser.add_argument("--base-checkpoint", default=DEFAULT_BASE)
    parser.add_argument("--mined-dir", default=DEFAULT_MINED, help="dir holding scenes_*.json from mine_hard_scenes.py")
    parser.add_argument("--scene-list", default="all", help="scenes_<name>.json to train on ('' = every cached scene)")
    parser.add_argument("--heldout-list", default="heldout", help="scenes_<name>.json used for model selection")
    parser.add_argument("--output-dir", default="", help="default: <mined-dir>/arms/<tag>")
    parser.add_argument("--candidate-select", default="all64", choices=["all64", "top10bottom10"])
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--bottom-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=12, help="per-GPU batch (12 x 16 => global 192, same as joint)")
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--scorer-lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup", type=int, default=100, help="linear LR warmup optimizer-steps")
    parser.add_argument("--max-steps", type=int, default=2000, help="0 = fall back to --epochs")
    parser.add_argument("--epochs", type=int, default=0, help="used only when --max-steps is 0")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=100, help="held-out eval + best-checkpoint cadence (0 = only at the end)")
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--find-unused", action="store_true")
    return parser.parse_args()


def load_base_checkpoint(path: Path) -> dict[str, Any]:
    """mmap the base checkpoint so the 1.2 GB generator never reaches RSS."""
    for kwargs in ({"mmap": True, "weights_only": True}, {"weights_only": True}, {"weights_only": False}):
        try:
            return torch.load(path, map_location="cpu", **kwargs)
        except Exception:
            continue
    raise RuntimeError(f"cannot load base checkpoint {path}")


def read_scene_list(mined_dir: Path, name: str) -> list[str] | None:
    if not name:
        return None
    path = Path(mined_dir) / f"scenes_{name}.json"
    if not path.exists():
        raise RuntimeError(f"missing scene list {path}; run mine_hard_scenes.py first")
    blob = json.loads(path.read_text())
    scenes = blob.get("scenes") if isinstance(blob, dict) else blob
    if not isinstance(scenes, list) or not scenes:
        raise RuntimeError(f"{path} contains no scenes")
    return [str(s) for s in scenes]


def fingerprint(state: dict[str, torch.Tensor]) -> str:
    keys = sorted(state)
    if not keys:
        return hashlib.sha256(b"").hexdigest()
    picks = [keys[0], keys[len(keys) // 2], keys[-1]]
    digest = hashlib.sha256()
    for key in picks:
        digest.update(key.encode())
        digest.update(state[key].detach().contiguous().view(-1).numpy().tobytes())
    return digest.hexdigest()


class ScorerCacheDataset(Dataset):
    """Cached (scene, proposals, oracle labels); optionally subsampled to top+bottom."""

    def __init__(self, scene: torch.Tensor, trajectory: torch.Tensor, raw7: torch.Tensor,
                 final_valid: torch.Tensor, mode: str, top_k: int, bottom_k: int) -> None:
        self.scene = scene
        self.trajectory = trajectory
        self.raw7 = raw7
        self.final_valid = final_valid
        self.mode = mode
        self.top_k = top_k
        self.bottom_k = bottom_k

    def __len__(self) -> int:
        return int(self.scene.shape[0])

    def _keep(self, raw7: torch.Tensor) -> torch.Tensor | None:
        if self.mode != "top10bottom10":
            return None
        candidates = raw7.shape[0]
        top = min(self.top_k, candidates)
        bottom = min(self.bottom_k, candidates - top)
        order = torch.argsort(raw7[:, PDMS], descending=True, stable=True)
        if bottom <= 0:
            return order[:top]
        return torch.cat([order[:top], order[candidates - bottom:]])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        trajectory = self.trajectory[index]
        raw7 = self.raw7[index]
        final_valid = self.final_valid[index]
        keep = self._keep(raw7)
        if keep is not None:
            trajectory = trajectory.index_select(0, keep)
            raw7 = raw7.index_select(0, keep)
            final_valid = final_valid.index_select(0, keep)
        return {"scene": self.scene[index], "trajectory": trajectory,
                "raw7": raw7, "final_valid": final_valid}


def load_rank_rows(args, dctx, train_ids: set[str] | None, heldout_ids: set[str]) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Concatenate this rank's cache shards, split into train / held-out rows."""
    shard_files = sorted(Path(args.cache_dir).glob("shard_*.pt"))
    if not shard_files:
        raise RuntimeError(f"no shard_*.pt under {args.cache_dir}; run build_scorer_cache.py first")
    mine = shard_files[dctx.rank::dctx.world]
    if not mine:
        raise RuntimeError(f"rank {dctx.rank} got no shard files ({len(shard_files)} files, world {dctx.world})")
    keys = ("scene", "trajectory", "raw7", "final_valid")
    train_rows: dict[str, list[torch.Tensor]] = {k: [] for k in keys}
    held_rows: dict[str, list[torch.Tensor]] = {k: [] for k in keys}
    seen: set[str] = set()
    for path in mine:
        try:
            shard = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        except Exception:
            shard = torch.load(path, map_location="cpu", weights_only=False)
        scene_ids = [str(s) for s in shard["scene_ids"]]
        for row, scene_id in enumerate(scene_ids):
            if scene_id in seen:
                continue
            in_train = (train_ids is None) or (scene_id in train_ids)
            in_held = scene_id in heldout_ids
            if not in_train and not in_held:
                continue
            seen.add(scene_id)
            target = held_rows if in_held else train_rows
            target["scene"].append(shard["scene"][row])
            target["trajectory"].append(shard["trajectory"][row])
            target["raw7"].append(shard["raw7"][row])
            target["final_valid"].append(shard["final_valid"][row])
    empty = {k: torch.empty(0) for k in keys}
    train = {k: (torch.stack(v) if v else empty[k]) for k, v in train_rows.items()}
    held = {k: (torch.stack(v) if v else empty[k]) for k, v in held_rows.items()}
    print(json.dumps({"event": "rank_rows", "rank": dctx.rank, "shard_files": len(mine),
                      "train": int(train["scene"].shape[0]), "heldout": int(held["scene"].shape[0])}), flush=True)
    return train, held


@torch.no_grad()
def evaluate(scorer: ScratchDrivoRScorer, rows: dict[str, torch.Tensor], batch_size: int,
             device: torch.device, dctx) -> dict[str, float]:
    """Held-out metrics on ALL 64 candidates; one all_reduce for the whole pass."""
    sums = torch.zeros(7, dtype=torch.float64)
    count = int(rows["scene"].shape[0])
    if count:
        scorer.eval()
        for start in range(0, count, batch_size):
            stop = min(start + batch_size, count)
            scene = rows["scene"][start:stop].to(device)
            trajectory = rows["trajectory"][start:stop].to(device)
            raw7 = rows["raw7"][start:stop].to(device)
            logits = scorer(scene, trajectory, None)
            selected, _utility = scorer.select(logits)
            final_score = raw7[..., PDMS]
            best = final_score.max(dim=1).values
            chosen = final_score.gather(1, selected.unsqueeze(1)).squeeze(1)
            order = torch.argsort(final_score, descending=True, stable=True)
            ranks = torch.empty_like(order)
            arange = torch.arange(final_score.shape[1], device=device).expand_as(order)
            ranks.scatter_(1, order, arange)
            rank_of_selected = ranks.gather(1, selected.unsqueeze(1)).squeeze(1).to(torch.float64)
            sums[0] += chosen.double().sum().cpu()
            sums[1] += best.double().sum().cpu()
            sums[2] += (selected == final_score.argmax(dim=1)).double().sum().cpu()
            sums[3] += rank_of_selected.sum().cpu()
            sums[4] += float(stop - start)
            sums[5] += chosen.double().sum().cpu() - final_score.mean(dim=1).double().sum().cpu()
            sums[6] += (chosen >= 0.8).double().sum().cpu()
        scorer.train()
    sums = sums.to(device)
    sums = rank.all_reduce_sum(sums, dctx).double().cpu()
    denom = max(float(sums[4]), 1.0)
    return {
        "heldout_scenes": int(sums[4].item()),
        "heldout_selected_pdms": round(float(sums[0]) / denom, 4),
        "heldout_oracle_best_pdms": round(float(sums[1]) / denom, 4),
        "heldout_oracle_match_rate": round(float(sums[2]) / denom, 4),
        "heldout_mean_rank_of_selected": round(float(sums[3]) / denom, 3),
        "heldout_gain_over_mean64": round(float(sums[5]) / denom, 4),
        "heldout_frac_selected_ge_0.80": round(float(sums[6]) / denom, 4),
    }


def provenance(args, base: dict[str, Any], base_model_fingerprint: str, kind: str,
               opt_step: int, best_metric: float | None) -> dict[str, Any]:
    return {
        "kind": kind,
        "tag": args.tag,
        "base_checkpoint": str(Path(args.base_checkpoint).resolve()),
        "base_opt_step": int(base.get("opt_step", -1)),
        "representation": "frozen: DirectedPairDiT never instantiated; base model tensors reused verbatim",
        "base_model_fingerprint": base_model_fingerprint,
        "scene_list": args.scene_list,
        "heldout_list": args.heldout_list,
        "candidate_select": args.candidate_select,
        "top_k": args.top_k,
        "bottom_k": args.bottom_k,
        "scorer_lr": args.scorer_lr,
        "weight_decay": args.weight_decay,
        "warmup": args.warmup,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "loss": "official DrivoRLoss.score_loss via ScratchDrivoRScorer.source_loss (unchanged)",
        "lambda_rank": 0.0,
        "heldout_selected_pdms": best_metric,
        "finetune_opt_step": opt_step,
    }


def _atomic_save(payload: dict[str, Any], path: Path, dctx) -> Path:
    tmp = path.with_suffix(f".tmp{dctx.rank}.pt")
    torch.save(payload, tmp)
    tmp.replace(path)
    return path


def save_scorer_state(output_dir: Path, scorer_state: dict[str, torch.Tensor], opt_state: dict[str, Any],
                      opt_step: int, kind: str, prov: dict[str, Any], dctx) -> Path:
    """Small (~7 MB) psi-only snapshot; written often so a killed arm is recoverable."""
    return _atomic_save({"scorer": scorer_state, "opt_scorer": opt_state,
                         "finetune_opt_step": opt_step, "provenance": prov},
                        output_dir / f"scorer_{kind}.pt", dctx)


def save_full_checkpoint(output_dir: Path, base: dict[str, Any], scorer_state: dict[str, torch.Tensor],
                         opt_state: dict[str, Any], opt_step: int, kind: str,
                         prov: dict[str, Any], dctx) -> Path:
    """Merged checkpoint loadable by eval_navtest_official_pdms.py with strict=True.

    ``base['model']`` is written verbatim (it was never touched, never even
    instantiated), so the representation in the evaluated model is provably the
    frozen ``joint@17500`` one.  ``opt_generator`` is deliberately dropped: the
    merged file is only ever read by the evaluator, and keeping it would add
    2.4 GB per arm.  Only written at the end of an arm.
    """
    config = dict(base.get("config", {}) or {})
    config["provenance"] = prov
    payload = {"model": base["model"], "scorer": scorer_state, "opt_scorer": opt_state,
               "epoch": base.get("epoch", -1), "opt_step": base.get("opt_step", -1),
               "finetune_opt_step": opt_step, "config": config}
    return _atomic_save(payload, output_dir / f"checkpoint_{kind}.pt", dctx)


def main() -> int:
    args = parse_args()
    dctx = trainer.setup_distributed()
    device = dctx.device
    trainer.seed_everything(args.seed)

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.mined_dir) / "arms" / args.tag
    if dctx.main:
        output_dir.mkdir(parents=True, exist_ok=True)
    rank.barrier(dctx)

    train_ids = read_scene_list(Path(args.mined_dir), args.scene_list)
    heldout_ids = set(read_scene_list(Path(args.mined_dir), args.heldout_list) or [])
    train_set = set(train_ids) if train_ids is not None else None
    if train_set is not None and (train_set & heldout_ids):
        raise RuntimeError("training list overlaps the held-out list; model selection would leak")

    rows, held_rows = load_rank_rows(args, dctx, train_set, heldout_ids)
    n_train = int(rows["scene"].shape[0])
    if n_train < args.batch_size:
        raise RuntimeError(f"rank {dctx.rank} has only {n_train} training rows; "
                           f"need >= batch_size ({args.batch_size}). Rebuild the cache with more shards.")
    steps_local = torch.tensor([n_train // args.batch_size], dtype=torch.long, device=device)
    common_steps = int(rank.all_reduce_min(steps_local, dctx).item())
    if common_steps < 1:
        raise RuntimeError(f"some rank has fewer than {args.batch_size} training rows (min steps {common_steps})")

    if args.max_steps > 0:
        total_steps = args.max_steps
        epochs_needed = max(1, math.ceil(args.max_steps / common_steps))
    elif args.epochs > 0:
        epochs_needed = args.epochs
        total_steps = common_steps * epochs_needed
    else:
        epochs_needed = 0
        total_steps = 0

    base = load_base_checkpoint(Path(args.base_checkpoint))
    scorer = ScratchDrivoRScorer(d_model=256, heads=1, ffn_dim=1024, decoder_layers=4).to(device)
    scorer.load_state_dict(base["scorer"], strict=True)
    if scorer.initialization != "scratch_random" or scorer.checkpoint_loaded:
        raise AssertionError("scorer must remain the scratch-random DrivoR scorer (no DrivoR pretrained weights)")
    base_model_fingerprint = fingerprint(base["model"]) if dctx.main else ""
    for parameter in scorer.parameters():
        parameter.requires_grad_(True)
    scorer_parameters = list(scorer.parameters())
    opt = torch.optim.AdamW(scorer_parameters, lr=args.scorer_lr, weight_decay=args.weight_decay)

    net: nn.Module = rank.ScorerModule(scorer)
    if dctx.enabled:
        net = DDP(net, device_ids=[dctx.local_rank], output_device=dctx.local_rank,
                  find_unused_parameters=args.find_unused, broadcast_buffers=False)

    dataset = ScorerCacheDataset(rows["scene"], rows["trajectory"], rows["raw7"], rows["final_valid"],
                                 args.candidate_select, args.top_k, args.bottom_k)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
                        num_workers=0, collate_fn=rank.cache_collate)
    trainable = sum(p.numel() for p in scorer_parameters)
    trainer.log0(dctx, {"event": "setup", "tag": args.tag, "world": dctx.world, "rank_train_rows": n_train,
                        "common_steps_per_epoch": common_steps, "epochs_needed": epochs_needed,
                        "total_opt_steps": total_steps, "trainable_scorer": trainable,
                        "candidate_select": args.candidate_select,
                        "train_m": int(args.top_k + args.bottom_k) if args.candidate_select == "top10bottom10" else 64,
                        "scorer_contract": scorer.feature_audit, "lambda_rank": 0.0})

    history: list[dict[str, Any]] = []
    best_metric = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    best_opt_state: dict[str, Any] | None = None
    best_step = 0
    opt_step = 0
    start = time.time()

    def snapshot() -> dict[str, torch.Tensor]:
        return {key: value.detach().cpu().clone() for key, value in scorer.state_dict().items()}

    def opt_snapshot() -> dict[str, Any]:
        # deepcopy: AdamW mutates exp_avg/exp_avg_sq in place, so a plain
        # state_dict() reference would silently drift away from the "best" step.
        return copy.deepcopy(opt.state_dict())

    def prov(kind: str, step: int, metric: float | None) -> dict[str, Any]:
        return provenance(args, base, base_model_fingerprint, kind, step, metric)

    def run_eval(kind: str) -> dict[str, float]:
        metrics = evaluate(scorer, held_rows, args.eval_batch_size, device, dctx)
        record = {"event": "eval", "kind": kind, "opt_step": opt_step, **metrics}
        history.append(record)
        trainer.log0(dctx, record)
        return metrics

    net.train()
    if held_rows["scene"].shape[0] > 0 or dctx.enabled:
        initial = run_eval("initial")
        if dctx.main:
            best_metric = initial["heldout_selected_pdms"]
            best_state, best_opt_state = snapshot(), opt_snapshot()
            save_scorer_state(output_dir, best_state, best_opt_state, opt_step, "initial",
                              prov("initial", opt_step, best_metric), dctx)

    stop = False
    for epoch in range(epochs_needed):
        for batch in loader:
            scene = batch["scene"].to(device, non_blocking=True)
            trajectory = batch["trajectory"].to(device, non_blocking=True)
            raw7 = batch["raw7"].to(device, non_blocking=True)
            final_valid = batch["final_valid"].to(device, non_blocking=True)
            factor = rank.schedule_factor(opt_step + 1, max(total_steps, 1), args.warmup)
            for group in opt.param_groups:
                group["lr"] = args.scorer_lr * factor
            logits = net(scene, trajectory)
            labels = rank.make_labels(raw7, final_valid)
            loss = scorer.source_loss(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(scorer_parameters, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            opt_step += 1

            if args.log_every > 0 and (opt_step % args.log_every == 0 or opt_step == total_steps):
                record = {"event": "step", "tag": args.tag, "epoch": epoch, "opt_step": opt_step,
                          "total_opt_steps": total_steps, "lr": round(args.scorer_lr * factor, 8),
                          "loss_score": round(float(loss.detach()), 6),
                          "train_m": int(trajectory.shape[1]),
                          "sec_per_step": round((time.time() - start) / opt_step, 4)}
                history.append(record)
                trainer.log0(dctx, record)
            if args.eval_every > 0 and opt_step % args.eval_every == 0:
                metrics = run_eval("periodic")
                if dctx.main and metrics["heldout_selected_pdms"] > best_metric:
                    best_metric = metrics["heldout_selected_pdms"]
                    best_state, best_opt_state, best_step = snapshot(), opt_snapshot(), opt_step
                    path = save_scorer_state(output_dir, best_state, best_opt_state, opt_step, "best",
                                             prov("best", opt_step, best_metric), dctx)
                    trainer.log0(dctx, {"event": "best_saved", "opt_step": opt_step,
                                        "heldout_selected_pdms": best_metric, "path": str(path)})
            if args.save_every > 0 and opt_step % args.save_every == 0 and dctx.main:
                save_scorer_state(output_dir, snapshot(), opt.state_dict(), opt_step, "last",
                                  prov("last", opt_step, best_metric), dctx)
            if opt_step >= total_steps:
                stop = True
                break
        if stop:
            break

    final = run_eval("final")
    if dctx.main:
        if best_state is None or final["heldout_selected_pdms"] > best_metric:
            best_metric = final["heldout_selected_pdms"]
            best_state, best_opt_state, best_step = snapshot(), opt_snapshot(), opt_step
        if best_step == 0 and total_steps > 0:
            # The heldout-best snapshot is the untouched initial psi: this arm
            # trained but never beat its own starting point, so checkpoint_best.pt
            # will evaluate as the baseline.  Say so loudly instead of shipping a
            # silent no-op.
            trainer.log0(dctx, {"event": "warning_best_is_initial", "tag": args.tag,
                                "total_opt_steps": total_steps,
                                "initial_heldout_selected_pdms": history[0]["heldout_selected_pdms"] if history else None,
                                "final_heldout_selected_pdms": final["heldout_selected_pdms"],
                                "note": "no eval beat the initial psi; checkpoint_best.pt == base psi"})
        best_path = save_full_checkpoint(output_dir, base, best_state, best_opt_state, best_step, "best",
                                         prov("best", best_step, best_metric), dctx)
        last_path = save_full_checkpoint(output_dir, base, snapshot(), opt.state_dict(), opt_step, "last",
                                         prov("last", opt_step, best_metric), dctx)
        summary = {
            "tag": args.tag, "opt_step": opt_step, "best_step": best_step, "total_opt_steps": total_steps,
            "base_checkpoint": str(Path(args.base_checkpoint).resolve()),
            "base_opt_step": int(base.get("opt_step", -1)),
            "base_model_fingerprint": base_model_fingerprint,
            "scene_list": args.scene_list, "train_rows_rank0": n_train,
            "candidate_select": args.candidate_select,
            "best_heldout_selected_pdms": best_metric,
            "final": final, "initial": history[0] if history else None,
            "checkpoint_best": str(best_path), "checkpoint_last": str(last_path),
            "elapsed_s": round(time.time() - start, 1),
        }
        (output_dir / "history.json").write_text(json.dumps({"summary": summary, "history": history}, indent=2) + "\n")
        trainer.log0(dctx, {"event": "arm_done", **{k: v for k, v in summary.items() if k != "final"}})
    rank.finalize(dctx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
