"""Frozen-generator cache builder for the scorer-only fine-tune experiments.

One pass over navtrain with ``joint@17500`` fully frozen (``eval()``, no grad) and
persists, per scene, everything a scorer-only run needs:

    scene [66,256]        NVS DINOv3 Scene K/V context (detached by contract)
    trajectory [64,8,3]   the 64 proposals the scorer will rank
    raw7 [64,7]           DrivoR oracle labels (official ``get_sub_score``)
    final_valid [64]      isfinite(PDMS)
    selected_index        argmax of the DrivoR utility under the CURRENT psi

Because the generator runs in ``eval()`` mode this is the deterministic inference
trajectory pool -- identical in kind to what ``eval_navtest_official_pdms.py``
produces -- so mined "hard" scenes are hard under the real deployment condition.

Strictly additive: imports ``run_navtrain_wan_joint_ddp`` (index / dataset /
collate / model) and ``run_navtrain_wan_scorer_rank_ddp`` (dist helpers) and
never edits them.  Launch with
``python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=16``.

Shards are named ``shard_{rank:02d}_{chunk:04d}.pt`` where a chunk is a fixed
slice of the rank's ``DistributedSampler(shuffle=False)`` index list, so
``--skip-existing`` resumes exactly where a killed run stopped.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, DistributedSampler, Subset

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_navtrain_wan_joint_ddp as trainer  # noqa: E402
import run_navtrain_wan_scorer_rank_ddp as rank  # noqa: E402
from nvs_v6_wan.modes import ImmutableModeBuffer, StructuredSourceInitializer  # noqa: E402
from nvs_v6_wan.oracle import DrivoRBatchMetricCacheOracle  # noqa: E402
from nvs_v6_wan.scorer import ScratchDrivoRScorer  # noqa: E402

DEFAULT_CHECKPOINT = "artifacts/navtrain_wan_joint_ddp/checkpoint.pt"
DEFAULT_INDEX_DIR = "artifacts/navtrain_wan_joint_ddp"
DEFAULT_OUTPUT = "artifacts/scorer_hard_mining/cache"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="frozen joint generator+scorer checkpoint")
    parser.add_argument("--expect-opt-step", type=int, default=0, help="abort unless the checkpoint opt_step equals this (0 = no check)")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--index-dir", default=DEFAULT_INDEX_DIR,
                        help="sidecar scene-index dir; defaults to the joint run so its indexes are reused verbatim")
    parser.add_argument("--batch-size", type=int, default=8, help="per-GPU inference batch")
    parser.add_argument("--workers", type=int, default=6, help="DataLoader workers per rank")
    parser.add_argument("--shard-scenes", type=int, default=2048, help="scenes per shard file (rounded down to a batch multiple)")
    parser.add_argument("--limit-scenes", type=int, default=0, help="cap the scene index (0 = full navtrain)")
    parser.add_argument("--skip-existing", action="store_true", help="do not recompute shards already on disk")
    parser.add_argument("--depth", type=int, default=0, help="0 = read from the checkpoint config")
    parser.add_argument("--dino-checkpoint", default="", help="empty = read from the checkpoint config")
    parser.add_argument("--candidates", type=int, default=64)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--log-every", type=int, default=20, help="print progress this many batches")
    parser.add_argument("--multigt-root", default=trainer.DEFAULT_MULTIGT)
    parser.add_argument("--sensor-root", default=trainer.DEFAULT_SENSOR)
    parser.add_argument("--log-root", default=trainer.DEFAULT_LOG_ROOT)
    parser.add_argument("--cache-root", default=trainer.DEFAULT_CACHE)
    return parser.parse_args()


def load_frozen(args, dctx) -> tuple[Any, ScratchDrivoRScorer, int, int]:
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "model" not in ckpt or "scorer" not in ckpt:
        raise RuntimeError(f"{args.checkpoint} is not a joint stage-2 checkpoint")
    config = ckpt.get("config", {}) or {}
    opt_step = int(ckpt.get("opt_step", -1))
    epoch = int(ckpt.get("epoch", -1))
    if args.expect_opt_step and opt_step != args.expect_opt_step:
        raise RuntimeError(f"checkpoint opt_step={opt_step}, expected {args.expect_opt_step}")
    depth = args.depth or int(config.get("depth", 8))
    dino = args.dino_checkpoint or config.get("dino_checkpoint", trainer.DEFAULT_DINO)
    model = trainer.DirectedPairDiT(
        d_model=256, heads=8, ffn_dim=1024, depth=depth,
        dino_checkpoint=dino, dino_model_name="vit_large_patch16_dinov3.lvd1689m",
        dino_lora_rank=16, dino_lora_alpha=16.0,
    ).to(dctx.device)
    model.load_state_dict(ckpt["model"], strict=True)
    scorer = ScratchDrivoRScorer(d_model=256, heads=1, ffn_dim=1024, decoder_layers=4).to(dctx.device)
    scorer.load_state_dict(ckpt["scorer"], strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in scorer.parameters():
        parameter.requires_grad_(False)
    model.eval()
    scorer.eval()
    if scorer.initialization != "scratch_random" or scorer.checkpoint_loaded:
        raise AssertionError("scorer must remain the scratch-random DrivoR scorer")
    return model, scorer, opt_step, epoch


@torch.no_grad()
def build_chunk(args, dctx, model, scorer, oracle, modes, initializer,
                dataset, chunk_indices: list[int]) -> dict[str, Any]:
    """Forward + oracle over one chunk; return the shard payload (rows already filtered)."""
    loader = DataLoader(
        Subset(dataset, chunk_indices),
        batch_size=args.batch_size, shuffle=False, drop_last=False,
        num_workers=args.workers, collate_fn=trainer.collate,
        pin_memory=False, persistent_workers=False,
    )
    scenes: list[torch.Tensor] = []
    trajectories: list[torch.Tensor] = []
    raw7s: list[torch.Tensor] = []
    valids: list[torch.Tensor] = []
    selecteds: list[int] = []
    ids: list[str] = []
    skipped = 0
    seen: set[str] = set()
    for step, batch in enumerate(loader):
        observation, _targets, scene_ids = trainer.batch_to_device(batch, dctx.device)
        size = observation.rgb_history.shape[0]
        mode_codes = modes.expand_for_batch(size, dctx.device)
        source = initializer.clean(observation.ego_history, mode_codes)
        outputs = model.forward_pairs(observation, source, mode_codes)
        trajectory = outputs.trajectory.detach()
        logits = scorer(outputs.scene_context, trajectory, None)
        selected, _utility = scorer.select(logits)
        try:
            labels = oracle.evaluate(scene_ids, trajectory)
        except Exception:
            skipped += size
            continue
        scene_cpu = outputs.scene_context.detach().to("cpu", torch.float32)
        traj_cpu = trajectory.to("cpu", torch.float32)
        raw7_cpu = labels.raw7.to("cpu", torch.float32)
        valid_cpu = labels.final_valid.to("cpu")
        sel_cpu = selected.detach().to("cpu", torch.long)
        failed = {f.get("scene_id") for f in labels.failure_diagnostics}
        for row, scene_id in enumerate(scene_ids):
            row_ok = bool(valid_cpu[row].all()) and scene_id not in failed and scene_id not in seen
            if not row_ok:
                skipped += 1
                continue
            seen.add(scene_id)
            scenes.append(scene_cpu[row])
            trajectories.append(traj_cpu[row])
            raw7s.append(raw7_cpu[row])
            valids.append(valid_cpu[row])
            selecteds.append(int(sel_cpu[row]))
            ids.append(scene_id)
        if args.log_every > 0 and (step + 1) % args.log_every == 0:
            trainer.log0(dctx, {"event": "chunk_progress", "rank": dctx.rank, "batch": step + 1,
                                "cached": len(ids), "skipped": skipped})
    if not ids:
        return {}
    return {
        "scene": torch.stack(scenes),
        "trajectory": torch.stack(trajectories),
        "raw7": torch.stack(raw7s),
        "final_valid": torch.stack(valids),
        "selected_index": torch.tensor(selecteds, dtype=torch.int16),
        "scene_ids": ids,
        "skipped": skipped,
        "base_opt_step": -1,
    }


def main() -> int:
    args = parse_args()
    if args.candidates != 64:
        raise ValueError("the DrivoR online oracle requires exactly 64 candidates")

    dctx = trainer.setup_distributed()
    trainer.seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    index_dir = Path(args.index_dir)
    if dctx.main:
        output_dir.mkdir(parents=True, exist_ok=True)
        index_dir.mkdir(parents=True, exist_ok=True)
    rank.barrier(dctx)

    index_start = time.time()
    limit_scenes = args.limit_scenes if args.limit_scenes > 0 else None
    if dctx.main:
        candidates = trainer.build_scene_index(args.multigt_root, args.cache_root, args.log_root, index_dir, limit_scenes)
    rank.barrier(dctx)
    if not dctx.main:
        candidates = trainer.build_scene_index(args.multigt_root, args.cache_root, args.log_root, index_dir, limit_scenes)
    if len(candidates) < args.batch_size * dctx.world:
        raise RuntimeError(f"only {len(candidates)} scenes resolved; need >= batch_size*world ({args.batch_size * dctx.world})")
    trainer.log0(dctx, {"event": "index_built", "scenes": len(candidates),
                        "index_dir": str(index_dir), "elapsed_s": round(time.time() - index_start, 2)})

    dataset = trainer.NavtrainSceneDataset(candidates, args.sensor_root, args.multigt_root, args.cache_root, args.log_root)
    if dctx.enabled:
        sampler = DistributedSampler(dataset, num_replicas=dctx.world, rank=dctx.rank, shuffle=False, drop_last=True)
        rank_indices = list(iter(sampler))
    else:
        rank_indices = list(range(len(dataset)))
    if not rank_indices:
        raise RuntimeError(f"rank {dctx.rank} received zero scenes; reduce --nproc_per_node or raise --limit-scenes")

    chunk_len = max(args.batch_size, (args.shard_scenes // args.batch_size) * args.batch_size)
    chunks = [rank_indices[start:start + chunk_len] for start in range(0, len(rank_indices), chunk_len)]

    model, scorer, base_opt_step, base_epoch = load_frozen(args, dctx)
    oracle = DrivoRBatchMetricCacheOracle(
        cache_paths={c.token: c.cache_path for c in candidates},
        evaluator_version=trainer.ORACLE_VERSION,
    )
    modes = ImmutableModeBuffer().subset(args.candidates).to(dctx.device)
    initializer = StructuredSourceInitializer(jitter_std=0.0)
    trainer.log0(dctx, {"event": "setup", "world": dctx.world, "rank_indices": len(rank_indices),
                        "chunks": len(chunks), "chunk_len": chunk_len, "base_opt_step": base_opt_step,
                        "base_epoch": base_epoch, "base_checkpoint": str(args.checkpoint)})

    start = time.time()
    written: list[str] = []
    reused = 0
    cached = 0
    skipped = 0
    for chunk_id, chunk_indices in enumerate(chunks):
        shard_path = output_dir / f"shard_{dctx.rank:02d}_{chunk_id:04d}.pt"
        if shard_path.exists() and args.skip_existing:
            reused += 1
            try:
                cached += int(torch.load(shard_path, map_location="cpu", weights_only=True)["scene"].shape[0])
            except Exception:
                pass
            continue
        payload = build_chunk(args, dctx, model, scorer, oracle, modes, initializer, dataset, chunk_indices)
        if not payload:
            skipped += len(chunk_indices)
            trainer.log0(dctx, {"event": "chunk_empty", "rank": dctx.rank, "chunk": chunk_id})
            continue
        payload["base_opt_step"] = base_opt_step
        tmp = shard_path.with_suffix(f".tmp{shard_path.suffix}.{dctx.rank}")
        torch.save(payload, tmp)
        tmp.replace(shard_path)
        written.append(shard_path.name)
        cached += len(payload["scene_ids"])
        skipped += int(payload["skipped"])
        print(json.dumps({"event": "chunk_written", "rank": dctx.rank, "chunk": chunk_id,
                          "scenes": len(payload["scene_ids"]), "skipped": int(payload["skipped"]),
                          "elapsed_s": round(time.time() - start, 1)}), flush=True)

    local = {"rank": dctx.rank, "written": written, "reused": reused, "cached": cached,
             "skipped": skipped, "elapsed_s": round(time.time() - start, 1)}
    print(json.dumps({"event": "rank_done", **local}), flush=True)
    rank.barrier(dctx)
    if dctx.main:
        shards = sorted(p.name for p in output_dir.glob("shard_*.pt"))
        (output_dir / "_builder_meta.json").write_text(json.dumps({
            "base_checkpoint": str(Path(args.checkpoint).resolve()),
            "base_opt_step": base_opt_step,
            "base_epoch": base_epoch,
            "oracle_version": trainer.ORACLE_VERSION,
            "index_dir": str(Path(args.index_dir).resolve()),
            "roots": {"multigt": args.multigt_root, "sensor": args.sensor_root,
                      "log": args.log_root, "cache": args.cache_root},
            "world": dctx.world,
            "scenes_indexed": len(candidates),
            "shards": shards,
            "shard_files": len(shards),
            "args": vars(args),
        }, indent=2) + "\n")
        trainer.log0(dctx, {"event": "build_done", "shard_files": len(shards),
                            "elapsed_s": round(time.time() - start, 1)})
    rank.finalize(dctx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
