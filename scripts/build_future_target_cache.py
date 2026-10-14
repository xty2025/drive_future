"""Build the offline Future-Register target cache Stage 1 trains from.

One frozen-Teacher pass per scene, ever.  The register target cannot change
during training -- ``config_v6_wan_e2e.yaml`` puts ``teacher`` and
``target_compressor`` in ``stage1.freeze`` and ``stage2.freeze`` -- so decoding
four ``future.mp4`` files and running DINOv3 over 32 frames on every epoch is
pure repeated work.  This script does it once and writes ``[K,4,256]`` float32
plus the metric trajectories, ~66 KB per scene, ~6.8 GB for the 103075-scene
corpus.

Two ordering constraints shape the run:

* The compressor is fit **once, on rank 0, over training scenes only**, and every
  rank then loads that same ``compressor.pt``.  An SVD basis is only defined up to
  column order and sign, so 32 independently-fit compressors would emit 32
  mutually incomparable target spaces and the all-reduced Stage-1 gradient would
  average unrelated numbers.
* The train/eval split is decided **before** the fit and shared with the trainer
  through ``split_scene_tokens``, so no eval scene can leak into the basis.

Sharding is ``tokens[rank::world]``, which needs no coordination once the index
sidecar exists.  Under torchrun the rank comes from the environment; a plain
``python`` invocation uses ``--shard-index/--shard-count`` so a single GPU can
build a subset for a smoke run.

Run with the repository's standard PYTHONPATH, e.g.::

    PYTHONPATH="$PWD/src:$PWD/vendor/DrivoR:$PWD/vendor/DrivoR/nuplan-devkit" \\
      python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=16 \\
      scripts/build_future_target_cache.py --target-cache-root artifacts/future_target_cache
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist


from nvs_v6_wan_e2e.multiview_data import (
    DEFAULT_CACHE_ROOT,
    DEFAULT_LOG_ROOT,
    DEFAULT_MULTIVIEW_ROOT,
    DEFAULT_SENSOR_ROOT,
    MultiViewSample,
    iter_ready_scenes,
    load_multiview_sample,
)
from nvs_v6_wan_e2e.target_cache import (
    atomic_json_write,
    atomic_torch_save,
    build_scene_payload,
    compressor_path,
    list_cached_scenes,
    manifest_path,
    ready_scenes_path,
    save_compressor,
    scene_target_path,
    split_scene_tokens,
    load_compressor,
)
from nvs_v6_wan_e2e.teacher import (
    NATIVE_FUTURE_IMAGE_HW,
    Ordered2x2Compressor,
    TimmDinoV3Teacher,
    ordered_quadrant_vectors,
)
from nvs_v6_wan_e2e.vision import DRIVOR_OFFICIAL_IMAGE_HW

DEFAULT_DINO_CHECKPOINT = "/mnt/LLM_weight/timm/vit_large_patch16_dinov3.lvd1689m/model.safetensors"


@dataclass
class Dist:
    rank: int
    world: int
    local_rank: int
    device: torch.device
    enabled: bool

    @property
    def main(self) -> bool:
        return self.rank == 0


def setup_distributed(shard_index: int, shard_count: int) -> Dist:
    """torchrun ranks when present, otherwise the explicit ``--shard-*`` pair."""

    has_env = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", shard_index))
    world = int(os.environ.get("WORLD_SIZE", shard_count))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if not torch.cuda.is_available():
        raise RuntimeError("building Teacher targets requires the configured CUDA/PPU device")
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    enabled = False
    if has_env and world > 1:
        # Rank 0 walks all 103075 scene directories and fits the SVD while every
        # other rank waits; a short timeout would abort on that legitimate skew.
        dist.init_process_group(backend="nccl", timeout=timedelta(hours=6))
        enabled = True
        world = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
    if not 0 <= rank < world:
        raise ValueError(f"rank {rank} outside world {world}")
    return Dist(rank=rank, world=world, local_rank=local_rank, device=device, enabled=enabled)


def log(dctx: Dist, payload: dict[str, Any], all_ranks: bool = False) -> None:
    if dctx.main or all_ranks:
        print(json.dumps({"rank": dctx.rank, **payload}), flush=True)


def barrier(dctx: Dist) -> None:
    if dctx.enabled:
        dist.barrier()


def ready_scene_index(dctx: Dist, args: argparse.Namespace) -> list[str]:
    """Rank 0 walks the corpus once; the sidecar spares 31 ranks the same walk.

    ``scene_is_ready`` costs four filesystem reads per directory, so scanning
    103075 scenes is minutes of work that must not be repeated per rank.  Rank 0
    writes the sidecar *before* the caller's barrier, and every other rank reads it
    after -- reading first would race the write.
    """

    path = ready_scenes_path(args.target_cache_root)
    meta = {
        "multiview_root": str(Path(args.multiview_root)),
        "cache_root": str(Path(args.cache_root)),
        "limit": None if args.limit is None else int(args.limit),
        "skip_degenerate": bool(args.skip_degenerate),
    }
    cached = _read_index_sidecar(path, meta)
    if cached is not None:
        log(dctx, {"event": "reused_scene_index", "scenes": len(cached), "path": str(path)})
        return cached
    if not dctx.main:
        raise FileNotFoundError(
            f"{path} is missing or stale for {meta}; let rank 0 build it first "
            "(under torchrun it does, behind a barrier)"
        )
    started = time.time()
    tokens = [
        scene_dir.name
        for scene_dir in iter_ready_scenes(
            args.multiview_root,
            args.cache_root,
            limit=args.limit,
            require_distinct_labels=args.skip_degenerate,
        )
    ]
    atomic_json_write({"_meta": meta, "tokens": tokens}, path)
    log(dctx, {"event": "built_scene_index", "scenes": len(tokens), "elapsed_s": round(time.time() - started, 1), "path": str(path)})
    return tokens


def _read_index_sidecar(path: Path, meta: dict[str, Any]) -> list[str] | None:
    if not path.exists():
        return None
    try:
        blob = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    if not isinstance(blob, dict) or blob.get("_meta") != meta or not isinstance(blob.get("tokens"), list):
        return None
    return [str(token) for token in blob["tokens"]]


def read_scene_index(dctx: Dist, args: argparse.Namespace) -> list[str]:
    """Post-barrier read: the sidecar exists now, on every rank."""

    meta = {
        "multiview_root": str(Path(args.multiview_root)),
        "cache_root": str(Path(args.cache_root)),
        "limit": None if args.limit is None else int(args.limit),
        "skip_degenerate": bool(args.skip_degenerate),
    }
    tokens = _read_index_sidecar(ready_scenes_path(args.target_cache_root), meta)
    if tokens is None:
        raise FileNotFoundError(
            f"{ready_scenes_path(args.target_cache_root)} is missing or stale for {meta}; "
            "rank 0 was supposed to have written it before the barrier"
        )
    return tokens



def _ordered_block(sample: MultiViewSample, teacher: TimmDinoV3Teacher, device: torch.device) -> torch.Tensor:
    """``[K,4,8192]`` quadrant-pooled Teacher features for one scene.

    Each GT's eight-frame grid is reduced as it is produced and never retained: a
    grid is 50 MB float32 at the future videos' native 29x52 while its ordered
    vectors are 131 KB, which is what lets the fit set stay in RAM.
    """

    if sample.future_rgb is None:
        raise ValueError(f"{sample.scene_id} was loaded without future RGB")
    blocks = []
    for gt_index in range(sample.future_rgb.shape[1]):
        grid = teacher(sample.future_rgb[:, gt_index].to(device)).float().cpu()
        blocks.append(ordered_quadrant_vectors(grid))
        del grid
    return torch.cat(blocks, dim=0)


def _load_sample(args: argparse.Namespace, scene_id: str, log_cache: dict[str, list[dict]]) -> MultiViewSample:
    scene_dir = Path(args.multiview_root) / scene_id
    metadata = json.loads((scene_dir / "scene_metadata.json").read_text())
    log_name = str(metadata["log_name"])
    if log_name not in log_cache:
        with (Path(args.log_root) / f"{log_name}.pkl").open("rb") as handle:
            log_cache[log_name] = pickle.load(handle)
        # Consecutive tokens usually share a log, but an unbounded cache would grow
        # with the corpus; two entries covers the common run without leaking.
        while len(log_cache) > 2:
            log_cache.pop(next(iter(log_cache)))
    return load_multiview_sample(
        scene_dir,
        log_root=args.log_root,
        sensor_root=args.sensor_root,
        cache_root=args.cache_root,
        image_hw=tuple(args.image_hw),
        future_image_hw=tuple(args.teacher_image_hw),
        log_frames=log_cache[log_name],
    )


def fit_compressor(dctx: Dist, args: argparse.Namespace, train_tokens: list[str], teacher: TimmDinoV3Teacher) -> dict[str, Any]:
    """Rank 0 only: SVD over the ordered vectors of ``--fit-scenes`` train scenes.

    ``Ordered2x2Compressor.fit`` treats every ordered chunk as its own row, so the
    rank is ``min(target_dim, 16 * scenes)``: 16 scenes already saturate a 256-d
    target.  ``--fit-scenes`` defaults well past that because a basis estimated
    from 256 rows is noisy even when it is full rank.
    """

    fit_count = min(int(args.fit_scenes), len(train_tokens))
    if fit_count < 1:
        raise ValueError("the compressor fit needs at least one training scene")
    log_cache: dict[str, list[dict]] = {}
    blocks: list[torch.Tensor] = []
    fitted: list[str] = []
    started = time.time()
    for scene_id in train_tokens[:fit_count]:
        try:
            sample = _load_sample(args, scene_id, log_cache)
            blocks.append(_ordered_block(sample, teacher, dctx.device))
        except (FileNotFoundError, ValueError, RuntimeError) as error:
            log(dctx, {"event": "fit_scene_skipped", "scene": scene_id, "reason": str(error)})
            continue
        fitted.append(scene_id)
        del sample
    if not blocks:
        raise RuntimeError("no training scene could be loaded for the compressor fit")
    fit_vectors = torch.cat(blocks, dim=0)
    compressor = Ordered2x2Compressor(output_dim=int(args.target_dim))
    compressor.fit(
        fit_vectors,
        source_hash="+".join(fitted),
        split="train",
    )
    fit_rows = int(fit_vectors.shape[0] * fit_vectors.shape[1])
    audit = {
        "teacher_checkpoint": str(args.dino_checkpoint),
        "teacher_image_hw": [int(args.teacher_image_hw[0]), int(args.teacher_image_hw[1])],
        "teacher_grid_hw": [int(value) for value in teacher.grid_hw],
        "compressor_input_dim": int(compressor.state.input_dim),
        "compressor_output_dim": int(compressor.state.output_dim),
        "compressor_rank": int(min(args.target_dim, fit_rows)),
        "compressor_fit_rows": fit_rows,
        "compressor_fit_scenes": len(fitted),
        "compressor_full_rank": bool(min(args.target_dim, fit_rows) >= args.target_dim),
        "compressor_fingerprint": compressor.fingerprint(),
        "fit_scene_ids": fitted,
        "fit_elapsed_s": round(time.time() - started, 1),
    }
    save_compressor(args.target_cache_root, compressor, audit)
    del blocks, fit_vectors
    gc.collect()
    log(dctx, {"event": "compressor_fit", **{key: value for key, value in audit.items() if key != "fit_scene_ids"}})
    return audit


def build_shard(
    dctx: Dist,
    args: argparse.Namespace,
    tokens: list[str],
    teacher: TimmDinoV3Teacher,
    compressor: Ordered2x2Compressor,
) -> dict[str, Any]:
    shard = tokens[dctx.rank :: dctx.world]
    fingerprint = compressor.fingerprint()
    log_cache: dict[str, list[dict]] = {}
    built = 0
    reused = 0
    failed: list[dict[str, str]] = []
    started = time.time()
    for position, scene_id in enumerate(shard):
        target = scene_target_path(args.target_cache_root, scene_id)
        if target.exists() and not args.overwrite:
            reused += 1
            continue
        try:
            sample = _load_sample(args, scene_id, log_cache)
            block = _ordered_block(sample, teacher, dctx.device)
            register = compressor.transform_ordered(block)
            payload = build_scene_payload(
                future_register=register,
                future_chunk_valid=sample.targets.future_chunk_valid[0],
                future_quality=sample.targets.future_quality[0],
                gt_trajectory=sample.targets.gt_trajectory[0],
                gt_valid=sample.targets.gt_valid[0],
                observation=sample.observation,
                pair_ids=list(sample.targets.pair_ids[0]),
                metadata=sample.metadata,
                compressor_fingerprint=fingerprint,
            )
            atomic_torch_save(payload, target)
            built += 1
            del block, register, sample
        except Exception as error:  # a bad scene must not abort the whole shard
            failed.append({"scene": scene_id, "reason": f"{type(error).__name__}: {error}"})
            continue
        if (position + 1) % int(args.log_every) == 0:
            rate = (position + 1) / max(1e-9, time.time() - started)
            log(
                dctx,
                {
                    "event": "shard_progress",
                    "position": position + 1,
                    "shard_scenes": len(shard),
                    "built": built,
                    "reused": reused,
                    "failed": len(failed),
                    "scenes_per_s": round(rate, 3),
                    "eta_s": round((len(shard) - position - 1) / max(1e-9, rate), 1),
                },
                all_ranks=True,
            )
    return {
        "shard_scenes": len(shard),
        "built": built,
        "reused": reused,
        "failed": failed[:20],
        "failed_count": len(failed),
        "elapsed_s": round(time.time() - started, 1),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target-cache-root", required=True, help="destination for compressor.pt, targets/ and _manifest.json")
    parser.add_argument("--multiview-root", default=DEFAULT_MULTIVIEW_ROOT)
    parser.add_argument("--log-root", default=DEFAULT_LOG_ROOT)
    parser.add_argument("--sensor-root", default=DEFAULT_SENSOR_ROOT)
    parser.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--dino-checkpoint", default=DEFAULT_DINO_CHECKPOINT)
    parser.add_argument("--image-hw", type=int, nargs=2, default=list(DRIVOR_OFFICIAL_IMAGE_HW), help="history size recorded in each payload; the trainer must ask for the same one")
    parser.add_argument("--teacher-image-hw", type=int, nargs=2, default=list(NATIVE_FUTURE_IMAGE_HW), help="must equal the future videos' native size; the Teacher refuses to interpolate")
    parser.add_argument("--target-dim", type=int, default=256)
    parser.add_argument("--fit-scenes", type=int, default=512, help="train scenes the SVD basis is estimated from; rank saturates at 16")
    parser.add_argument("--eval-scenes", type=int, default=256, help="held out of training and of the compressor fit")
    parser.add_argument("--limit", type=int, default=None, help="cap the scene index; useful for a first subset build")
    parser.add_argument("--skip-degenerate", action="store_true", help="drop scenes whose Multi-GT trajectory labels collapse")
    parser.add_argument("--overwrite", action="store_true", help="rebuild scenes that already have a cached target")
    parser.add_argument("--rebuild-index", action="store_true", help="ignore any _ready_scenes.json sidecar and rescan the corpus")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--shard-index", type=int, default=0, help="only used when torchrun did not set RANK")
    parser.add_argument("--shard-count", type=int, default=1, help="only used when torchrun did not set WORLD_SIZE")
    args = parser.parse_args()
    args.target_cache_root = str(Path(args.target_cache_root))
    Path(args.target_cache_root).mkdir(parents=True, exist_ok=True)
    if args.rebuild_index and ready_scenes_path(args.target_cache_root).exists():
        ready_scenes_path(args.target_cache_root).unlink()
    return args


def main() -> int:
    args = parse_args()
    dctx = setup_distributed(args.shard_index, args.shard_count)
    torch.manual_seed(args.seed)
    log(dctx, {"event": "start", "world": dctx.world, "device": str(dctx.device), "target_cache_root": args.target_cache_root})

    if dctx.main:
        ready_scene_index(dctx, args)
    barrier(dctx)
    tokens = read_scene_index(dctx, args)
    if not tokens:
        raise FileNotFoundError(f"no ready Multi-GT scene under {args.multiview_root}")
    train_tokens, eval_tokens = split_scene_tokens(tokens, args.eval_scenes, args.seed)
    log(dctx, {"event": "split", "scenes": len(tokens), "train": len(train_tokens), "eval": len(eval_tokens)})

    teacher = TimmDinoV3Teacher(checkpoint=args.dino_checkpoint, image_hw=tuple(args.teacher_image_hw)).to(dctx.device)
    if not compressor_path(args.target_cache_root).exists() or args.overwrite:
        # Only rank 0 fits, and only it may: the basis has to be one basis.
        if dctx.main:
            fit_compressor(dctx, args, train_tokens, teacher)
        barrier(dctx)
    compressor, fit_audit = load_compressor(args.target_cache_root)
    log(
        dctx,
        {
            "event": "compressor_loaded",
            "fingerprint": compressor.fingerprint(),
            "rank": fit_audit.get("compressor_rank"),
            "fit_rows": fit_audit.get("compressor_fit_rows"),
            "full_rank": fit_audit.get("compressor_full_rank"),
        },
    )

    result = build_shard(dctx, args, tokens, teacher, compressor)
    barrier(dctx)
    if dctx.main:
        cached = list_cached_scenes(args.target_cache_root)
        atomic_json_write(
            {
                "format": 1,
                "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "world_size": dctx.world,
                "multiview_root": str(Path(args.multiview_root)),
                "sensor_root": str(Path(args.sensor_root)),
                "log_root": str(Path(args.log_root)),
                "cache_root": str(Path(args.cache_root)),
                "image_hw": [int(args.image_hw[0]), int(args.image_hw[1])],
                "teacher_image_hw": [int(args.teacher_image_hw[0]), int(args.teacher_image_hw[1])],
                "target_dim": int(args.target_dim),
                "seed": int(args.seed),
                "eval_scenes": int(args.eval_scenes),
                "skip_degenerate": bool(args.skip_degenerate),
                "scene_index_count": len(tokens),
                "train_scene_count": len(train_tokens),
                "eval_scene_count": len(eval_tokens),
                "cached_scene_count": len(cached),
                "compressor": fit_audit,
                "compressor_fingerprint": compressor.fingerprint(),
            },
            manifest_path(args.target_cache_root),
        )
        log(dctx, {"event": "done", "cached_scenes": len(cached), "scene_index": len(tokens), "manifest": str(manifest_path(args.target_cache_root))})
    log(dctx, {"event": "shard_done", **result}, all_ranks=True)
    if dctx.enabled:
        dist.destroy_process_group()
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
