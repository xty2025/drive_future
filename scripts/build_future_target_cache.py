"""Build the offline Future-Register target cache Stage 1 trains from.

One frozen-Teacher pass per scene, ever.  The register target cannot change
during training -- ``config_v6_wan_e2e.yaml`` puts ``teacher`` and
``target_standardizer`` in ``stage1.freeze`` and ``stage2.freeze`` -- so decoding
four ``future.mp4`` files and running DINOv3 over 32 frames on every epoch is
pure repeated work.  This script does it once and writes ``[K,32,D]`` float32 plus
the metric trajectories -- four DINOv3 prefix-query-extracted, standardized tokens
per future frame at +0.5..+4.0s, frame-major, 512 KB of register data at ``K=4``
and DINOv3-Large's ``D=1024``.

Two ordering constraints shape the run:

* The standardizer is fit **once, on rank 0, over training scenes only**, and every
  rank then loads that same ``standardizer.pt``.  A diagonal mean/std has no basis
  ambiguity, but two fits on different subsets still differ by a per-dimension
  rescaling, and targets built under different rescalings are not comparable -- the
  all-reduced Stage-1 gradient would average them anyway.
* The train/eval split is decided **before** the fit and shared with the trainer
  through ``split_scene_tokens``, so no eval scene can leak into it.

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
    PAYLOAD_FORMAT,
    atomic_json_write,
    atomic_torch_save,
    build_scene_payload,
    list_cached_scenes,
    load_standardizer,
    manifest_path,
    ready_scenes_path,
    save_standardizer,
    scene_target_path,
    split_scene_tokens,
    standardizer_path,
)
from nvs_v6_wan_e2e.teacher import (
    MAX_QUERY_ATTENTION_ENTROPY,
    NATIVE_FUTURE_IMAGE_HW,
    QUERY_HEADS,
    QUERY_TEMPERATURE,
    FutureQueryExtractor,
    TargetStandardizer,
    TimmDinoV3Teacher,
    measure_target_quality,
    query_future_register,
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
        # Rank 0 walks all 103075 scene directories and fits the target mean/std
        # while every other rank waits; a short timeout would abort on that
        # legitimate skew.
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



def _query_block(
    sample: MultiViewSample,
    teacher: TimmDinoV3Teacher,
    extractor: FutureQueryExtractor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``([K,32,D] query tokens, [K*32,heads] normalized entropies)`` for one scene.

    Each GT's eight-frame patch grid is reduced as it is produced and never retained: a
    grid is 49 MB float32 at the future videos' native 29x52 while its thirty-two query
    tokens are 128 KB, which is what lets the fit set stay in RAM.
    """

    if sample.future_rgb is None:
        raise ValueError(f"{sample.scene_id} was loaded without future RGB")
    return query_future_register(teacher, extractor, sample.future_rgb, device)


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


def fit_standardizer(
    dctx: Dist,
    args: argparse.Namespace,
    train_tokens: list[str],
    teacher: TimmDinoV3Teacher,
    extractor: FutureQueryExtractor,
) -> dict[str, Any]:
    """Rank 0 only: mean/std over the query tokens of ``--fit-scenes`` train scenes.

    ``TargetStandardizer.fit`` treats every register token of every GT as its own row,
    so at ``K = 4`` the fit sees ``128 * scenes`` rows -- 65536 rows for 1024 dimensions
    at the default, which pins each per-dimension std to ~0.4% relative error.  Those
    rows pool the four queries of a frame together, which is what the probe measured its
    numbers under, so the standardization the target is audited in and the one it is
    trained in are the same map.  Unlike the SVD basis this replaced, more scenes cannot
    make the target *degenerate*; they can only estimate the scaling better.
    ``--fit-scenes`` is therefore a precision knob, not a correctness gate, and nothing
    here refuses a small fit.

    What is refused is a flat readout, via
    :data:`~nvs_v6_wan_e2e.teacher.MAX_QUERY_ATTENTION_ENTROPY`.  See
    ``measure_target_quality`` for why the separation ratio and the trajectory
    correlation are audited but not gated.
    """

    fit_count = min(int(args.fit_scenes), len(train_tokens))
    if fit_count < 1:
        raise ValueError("the standardizer fit needs at least one training scene")
    log_cache: dict[str, list[dict]] = {}
    blocks: list[torch.Tensor] = []
    trajectories: list[torch.Tensor] = []
    entropy_sum = 0.0
    entropy_max = 0.0
    entropy_rows = 0
    fitted: list[str] = []
    degenerate = 0
    started = time.time()
    last_progress = started
    for scene_id in train_tokens[:fit_count]:
        try:
            sample = _load_sample(args, scene_id, log_cache)
            block, entropies = _query_block(sample, teacher, extractor, dctx.device)
        except (FileNotFoundError, ValueError, RuntimeError) as error:
            log(dctx, {"event": "fit_scene_skipped", "scene": scene_id, "reason": str(error)})
            continue
        blocks.append(block)
        trajectories.append(sample.targets.gt_trajectory[0].float().cpu())
        degenerate += int(bool(sample.metadata["degenerate_multi_gt"]))
        entropy_sum += float(entropies.sum())
        entropy_max = max(entropy_max, float(entropies.max()))
        entropy_rows += int(entropies.shape[0] * entropies.shape[1])
        fitted.append(scene_id)
        del sample, block, entropies
        # Only rank 0 fits; the other ranks are already parked in ``barrier()``, so a
        # quiet fit is indistinguishable from a NCCL hang from the outside -- the
        # StateMonitor starts warning at 60 s of no collective progress.  At the
        # default ``--fit-scenes 512`` this loop is 15-25 min of exactly that, so it
        # emits a liveness line every 30 s instead.
        if time.time() - last_progress >= 30.0:
            elapsed = time.time() - started
            last_progress = time.time()
            log(
                dctx,
                {
                    "event": "standardizer_fit_progress",
                    "scenes": len(fitted),
                    "of": fit_count,
                    "elapsed_s": round(elapsed, 1),
                    "eta_s": round(elapsed / len(fitted) * (fit_count - len(fitted)), 1),
                },
            )
    if not blocks:
        raise RuntimeError("no training scene could be loaded for the standardizer fit")
    fit_rows = torch.cat([block.reshape(-1, teacher.feature_dim) for block in blocks], dim=0)
    standardizer = TargetStandardizer(
        feature_dim=int(teacher.feature_dim),
        temperature=float(args.query_temperature),
        heads=int(args.query_heads),
    )
    standardizer.fit(fit_rows, source_hash="+".join(fitted), split="train")
    entropy_mean = entropy_sum / max(entropy_rows, 1)
    audit = {
        "teacher_checkpoint": str(args.dino_checkpoint),
        "teacher_image_hw": [int(args.teacher_image_hw[0]), int(args.teacher_image_hw[1])],
        "teacher_grid_hw": [int(value) for value in teacher.grid_hw],
        "register_target_dim": int(teacher.feature_dim),
        "query_temperature": float(args.query_temperature),
        "query_heads": int(args.query_heads),
        # The extractor's one silent failure mode.  A uniform readout *is* the mean
        # pool it replaced, keeps every downstream number finite, and discards the
        # 81% of the GT-discriminating variance that lives in the spatial pattern.
        "query_attention_entropy_mean": round(entropy_mean, 6),
        "query_attention_entropy_max": round(entropy_max, 6),
        "query_attention_entropy_rows": entropy_rows,
        "standardizer_fit_rows": int(fit_rows.shape[0]),
        "standardizer_fit_scenes": len(fitted),
        # Degenerate scenes (duplicate Multi-GT labels) stay in the fit -- their rows
        # are legitimate mean/std data -- but their two futures are identical, so they
        # pull ``separation_ratio`` down.  Counted so a low ratio can be attributed
        # rather than guessed at; pessimistic is the safe direction for an audit.
        "standardizer_fit_degenerate_scenes": degenerate,
        "standardizer_clamped_components": int(standardizer.state.clamped_components),
        "standardizer_fingerprint": standardizer.fingerprint(),
        "fit_scene_ids": fitted,
        "fit_elapsed_s": round(time.time() - started, 1),
    }
    # Quality is measured on the standardized targets: standardization is a
    # per-dimension affine map, so it moves both numbers, and the standardized space
    # is the one the Stage-1 L1 lives in.
    audit.update(
        measure_target_quality(
            [standardizer.transform(block) for block in blocks],
            trajectories,
        )
    )
    # Log before saving, so a rejected fit still leaves its numbers on the record.
    log(dctx, {"event": "standardizer_fit", **{key: value for key, value in audit.items() if key != "fit_scene_ids"}})
    if entropy_mean > MAX_QUERY_ATTENTION_ENTROPY and not args.allow_degenerate_query:
        raise ValueError(
            f"refusing to persist a target whose query attention is {entropy_mean:.4f} of uniform "
            f"(max frame {float(entropy_max):.4f}, ceiling {MAX_QUERY_ATTENTION_ENTROPY}): the readout has "
            f"degenerated into a mean pool over {teacher.grid_hw[0]}x{teacher.grid_hw[1]} patches, which keeps "
            f"only the 19% of the GT-discriminating variance that is a per-frame average. Lower "
            f"--query-temperature below {args.query_temperature}, or pass --allow-degenerate-query to "
            f"persist it anyway."
        )
    save_standardizer(args.target_cache_root, standardizer, audit)
    del blocks, trajectories, fit_rows
    gc.collect()
    return audit


def build_shard(
    dctx: Dist,
    args: argparse.Namespace,
    tokens: list[str],
    teacher: TimmDinoV3Teacher,
    extractor: FutureQueryExtractor,
    standardizer: TargetStandardizer,
) -> dict[str, Any]:
    shard = tokens[dctx.rank :: dctx.world]
    fingerprint = standardizer.fingerprint()
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
            block, _ = _query_block(sample, teacher, extractor, dctx.device)
            register = standardizer.transform(block)
            payload = build_scene_payload(
                future_register=register,
                future_chunk_valid=sample.targets.future_chunk_valid[0],
                future_quality=sample.targets.future_quality[0],
                gt_trajectory=sample.targets.gt_trajectory[0],
                gt_valid=sample.targets.gt_valid[0],
                observation=sample.observation,
                pair_ids=list(sample.targets.pair_ids[0]),
                metadata=sample.metadata,
                target_fingerprint=fingerprint,
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


def shard_produced_nothing(result: dict[str, Any], cached_scene_count: int | None = None) -> bool:
    """Whether this build wrote no usable target at all, and must not report success.

    A per-scene ``except`` is required -- one unreadable scene out of 103075 must not
    condemn the whole build, and a resumed run legitimately produces zero *new*
    payloads while reusing thousands.  But the same tolerance hid a systematic fault:
    ``build_scene_payload`` read ``history_paths`` from ``metadata``, where the loader
    never puts it, so every scene raised ``KeyError``, and the run still printed
    ``done``, wrote a manifest beside a ``targets/`` that did not exist, and exited 0.

    The distinction is between a few bad scenes and *no cache at all*.  A systematic
    fault is total, so "this shard has scenes and produced neither a new nor a reused
    payload" and "the destination holds no target" are each sufficient.  Only rank 0
    scans the destination, so the second clause is applied there alone; a shard whose
    slice is empty because ``world`` exceeds the scene count is not a failure either.
    """

    produced = int(result["built"]) + int(result["reused"])
    if bool(int(result["shard_scenes"])) and produced == 0:
        return True
    return cached_scene_count is not None and int(cached_scene_count) == 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target-cache-root", required=True, help="destination for standardizer.pt, targets/ and _manifest.json")
    parser.add_argument("--multiview-root", default=DEFAULT_MULTIVIEW_ROOT)
    parser.add_argument("--log-root", default=DEFAULT_LOG_ROOT)
    parser.add_argument("--sensor-root", default=DEFAULT_SENSOR_ROOT)
    parser.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--dino-checkpoint", default=DEFAULT_DINO_CHECKPOINT)
    parser.add_argument("--image-hw", type=int, nargs=2, default=list(DRIVOR_OFFICIAL_IMAGE_HW), help="history size recorded in each payload; the trainer must ask for the same one")
    parser.add_argument("--teacher-image-hw", type=int, nargs=2, default=list(NATIVE_FUTURE_IMAGE_HW), help="must equal the future videos' native size; the Teacher refuses to interpolate")
    parser.add_argument(
        "--query-temperature",
        type=float,
        default=QUERY_TEMPERATURE,
        help="cosine-attention temperature of the prefix-query readout; part of the target fingerprint",
    )
    parser.add_argument(
        "--query-heads",
        type=int,
        default=QUERY_HEADS,
        help="query heads the patch grid is split over; part of the target fingerprint",
    )
    parser.add_argument(
        "--fit-scenes",
        type=int,
        default=512,
        help="train scenes the target mean/std are estimated from; a precision knob, not a rank gate",
    )
    parser.add_argument(
        "--allow-degenerate-query",
        action="store_true",
        help="persist the target even when the query attention is close to uniform, i.e. the readout has become a mean pool",
    )
    parser.add_argument("--eval-scenes", type=int, default=256, help="held out of training and of the standardizer fit")
    parser.add_argument("--limit", type=int, default=None, help="cap the scene index; useful for a first subset build")
    parser.add_argument("--skip-degenerate", action="store_true", help="drop scenes whose Multi-GT trajectory labels collapse")
    parser.add_argument("--overwrite", action="store_true", help="rebuild scenes that already have a cached target, and refit the standardizer")
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
    if not standardizer_path(args.target_cache_root).exists() or args.overwrite:
        # Only rank 0 fits, and only it may: the scaling has to be one scaling.
        if dctx.main:
            fit_standardizer(
                dctx,
                args,
                train_tokens,
                teacher,
                FutureQueryExtractor(teacher.feature_dim, args.query_temperature, args.query_heads),
            )
        barrier(dctx)
    standardizer, fit_audit = load_standardizer(args.target_cache_root)
    # The readout's temperature and heads are not recoverable from ``mean``/``scale``,
    # so a resumed run that changed either would silently build the second half of the
    # cache in a different target space than the first half -- same width, same shapes,
    # different tensors.  Refuse instead; ``--overwrite`` refits and so is exempt.
    if standardizer.temperature != float(args.query_temperature) or standardizer.heads != int(args.query_heads):
        raise ValueError(
            f"{standardizer_path(args.target_cache_root)} was fit at temperature "
            f"{standardizer.temperature} with {standardizer.heads} heads, but this run asks for "
            f"{args.query_temperature} with {args.query_heads}. Pass --overwrite to refit and rebuild "
            f"every scene, or drop the flags to extend the existing cache."
        )
    # Built from the *loaded* state, never from args, so the shard and the fit cannot
    # disagree about which readout produced the cache.
    extractor = FutureQueryExtractor(
        int(standardizer.state.feature_dim),
        float(standardizer.temperature),
        int(standardizer.heads),
    )
    log(
        dctx,
        {
            "event": "standardizer_loaded",
            "fingerprint": standardizer.fingerprint(),
            "register_target_dim": int(standardizer.state.feature_dim),
            "query_temperature": standardizer.temperature,
            "query_heads": standardizer.heads,
            "fit_rows": fit_audit.get("standardizer_fit_rows"),
            "fit_scenes": fit_audit.get("standardizer_fit_scenes"),
            "attention_entropy_mean": fit_audit.get("query_attention_entropy_mean"),
            "attention_entropy_max": fit_audit.get("query_attention_entropy_max"),
            "clamped_components": fit_audit.get("standardizer_clamped_components"),
            "separation_ratio": fit_audit.get("separation_ratio"),
            "trajectory_distance_pearson": fit_audit.get("trajectory_distance_pearson"),
            "trajectory_distance_spearman": fit_audit.get("trajectory_distance_spearman"),
        },
    )

    result = build_shard(dctx, args, tokens, teacher, extractor, standardizer)
    barrier(dctx)
    # Decided per rank, and only after the barrier: raising inside ``build_shard``
    # would leave the other ranks blocked in the collective and turn a legible
    # failure into a hang.  ``torchrun`` fails the job if any rank returns non-zero.
    cached = list_cached_scenes(args.target_cache_root) if dctx.main else None
    total_failure = shard_produced_nothing(result, None if cached is None else len(cached))
    if dctx.main:
        # Written even on failure, with ``cached_scene_count`` 0 and the reason count:
        # the manifest is the durable record of what the run did, and a partial
        # failure that a per-scene ``except`` tolerated must outlive its log line.
        atomic_json_write(
            {
                "format": PAYLOAD_FORMAT,
                "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "world_size": dctx.world,
                "multiview_root": str(Path(args.multiview_root)),
                "sensor_root": str(Path(args.sensor_root)),
                "log_root": str(Path(args.log_root)),
                "cache_root": str(Path(args.cache_root)),
                "image_hw": [int(args.image_hw[0]), int(args.image_hw[1])],
                "teacher_image_hw": [int(args.teacher_image_hw[0]), int(args.teacher_image_hw[1])],
                "register_target_dim": int(standardizer.state.feature_dim),
                "seed": int(args.seed),
                "eval_scenes": int(args.eval_scenes),
                "skip_degenerate": bool(args.skip_degenerate),
                "scene_index_count": len(tokens),
                "train_scene_count": len(train_tokens),
                "eval_scene_count": len(eval_tokens),
                "cached_scene_count": len(cached),
                "failed_scene_count": int(result["failed_count"]),
                "standardizer": fit_audit,
                "target_fingerprint": standardizer.fingerprint(),
            },
            manifest_path(args.target_cache_root),
        )
    log(dctx, {"event": "shard_done", **result}, all_ranks=True)
    if dctx.enabled:
        dist.destroy_process_group()
    if total_failure:
        log(
            dctx,
            {
                "event": "no_cached_target",
                "shard_scenes": result["shard_scenes"],
                "built": result["built"],
                "reused": result["reused"],
                "failed_count": result["failed_count"],
                "first_reasons": result["failed"][:5],
            },
            all_ranks=True,
        )
        return 1
    if dctx.main:
        log(dctx, {"event": "done", "cached_scenes": len(cached), "scene_index": len(tokens), "manifest": str(manifest_path(args.target_cache_root))})
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
