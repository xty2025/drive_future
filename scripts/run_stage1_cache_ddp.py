"""Stage-1 production trainer: cached Future-Register targets, sharded over ranks.

This is the 16-GPU Stage 1.  ``run_stage1_multiview.py`` stays as the smoke and
diagnostic harness, and the two are not interchangeable:

* That script decodes four ``future.mp4`` files and runs the frozen Teacher over 32
  frames **per scene per run** -- ~5.7 s of GPU time, measured, which is 163
  GPU-hours over the 103075-scene corpus and is repeated on every restart.  This one
  reads ``{target_cache_root}/targets/{token[:2]}/{token}.pt``, the 512 KB
  ``[K,32,D]`` block Stage 0 already paid for.
* That script holds every scene resident -- it moves the whole sample list to the
  device before the first step and pre-collates all of it -- which is why the
  launcher's RAM estimate is 195 MB per scene and caps a run at a few hundred.  This
  one streams through ``Stage1SceneDataset``, which keeps the history as ``uint8`` on
  the host (11.6 MB per scene) and decodes five JPEGs in ``__getitem__``.
* That script is single-process.  This one is a real ``torchrun`` job, so Stage 1 can
  use the same 16 cards Stage 0 and Stage 2 already use.

What the two deliberately share is every *number*.  The metrics, the baselines and
the verdict rule live in :mod:`nvs_v6_wan_e2e.stage1_eval`, so ``paired_future_l1``
or ``learned_trajectory_conditional_future`` means the same thing in both logs --
that rule is the part that decides whether Stage 1 learned anything, and a second
copy of it in a second script is exactly how it would stop meaning that.

The train/eval split is **the builder's**, re-derived from the same
``_ready_scenes.json`` sidecar with the manifest's own ``eval_scenes`` and ``seed``
through :func:`~nvs_v6_wan_e2e.target_cache.split_scene_tokens`.  Recomputing it from
CLI flags would be a leak, not a typo: the standardizer was fit on the builder's
train scenes, so a scene this script calls "eval" could have contributed to the
mean/std its own targets were standardized by.

Run with the repository's standard PYTHONPATH, e.g.::

    PYTHONPATH="$PWD/src:$PWD/vendor/DrivoR:$PWD/vendor/DrivoR/nuplan-devkit" \\
      python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=16 \\
      scripts/run_stage1_cache_ddp.py --target-cache-root artifacts/future_target_cache \\
      --max-steps 20000 --batch-size 4 --workers 6
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from nvs_v6_wan_e2e.data import Observation, PairTargets, REGISTER_TOKENS
from nvs_v6_wan_e2e.losses import identity_future_l1
from nvs_v6_wan_e2e.model import DirectedPairDiT, parameter_groups
from nvs_v6_wan_e2e.modes import ImmutableModeBuffer, StructuredSourceInitializer
from nvs_v6_wan_e2e.scorer import ScratchDrivoRScorer
from nvs_v6_wan_e2e.stage1_eval import (
    CONDITIONING_SUMMARY_KEYS,
    future_fit_metrics,
    register_verdict,
    streaming_future_alignment,
    trajectory_errors,
)
from nvs_v6_wan_e2e.stages import build_optimizers, schedule_factor
from nvs_v6_wan_e2e.target_cache import (
    MIN_GT_SPREAD_M,
    Stage1SceneDataset,
    collate_stage1_batch,
    filter_usable_tokens,
    list_cached_scenes,
    load_standardizer,
    manifest_path,
    observation_to_device,
    ready_scenes_path,
    split_scene_tokens,
    standardizer_path,
    targets_to_device,
)
from nvs_v6_wan_e2e.vision import (
    DRIVOR_LORA_RANK,
    DRIVOR_NUM_SCENE_TOKENS,
    DRIVOR_OFFICIAL_IMAGE_HW,
)

DEFAULT_DINO_CHECKPOINT = "/mnt/LLM_weight/timm/vit_large_patch16_dinov3.lvd1689m/model.safetensors"
DEFAULT_WAN_BACKBONE = "checkpoints/ActionDiT_Wan22_256hdim_3outdim.pt"


# --------------------------------------------------------------------------- #
# Distributed helpers -- the same shape as build_future_target_cache.py's, so a
# Stage 0 log and a Stage 1 log read alike.
# --------------------------------------------------------------------------- #
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


def setup_distributed() -> Dist:
    """torchrun/nccl when the environment says so, otherwise a plain single process.

    ``enabled`` follows ``world > 0``, not ``world > 1``, matching Stage 2: a
    ``--nproc_per_node=1`` torchrun still wraps in DDP and still builds the reducer.
    That is what makes the "every trainable Parameter receives a gradient" contract
    testable on one card -- a bare ``python`` run would skip the wrap and pass
    vacuously, and the failure it hides only appears at 16 ranks, as a hang in the
    all-reduce with no rank left to print anything.
    """

    has_env = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 1 on cached Multi-GT targets requires the configured CUDA/PPU device")
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    enabled = False
    if has_env and world > 0:
        # Rank 0 reads the scene index and the mean-target sample while the others
        # wait at the barrier; a short timeout would abort on that legitimate skew.
        dist.init_process_group(backend="nccl", timeout=timedelta(hours=6))
        enabled = True
        world = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
    if not 0 <= rank < world:
        raise ValueError(f"rank {rank} outside world {world}")
    return Dist(rank=rank, world=world, local_rank=local_rank, device=device, enabled=enabled)


def log0(dctx: Dist, payload: dict[str, Any]) -> None:
    if dctx.main:
        print(json.dumps({"rank": dctx.rank, **payload}), flush=True)


def barrier(dctx: Dist) -> None:
    if dctx.enabled:
        dist.barrier()


def all_reduce_sum(value: torch.Tensor, dctx: Dist) -> torch.Tensor:
    if dctx.enabled:
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# The DDP-visible Stage-1 forward.
# --------------------------------------------------------------------------- #
class Stage1Module(nn.Module):
    """``forward_register_only`` behind a ``forward`` DDP can see.

    ``stages.stage1_step`` cannot be used here.  It calls
    ``model.forward_register_only`` directly, which bypasses ``DDP.forward`` -- and
    the reducer's ``prepare_for_backward`` runs *there*, tracing the returned tensors.
    Reaching into ``net.module`` would produce gradients that are never all-reduced,
    i.e. 16 ranks training 16 different models while every loss curve still looks
    healthy.  So the register read-out is the module's own forward, the loss stays
    outside it exactly as Stage 2 does, and ``stage1_step`` keeps serving the
    single-process harness.

    The training semantics are unchanged: clean GT trajectory at ``sigma = 0``, the
    mode code zeroed so no row can be identified from its index, the **stage1**
    segment mask, and ``identity_future_l1`` pairing register row ``k`` with GT
    ``k``'s own compressed future.
    """

    def __init__(self, model: DirectedPairDiT) -> None:
        super().__init__()
        self.model = model

    def forward(self, observation: Observation, source: torch.Tensor, modes: torch.Tensor) -> torch.Tensor:
        return self.model.forward_register_only(observation, source, modes)


@dataclass
class CacheEvalSample:
    """The three fields ``stage1_eval.EvalSample`` asks for, built from a payload.

    ``observation.rgb_history`` stays ``uint8`` on the host: an eval set is scored
    scene by scene, and ``future_fit_metrics`` moves each one to the device itself.
    Holding a few hundred scenes as float32 instead would cost 46.3 MB each next to
    a model that already owns most of the card.
    """

    scene_id: str
    observation: Observation
    targets: PairTargets


def load_eval_samples(
    dataset: Stage1SceneDataset,
    tokens: list[str],
    limit: int,
) -> tuple[list[CacheEvalSample], list[dict[str, str]]]:
    """Materialize up to ``limit`` cached scenes for the metric pass.

    ``scene_id`` is read from the collated batch rather than from ``tokens[index]``:
    ``Stage1SceneDataset.__getitem__`` walks forward to the next token when a decode
    fails, precisely so a bad scene cannot change how many batches a rank produces.
    Taking the id from the input list would then silently score the wrong scene.
    """

    samples: list[CacheEvalSample] = []
    skipped: list[dict[str, str]] = []
    for index, token in enumerate(tokens):
        if len(samples) >= limit:
            break
        try:
            observation, targets, scene_ids = collate_stage1_batch([dataset[index]])
        except Exception as error:
            skipped.append({"scene": token, "reason": f"{type(error).__name__}: {error}"})
            continue
        samples.append(CacheEvalSample(scene_ids[0], observation, targets))
    if not samples:
        raise RuntimeError(f"no eval scene could be loaded: {skipped[:5]}")
    return samples, skipped


def mean_target_from_cache(
    dctx: Dist,
    dataset: Stage1SceneDataset,
    train_tokens: list[str],
    limit: int,
) -> tuple[torch.Tensor, int]:
    """The dataset-average register, ``[REGISTER_TOKENS, D]``, broadcast from rank 0.

    This is the "learn nothing" reference ``constant_target_l1`` and the
    ``collapse_toward_mean`` detector are measured against.  It cannot be assumed to
    be zero even though the standardizer makes the target zero-mean over its *fit*
    scenes: the fit covers ``--fit-scenes`` of them, this run trains on all the
    cached ones, and the difference between the two averages is the drift a collapse
    would hide inside.

    Computed on rank 0 and broadcast, so all ranks report the same baseline and a
    per-rank sample of a shuffled index cannot move the number between evaluations.
    ``float32`` rather than ``float64`` throughout: the average of a few hundred
    scenes needs no extra precision, and a double-precision collective is the one
    datatype an accelerator's NCCL shim is least likely to have exercised.
    """

    count = min(max(1, int(limit)), len(train_tokens))
    if not dctx.main:
        # Shape is fixed by the standardizer, so the receivers can allocate without a
        # second broadcast; only the values travel.
        width = int(dataset.load_payload(train_tokens[0])["future_register"].shape[-1])
        register = torch.zeros(REGISTER_TOKENS, width, dtype=torch.float32, device=dctx.device)
        dist.broadcast(register, src=0)
        return register.cpu(), count
    blocks = []
    for index in range(count):
        try:
            blocks.append(dataset.load_payload(train_tokens[index])["future_register"].float())
        except Exception:
            continue
    if not blocks:
        raise RuntimeError("no training payload could be read for the mean-target baseline")
    count = len(blocks)
    register = torch.stack(blocks).mean(dim=(0, 1))
    if dctx.enabled:
        register = register.to(dctx.device)
        dist.broadcast(register, src=0)
        register = register.cpu()
    return register, count


def save_checkpoint(
    model: DirectedPairDiT,
    optimizer: torch.optim.Optimizer,
    opt_step: int,
    register_target_dim: int,
    scene_ids: list[str],
    args: argparse.Namespace,
    output_dir: Path,
    dctx: Dist,
) -> None:
    """Rank 0 only, in the payload shape Stage 2 loads with ``strict=True``.

    ``register_target_dim`` rides along because it is a property of the Teacher
    checkpoint these weights were trained against, not of a command line: Stage 2 has
    to rebuild a future head of exactly this width before it can load them, and
    ``d_model`` no longer tells it what that is.  ``scene_ids`` is the cached token
    list this run trained on, kept because the smoke harness records the same thing
    and a Stage 2 log that names its Stage 1 provenance is easier to audit than one
    that only names a file.
    """

    if not dctx.main:
        return
    torch.save(
        {
            "model": model.state_dict(),
            "stage": 1,
            "register_target_dim": int(register_target_dim),
            "scene_ids": list(scene_ids),
            "opt_step": int(opt_step),
            "optimizer": optimizer.state_dict(),
            "config": {key: value for key, value in vars(args).items()},
        },
        output_dir / "stage1_checkpoint.pt",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target-cache-root", required=True, help="the Stage-0 cache: standardizer.pt, targets/ and _manifest.json")
    parser.add_argument("--output-dir", default="artifacts/stage1_cache_ddp")
    # ---- model: every one of these must equal Stage 2's, or its strict load fails ---- #
    parser.add_argument("--dino-checkpoint", default=DEFAULT_DINO_CHECKPOINT)
    parser.add_argument("--wan-backbone", default=DEFAULT_WAN_BACKBONE)
    parser.add_argument("--target-dim", type=int, default=256, help="d_model: the DiT hidden width. The register target width comes from the cache's standardizer and is not a knob here")
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--num-scene-tokens", type=int, default=DRIVOR_NUM_SCENE_TOKENS)
    parser.add_argument("--lora-rank", type=int, default=DRIVOR_LORA_RANK)
    parser.add_argument("--grad-checkpointing", action="store_true")
    parser.add_argument(
        "--precision",
        choices=("fp32", "bf16"),
        default="fp32",
        help=(
            "autocast dtype for the forward passes. The scene encoder is DINOv3 ViT-L/16 at "
            "672x1148 (3024 patches/image, five images per scene) and dominates the step; "
            "measured on one card, bf16 is 3.97x faster than fp32 and cuts peak memory "
            "29.69 -> 19.72 GiB per scene. fp32 stays the default so an existing command "
            "line reproduces the run it was written for."
        ),
    )
    # ---- data ---- #
    parser.add_argument("--image-hw", type=int, nargs=2, default=None, help="defaults to the size the cache was built at; Stage1SceneDataset rejects any payload that disagrees")
    parser.add_argument("--skip-degenerate", action="store_true", help="drop scenes whose Multi-GT trajectory labels collapse, or whose K trajectories are closer than --min-gt-spread-m")
    parser.add_argument("--min-gt-spread-m", type=float, default=MIN_GT_SPREAD_M, help="metres of mean pairwise GT separation a scene needs to teach trajectory conditioning; one threshold reaches both the split filter and the dataset, which must not disagree")
    parser.add_argument("--workers", type=int, default=6, help="DataLoader workers per rank; each decodes five JPEGs per scene")
    # ---- optimization ---- #
    parser.add_argument("--batch-size", type=int, default=4, help="scenes per rank per micro-batch")
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0, help="stop after this many optimizer steps; 0 runs every epoch")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)
    parser.add_argument("--find-unused", action="store_true", help="DDP find_unused_parameters=True (safer, slower)")
    # ---- evaluation ---- #
    parser.add_argument("--eval-metric-scenes", type=int, default=32, help="held-out scenes the verdict is computed on, rank 0 only")
    parser.add_argument("--train-metric-scenes", type=int, default=32, help="training scenes the verdict is computed on, rank 0 only")
    parser.add_argument("--mean-target-scenes", type=int, default=512, help="training payloads the dataset-average baseline is fit on")
    parser.add_argument("--eval-every", type=int, default=1000, help="optimizer steps between verdict passes; 0 evaluates only at the end")
    parser.add_argument("--candidates", type=int, default=64, help="anchor codebook size; only --check-scorer reads it, Stage 1 never does")
    parser.add_argument("--check-scorer", action="store_true")
    # ---- bookkeeping ---- #
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--seed", type=int, default=17, help="model init and DataLoader order; it does NOT choose the train/eval split, the builder's manifest does")
    args = parser.parse_args()
    args.target_cache_root = str(Path(args.target_cache_root))
    if not standardizer_path(args.target_cache_root).exists():
        raise FileNotFoundError(
            f"no standardizer.pt under {args.target_cache_root}; run scripts/build_future_target_cache.py first"
        )
    return args


def resolve_scene_sets(dctx: Dist, args: argparse.Namespace, fingerprint: str) -> dict[str, Any]:
    """Rank 0 reads the cache and the builder's split; every rank gets the result.

    Three things here are expensive and identical on all ranks, so they run once:
    ``list_cached_scenes`` is a directory walk over up to 103075 files, and
    ``filter_usable_tokens`` with a fingerprint check opens every surviving payload.
    At 16 ranks that is 16 copies of the same scan and 16 copies of the same answer,
    which is why the result is broadcast rather than recomputed.

    The split is re-derived from ``_ready_scenes.json`` with the manifest's own
    ``eval_scenes``/``seed``, so it is bit-identical to the one the standardizer was
    fit under.  Only then is it intersected with the cached tokens: a partial cache
    shrinks both halves without moving a scene between them, which is what keeps an
    eval scene out of the mean/std its targets were standardized by.

    Without the sidecar there is no way to recover the builder's ordering, so the
    split falls back to the cached list under ``--seed`` and says so out loud -- the
    numbers are still comparable to each other, they just cannot be claimed
    leak-free against the standardizer fit.
    """

    if not dctx.main:
        shared: list[Any] = [None]
        dist.broadcast_object_list(shared, src=0)
        return shared[0]

    sidecar = ready_scenes_path(args.target_cache_root)
    manifest = manifest_path(args.target_cache_root)
    blob = json.loads(manifest.read_text()) if manifest.exists() else {}
    cached = list_cached_scenes(args.target_cache_root)
    if not cached:
        raise FileNotFoundError(f"{args.target_cache_root}/targets holds no cached scene")
    image_hw = (
        tuple(args.image_hw)
        if args.image_hw
        else tuple(int(value) for value in blob.get("image_hw", DRIVOR_OFFICIAL_IMAGE_HW))
    )

    provenance: dict[str, Any] = {"sidecar": str(sidecar), "manifest": str(manifest),
                                   "cached_scenes": len(cached)}
    eval_scenes = int(blob.get("eval_scenes", 256))
    seed = int(blob.get("seed", args.seed))
    if sidecar.exists():
        tokens = [str(token) for token in json.loads(sidecar.read_text()).get("tokens", [])]
        provenance.update({"source": "ready_scenes_sidecar", "index_scenes": len(tokens)})
    else:
        tokens = list(cached)
        eval_scenes = min(eval_scenes, max(1, len(tokens) // 8))
        provenance.update({
            "source": "cached_tokens_fallback",
            "index_scenes": len(tokens),
            "warning": "no _ready_scenes.json, so the builder's ordering is unrecoverable and an eval "
                       "scene may have taken part in the standardizer fit",
        })
    if not tokens:
        raise FileNotFoundError(f"{sidecar} holds no scene token")
    train_tokens, eval_tokens = split_scene_tokens(tokens, eval_scenes, seed)
    provenance.update({"eval_scenes_requested": eval_scenes, "split_seed": seed})

    cached_set = set(cached)
    train_split = [token for token in train_tokens if token in cached_set]
    eval_split = [token for token in eval_tokens if token in cached_set]
    if not train_split:
        raise RuntimeError(f"no cached training scene out of {len(train_tokens)} split tokens")

    # The standardizer's own fingerprint, not the manifest's copy of it: the manifest
    # is written after the shard finishes, so a run killed between the two would leave
    # a manifest whose fingerprint no longer describes the payloads next to it.
    recorded = blob.get("target_fingerprint")
    if recorded and recorded != fingerprint:
        log0(dctx, {"event": "warn", "message": f"manifest fingerprint {recorded} != standardizer {fingerprint}; "
                                                "filtering by the standardizer, which is what the payloads were built under"})
    usable_train, rejected = filter_usable_tokens(
        train_split, args.target_cache_root,
        skip_degenerate=args.skip_degenerate, target_fingerprint=fingerprint,
        min_gt_spread_m=args.min_gt_spread_m,
    )
    usable_eval, eval_rejected = filter_usable_tokens(
        eval_split, args.target_cache_root,
        skip_degenerate=args.skip_degenerate, target_fingerprint=fingerprint,
        min_gt_spread_m=args.min_gt_spread_m,
    )
    rejected.extend(eval_rejected)
    if not usable_train:
        raise RuntimeError(f"every cached training scene was rejected: {rejected[:5]}")
    provenance.update({"train_split": len(train_split), "eval_split": len(eval_split),
                       "train": len(usable_train), "eval": len(usable_eval),
                       "fingerprint_filtered": fingerprint,
                       "rejected": len(rejected), "first_rejected": rejected[:5]})
    log0(dctx, {"event": "split", **provenance})
    shared = {"train": usable_train, "eval": usable_eval, "provenance": provenance,
              "image_hw": list(image_hw)}
    if dctx.enabled:
        dist.broadcast_object_list([shared], src=0)
    return shared


def main() -> int:
    args = parse_args()
    dctx = setup_distributed()
    seed_everything(args.seed)
    device = dctx.device
    output_dir = Path(args.output_dir)
    if dctx.main:
        output_dir.mkdir(parents=True, exist_ok=True)
    barrier(dctx)
    log0(dctx, {"event": "start", "world": dctx.world, "device": str(device),
                "target_cache_root": args.target_cache_root})

    standardizer, fit_audit = load_standardizer(args.target_cache_root)
    register_target_dim = int(standardizer.state.feature_dim)
    fingerprint = standardizer.fingerprint()
    log0(dctx, {
        "event": "standardizer_loaded",
        "fingerprint": fingerprint,
        "register_target_dim": register_target_dim,
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
    })

    scene_sets = resolve_scene_sets(dctx, args, fingerprint)
    usable_train = scene_sets["train"]
    usable_eval = scene_sets["eval"]
    split_provenance = scene_sets["provenance"]
    image_hw = tuple(int(value) for value in scene_sets["image_hw"])
    log0(dctx, {"event": "usable", "image_hw": list(image_hw), "train": len(usable_train),
                "eval": len(usable_eval), "rejected": split_provenance["rejected"],
                "first_rejected": split_provenance["first_rejected"]})

    train_dataset = Stage1SceneDataset(
        usable_train, args.target_cache_root, image_hw=image_hw,
        skip_degenerate=args.skip_degenerate, target_fingerprint=fingerprint,
        min_gt_spread_m=args.min_gt_spread_m,
    )
    # Only built when there is something held out.  A dataset that fell back to
    # training scenes would satisfy the constructor and then report train scenes as
    # "eval", which is the one mistake this split exists to prevent -- so the empty
    # case stays empty and the verdict reports eval as None instead.
    eval_dataset = (
        Stage1SceneDataset(
            usable_eval, args.target_cache_root, image_hw=image_hw,
            skip_degenerate=args.skip_degenerate, target_fingerprint=fingerprint,
            min_gt_spread_m=args.min_gt_spread_m,
        )
        if usable_eval
        else None
    )
    mean_target, mean_target_scenes = mean_target_from_cache(dctx, train_dataset, usable_train, args.mean_target_scenes)
    log0(dctx, {"event": "mean_target", "scenes": mean_target_scenes,
                "shape": list(mean_target.shape), "abs_mean": round(float(mean_target.abs().mean()), 6)})

    sampler = (
        DistributedSampler(train_dataset, num_replicas=dctx.world, rank=dctx.rank, shuffle=True, drop_last=True)
        if dctx.enabled else None
    )
    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=args.workers,
        collate_fn=collate_stage1_batch,
        drop_last=True,
        pin_memory=False,
        persistent_workers=args.workers > 0,
        prefetch_factor=2 if args.workers > 0 else None,
    )
    if not len(loader):
        raise RuntimeError(
            f"rank {dctx.rank} got no batch: {len(usable_train)} scenes over {dctx.world} ranks at "
            f"batch_size={args.batch_size} with drop_last=True. Lower --batch-size or --nproc_per_node, "
            "or finish the Stage-0 build so there are more cached scenes."
        )

    # ---- model ------------------------------------------------------------- #
    model = DirectedPairDiT(
        d_model=args.target_dim,
        heads=args.heads,
        ffn_dim=4 * args.target_dim,
        depth=args.depth,
        register_target_dim=register_target_dim,
        num_scene_tokens=args.num_scene_tokens,
        dino_checkpoint=args.dino_checkpoint,
        dino_image_hw=tuple(image_hw),
        dino_lora_rank=args.lora_rank,
        dino_grad_checkpointing=args.grad_checkpointing,
        wan_backbone_path=args.wan_backbone if Path(args.wan_backbone).exists() else None,
    ).to(device)
    if args.wan_backbone and not Path(args.wan_backbone).exists():
        log0(dctx, {"event": "warn", "message": f"Wan backbone {args.wan_backbone} not found; theta starts random"})
    # Built even though Stage 1 freezes it: ``build_optimizers`` is the one place the
    # freeze, the parameter groups and the disjointness assertions live, and giving it
    # a stub scorer here would mean a second, weaker copy of those checks.
    scorer = ScratchDrivoRScorer(scene_tokens=model.scene_tokens).to(device)

    resumed_step = 0
    resumed_optimizer: dict[str, Any] | None = None
    if args.resume:
        # Loaded once and read twice: the optimizer state has to go in after
        # ``build_optimizers`` has decided which parameters are trainable, and reading
        # the file again to get it would pay a second full unpickle of the weights.
        resume_ckpt = torch.load(args.resume, map_location="cpu", weights_only=True)
        model.load_state_dict(resume_ckpt["model"], strict=True)
        resumed_optimizer = resume_ckpt.get("optimizer")
        resumed_step = int(resume_ckpt.get("opt_step", 0))
        log0(dctx, {"event": "resumed", "checkpoint": str(args.resume), "prior_opt_step": resumed_step,
                    "optimizer_state_in_payload": resumed_optimizer is not None})
        # The launcher now passes --resume whenever a checkpoint exists, which makes the
        # easy mistake a silent one: resuming a run under a different global batch, lr or
        # cache root still loads cleanly via strict=True, but the numbers logged before and
        # after that step no longer describe the same optimization.  Warn instead of
        # aborting -- raising --max-steps to extend a run is a legitimate resume.
        saved_config = resume_ckpt.get("config") or {}
        benign = {"epochs", "max_steps", "eval_every", "log_every", "save_every", "output_dir", "resume"}
        current_config = vars(args)
        drifted = sorted(key for key in saved_config
                         if key not in benign and key in current_config and saved_config[key] != current_config[key])
        if drifted:
            log0(dctx, {"event": "resume_config_drift", "checkpoint": str(args.resume),
                        "drifted": {key: {"checkpoint": saved_config[key], "now": current_config[key]} for key in drifted}})
        del resume_ckpt

    # Before the DDP wrap: this is what applies the Stage-1 freeze, and the reducer is
    # built from the parameters still marked trainable at construction time.
    optimizer, _ = build_optimizers(model, scorer, stage=1, lr=args.lr)
    if resumed_optimizer is not None:
        optimizer.load_state_dict(resumed_optimizer)
    generator_parameters = [parameter for group in parameter_groups(model).values() for parameter in group]

    net = Stage1Module(model).to(device)
    if dctx.enabled:
        net = DDP(net, device_ids=[dctx.local_rank], output_device=dctx.local_rank,
                  find_unused_parameters=args.find_unused, broadcast_buffers=False)

    # One autocast context per training forward.  fp32 resolves to nullcontext, so the
    # default command line runs the exact op sequence it did before this flag existed.
    #
    # Two deliberate limits on where it applies:
    #   * the L1 is taken *outside* it, on ``.float()`` predictions -- a bf16 reduction
    #     over 32x1024 elements per row loses the low-order differences the loss lives on;
    #   * ``verdict_pass`` stays fp32 even under ``--precision bf16``.  Autocast never
    #     touches master weights, so the fp32 forward is a well-defined reference for the
    #     same parameters, and it has to be: the verdict reads ``paired_future_l1``
    #     against ``trajectory_blind_l1`` at a 6e-5 separation, while bf16 at 0.67 has
    #     ~2.7e-3 of resolution and would drown the comparison in rounding.
    def autocast_ctx():
        return torch.autocast("cuda", dtype=torch.bfloat16) if args.precision == "bf16" else nullcontext()

    trainable = {name: parameter for name, parameter in model.named_trainable_parameters()}
    groups = parameter_groups(model)
    model_audit = {
        "token_layout": model.token_layout(),
        "scene_tokens": int(model.scene_tokens),
        "action_tokens": int(model.action_tokens),
        "register_tokens": int(model.register_tokens),
        "register_target_dim": register_target_dim,
        "dit_token_order": ["scene", "action", "register"],
        "dit_sequence_tokens": int(model.scene_tokens + model.action_tokens + model.register_tokens),
        "self_attention_mask": "single_segment_block",
        "segment_visibility": [list(row) for row in model.theta.SEGMENT_VISIBILITY],
        "action_reads_register": bool(model.theta.SEGMENT_VISIBILITY[1][2]),
        "register_reads_action": bool(model.theta.SEGMENT_VISIBILITY[2][1]),
        "wan_backbone": model.wan_backbone_info,
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in trainable.values()),
        "trainable_tensors": len(trainable),
        "trainable_parameters_by_group": {key: sum(p.numel() for p in value) for key, value in groups.items()},
        "ddp_enabled": bool(dctx.enabled),
        "ddp_find_unused_parameters": bool(args.find_unused),
        "precision": args.precision,
        "precision_scope": "training_forward_only; verdict_pass stays fp32",
        "grad_checkpointing": bool(args.grad_checkpointing),
    }
    log0(dctx, {"event": "model", **model_audit})

    accum = max(1, args.grad_accum)
    steps_per_epoch = len(loader)
    opt_steps_per_epoch = max(1, steps_per_epoch // accum)
    total_opt_steps = opt_steps_per_epoch * args.epochs
    if args.max_steps > 0:
        total_opt_steps = min(total_opt_steps, args.max_steps)
    log0(dctx, {"event": "schedule", "steps_per_epoch": steps_per_epoch, "grad_accum": accum,
                "epochs": args.epochs, "total_opt_steps": total_opt_steps, "resumed_at": resumed_step,
                "global_batch_scenes": args.batch_size * dctx.world * accum})

    # ---- verdict inputs ---------------------------------------------------- #
    # Rank 0 only, between barriers: the eval pass is two forwards per scene and the
    # other ranks have nothing to reduce against, so sharding it would cost more
    # synchronization than it saves.  The samples are materialized on rank 0 alone for
    # the same reason -- each one is a five-JPEG decode plus a payload read, and 15
    # ranks doing it to sit at a barrier afterwards is pure waste.
    if dctx.main:
        train_eval_samples, train_skipped = load_eval_samples(train_dataset, usable_train, args.train_metric_scenes)
        eval_samples, eval_skipped = (
            load_eval_samples(eval_dataset, usable_eval, args.eval_metric_scenes)
            if eval_dataset is not None
            else ([], [])
        )
    else:
        train_eval_samples, train_skipped, eval_samples, eval_skipped = [], [], [], []
    cross_train = [train_eval_samples[(index + 1) % len(train_eval_samples)] for index in range(len(train_eval_samples))]
    cross_eval = [eval_samples[(index + 1) % len(eval_samples)] for index in range(len(eval_samples))] if eval_samples else None
    log0(dctx, {"event": "verdict_inputs", "train_scenes": len(train_eval_samples),
                "eval_scenes": len(eval_samples), "train_skipped": train_skipped[:5],
                "eval_skipped": eval_skipped[:5],
                "scene_blind_control": "eval scenes are rotated by one, so scene_blind_l1 scores a prediction against another scene's target"})

    # Every verdict pass in order, so the intermediate ones survive into
    # ``stage1_cache_ddp_result.json``.  ``log0`` writes them to stdout too, but that
    # stream is interleaved with 15 ranks of DataLoader warnings and cannot be parsed
    # reliably; a 20-epoch run's held-out curve is the whole point of evaluating once
    # per epoch, and it should not live only in a log.
    verdict_history: list[dict[str, Any]] = []

    def verdict_pass(opt_step: int, phase: str) -> dict[str, Any]:
        """Rank 0 scores the register; every rank waits at the barrier afterwards."""

        barrier(dctx)
        report: dict[str, Any] = {}
        if dctx.main:
            train_metrics = future_fit_metrics(train_eval_samples, model, mean_target, cross_train)
            eval_metrics = future_fit_metrics(eval_samples, model, mean_target, cross_eval) if eval_samples else None
            report = {
                "event": "verdict",
                "opt_step": opt_step,
                "phase": phase,
                "train": {key: value for key, value in train_metrics.items() if key != "scenes"},
                "eval": None if eval_metrics is None else {key: value for key, value in eval_metrics.items() if key != "scenes"},
                "register_verdict": register_verdict(train_metrics, eval_metrics),
                "trajectory_conditioning": {key: train_metrics[key] for key in CONDITIONING_SUMMARY_KEYS},
            }
            verdict_history.append(report)
            log0(dctx, report)
        barrier(dctx)
        return report

    before = verdict_pass(resumed_step, "before")
    net.train()

    # ---- training loop ----------------------------------------------------- #
    run_start = time.time()
    window_start = run_start
    window_start_step = resumed_step
    opt_step = resumed_step
    micro = 0
    stop = False
    history: list[dict[str, Any]] = []
    losses: list[float] = []
    cosines: list[float] = []
    relatives: list[float] = []
    rows: list[int] = []
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        epoch_start = time.time()
        for observation, targets, _scene_ids in loader:
            observation = observation_to_device(observation, device)
            targets = targets_to_device(targets, device)
            # The clean GT trajectory is the sigma = 0 end of the same flow Stage 2
            # noises, and the mode code is zeroed rather than per-row: a code constant
            # per row index would let the DiT name its GT from the row alone and skip
            # reading the trajectory the future queries attend to.
            source = targets.gt_trajectory
            modes = torch.zeros((*source.shape[:2], 2), device=device, dtype=source.dtype)

            factor = schedule_factor(opt_step + 1, total_opt_steps, args.warmup)
            for group in optimizer.param_groups:
                group["lr"] = args.lr * factor

            with autocast_ctx():
                predicted = net(observation, source, modes)
            loss = identity_future_l1(predicted.float(), targets)
            is_boundary = ((micro + 1) % accum) == 0
            sync_ctx = net.no_sync() if (dctx.enabled and accum > 1 and not is_boundary) else nullcontext()
            with sync_ctx:
                (loss / accum).backward()
            losses.append(float(loss.detach()))
            # Reuses the register the loss already produced, so this costs three
            # elementwise reductions over [B,K,32,1024] rather than another forward.
            alignment = streaming_future_alignment(predicted, targets)
            cosines.append(alignment["future_cosine_similarity"])
            relatives.append(alignment["future_relative_l1"])
            rows.append(int(targets.gt_valid.sum()))
            micro += 1
            if not is_boundary:
                continue

            if args.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(generator_parameters, args.clip_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            opt_step += 1

            if opt_step % args.log_every == 0 or opt_step == total_opt_steps:
                window = losses[-args.log_every:]
                cos_window = cosines[-args.log_every:]
                rel_window = relatives[-args.log_every:]
                # Each rank contributes exactly one window mean, so the all-reduce
                # counter is 1 per rank.  Passing ``len(window)`` instead divided the
                # cross-rank mean by the micro-steps per rank as well, reporting a loss
                # ``log_every``x smaller than the one actually being optimized.
                # All three series ride a single all-reduce; a second collective per
                # logged step would cost more synchronization than the window it reports.
                reduced = all_reduce_sum(
                    torch.tensor(
                        [
                            float(np.mean(window)),
                            float(np.mean(cos_window)),
                            float(np.mean(rel_window)),
                            1.0,
                        ],
                        device=device,
                    ),
                    dctx,
                )
                ranks = max(float(reduced[3]), 1.0)
                now = time.time()
                window_steps = opt_step - window_start_step
                record = {
                    "event": "step",
                    "epoch": epoch,
                    "opt_step": opt_step,
                    "total_opt_steps": total_opt_steps,
                    "lr": round(args.lr * factor, 8),
                    "loss_future": round(float(losses[-1]), 6),
                    "mean_loss_future": round(float(np.mean(window)), 6),
                    "all_rank_mean_loss_future": round(float(reduced[0] / ranks), 6),
                    # The verdict's ``future_cosine_similarity`` and
                    # ``future_relative_l1``, computed on the training batch instead of
                    # on the held-out scenes: 1.0 / 0.0 is a perfect future read, and
                    # ``relative_l1`` crossing 1.0 means the register is worse than
                    # emitting zeros.  These are train-space readings -- eval targets
                    # are standardized by train-fitted statistics, so only the eval
                    # numbers in ``verdict`` are comparable across the two splits.
                    "cos_future": round(float(cosines[-1]), 6),
                    "mean_cos_future": round(float(np.mean(cos_window)), 6),
                    "all_rank_mean_cos_future": round(float(reduced[1] / ranks), 6),
                    "relative_l1_future": round(float(relatives[-1]), 6),
                    "mean_relative_l1_future": round(float(np.mean(rel_window)), 6),
                    "all_rank_mean_relative_l1_future": round(float(reduced[2] / ranks), 6),
                    "supervised_gt_rows": int(rows[-1]),
                    "global_batch_scenes": args.batch_size * dctx.world * accum,
                    # ``sec_per_step`` is the cumulative mean since (re)start and is kept
                    # under its old name for anything already parsing it, but on its own it
                    # cannot show a slowdown: averaged over 1000 steps, a 2x regression
                    # starting at step 900 moves it by 10%.  The window mean over just the
                    # ``log_every`` steps since the last record is what exposes that.
                    "sec_per_step": round((now - run_start) / max(1, opt_step - resumed_step), 3),
                    "sec_per_step_window": round((now - window_start) / window_steps, 3) if window_steps > 0 else None,
                }
                window_start, window_start_step = now, opt_step
                history.append(record)
                log0(dctx, record)

            if args.eval_every > 0 and opt_step % args.eval_every == 0 and opt_step < total_opt_steps:
                verdict_pass(opt_step, "intermediate")
                net.train()
            if args.save_every > 0 and (opt_step % args.save_every == 0 or opt_step == total_opt_steps):
                save_checkpoint(model, optimizer, opt_step, register_target_dim, usable_train, args, output_dir, dctx)
                barrier(dctx)
            if args.max_steps > 0 and opt_step >= total_opt_steps:
                stop = True
                break
        if stop:
            break
        log0(dctx, {"event": "epoch_done", "epoch": epoch, "opt_step": opt_step,
                    "epoch_sec": round(time.time() - epoch_start, 2)})

    after = verdict_pass(opt_step, "after")
    save_checkpoint(model, optimizer, opt_step, register_target_dim, usable_train, args, output_dir, dctx)
    barrier(dctx)

    scorer_audit = None
    if args.check_scorer and dctx.main:
        observation, targets, _ = collate_stage1_batch([train_dataset[0]])
        observation = observation_to_device(observation, device)
        targets = targets_to_device(targets, device)
        modes = ImmutableModeBuffer().subset(args.candidates).to(device)
        initializer = StructuredSourceInitializer(jitter_std=0.0)
        single = observation
        mode_codes = modes.expand_for_batch(1, device)
        source = initializer.clean(single.ego_history, mode_codes)
        with torch.no_grad():
            outputs = model.forward_pairs(single, source, mode_codes)
            logits = scorer(outputs.scene_context, outputs.trajectory)
            selected, utility = scorer.select(logits)
        gt_count = int(targets.gt_valid[0].sum())
        scorer_audit = {
            "scene_context_shape": list(outputs.scene_context.shape),
            "scene_tokens_expected": int(model.scene_tokens),
            "trajectory_shape": list(outputs.trajectory.shape),
            "trajectory_units": "metres/radians, DrivoR-native, no rescaling",
            "predicted_register_shape": list(outputs.predicted_register.shape),
            "score_logits_shape": list(logits.shape),
            "logits_finite": bool(torch.isfinite(logits).all()),
            "selected_candidate": int(selected.reshape(-1)[0]),
            "selected_utility": float(utility.reshape(-1)[0]),
            "proposals_vs_multigt": trajectory_errors(
                outputs.trajectory[0].cpu(), targets.gt_trajectory[0, :gt_count].cpu()
            ),
            "scorer_parameters": sum(p.numel() for p in scorer.parameters()),
            "note": "the scorer is frozen in Stage 1; this is the structural check that Stage 2's inputs exist",
        }

    # Rank 0 spends real time in the scorer audit while the other ranks have nothing
    # left to do; meeting here keeps them from reaching destroy_process_group first
    # and tearing down the group underneath it.
    barrier(dctx)
    if dctx.main:
        summary = {
            "stage": 1,
            "trainer": "cache_ddp",
            "device": str(device),
            "world": dctx.world,
            "stage1_pairing": "identity_one_register_row_per_gt",
            "stage1_register_conditioning": "clean_gt_trajectory_at_sigma_zero_zeroed_mode_code",
            "target_cache_root": args.target_cache_root,
            "target_fingerprint": fingerprint,
            "standardizer_fit": fit_audit,
            "split": split_provenance,
            "train_scenes": len(usable_train),
            "eval_scenes": len(usable_eval),
            "model": model_audit,
            "schedule": {"steps_per_epoch": steps_per_epoch, "grad_accum": accum, "epochs": args.epochs,
                         "total_opt_steps": total_opt_steps, "resumed_at": resumed_step,
                         "global_batch_scenes": args.batch_size * dctx.world * accum},
            "opt_step": opt_step,
            "loss_future_first": losses[0] if losses else None,
            "loss_future_last": losses[-1] if losses else None,
            "mean_loss_future": float(np.mean(losses)) if losses else None,
            "all_losses_finite": bool(np.isfinite(losses).all()) if losses else False,
            "loss_decreased": bool(losses and losses[-1] < losses[0]),
            "supervised_gt_rows_per_step": rows[-args.log_every:] if rows else [],
            "mean_target_scenes": mean_target_scenes,
            "mean_target_abs_mean": float(mean_target.abs().mean()),
            "verdict_before": before,
            "verdict_after": after,
            "verdict_history": verdict_history,
            "scorer_audit": scorer_audit,
            "history": history,
            "elapsed_s": round(time.time() - run_start, 2),
            "config": {key: value for key, value in vars(args).items()},
        }
        (output_dir / "stage1_cache_ddp_result.json").write_text(json.dumps(summary, indent=2) + "\n")
        log0(dctx, {"event": "done", "opt_step": opt_step, "elapsed_s": summary["elapsed_s"],
                    "output": str(output_dir / "stage1_cache_ddp_result.json"),
                    "checkpoint": str(output_dir / "stage1_checkpoint.pt"),
                    "register_verdict": after.get("register_verdict")})
    if dctx.enabled:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
