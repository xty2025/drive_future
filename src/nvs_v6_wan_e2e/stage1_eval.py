"""Stage-1's evaluation contract: the controls that make a low loss meaningful.

Both Stage-1 trainers -- ``run_stage1_multiview.py`` (a few resident scenes, inline
Teacher, smoke/diagnostic) and ``run_stage1_cache_ddp.py`` (the cached, sharded
production run) -- call this module, so a number in one log means the same thing in
the other.  Keeping the metrics in the scripts instead would put two copies of a
verdict rule next to each other, and the rule is exactly the part that must not
drift: it is what decides whether Stage 1 learned anything.

Nothing here is Stage-1 *training*.  The step itself lives in :mod:`.stages`.
"""

from __future__ import annotations

import itertools
from typing import Any, Protocol, Sequence, runtime_checkable

import numpy as np
import torch

from .data import FUTURE_FRAMES, Observation, PairTargets, REGISTER_TOKENS
from .model import DirectedPairDiT
from .target_cache import observation_to_device


@runtime_checkable
class EvalSample(Protocol):
    """The three fields the metrics read; ``MultiViewSample`` satisfies it.

    A Protocol rather than a concrete type because the two trainers build their
    samples differently -- one from a decoded ``future.mp4``, the other from a cached
    payload -- and neither should have to construct the other's dataclass to be
    measured.
    """

    @property
    def scene_id(self) -> str: ...

    @property
    def observation(self) -> Observation: ...

    @property
    def targets(self) -> PairTargets: ...


def trajectory_errors(predicted: torch.Tensor, ground_truth: torch.Tensor) -> dict[str, float]:
    """Metric ADE/FDE of proposals against the Multi-GT set (no rescaling)."""

    per_pair_ade = (predicted[:, None, :, :2] - ground_truth[None, :, :, :2]).norm(dim=-1).mean(dim=-1)
    per_pair_fde = (predicted[:, None, -1, :2] - ground_truth[None, :, -1, :2]).norm(dim=-1)
    return {
        "min_ade_over_gt_m": float(per_pair_ade.min(dim=1).values.mean()),
        "best_pair_ade_m": float(per_pair_ade.min()),
        "best_pair_fde_m": float(per_pair_fde.min()),
        "mean_speed_mps": float(predicted[:, 1:, :2].sub(predicted[:, :-1, :2]).norm(dim=-1).mean() / 0.5),
    }


def trajectory_spread(trajectories: torch.Tensor) -> float:
    """Mean pairwise ADE, in metres, between one scene's own GT trajectories.

    This is how much the *inputs* of a scene differ.  Compared against
    ``target_spread_across_gt`` it says whether the compressed futures separate by
    at least as much as the trajectories that produced them do.  A single-GT scene
    has no pair to compare and reports NaN.
    """

    if trajectories.shape[0] < 2:
        return float("nan")
    xy = trajectories[:, :, :2]
    per_step = (xy[:, None] - xy[None, :]).norm(dim=-1)
    upper = torch.triu_indices(per_step.shape[0], per_step.shape[0], offset=1)
    return float(per_step[upper[0], upper[1]].mean())


#: The headline before/after summary: the L1, the tightest model-independent baseline,
#: the model-side trajectory-blind control, the two direct "how far from the real
#: future" readings (cosine similarity and the dimensionless relative L1), and the two
#: background-removed readings that isolate the part only the trajectory can explain.
CONDITIONING_SUMMARY_KEYS = (
    "mean_paired_future_l1",
    "mean_future_cosine_similarity",
    "mean_future_relative_l1",
    "mean_residual_relative_l1",
    "mean_residual_cos_future",
    "mean_scene_mean_target_l1",
    "mean_trajectory_blind_l1",
    "mean_trajectory_conditioning_fidelity",
    "mean_trajectory_response_alignment",
)


@torch.no_grad()
def future_fit_metrics(
    samples: Sequence[EvalSample],
    model: DirectedPairDiT,
    mean_target: torch.Tensor | None = None,
    cross_samples: Sequence[EvalSample] | None = None,
) -> dict[str, Any]:
    """Stage-1's objective plus the controls that make a low loss meaningful.

    The register is conditioned on each GT trajectory, so row ``k`` is scored
    against GT ``k``'s own compressed future.  A low paired error alone cannot
    distinguish "predicts this trajectory's future" from "predicts the average
    future", so the controls are recorded in two families.

    Model-independent baselines, the only ones ``mean_paired_future_l1`` can honestly
    be compared against: ``scene_mean_target_l1`` is the error of predicting this
    scene's own per-token average future -- averaged over its GTs at each of the
    ``REGISTER_TOKENS`` register positions, so perfect scene reading and zero
    trajectory reading -- and ``constant_target_l1`` is ``|mean_target - own_target|``, the
    dataset-average predictor.  ``scene_mean_target_l1`` is the tighter of the two
    and is the one that decides whether the trajectory input earned its place.

    Model-side controls: ``trajectory_blind_l1`` re-runs the scene with every row
    conditioned on GT 0's trajectory, ``shuffled_pair_l1`` scores row ``k`` against
    GT ``k+1``, and ``scene_blind_l1`` scores it against another scene's targets.
    ``distance_to_mean_target_l1`` is not a baseline at all -- a small value is the
    signature of collapse toward the dataset average.

    ``trajectory_conditioning_fidelity`` and ``trajectory_response_alignment`` say
    how the register reacts to swapping the trajectory: how far it moves relative to
    how far the target moves, and the cosine between the two displacements.  Both
    are scale-free; fidelity is 0 for a trajectory-blind register and 1.0 for one
    that reproduces the targets exactly, and alignment is 0 for a displacement in a
    random direction of the register's 1024 dimensions.

    "How far is the predicted future from the real one" gets two direct answers,
    because an L1 in a standardized 1024-d space has no intuitive scale on its own:
    ``future_cosine_similarity`` is the per-valid-token cosine between prediction and
    target (1.0 = same direction, 0 = orthogonal, i.e. the target's own information
    is entirely absent -- this is the quantity FastWAM trains against as
    ``1 - cosine``) and ``future_relative_l1`` divides the paired L1 by the target's
    own mean magnitude, so 0.0 is perfect and 1.0 is "as wrong as predicting zero".
    Neither gates the verdict, but ``future_relative_l1`` does carry one hard,
    calibrated bar by construction: **below 1.0 means the register beats emitting
    nothing**, and above 1.0 means its output is actively worse than silence.  That
    bar matters most on ``eval``, whose targets are standardized by statistics fit on
    train only, so their magnitudes are not comparable to train's.  What has no
    *absolute* threshold is "good *enough*"; its data-derived anchor is
    ``scene_mean_target_l1``, the loss of memorizing each scene's mean future while
    ignoring the trajectory entirely.  A register that cannot beat it has learned
    nothing trajectory-conditional -- and it is that comparison, not any fixed
    number, that carries over, because every absolute value recorded against the
    retired whitened-PCA ``[K,8,256]`` target (``mean_target_abs_mean`` 2.75 on eval
    against 0.76 on train, raw eval L1 2.78 / relative 1.01, and a scene-mean anchor
    of 0.5839 against a trained register at 0.7515) measures a different quantity in
    a different space and must not be reused as a threshold.

    Each observation is moved to the model's device inside the loop rather than
    beforehand.  The cached trainer holds its history as ``uint8`` on the host --
    11.6 MB per scene instead of 46.3 MB -- and an eval set of a few hundred scenes
    would not fit resident as float32 next to the model.  ``observation_to_device``
    is idempotent, so a caller that already moved its samples pays only a no-op.
    """

    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    rows = []
    for index, sample in enumerate(samples):
        observation = observation_to_device(sample.observation, device)
        targets = sample.targets
        gt_count = int(targets.gt_valid[0].sum())
        source = targets.gt_trajectory[:, :gt_count].to(device)
        # Zeros, not a per-row code: the DiT requires a [B,M,2] mode input, and a
        # code constant per row index would let it identify the GT from the row
        # alone instead of reading the trajectory the future queries attend to.
        modes = torch.zeros(1, gt_count, 2, device=device, dtype=source.dtype)
        predicted = model.forward_register_only(observation, source, modes)[0]
        own_target = targets.future_register[0, :gt_count].to(device)
        chunk_valid = targets.future_chunk_valid[0, :gt_count].to(device)
        weights = targets.future_quality[0, :gt_count].to(device) * chunk_valid.to(predicted.dtype)

        def weighted_error(prediction: torch.Tensor, target: torch.Tensor) -> float:
            error = (prediction - target).abs().mean(dim=-1)
            return float((error * weights).sum() / weights.sum().clamp_min(1e-8))

        blind_source = source[:, :1].expand(-1, gt_count, -1, -1).contiguous()
        blind_predicted = model.forward_register_only(observation, blind_source, modes)[0]

        scene_blind = float("nan")
        if cross_samples is not None:
            cross = cross_samples[index]
            pairs = min(gt_count, int(cross.targets.gt_valid[0].sum()))
            if pairs:
                cross_weights = weights[:pairs]
                cross_target = cross.targets.future_register[0, :pairs].to(predicted.device)
                error = (predicted[:pairs] - cross_target).abs().mean(dim=-1)
                scene_blind = float((error * cross_weights).sum() / cross_weights.sum().clamp_min(1e-8))

        response, spread, alignment = [], [], []
        for left, right in itertools.combinations(range(gt_count), 2):
            both = chunk_valid[left] & chunk_valid[right]
            if not bool(both.any()):
                continue
            moved = (predicted[left, both] - predicted[right, both]).flatten()
            differs = (own_target[left, both] - own_target[right, both]).flatten()
            response.append(float(moved.abs().mean()))
            spread.append(float(differs.abs().mean()))
            alignment.append(float(torch.nn.functional.cosine_similarity(moved, differs, dim=0)))

        # Direct "how far from the real future" readings, per valid register token,
        # over the same quality x validity weighting the L1 uses.
        cosine_rows = []
        for row in range(gt_count):
            valid = chunk_valid[row]
            if not bool(valid.any()):
                continue
            cosine_rows.append(
                float(
                    torch.nn.functional.cosine_similarity(
                        predicted[row, valid].float(), own_target[row, valid].float(), dim=-1
                    ).mean()
                )
            )
        paired_l1 = weighted_error(predicted, own_target)
        target_magnitude = weighted_error(
            torch.zeros_like(own_target), own_target
        )  # |0 - target| weighted == the target's own mean magnitude

        # This scene's own GT futures averaged over the GT axis at each token
        # position: what a register that read the scene perfectly and the trajectory
        # not at all would output.  Per token, not per frame -- the axis holds four
        # queries per frame and they read different regions, so collapsing them into
        # one frame-level average would hand the baseline information the register
        # being scored does not have.
        scene_mean = (own_target * weights.unsqueeze(-1)).sum(dim=0) / weights.sum(dim=0).clamp_min(1e-8).unsqueeze(-1)

        # Background-removed reading.  Measured on 400 cached scenes, the scene mean is
        # not a small nuisance term: it accounts for 79.3% of the target's raw L2 energy
        # (11.8% per-token global DC + 67.5% scene-specific) and buying it alone takes
        # the L1 from 0.792 (predict zero) to 0.318, i.e. 59.8% of the entire objective.
        # Only the remaining 40.2% can come from reading the trajectory.  So the raw
        # ``future_cosine_similarity`` is dominated by a direction the model gets right
        # for free -- a failed 8-token run read cos 0.6555 while both gains were
        # negative.  Subtracting each side's *own* scene mean leaves only what the
        # trajectory could have caused, and it is calibrated where cosine is not: a
        # register that reads the scene and ignores the trajectory has zero residual on
        # both sides, so ``residual_relative_l1`` lands on exactly 1.0.  Below 1.0 is
        # the first honest evidence of trajectory-conditional prediction.
        weight_sum = weights.sum(dim=0).clamp_min(1e-8).unsqueeze(-1)
        prediction_scene_mean = (predicted * weights.unsqueeze(-1)).sum(dim=0) / weight_sum
        prediction_residual = predicted - prediction_scene_mean
        target_residual = own_target - scene_mean
        anchor_l1 = weighted_error(scene_mean.expand_as(own_target), own_target)
        residual_l1 = weighted_error(prediction_residual, target_residual)
        # Both residual readings need at least two GT rows.  With one row the scene mean
        # *is* that row, so ``anchor_l1`` is ``(x*w)/w`` rounding (~1e-7) rather than an
        # exact zero and a ``> 0`` test would divide one noise quantity by another.
        residual_defined = gt_count > 1 and anchor_l1 > 1e-6
        residual_cosines = []
        # A single-GT scene has no within-scene variation to explain: its scene mean
        # *is* its only target, so both residuals are identically zero and their cosine
        # is undefined rather than bad.  Guarded exactly like ``trajectory_blind_l1``
        # and ``shuffled_pair_l1`` below, so ``nanmean`` drops the scene instead of
        # letting a structurally empty one drag the reading down.
        if residual_defined:
            for row in range(gt_count):
                valid = chunk_valid[row]
                if not bool(valid.any()):
                    continue
                residual_cosines.append(
                    float(
                        torch.nn.functional.cosine_similarity(
                            prediction_residual[row, valid].float(), target_residual[row, valid].float(), dim=-1
                        ).mean()
                    )
                )

        # Horizon-resolved gain.  The trajectory-attributable share of a frame rises
        # monotonically with horizon -- measured 30.6% of the L1 at +0.5s against 46.9%
        # at +4.0s -- because the near future is mostly the scene continuing and the far
        # future is mostly where the trajectory went.  A single pooled gain hides which
        # end is moving, and the near end is where a background reader scores best.
        queries_per_frame = REGISTER_TOKENS // FUTURE_FRAMES
        token_error = (predicted - own_target).abs().mean(dim=-1)
        token_anchor = target_residual.abs().mean(dim=-1)
        frame_gains = []
        for frame in range(FUTURE_FRAMES):
            slab = slice(frame * queries_per_frame, (frame + 1) * queries_per_frame)
            frame_weight = weights[:, slab].sum()
            if float(frame_weight) <= 0:
                frame_gains.append(float("nan"))
                continue
            paired = float((token_error[:, slab] * weights[:, slab]).sum() / frame_weight)
            anchor = float((token_anchor[:, slab] * weights[:, slab]).sum() / frame_weight)
            frame_gains.append((anchor - paired) / anchor if anchor > 0 else float("nan"))

        mean_errors, constant_errors = [], []
        if mean_target is not None:
            reference = mean_target.to(predicted.device)
            for row in range(gt_count):
                valid = chunk_valid[row]
                if not bool(valid.any()):
                    continue
                mean_errors.append(float((predicted[row, valid] - reference[valid]).abs().mean()))
                constant_errors.append(float((reference[valid] - own_target[row, valid]).abs().mean()))

        rows.append(
            {
                "scene_id": sample.scene_id,
                "gt_count": gt_count,
                "paired_future_l1": paired_l1,
                "future_cosine_similarity": float(np.mean(cosine_rows)) if cosine_rows else float("nan"),
                "future_relative_l1": (
                    paired_l1 / target_magnitude if target_magnitude > 0 else float("nan")
                ),
                "target_abs_mean": target_magnitude,
                "scene_mean_target_l1": anchor_l1,
                "residual_l1": residual_l1 if residual_defined else float("nan"),
                "residual_relative_l1": residual_l1 / anchor_l1 if residual_defined else float("nan"),
                "residual_cos_future": float(np.mean(residual_cosines)) if residual_cosines else float("nan"),
                **{f"frame{frame}_gain_over_scene_mean": value for frame, value in enumerate(frame_gains, start=1)},
                "constant_target_l1": float(np.mean(constant_errors)) if constant_errors else float("nan"),
                "shuffled_pair_l1": (
                    weighted_error(predicted, torch.roll(own_target, -1, dims=0)) if gt_count > 1 else float("nan")
                ),
                "trajectory_blind_l1": (
                    weighted_error(blind_predicted, own_target) if gt_count > 1 else float("nan")
                ),
                "scene_blind_l1": scene_blind,
                "distance_to_mean_target_l1": float(np.mean(mean_errors)) if mean_errors else float("nan"),
                "register_response_to_trajectory": float(np.mean(response)) if response else float("nan"),
                "target_spread_across_gt": float(np.mean(spread)) if spread else float("nan"),
                "trajectory_conditioning_fidelity": (
                    float(np.sum(response) / np.sum(spread)) if spread and float(np.sum(spread)) > 0 else float("nan")
                ),
                "trajectory_response_alignment": float(np.mean(alignment)) if alignment else float("nan"),
                "register_std_across_candidates": float(predicted.std(dim=0).mean()),
                "register_abs_mean": float(predicted.abs().mean()),
                "gt_trajectory_spread_ade_m": trajectory_spread(source[0].cpu()),
            }
        )
    model.train(was_training)

    keys = [key for key in rows[0] if key != "scene_id"]
    summary = {f"mean_{key}": float(np.nanmean([row[key] for row in rows])) for key in keys}
    summary["scenes"] = rows
    return summary


@torch.no_grad()
def streaming_future_alignment(predicted: torch.Tensor, targets: PairTargets) -> dict[str, float]:
    """The two "how far from the real future" readings, vectorized over a whole batch.

    ``future_fit_metrics`` produces the same two numbers but loops over scenes in
    Python and re-runs the model's forward pass for every control, which is far too
    expensive to call on each training micro-step.  This is the training-loop
    readout: it reuses the ``predicted`` register the loss already computed and
    returns ``future_cosine_similarity`` (the per-valid-token cosine between
    prediction and target, averaged the same way -- token, then GT row, then scene)
    and ``future_relative_l1`` (the quality-weighted paired L1 divided by the
    target's own quality-weighted magnitude, so 1.0 is exactly "as wrong as
    predicting zero").

    The averaging order is what makes these comparable to the verdict numbers rather
    than merely similar: the cosine is scene-balanced and GT-row-balanced and, as in
    ``future_fit_metrics``, uses only ``future_chunk_valid`` -- *not* the quality
    weight -- while the relative L1 uses both.  Mixing the two masks would let a
    handful of high-quality tokens in one scene dominate a step reading.

    Returns 0.0 rather than NaN for both when the batch holds no valid token, because
    the trainer all-reduces these across ranks and one NaN would poison every rank's
    reading for that step.  Callers can detect the empty case via ``valid_scenes``.
    """

    target = targets.future_register
    if predicted.shape != target.shape:
        raise ValueError(
            f"predicted register {tuple(predicted.shape)} does not match target {tuple(target.shape)}"
        )

    pred = predicted.float()
    own_target = target.to(pred.device, pred.dtype)
    chunk_valid = targets.future_chunk_valid.to(pred.device).bool()
    token_valid = targets.gt_valid.to(pred.device).bool().unsqueeze(-1) & chunk_valid
    weights = targets.future_quality.to(pred.device).float() * token_valid.to(pred.dtype)

    cosine = torch.nn.functional.cosine_similarity(pred, own_target, dim=-1)
    token_mask = token_valid.to(pred.dtype)
    per_row = (cosine * token_mask).sum(dim=-1) / token_mask.sum(dim=-1).clamp_min(1.0)
    row_mask = token_valid.any(dim=-1).to(pred.dtype)
    per_scene_cos = (per_row * row_mask).sum(dim=-1) / row_mask.sum(dim=-1).clamp_min(1.0)

    error = (pred - own_target).abs().mean(dim=-1)
    numerator = (error * weights).sum(dim=(1, 2))
    magnitude = (own_target.abs().mean(dim=-1) * weights).sum(dim=(1, 2))
    per_scene_rel = numerator / magnitude.clamp_min(1e-8)
    has_scene = row_mask.any(dim=-1).to(pred.dtype)
    has_magnitude = (has_scene > 0) & (magnitude > 0)
    rel_mask = has_magnitude.to(pred.dtype)

    # The background-removed twin, cheap enough to log every step.  Subtracting each
    # side's own per-scene mean leaves only what the trajectory could have caused, so
    # 1.0 is exactly "reads the scene, ignores the trajectory" -- the reading the raw
    # cosine cannot give because the scene mean carries 59.8% of the L1 objective.
    scene_weight = weights.sum(dim=1).clamp_min(1e-8).unsqueeze(-1)
    pred_residual = pred - (pred * weights.unsqueeze(-1)).sum(dim=1, keepdim=True) / scene_weight.unsqueeze(1)
    target_residual = own_target - (own_target * weights.unsqueeze(-1)).sum(dim=1, keepdim=True) / scene_weight.unsqueeze(1)
    residual_error = (pred_residual - target_residual).abs().mean(dim=-1)
    residual_numerator = (residual_error * weights).sum(dim=(1, 2))
    residual_magnitude = (target_residual.abs().mean(dim=-1) * weights).sum(dim=(1, 2))
    per_scene_residual_rel = residual_numerator / residual_magnitude.clamp_min(1e-8)
    # Degeneracy is decided by the *row count*, not by a magnitude threshold.  A scene
    # with one valid GT row has a scene mean equal to that row, so both residuals are
    # zero in exact arithmetic -- but the mean is formed as ``(x*w)/w``, which leaves
    # ~1e-7 of rounding, and a ``> 0`` test then admits a scene whose cosine is pure
    # float noise (measured 0.0547 against 0.82 for its siblings, enough to move a
    # four-scene average by 0.19).  ``future_fit_metrics`` guards the same case with
    # ``gt_count > 1``; this is its vectorized equivalent.
    residual_mask = (token_valid.any(dim=-1).sum(dim=-1) >= 2) & (residual_magnitude > 1e-6)

    residual_cosine = torch.nn.functional.cosine_similarity(pred_residual, target_residual, dim=-1)
    residual_per_row = (residual_cosine * token_mask).sum(dim=-1) / token_mask.sum(dim=-1).clamp_min(1.0)
    # Rows collapse with ``row_mask`` ([B,K]); the scene-level mask is [B] and only
    # applies when scenes collapse.  Using the [B] mask here instead would broadcast
    # against the K axis, which is silent whenever B == K and wrong always.
    per_scene_residual_cos = (residual_per_row * row_mask).sum(dim=-1) / row_mask.sum(dim=-1).clamp_min(1.0)
    # A scene whose residual magnitude is zero -- a single valid GT row, whose scene
    # mean *is* its only target -- has an undefined residual cosine rather than a bad
    # one, so it drops out of both residual readings instead of reading as failure.
    residual_scene_mask = residual_mask.to(pred.dtype)

    return {
        "future_cosine_similarity": float((per_scene_cos * has_scene).sum() / has_scene.sum().clamp_min(1.0)),
        "future_relative_l1": float((per_scene_rel * rel_mask).sum() / rel_mask.sum().clamp_min(1.0)),
        "residual_relative_l1": float(
            (per_scene_residual_rel * residual_scene_mask).sum() / residual_scene_mask.sum().clamp_min(1.0)
        ),
        "residual_cos_future": float(
            (per_scene_residual_cos * residual_scene_mask).sum() / residual_scene_mask.sum().clamp_min(1.0)
        ),
        "valid_scenes": int(has_scene.sum()),
    }


def register_verdict(train: dict[str, Any], evaluation: dict[str, Any] | None, margin: float = 0.02) -> dict[str, Any]:
    """Turn the controls into a single honest statement about the Future Register.

    A margin is required because "paired < baseline" by a fraction of a percent is
    what an untrained register also produces; the earlier four-scene run sat at 1.8%
    and was not evidence of scene-conditional prediction.

    Every ``gain_over_*`` here compares ``mean_paired_future_l1`` against a
    model-independent baseline or a deliberately degraded re-run of the same model,
    never against ``distance_to_mean_target_l1``.  That distance is a collapse
    detector -- how far the register drifted toward the dataset average -- and is
    reported separately as ``collapse_toward_mean``, the ratio of that drift to the
    paired error.  The old ``gain_over_mean_predictor`` compared against it and was
    meaningless as a skill claim.

    ``gain_over_scene_mean_target`` carries the Stage-1 claim: the scene's own
    per-token average future is what a register that read the scene perfectly and
    the trajectory not at all would output, so beating it is the only evidence that
    the trajectory input changed the prediction for the better.  The two
    scale-free numbers must agree with it -- a register can lower its L1 by
    shrinking toward the average, which shows up as fidelity below 1 and alignment
    below 0 even when the paired error falls.
    """

    gains = (
        "scene_mean_target_l1",
        "constant_target_l1",
        "scene_blind_l1",
        "shuffled_pair_l1",
        "trajectory_blind_l1",
    )

    def beats(metrics: dict[str, Any], key: str) -> float:
        paired = float(metrics["mean_paired_future_l1"])
        baseline = float(metrics[f"mean_{key}"])
        return (baseline - paired) / baseline if baseline > 0 else float("nan")

    def collapse(metrics: dict[str, Any]) -> float:
        paired = float(metrics["mean_paired_future_l1"])
        drift = float(metrics["mean_distance_to_mean_target_l1"])
        return drift / paired if paired > 0 else float("nan")

    def describe(metrics: dict[str, Any], prefix: str) -> dict[str, Any]:
        reported: dict[str, Any] = {
            f"{prefix}_gain_over_{key.removesuffix('_l1')}": beats(metrics, key) for key in gains
        }
        reported[f"{prefix}_collapse_toward_mean"] = collapse(metrics)
        reported[f"{prefix}_trajectory_conditioning_fidelity"] = float(metrics["mean_trajectory_conditioning_fidelity"])
        reported[f"{prefix}_trajectory_response_alignment"] = float(metrics["mean_trajectory_response_alignment"])
        # Reported, not gated: the cosine is the direct "how much of the real future is
        # in the prediction" reading (FastWAM trains 1 - cosine at weight 0.1) and the
        # relative L1 makes the standardized-space error dimensionless.  relative_l1 does
        # carry one calibrated bar by construction -- 1.0 is the all-zero predictor --
        # but "good enough" is anchored by scene_mean_target_l1 below, not by these.
        reported[f"{prefix}_future_cosine_similarity"] = float(metrics["mean_future_cosine_similarity"])
        reported[f"{prefix}_future_relative_l1"] = float(metrics["mean_future_relative_l1"])
        # The background-removed pair, and the ones to read first.  The scene mean of the
        # target carries 59.8% of the L1 objective (measured on 400 cached scenes: 0.792
        # predicting zero -> 0.318 predicting the scene mean), so the raw cosine above is
        # dominated by a direction a scene reader gets for free -- the retired 8-token run
        # read cos 0.6555 while both gains were negative.  ``residual_relative_l1`` is
        # calibrated where cosine is not: 1.0 is exactly "reads the scene, ignores the
        # trajectory", 0.0 is perfect, and above 1.0 the trajectory made things worse.
        reported[f"{prefix}_residual_relative_l1"] = float(metrics["mean_residual_relative_l1"])
        reported[f"{prefix}_residual_cos_future"] = float(metrics["mean_residual_cos_future"])
        # Named for the baseline it actually gates on.  It used to be called
        # beats_trajectory_blind_baseline, which read as a claim about
        # gain_over_trajectory_blind -- a different, weaker control -- and reported
        # True for a register that was trajectory-blind by every other measure.
        # scene_mean_target_l1 is the tighter baseline: beating it is what proves the
        # trajectory input earned its place.
        reported[f"{prefix}_beats_scene_mean_baseline"] = bool(
            float(reported[f"{prefix}_gain_over_scene_mean_target"]) >= margin
        )
        reported[f"{prefix}_learned_trajectory_conditional_future"] = bool(
            reported[f"{prefix}_beats_scene_mean_baseline"]
            and all(float(reported[f"{prefix}_gain_over_{key.removesuffix('_l1')}"]) >= margin for key in gains)
            and float(reported[f"{prefix}_trajectory_response_alignment"]) >= 0.5
            and float(reported[f"{prefix}_trajectory_conditioning_fidelity"]) >= 0.25
        )
        return reported

    verdict: dict[str, Any] = {"margin_required": margin}
    verdict.update(describe(train, "train"))
    if evaluation is not None:
        verdict.update(describe(evaluation, "eval"))
        verdict["generalizes_beyond_fitted_scenes"] = bool(verdict["eval_learned_trajectory_conditional_future"])
    return verdict
