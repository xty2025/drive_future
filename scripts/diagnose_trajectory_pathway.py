#!/usr/bin/env python
"""Measure how much of the Stage-1 register's input actually depends on the trajectory.

The 400-step, 8-scene overfit produced a register that predicts the future well and
ignores which future: ``paired_l1``, ``trajectory_blind_l1`` and ``shuffled_pair_l1``
agree to four decimals while ``scene_blind_l1`` is 3x worse.  That is a scene-
conditional, trajectory-blind predictor.  This script answers the structural question
behind it -- how much trajectory signal is even present at each hop of the path the
register has to read it through -- so the fix can be aimed rather than guessed at.

The path is fixed by the architecture.  ``encode_future`` deliberately takes no
trajectory: register tokens are ``future_queries + ego``, identical across the K GT
rows of one scene, so the *only* route by which row k can learn it is row k's GT is
self-attention onto the action segment.  Three things can therefore starve it, and
each is measured here rather than assumed:

1. **The action token barely varies across GT rows.**  It is a sum of four terms and
   three of them -- ``ego``, ``time_queries``, ``mode`` -- are constant across rows,
   leaving only ``action_encoder(norm_odo(source))`` to carry the difference.  If the
   constant part dominates, the K rows present near-identical keys and there is
   nothing to attend *to*.  Reported as ``action_cross_gt_rms_fraction``.
2. **The register spends its attention mass on the scene.**  81 scene tokens against
   8 action tokens, and the scene is the high-variance DINOv3 feature while the action
   term is held at Wan's latent magnitude by design.  Reported per block as the mass
   landing on each segment.
3. **The register output varies less across rows than its target does.**  The end-to-
   end version of 1 and 2: whatever survives the path, compared against the spread the
   loss is asking for.  Reported as ``register_cross_gt_rms`` next to
   ``target_cross_gt_rms``.

Run it against a Stage-1 checkpoint and against a fresh model; the difference is what
400 steps actually bought.  ``--json`` writes the numbers for the run log.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

from nvs_v6_wan_e2e.data import FUTURE_FRAMES
from nvs_v6_wan_e2e.flow import norm_odo
from nvs_v6_wan_e2e.model import DirectedPairDiT
from nvs_v6_wan_e2e.target_cache import (
    Stage1SceneDataset,
    collate_stage1_batch,
    load_standardizer,
    observation_to_device,
    ready_scenes_path,
    targets_to_device,
)
from nvs_v6_wan_e2e.vision import DRIVOR_LORA_RANK, DRIVOR_NUM_SCENE_TOKENS
from nvs_v6_wan_e2e.wan_dit import detrend_trajectory, rope_apply

DEFAULT_DINO_CHECKPOINT = "/mnt/LLM_weight/timm/vit_large_patch16_dinov3.lvd1689m/model.safetensors"
DEFAULT_WAN_BACKBONE = "checkpoints/ActionDiT_Wan22_256hdim_3outdim.pt"


def rms(x: torch.Tensor) -> float:
    """Root mean square over every element -- the scale contract's own unit."""
    return float(x.detach().double().pow(2).mean().sqrt())


def cross_gt_rms(x: torch.Tensor) -> float:
    """RMS of the across-GT-row std of ``[B, M, ...]``.

    Zero means the tensor is identical for every GT of a scene, i.e. it carries no
    information about *which* future is being asked for.  This is the quantity the
    whole diagnosis turns on, so it is measured on the actual activations rather than
    reasoned about from the parameter shapes.
    """
    if x.shape[1] < 2:
        return float("nan")
    return float(x.detach().double().std(dim=1, unbiased=False).pow(2).mean().sqrt())


def decompose_action_tokens(model: DirectedPairDiT, observation, source, modes) -> dict[str, Any]:
    """Split ``encode_action``'s sum into its four terms and measure each.

    Recomputes the terms rather than reading them off a hook: the sum is built inline
    in ``encode_action``, so a hook would only see the result, and the result is
    exactly what cannot distinguish "the trajectory term is small" from "the constant
    terms are large".

    Recomputing is also how this went stale once already: ``action_encoder`` gained a
    second channel (the detrended shape term) and this kept feeding it one, which
    failed as a ``mat1 and mat2 shapes cannot be multiplied``.  The comparison against
    ``encode_action`` at the end is what makes a second drift loud instead of silent.
    """
    theta = model.theta
    with torch.no_grad():
        clean_ego = model.eta.encode_clean_ego(observation)
        batch = source.shape[0]
        shape = detrend_trajectory(source) / theta.shape_range.to(device=source.device, dtype=source.dtype)
        trajectory = theta.action_encoder(torch.cat([norm_odo(source), shape], dim=-1))
        mode = theta._mode_features(modes).unsqueeze(2)
        ego = theta._ego_features(clean_ego).view(batch, 1, 1, theta.hidden_dim)
        time_queries = theta.time_queries.view(1, 1, theta.action_tokens, theta.hidden_dim)
        ego = ego.expand_as(trajectory)
        mode = mode.expand_as(trajectory)
        time_queries = time_queries.expand_as(trajectory)
        total = trajectory + ego + time_queries + mode

    terms = {"trajectory": trajectory, "ego": ego, "time_queries": time_queries, "mode": mode}
    constant = ego + time_queries + mode
    report: dict[str, Any] = {
        f"{name}_rms": rms(value) for name, value in terms.items()
    } | {
        f"{name}_cross_gt_rms": cross_gt_rms(value) for name, value in terms.items()
    }
    report["constant_part_rms"] = rms(constant)
    report["action_token_rms"] = rms(total)
    report["action_cross_gt_rms"] = cross_gt_rms(total)
    # The one number that says whether hop 1 is starved: the share of the action
    # token that differs between two GTs of the same scene.  A value near zero means
    # the K rows hand the register near-identical keys to attend to.
    report["action_cross_gt_rms_fraction"] = (
        report["action_cross_gt_rms"] / report["action_token_rms"] if report["action_token_rms"] > 0 else float("nan")
    )
    report["trajectory_to_constant_rms_ratio"] = (
        report["trajectory_rms"] / report["constant_part_rms"] if report["constant_part_rms"] > 0 else float("nan")
    )
    # The three constant terms must come out exactly zero here.  They are not asserted
    # away, because a nonzero would mean something per-row leaked into the
    # conditioning -- the shortcut ``encode_future``'s docstring says was removed.
    report["constant_terms_cross_gt_rms"] = {
        name: cross_gt_rms(value) for name, value in (("ego", ego), ("time_queries", time_queries), ("mode", mode))
    }
    with torch.no_grad():
        reference = theta.encode_action(source, modes, clean_ego)
    if not torch.allclose(total, reference, atol=1e-5, rtol=1e-5):
        raise AssertionError(
            "decompose_action_tokens no longer reproduces encode_action "
            f"(max abs diff {float((total - reference).abs().amax()):.3e}); the four terms "
            "it reports are not the four terms the model adds"
        )
    return report


class AttentionMassRecorder:
    """Re-derive each block's attention probabilities and bucket them by segment.

    ``SelfAttention.forward`` calls ``F.scaled_dot_product_attention``, which returns
    no weights, so the probabilities are recomputed from the same ``q``/``k`` the block
    itself builds -- same RoPE, same RMSNorm, same scale, same boolean mask -- and the
    block's real forward is then left to run untouched.  Recomputing is what keeps this
    honest: a separately constructed attention would be measuring a different function
    than the one that produced the register.
    """

    def __init__(self, theta: Any, segment_slices: dict[str, slice]) -> None:
        self.records: list[torch.Tensor] = []
        self.segment_slices = segment_slices
        self._saved: list[tuple[Any, Any]] = []
        for block in theta.blocks:
            attention = block.self_attn
            self._saved.append((attention, attention.forward))
            attention.forward = self._patched(attention, attention.forward)

    def _patched(self, attention: Any, original: Any) -> Any:
        def patched(x: torch.Tensor, freqs: torch.Tensor, self_attn_mask=None) -> torch.Tensor:
            with torch.no_grad():
                q = rope_apply(attention.norm_q(attention.q(x)), freqs, attention.num_heads)
                k = rope_apply(attention.norm_k(attention.k(x)), freqs, attention.num_heads)
                qs, ks = attention._split(q), attention._split(k)
                logits = torch.matmul(qs, ks.transpose(-1, -2)) / math.sqrt(qs.shape[-1])
                if self_attn_mask is not None:
                    logits = logits.masked_fill(~self_attn_mask, float("-inf"))
                self.records.append(logits.softmax(dim=-1).float().cpu())
            return original(x, freqs, self_attn_mask)

        return patched

    def close(self) -> None:
        for attention, original in self._saved:
            attention.forward = original
        self._saved.clear()

    def register_mass_by_segment(self, register_slice: slice) -> list[dict[str, float]]:
        """Per block, where the register rows' attention mass lands.

        Averaged over batch, heads and the register's own rows, so one number per
        segment per block.  Sums to 1 up to the masked-out segments, which the stage-1
        table closes for real: register can read all three, action cannot read
        register, scene reads only itself.
        """
        report = []
        for probs in self.records:
            rows = probs[:, :, register_slice, :]
            entry = {
                name: float(rows[:, :, :, span].sum(dim=-1).mean())
                for name, span in self.segment_slices.items()
            }
            entry["max_single_key"] = float(rows.max())
            report.append(entry)
        return report


def measure(model: DirectedPairDiT, observation, source, modes, targets) -> dict[str, Any]:
    device = next(model.parameters()).device
    theta = model.theta
    scene_tokens = model.scene_tokens
    action_tokens = model.action_tokens
    register_tokens = model.register_tokens
    total = scene_tokens + action_tokens + register_tokens
    slices = {
        "scene": slice(0, scene_tokens),
        "action": slice(scene_tokens, scene_tokens + action_tokens),
        "register": slice(scene_tokens + action_tokens, total),
    }

    action_report = decompose_action_tokens(model, observation, source, modes)

    recorder = AttentionMassRecorder(theta, slices)
    try:
        with torch.no_grad():
            predicted = model.forward_register_only(observation, source, modes)
            with torch.no_grad():
                clean_ego = model.eta.encode_clean_ego(observation)
                action_tokens_out = theta.encode_action(source, modes, clean_ego)
                register_in = theta.encode_future(action_tokens_out, clean_ego)
    finally:
        recorder.close()

    register_report = {
        "register_input_rms": rms(register_in),
        "register_input_cross_gt_rms": cross_gt_rms(register_in),
        # The register input is ``future_queries + ego`` and neither term depends on
        # the row, so this must be exactly zero.  It is measured, not asserted, because
        # it is the premise of the whole diagnosis: if it were nonzero the register
        # would have a per-row shortcut and the attention numbers below would not be
        # the binding constraint.
        "register_input_is_row_constant": bool(cross_gt_rms(register_in) == 0.0),
        "register_output_rms": rms(predicted),
        "register_output_cross_gt_rms": cross_gt_rms(predicted),
        "target_cross_gt_rms": cross_gt_rms(targets.future_register),
    }
    register_report["register_to_target_cross_gt_ratio"] = (
        register_report["register_output_cross_gt_rms"] / register_report["target_cross_gt_rms"]
        if register_report["target_cross_gt_rms"] > 0
        else float("nan")
    )
    return {
        "action_tokens": action_report,
        "register": register_report,
        "attention_mass_by_block": recorder.register_mass_by_segment(slices["register"]),
        "scene_tokens": scene_tokens,
        "action_token_count": action_tokens,
        "register_token_count": register_tokens,
        "sequence_tokens": total,
        "gt_rows": int(source.shape[1]),
        "future_frames": int(FUTURE_FRAMES),
    }


def load_scene(dataset: Stage1SceneDataset, index: int, device: torch.device):
    observation, targets, scene_ids = collate_stage1_batch([dataset[index]])
    observation = observation_to_device(observation, device)
    targets = targets_to_device(targets, device)
    source = targets.gt_trajectory
    modes = torch.zeros((*source.shape[:2], 2), device=device, dtype=source.dtype)
    return observation, source, modes, targets, scene_ids[0]


def build_model(args: argparse.Namespace, register_target_dim: int, image_hw: tuple[int, int], device) -> DirectedPairDiT:
    model = DirectedPairDiT(
        d_model=args.target_dim,
        heads=args.heads,
        ffn_dim=4 * args.target_dim,
        depth=args.depth,
        register_target_dim=register_target_dim,
        num_scene_tokens=args.num_scene_tokens,
        dino_checkpoint=args.dino_checkpoint,
        dino_image_hw=image_hw,
        dino_lora_rank=args.lora_rank,
        dino_grad_checkpointing=False,
        wan_backbone_path=args.wan_backbone if Path(args.wan_backbone).exists() else None,
    ).to(device)
    # The Stage-1 freeze lives in ``build_optimizers``; this script only ever runs under
    # ``no_grad``, so the flags are set directly to keep the parameter count in the
    # report matching the trainer's.
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if args.checkpoint:
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(payload["model"], strict=True)
    return model.eval()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target-cache-root", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path, default=None, help="Stage-1 payload; omit to measure a fresh model")
    parser.add_argument("--scenes", type=int, default=4, help="cached scenes to average over")
    parser.add_argument("--dino-checkpoint", default=DEFAULT_DINO_CHECKPOINT)
    parser.add_argument("--wan-backbone", default=DEFAULT_WAN_BACKBONE)
    parser.add_argument("--target-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--num-scene-tokens", type=int, default=DRIVOR_NUM_SCENE_TOKENS)
    parser.add_argument("--lora-rank", type=int, default=DRIVOR_LORA_RANK)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = json.loads((args.target_cache_root / "_manifest.json").read_text())
    image_hw = tuple(int(value) for value in manifest["image_hw"])
    register_target_dim = int(manifest["register_target_dim"])
    standardizer, _ = load_standardizer(args.target_cache_root)
    tokens = json.loads(ready_scenes_path(args.target_cache_root).read_text())["tokens"]
    dataset = Stage1SceneDataset(
        tokens,
        args.target_cache_root,
        image_hw=image_hw,
        skip_degenerate=True,
        target_fingerprint=standardizer.fingerprint(),
    )

    model = build_model(args, register_target_dim, image_hw, device)
    per_scene = []
    for index in range(min(args.scenes, len(dataset))):
        observation, source, modes, targets, scene_id = load_scene(dataset, index, device)
        if source.shape[1] < 2:
            continue
        report = measure(model, observation, source, modes, targets)
        report["scene_id"] = scene_id
        per_scene.append(report)

    if not per_scene:
        raise SystemExit("no cached scene with >= 2 GT rows; nothing to diagnose")

    def mean(path: str) -> float:
        values = []
        for report in per_scene:
            node: Any = report
            for key in path.split("."):
                node = node[int(key)] if isinstance(node, list) else node[key]
            if isinstance(node, (int, float)) and not math.isnan(node):
                values.append(float(node))
        return sum(values) / len(values) if values else float("nan")

    summary = {
        "checkpoint": str(args.checkpoint) if args.checkpoint else "fresh",
        "scenes_measured": len(per_scene),
        "sequence_tokens": per_scene[0]["sequence_tokens"],
        "gt_rows": per_scene[0]["gt_rows"],
        "hop1_action_token_rms": mean("action_tokens.action_token_rms"),
        "hop1_action_cross_gt_rms": mean("action_tokens.action_cross_gt_rms"),
        "hop1_action_cross_gt_rms_fraction": mean("action_tokens.action_cross_gt_rms_fraction"),
        "hop1_trajectory_rms": mean("action_tokens.trajectory_rms"),
        "hop1_constant_part_rms": mean("action_tokens.constant_part_rms"),
        "hop1_trajectory_to_constant_rms_ratio": mean("action_tokens.trajectory_to_constant_rms_ratio"),
        "hop2_register_input_cross_gt_rms": mean("register.register_input_cross_gt_rms"),
        "hop2_register_input_is_row_constant": all(
            bool(report["register"]["register_input_is_row_constant"]) for report in per_scene
        ),
        "hop2_attention_mass_on_scene_mean_over_blocks": sum(
            mean(f"attention_mass_by_block.{block}.scene") for block in range(args.depth)
        )
        / args.depth,
        "hop2_attention_mass_on_action_mean_over_blocks": sum(
            mean(f"attention_mass_by_block.{block}.action") for block in range(args.depth)
        )
        / args.depth,
        "hop2_attention_mass_on_action_last_block": mean(f"attention_mass_by_block.{args.depth - 1}.action"),
        "hop2_attention_mass_on_action_first_block": mean("attention_mass_by_block.0.action"),
        "hop2_attention_max_single_key": max(
            mean(f"attention_mass_by_block.{block}.max_single_key") for block in range(args.depth)
        ),
        "hop3_register_output_rms": mean("register.register_output_rms"),
        "hop3_register_output_cross_gt_rms": mean("register.register_output_cross_gt_rms"),
        "hop3_target_cross_gt_rms": mean("register.target_cross_gt_rms"),
        "hop3_register_to_target_cross_gt_ratio": mean("register.register_to_target_cross_gt_ratio"),
        "per_block_attention_mass_on_action": [
            mean(f"attention_mass_by_block.{block}.action") for block in range(args.depth)
        ],
    }
    print(json.dumps(summary, indent=2))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"summary": summary, "per_scene": per_scene}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
