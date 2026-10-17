#!/usr/bin/env python
"""Re-run the Stage 2 evaluation-path probe and print a PASS/FAIL table.

The contract suite cannot reach two things: the production 672x1148 forward, and
``run_navtrain_wan_e2e_ddp.py``'s own ``JointStage2Module.inference`` /
``save_checkpoint`` / ``load_joint_checkpoint``.  Those are the pieces an evaluator
actually runs, so this script is what the numbers quoted in ``AGENTS.md`` come from.

Nothing here touches NAVSIM data or the metric cache: the observation is synthetic
and the model is at scratch init, so the trajectory assertions are about *path
identity* (eval == train, no GT leak, bitwise reload), never about quality.  Do not
read any number below as a PDMS or ADE claim.

    source /usr/local/PPU_SDK/envsetup.sh ppu
    export PYTHONPATH=<repo>/src:<repo>/vendor/DrivoR:<repo>/vendor/DrivoR/nuplan-devkit
    python scripts/probe_stage2_eval_path.py
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import run_navtrain_wan_e2e_ddp as trainer  # noqa: E402
from nvs_v6_wan_e2e.config import EXPECTED_REGISTER_TARGET_DIM, EXPECTED_REGISTER_TOKENS  # noqa: E402
from nvs_v6_wan_e2e.data import Observation  # noqa: E402
from nvs_v6_wan_e2e.modes import ImmutableModeBuffer, StructuredSourceInitializer  # noqa: E402
from nvs_v6_wan_e2e.vision import DRIVOR_OFFICIAL_IMAGE_HW  # noqa: E402
from nvs_v6_wan_e2e.wan_dit import NvsActionDiT  # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(condition), detail))


class _StubOptimizer:
    """``save_checkpoint`` only calls ``state_dict()`` on the two optimizers."""

    def state_dict(self) -> dict:
        return {"state": {}, "param_groups": []}


def synthetic_observation(batch: int = 1, speed: float = 8.0) -> Observation:
    """DrivoR-resolution frames with a realistic observable speed.

    ``speed`` lands in ``ego_history[:, -1, 3]``, which is the vx the unicycle
    rollout reads; a random N(0,1) there is negative half the time and clamps to
    zero, which collapses the anchor grid and makes the spread assertion vacuous.
    """

    torch.manual_seed(3)
    ego = torch.randn(batch, 5, 7)
    ego[..., 3] = speed
    return Observation(
        rgb_history=torch.rand(batch, 5, 1, 3, *DRIVOR_OFFICIAL_IMAGE_HW),
        history_valid=torch.ones(batch, 5, 1, dtype=torch.bool),
        ego_history=ego,
        route_command=torch.tensor([[0.0, 1.0, 0.0, 0.0]] * batch),
        relative_ego_poses=torch.eye(4).expand(batch, 5, 4, 4).contiguous(),
    )


def probe_geometry() -> None:
    for kwargs, label in (({"target_dim": 128}, "target_dim=128"), ({"heads": 8}, "heads=8")):
        try:
            trainer.build_generator_and_scorer(**kwargs)
        except ValueError as error:
            check(f"wrong geometry rejected: {label}", True, str(error)[:90])
        else:
            check(f"wrong geometry rejected: {label}", False, "no ValueError raised")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", default="", help="where to write the round-trip checkpoint (a temp dir by default)")
    args = parser.parse_args()

    probe_geometry()

    # Seeded immediately before the build, not at the top of main(): probe_geometry
    # constructs two models of its own and would consume the stream otherwise.
    # Without this the scratch head is drawn from PyTorch's non-deterministic default
    # generator, and every reading that depends on it -- max|traj-anchor| in
    # particular -- becomes per-run noise instead of a regression signal.  Three
    # fresh unseeded builds measured future_head weight sums of +3.48 / -7.01 / -7.28.
    torch.manual_seed(0)
    # Built at the production split, not at the defaults: register_target_dim is the
    # Teacher's feature width (1024) while d_model stays 256, and a probe that let
    # both default would never exercise the widened future head -- the one thing the
    # split actually added.
    model, scorer = trainer.build_generator_and_scorer(
        predict_future=True, register_target_dim=EXPECTED_REGISTER_TARGET_DIM
    )
    model.eval(), scorer.eval()
    net = trainer.JointStage2Module(model, scorer)
    observation = synthetic_observation(1)
    codes = ImmutableModeBuffer().expand_for_batch(1)
    source = StructuredSourceInitializer(jitter_std=0.0).clean(observation.ego_history, codes)
    spread = float((source - source[:, :1]).abs().amax())
    print(f"anchor spread over {codes.shape[1]} modes (m/rad, vs mode 0): {spread:.4f}")
    check("the 64-mode anchor grid spans 38.4 m/rad", abs(spread - 38.4) < 1e-3, f"{spread:.4f}")

    # [scene | action | register], one mask for every entry point: Action never reads
    # Register, Register reads Action, Scene reads only itself.  Which mask an entry
    # point gets is the whole leakage contract, so it is recorded per call rather than
    # assumed -- the point of this probe is to catch a deployed path that silently ran
    # a different information flow from the one training used.
    seen: dict[str, object] = {}
    masks: dict[str, list] = {}
    current = "unset"

    def record_block(_module, args, kwargs):
        seen["block_kwargs"] = dict(kwargs)
        masks.setdefault(current, []).append(kwargs.get("self_attn_mask"))

    def record_self(_module, args, kwargs):
        seen["self_len"] = int(args[0].shape[1])
        seen["self_kwargs"] = dict(kwargs)
        masks.setdefault(current, []).append(kwargs.get("self_attn_mask"))

    def record_cross(_module, args, kwargs):
        seen["cross_len"] = int(args[1].shape[1])
        seen["cross_kwargs"] = dict(kwargs)

    handles = [
        model.theta.blocks[0].register_forward_pre_hook(record_block, with_kwargs=True),
        model.theta.blocks[0].self_attn.register_forward_pre_hook(record_self, with_kwargs=True),
        model.theta.blocks[0].cross_attn.register_forward_pre_hook(record_cross, with_kwargs=True),
    ]

    timestep = torch.full((1,), trainer.INFERENCE_SIGMA * model.flow.num_train_timesteps)
    try:
        with torch.no_grad():
            current = "forward_pairs"
            endpoint, trajectory, logits = net.inference(observation, source, codes)
            current = "forward_flow"
            flow = model.forward_flow(observation, source, codes, source, timestep)
            current = "forward_pairs"
            endpoint2, trajectory2, logits2 = net.inference(observation, source, codes)
            current = "forward_register_only"
            predicted_register = model.forward_register_only(observation, source, codes)
    finally:
        for handle in handles:
            handle.remove()

    sequence = model.scene_tokens + model.action_tokens + model.register_tokens
    # 81 + 8 + 32: the register segment is four DINOv3 prefix-query reads per future
    # frame, so the sequence is longer than the eight waypoints it predicts.  Spelled
    # from the production constant, not from a literal, so widening the readout moves
    # the expectation instead of quietly passing a stale one.
    expected_sequence = 81 + 8 + EXPECTED_REGISTER_TOKENS
    check(f"the production DiT sequence is [scene(81) | action(8) | register({EXPECTED_REGISTER_TOKENS})]",
          int(seen["self_len"]) == sequence == expected_sequence
          and int(seen["cross_len"]) == model.scene_tokens,
          f"self={seen['self_len']} cross={seen['cross_len']} expected={expected_sequence}")
    lengths = [model.scene_tokens, model.action_tokens, model.register_tokens]
    device = next(model.theta.parameters()).device
    table = model.theta.segment_attention_mask(lengths, device)

    def expand(visibility):
        segment_of_token = torch.cat([torch.full((length,), index, dtype=torch.long) for index, length in enumerate(lengths)])
        return torch.tensor(visibility, dtype=torch.bool)[segment_of_token[:, None], segment_of_token[None, :]]

    check("the one visibility table is the one the design review specifies",
          torch.equal(table, expand(NvsActionDiT.SEGMENT_VISIBILITY).to(device))
          and NvsActionDiT.SEGMENT_VISIBILITY == ((1, 0, 0), (1, 1, 0), (1, 1, 1))
          and NvsActionDiT.SEGMENT_VISIBILITY[1][2] == 0
          and NvsActionDiT.SEGMENT_VISIBILITY[2][1] == 1)
    check("the cross-attention context is the whole Scene and stays unmasked",
          seen["block_kwargs"].get("context_mask", None) is None
          and seen["cross_kwargs"].get("ctx_mask", None) is None,
          f"block={sorted(seen['block_kwargs'])} cross={sorted(seen['cross_kwargs'])}")
    for entry in ("forward_pairs", "forward_flow", "forward_register_only"):
        captured = masks.get(entry, [])
        check(f"{entry} runs the same mask on every theta forward",
              bool(captured)
              and all(mask is not None and mask.dtype is torch.bool and torch.equal(mask, table) for mask in captured),
              f"{len(captured)} forwards")
    action_slice = slice(model.scene_tokens, model.scene_tokens + model.action_tokens)
    register_slice = slice(model.scene_tokens + model.action_tokens, sequence)
    check("no deployed entry point lets Action read Register",
          all(not bool(mask[action_slice, register_slice].any())
              for captured in masks.values() for mask in captured if mask is not None))
    check("no Register -> Action bypass module survived on the model",
          not [name for name, _ in model.named_parameters() if name.startswith(("future_fuse", "ema_theta", "_ema_theta"))]
          and not hasattr(model, "enable_ema") and not hasattr(model, "update_ema"))
    # The register segment is four reads of one instant per frame, so its RoPE phase
    # must be the frame index repeated -- counting 0..31 would hand the trunk a
    # timestamp it can use to tell a frame's queries apart without reading anything.
    phases = model.theta.register_rope(model.register_tokens, device)
    per_frame = phases.reshape(-1, model.register_tokens // model.action_tokens, *phases.shape[1:])
    check("the register RoPE repeats one phase per future frame",
          model.register_tokens % model.action_tokens == 0
          and torch.equal(per_frame, per_frame[:, :1].expand_as(per_frame))
          and torch.equal(per_frame[:, 0], model.theta.rope_for(model.action_tokens, device)))
    # Compared against the production constant rather than against whatever the model
    # happens to hold, so a probe that silently built the pre-split geometry fails here
    # instead of reporting (1, 64, 32, 256) as a pass.
    check("Stage 1 returns one register token per query per future frame at the Teacher width",
          tuple(predicted_register.shape)[-2:] == (model.register_tokens, EXPECTED_REGISTER_TARGET_DIM)
          and model.theta.target_dim != model.d_model,
          f"{tuple(predicted_register.shape)} vs d_model={model.d_model}")
    check("the register read-out reads the trunk at d_model and writes the Teacher width",
          tuple(model.theta.future_head[1].weight.shape) == (EXPECTED_REGISTER_TARGET_DIM, model.d_model),
          f"future_head[1]={tuple(model.theta.future_head[1].weight.shape)}")

    check("inference == forward_flow(data=anchor, sigma=1) bitwise",
          torch.equal(endpoint, flow.pairs.endpoint) and torch.equal(trajectory, flow.pairs.trajectory))
    check("inference is deterministic across calls",
          torch.equal(endpoint, endpoint2) and torch.equal(trajectory, trajectory2) and torch.equal(logits, logits2))

    moved = float((trajectory - source).abs().amax())
    check("a scratch head stays near the anchor instead of collapsing", 0.0 < moved < 1.0,
          f"max|traj-anchor|={moved:.4f} m/rad at HEAD_INIT_STD=1e-3/33.37")

    # ``v_target`` is ``norm(anchor) - norm(data)`` and carries no sigma, so at the
    # inference setting (data=anchor) it must vanish, while a real GT still produces
    # a nonzero target at the same sigma.
    check("sigma=1 velocity target is exactly zero", float(flow.velocity_target.abs().max()) == 0.0,
          f"max|v_target|={float(flow.velocity_target.abs().max()):.3e}")
    with torch.no_grad():
        ground_truth = torch.randn_like(source)
        noised, v_target_gt = model.flow.noisy_action(source, ground_truth, timestep)
    leak = float((noised - source).abs().max())
    check("sigma=1 noisy action == anchor (no GT leak)", leak < 1e-5, f"max|diff|={leak:.3e}")
    check("v_target is sigma-independent, so a real GT still supervises at sigma=1",
          float(v_target_gt.abs().max()) > 0.0, f"max|v_target(gt)|={float(v_target_gt.abs().max()):.3f}")

    workdir = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(prefix="nvs_probe_"))
    workdir.mkdir(parents=True, exist_ok=True)
    try:
        config = argparse.Namespace(
            depth=8, heads=4, target_dim=256, register_target_dim=EXPECTED_REGISTER_TARGET_DIM,
            dino_checkpoint=trainer.DEFAULT_DINO, wan_backbone=trainer.DEFAULT_WAN_BACKBONE,
            stage1_checkpoint="", future_target="none", lambda_future=0.0,
        )
        trainer.save_checkpoint(net, model, scorer, _StubOptimizer(), _StubOptimizer(), 0, 7, config,
                                workdir, trainer.Dist(0, 1, 0, torch.device("cpu"), False))
        payload = torch.load(workdir / "checkpoint.pt", map_location="cpu", weights_only=True)
        check("checkpoint records predict_future from the model", payload.get("predict_future") is True,
              str(payload.get("predict_future")))
        check("checkpoint config is a weights_only-safe dict", isinstance(payload["config"], dict),
              f"{len(payload['config'])} keys")
        check("the payload carries no retired bypass tensors, so strict=True still loads it",
              "ema_theta" not in payload
              and not any(key.startswith(("future_fuse", "ema_theta", "_ema_theta")) for key in payload["model"]),
              f"model={len(payload['model'])} tensors, scorer={len(payload['scorer'])} tensors")

        _, reloaded_model, reloaded_scorer, meta = trainer.load_joint_checkpoint(
            workdir / "checkpoint.pt", device=torch.device("cpu")
        )
        reloaded_model.eval(), reloaded_scorer.eval()
        with torch.no_grad():
            back = trainer.JointStage2Module(reloaded_model, reloaded_scorer).inference(observation, source, codes)
        check("load_joint_checkpoint reproduces inference bitwise",
              torch.equal(back[0], endpoint) and torch.equal(back[1], trajectory) and torch.equal(back[2], logits))
        check("load_joint_checkpoint reports the single segment mask it will run",
              meta.get("self_attention_mask") == "single_segment_block"
              and meta.get("segment_visibility") == [[1, 0, 0], [1, 1, 0], [1, 1, 1]],
              f"visibility={meta.get('segment_visibility')}")
        check("load_joint_checkpoint resolves the trained sequence length",
              reloaded_model.predict_future is True
              and int(meta["dit_sequence_tokens"]) == expected_sequence
              and reloaded_model.scene_tokens + reloaded_model.action_tokens + reloaded_model.register_tokens
              == expected_sequence,
              f"opt_step={meta.get('opt_step')} dit_sequence_tokens={meta.get('dit_sequence_tokens')}")

        legacy = dict(payload)
        legacy.pop("predict_future")
        legacy["config"] = {**payload["config"], "stage1_checkpoint": "/nonexistent/stage1.pt"}
        torch.save(legacy, workdir / "legacy.pt")
        _, legacy_model, _, _ = trainer.load_joint_checkpoint(workdir / "legacy.pt", device=torch.device("cpu"))
        check("a pre-predict_future checkpoint falls back to stage1_checkpoint",
              legacy_model.predict_future is True)
    finally:
        if not args.workdir:
            shutil.rmtree(workdir, ignore_errors=True)

    print()
    for name, ok, detail in RESULTS:
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))
    passed = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"\n{passed}/{len(RESULTS)} passed")
    return 0 if passed == len(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
