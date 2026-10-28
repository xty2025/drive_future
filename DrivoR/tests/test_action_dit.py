"""Numerical contracts for the DrivoR + Action-DiT variant (CPU, no oracle)."""

import ast
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from omegaconf import OmegaConf

from navsim.agents.drivoR_action_dit.action_dit import ActionDiT, SharedSceneCrossAttention
from navsim.agents.drivoR_action_dit.flow import (
    ContinuousFlowSchedule, denormalize_trajectory, normalize_trajectory,
)
from navsim.agents.drivoR_action_dit.loss import DrivoRActionDiTLoss
from navsim.agents.drivoR_action_dit import model as model_module


def tiny_dit(**kwargs):
    return ActionDiT(hidden_dim=8, ffn_dim=16, freq_dim=8, num_heads=2,
                     attn_head_dim=4, num_layers=2, **kwargs)


def tiny_config():
    root = Path(__file__).resolve().parents[1]
    base = OmegaConf.load(root / "navsim/planning/script/config/common/agent/drivoR.yaml")
    extra = OmegaConf.load(root / "navsim/planning/script/config/common/agent/drivoR_action_dit.yaml")
    cfg = OmegaConf.merge(base.config, extra.config)
    cfg.tf_d_model = 8
    cfg.tf_d_ffn = 16
    cfg.scorer_ref_num = 2
    cfg.proposal_num = 64
    cfg.action_dit.update(dict(hidden_dim=8, ffn_dim=16, freq_dim=8, num_heads=2,
                               attn_head_dim=4, num_layers=2, pretrained_path="", num_inference_steps=2))
    return cfg


class TinyImageEncoder(nn.Module):
    def __init__(self, config, image_size, output_dim):
        super().__init__()
        self.neck = nn.Linear(3, output_dim)

    def forward(self, images):
        return self.neck(images.mean(dim=(-1, -2)))


def tiny_model(monkeypatch):
    monkeypatch.setattr(model_module, "NativePatchImageEncoder", TinyImageEncoder)
    return model_module.DrivoRActionDiTModel(tiny_config())


def test_default_sensor_config_reads_only_current_front_image(tmp_path, monkeypatch):
    from navsim.common import dataclasses

    cfg = tiny_config()
    names = ("cam_f0", "cam_l0", "cam_l1", "cam_l2", "cam_r0", "cam_r1", "cam_r2", "cam_b0")
    sensors = dataclasses.SensorConfig(**{
        name: OmegaConf.to_object(cfg[name]) for name in (*names, "lidar_pc")
    })
    assert sensors.get_sensors_at_iteration(3) == ["cam_f0"]
    assert all(sensors.get_sensors_at_iteration(index) == [] for index in range(3))
    requested = []

    def read_image(root, path):
        requested.append(path)
        return np.zeros((5, 7, 3), dtype=np.uint8)

    records = {name.upper(): {
        "data_path": f"{name.upper()}/frame.jpg",
        "sensor2lidar_rotation": np.eye(3),
        "sensor2lidar_translation": np.zeros(3),
        "cam_intrinsic": np.eye(3),
        "distortion": np.zeros(5),
    } for name in names}
    monkeypatch.setattr(dataclasses, "load_camera_image", read_image)
    cameras = dataclasses.Cameras.from_camera_dict(tmp_path, records, sensors.get_sensors_at_iteration(3))
    assert requested == ["CAM_F0/frame.jpg"]
    assert cameras.cam_f0.image is not None
    assert all(getattr(cameras, name).image is None for name in names if name != "cam_f0")


def test_absolute_normalization_constants_and_unclipped_roundtrip():
    low = torch.tensor([-1.57, -19.68, -1.67])
    high = low + torch.tensor([66.74, 42., 3.53])
    torch.testing.assert_close(normalize_trajectory(low), -torch.ones(3))
    torch.testing.assert_close(normalize_trajectory(high), torch.ones(3))
    trajectory = torch.tensor([[120., -40., 2.7], [0., 0., 0.]])
    normalized = normalize_trajectory(trajectory)
    assert normalized[0, 0] > 1 and normalized[0, 1] < -1
    torch.testing.assert_close(denormalize_trajectory(normalized), trajectory, atol=1e-5, rtol=1e-6)


def test_shifted_time_is_shared_across_candidates_and_has_reference_direction():
    schedule = ContinuousFlowSchedule()
    torch.manual_seed(17)
    actual = schedule.sample_timesteps(3, 64, "cpu")
    torch.manual_seed(17)
    u = torch.rand(3, 1)
    torch.testing.assert_close(actual, (1000 * 5 * u / (1 + 4 * u)).expand(3, 64))
    clean, noise = torch.randn(3, 64, 8, 3), torch.randn(3, 64, 8, 3)
    torch.testing.assert_close(schedule.add_noise(clean, noise, torch.zeros(3, 64)), clean)
    torch.testing.assert_close(schedule.add_noise(clean, noise, torch.full((3, 64), 1000.)), noise)
    noisy = schedule.add_noise(clean, noise, actual)
    # Above the loss floor, conversion gives the physical velocity noise-clean.
    torch.testing.assert_close(schedule.x_to_v(clean, noisy, actual), noise - clean, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("time", [0., 0.001, 1., 99., 100., 500., 1000.])
def test_x_to_v_loss_is_zero_for_perfect_prediction_even_below_floor(time):
    schedule = ContinuousFlowSchedule(sigma_min=0.1)
    clean = torch.randn(2, 64, 8, 3)
    times = torch.full((2, 64), time)
    noisy = schedule.add_noise(clean, torch.randn_like(clean), times)
    pred = clean.clone().requires_grad_()
    loss, predicted_v, target_v = schedule.velocity_loss(pred, clean, noisy, times)
    assert loss.item() == 0
    torch.testing.assert_close(predicted_v, target_v, rtol=0, atol=0)
    loss.backward()
    assert torch.isfinite(pred.grad).all()
    assert torch.count_nonzero(pred.grad) == 0


def test_velocity_error_matches_reference_weighted_x_error_and_stays_finite():
    schedule = ContinuousFlowSchedule(sigma_min=0.1)
    times = torch.tensor([[0.001, 1., 50., 500.]])
    clean = torch.randn(1, 4, 8, 3)
    noisy = schedule.add_noise(clean, torch.randn_like(clean), times)
    pred = (clean + 0.1).requires_grad_()
    loss, _, _ = schedule.velocity_loss(pred, clean, noisy, times)
    expected = (((clean - pred).square().mean((-2, -1)) /
                 schedule.sigma(times).clamp_min(0.1).square()) * schedule.training_weight(times)).mean()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(pred.grad).all()
    assert pred.grad.abs().max() < 2


def test_shared_cross_attention_equals_independent_candidate_calls_and_projects_kv_once():
    attn = SharedSceneCrossAttention(8, 2, 4, 1e-6)
    action = torch.randn(2, 4, 8, 8)
    scene = torch.randn(2, 11, 8)
    calls = []
    handle = attn.k.register_forward_pre_hook(lambda module, args: calls.append(tuple(args[0].shape)))
    actual = attn(action, scene)
    handle.remove()
    assert calls == [(2, 11, 8)]
    expected = torch.cat([attn(action[:, i:i + 1], scene) for i in range(4)], dim=1)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_candidates_do_not_read_each_other_in_action_dit():
    model = tiny_dit().eval()
    actions = torch.randn(1, 4, 8, 3)
    times, scene, ego = torch.full((1, 4), 500.), torch.randn(1, 11, 8), torch.randn(1, 8)
    original = model(actions, times, scene, ego)
    changed = actions.clone()
    changed[:, 1:] += 10
    new = model(changed, times, scene, ego)
    torch.testing.assert_close(original[:, 0], new[:, 0], rtol=1e-5, atol=1e-6)


def test_checkpoint_load_covers_backbone_and_compatible_io_and_rejects_missing_block(tmp_path):
    original = tiny_dit()
    state = original.state_dict()
    state["text_embedding.0.weight"] = torch.randn(8, 4096)
    payload = {"meta": original.checkpoint_meta, "backbone_state_dict": state}
    path = tmp_path / "test.pt"
    torch.save(payload, path)
    restored = tiny_dit()
    report = restored.load_pretrained(path)
    assert report["fraction"] == 1
    assert report["random_io_tensors"] == []
    assert report["omitted_text_tensors"] == ["text_embedding.0.weight"]
    torch.testing.assert_close(restored.blocks[1].self_attn.q.weight, original.blocks[1].self_attn.q.weight)
    del state["blocks.0.self_attn.q.weight"]
    torch.save(payload, path)
    with pytest.raises(ValueError, match="missing_backbone"):
        restored.load_pretrained(path)


def test_checkpoint_explicitly_reports_absent_io(tmp_path):
    original = tiny_dit()
    state = {k: v for k, v in original.state_dict().items() if not k.startswith(("action_encoder.", "head."))}
    path = tmp_path / "prepared.pt"
    torch.save({"meta": original.checkpoint_meta, "backbone_state_dict": state}, path)
    report = tiny_dit().load_pretrained(path)
    assert set(report["random_io_tensors"]) == {"action_encoder.weight", "action_encoder.bias", "head.weight", "head.bias"}


def test_multistep_euler_uses_nonzero_time_and_lands_on_perfect_x_prediction(monkeypatch):
    model = tiny_model(monkeypatch)
    # 60 steps also exercises sigma below the TRAINING floor of 0.1.
    model.num_inference_steps = 60
    clean = torch.randn(1, 64, 8, 3)
    seen = []
    class PerfectPredictor(nn.Module):
        def forward(self, noisy, times, scene, ego):
            seen.append(times[0, 0].item())
            return clean
    model.action_dit = PerfectPredictor()
    generated = model.sample_normalized(torch.randn(1, 11, 8), torch.randn(1, 8))
    assert len(seen) == 60 and seen[0] == 1000 and seen[-1] > 0
    assert seen[-1] < 100
    assert all(a > b for a, b in zip(seen, seen[1:]))
    torch.testing.assert_close(generated, clean, rtol=0, atol=0)


def test_two_gt_sets_share_noise_time_but_choose_independent_best_candidates(monkeypatch):
    model = tiny_model(monkeypatch)
    targets = {"trajectory": torch.randn(2, 8, 3), "trajectory_long": torch.randn(2, 8, 3)}
    seen = {}
    class TwoWinnerPredictor(nn.Module):
        def forward(self, noisy, times, scene, ego):
            seen["noisy"], seen["times"] = noisy, times
            normal = normalize_trajectory(targets["trajectory"])[:, None].expand(-1, 64, -1, -1)
            long = normalize_trajectory(targets["trajectory_long"])[:, None].expand(-1, 64, -1, -1)
            pred = torch.cat((normal + 0.2, long + 0.3), dim=1).clone()
            pred[:, 7] = normal[:, 7]
            pred[:, 64 + 42] = long[:, 42]
            return pred
    model.action_dit = TwoWinnerPredictor()
    loss, diagnostics = model.flow_training_loss(targets, torch.randn(2, 11, 8), torch.randn(2, 8))
    assert loss.item() == 0
    assert diagnostics["flow_normal_loss"] == 0 and diagnostics["flow_long_loss"] == 0
    times = seen["times"]
    assert seen["noisy"].shape == (2, 128, 8, 3)
    torch.testing.assert_close(times, times[:, :1].expand(-1, 128))
    sigma = (times[:, :64] / 1000)[..., None, None]
    c0 = normalize_trajectory(targets["trajectory"])[:, None]
    c1 = normalize_trajectory(targets["trajectory_long"])[:, None]
    # Reusing noise means the two interpolations differ only in their GT terms.
    torch.testing.assert_close(seen["noisy"][:, :64] - seen["noisy"][:, 64:],
                               ((1 - sigma) * (c0 - c1)).expand(-1, 64, -1, -1), atol=1e-6, rtol=1e-5)


def test_scorer_detaches_coordinates_but_updates_scene_and_ego(monkeypatch):
    model = tiny_model(monkeypatch).train()
    proposals = torch.randn(2, 64, 8, 3, requires_grad=True)
    scene = torch.randn(2, 11, 8, requires_grad=True)
    ego = torch.randn(2, 8, requires_grad=True)
    outputs = model.score_proposals(proposals, scene, ego)
    loss = sum(value.square().mean() for value in outputs["pred_logit"].values())
    loss.backward()
    assert proposals.grad is None
    assert scene.grad.abs().sum() > 0 and ego.grad.abs().sum() > 0
    assert model.pos_embed[0].weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.action_dit.parameters())


def test_generator_and_scorer_share_the_full_front_patch_context(monkeypatch):
    model = tiny_model(monkeypatch).eval()
    front_patches = torch.randn(1, 3936, 8)
    ego = torch.randn(1, 8)
    monkeypatch.setattr(model, "encode_scene", lambda _: (front_patches, ego))
    action_kv = []
    scorer_qkv = []
    action_hook = model.action_dit.blocks[0].cross_attn.k.register_forward_pre_hook(
        lambda module, args: action_kv.append(tuple(args[0].shape)))
    scorer_hook = model.scorer_attention.layers[0].cross_attn.attn.register_forward_pre_hook(
        lambda module, args, kwargs: scorer_qkv.append((
            tuple(kwargs["query"].shape), tuple(kwargs["key"].shape), tuple(kwargs["value"].shape))),
        with_kwargs=True)
    try:
        output = model({})
    finally:
        action_hook.remove()
        scorer_hook.remove()
    assert action_kv and all(shape == (1, 3936, 8) for shape in action_kv)
    assert scorer_qkv == [((1, 64, 8), (1, 3936, 8), (1, 3936, 8))]
    assert output["_scene_context"] is front_patches
    assert output["proposals"].shape == (1, 64, 8, 3)
    assert len(output["pred_logit"]) == 6


def test_training_forward_flow_and_official_six_head_loss_backward(monkeypatch):
    model = tiny_model(monkeypatch).train()
    features = {"camera_feature": torch.randn(2, 1, 3, 28, 28), "ego_status": torch.randn(2, 4, 11)}
    features["ego_status"][:, -1, :3] = 0
    targets = {"trajectory": torch.randn(2, 8, 3), "trajectory_long": torch.randn(2, 8, 3)}
    with torch.autocast("cpu", dtype=torch.bfloat16):
        pred = model(features, targets=targets)
        flow = pred["_flow_loss"]
        def oracle(targets, proposals, test):
            assert proposals.shape == (2, 64, 8, 3) and not test
            raw7 = torch.full((2, 64, 7), 0.75)
            return raw7[..., -1], raw7[..., -1].max(1).values, raw7, None, None, None
        loss_dict = DrivoRActionDiTLoss()(targets, pred, model._config, oracle, flow_loss=flow)
    loss_dict["loss"].backward()
    assert torch.isfinite(loss_dict["loss"])
    assert not pred["proposals"].requires_grad
    assert model.action_dit.head.weight.grad.abs().sum() > 0
    assert model.hist_encoding.weight.grad.abs().sum() > 0
    assert model.image_backbone.neck.weight.grad.abs().sum() > 0
    assert model.scorer.pred_score["comfort"][0].weight.grad.abs().sum() > 0


@pytest.mark.parametrize("long_poses", [0, 2])
def test_lightning_training_reuses_one_x_prediction_for_flow_and_scoring(monkeypatch, long_poses):
    from navsim.agents.abstract_agent import AbstractAgent
    from navsim.agents.drivoR_action_dit.agent import DrivoRActionDiTAgent
    from navsim.planning.training.agent_lightning_module import AgentLightningModule

    model = tiny_model(monkeypatch).train()
    model._config.long_trajectory_additional_poses = long_poses
    model.num_inference_steps = 10
    features = {"camera_feature": torch.randn(2, 1, 3, 28, 28), "ego_status": torch.randn(2, 4, 11)}
    targets = {"trajectory": torch.randn(2, 8, 3), "trajectory_long": torch.randn(2, 8, 3)}
    calls, images, scored = [], [], []

    def capture(module, args, output):
        calls.append((args[0].shape, output.detach().float().clone(), torch.is_grad_enabled()))

    hook = model.action_dit.register_forward_hook(capture)
    image_hook = model.image_backbone.register_forward_hook(lambda *args: images.append(1))
    monkeypatch.setattr(model, "sample_normalized", lambda *args, **kwargs: pytest.fail("Euler called during training"))
    monkeypatch.setattr(model, "flow_training_loss", lambda *args, **kwargs: pytest.fail("Extra flow forward called"))
    agent = DrivoRActionDiTAgent.__new__(DrivoRActionDiTAgent)
    AbstractAgent.__init__(agent)
    agent._drivor_model, agent._config, agent.loss = model, model._config, DrivoRActionDiTLoss()

    def oracle(targets, proposals, test):
        assert not test and not proposals.requires_grad
        scored.append(proposals.clone())
        raw7 = torch.full((2, 64, 7), 0.75)
        return raw7[..., -1], raw7[..., -1].max(1).values, raw7, None, None, None

    agent.compute_score = oracle
    lightning = AgentLightningModule(agent, console_log_interval=0, log_local_steps=False)
    monkeypatch.setattr(lightning, "log", lambda *args, **kwargs: None)
    try:
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss = lightning.training_step((features, targets), 0)
        loss.backward()
    finally:
        hook.remove()
        image_hook.remove()
    assert len(calls) == len(images) == len(scored) == 1
    assert calls[0][0] == (2, 128 if long_poses else 64, 8, 3)
    assert calls[0][2]
    # Every normal-GT X prediction is scored, in original candidate order.
    torch.testing.assert_close(scored[0], denormalize_trajectory(calls[0][1][:, :64]), rtol=0, atol=0)
    assert torch.isfinite(loss)
    assert model.action_dit.head.weight.grad.abs().sum() > 0
    assert model.scorer.pred_score["comfort"][0].weight.grad.abs().sum() > 0


def test_reused_x_predictions_stop_score_gradient_at_trajectory(monkeypatch):
    model = tiny_model(monkeypatch).train()
    features = {"camera_feature": torch.randn(2, 1, 3, 28, 28), "ego_status": torch.randn(2, 4, 11)}
    targets = {"trajectory": torch.randn(2, 8, 3), "trajectory_long": torch.randn(2, 8, 3)}
    pred = model(features, targets=targets)
    assert pred["_flow_loss"].requires_grad
    assert not pred["proposals"].requires_grad
    sum(value.square().mean() for value in pred["pred_logit"].values()).backward()
    assert all(parameter.grad is None for parameter in model.action_dit.parameters())
    assert model.hist_encoding.weight.grad.abs().sum() > 0
    assert model.image_backbone.neck.weight.grad.abs().sum() > 0


def test_feature_only_evaluation_keeps_ten_euler_calls(monkeypatch):
    model = tiny_model(monkeypatch).eval()
    model.num_inference_steps = 10
    features = {"camera_feature": torch.randn(1, 1, 3, 28, 28), "ego_status": torch.randn(1, 4, 11)}
    calls = []
    handle = model.action_dit.register_forward_pre_hook(
        lambda module, args: calls.append((args[0].shape, args[1][0, 0].item())))
    monkeypatch.setattr(model, "flow_training_outputs", lambda *args, **kwargs: pytest.fail("Training targets used in evaluation"))
    try:
        with torch.no_grad():
            pred = model(features)
    finally:
        handle.remove()
    assert len(calls) == 10
    assert all(shape == (1, 64, 8, 3) for shape, _ in calls)
    assert calls[0][1] == 1000 and calls[-1][1] > 0
    assert pred["proposals"].shape == (1, 64, 8, 3)
    assert "_flow_loss" not in pred


def test_wan_block_matches_reference_code_when_reference_checkout_is_available():
    reference = Path("/cpfs29jb/data/zhaozc/project/FastWam_re/src/fastwam/models/wan22/wan_video_dit.py")
    if not reference.is_file():
        pytest.skip("External FastWAM reference checkout unavailable")
    # Load only standalone attention/block definitions, with no text/VAE/model
    # dependencies or side effects from importing the external FastWAM package.
    from einops import rearrange
    from typing import Optional
    wanted = {"flash_attention", "modulate", "precompute_freqs_cis", "rope_apply",
              "RMSNorm", "SelfAttention", "CrossAttention", "GateModule", "DiTBlock"}
    tree = ast.parse(reference.read_text())
    tree.body = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in wanted]
    namespace = dict(torch=torch, nn=nn, F=F, rearrange=rearrange, Optional=Optional)
    exec(compile(tree, str(reference), "exec"), namespace)
    native = namespace["DiTBlock"](hidden_dim=8, attn_head_dim=4, num_heads=2, ffn_dim=16, eps=1e-6)
    block = tiny_dit().blocks[0]
    native.load_state_dict(block.state_dict(), strict=True)
    x = torch.randn(2, 4, 8, 8)
    scene = torch.randn(2, 11, 8)
    modulation = torch.randn(2, 4, 6, 8)
    actual = block(x, scene, modulation)
    freqs = namespace["precompute_freqs_cis"](4, end=8).view(8, 1, -1)
    expected = native(x.reshape(8, 8, 8), scene[:, None].expand(-1, 4, -1, -1).reshape(8, 11, 8),
                      modulation.reshape(8, 6, 8), freqs).reshape_as(actual)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("variant,batch_size,num_gpus,accumulation", [
    ("drivor", 16, 4, 1),
    ("drivor", 1, 64, 1),
    ("action_dit", 1, 4, 16),
    ("action_dit", 1, 64, 1),
])
def test_optimizer_lr_and_schedule_preserve_global_batch(variant, batch_size, num_gpus, accumulation):
    from navsim.agents.drivoR.drivor_agent import DrivoRAgent
    from navsim.agents.drivoR_action_dit.agent import DrivoRActionDiTAgent
    agent_type = DrivoRAgent if variant == "drivor" else DrivoRActionDiTAgent
    agent = agent_type.__new__(agent_type)
    nn.Module.__init__(agent)
    agent._drivor_model = nn.Linear(3, 8)
    agent._lr_args = dict(name="AdamW", base_lr=0.0002, base_batch_size=64)
    agent.batch_size, agent.num_gpus, agent.accumulate_grad_batches = batch_size, num_gpus, accumulation
    agent.scheduler_args = OmegaConf.create(dict(dataset_size=85000, num_epochs=25))
    optimizers, schedules = agent.get_optimizers()
    assert optimizers[0].defaults["lr"] == pytest.approx(0.0002)
    schedule = schedules[0]["scheduler"]
    assert schedule._milestones[0] + schedule._schedulers[1].T_max == 33225
    assert schedules[0]["interval"] == "step"
