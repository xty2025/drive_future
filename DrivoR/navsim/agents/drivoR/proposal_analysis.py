"""Run DrivoR checkpoints on explicit scene tokens and keep every proposal and scorer output.

Used by the proposal visualizations and the scorer oracle-gap analysis. The agent is built with
the README NAVSIM-v1 evaluation config (scripts/evaluation/run_drivor_pdm_score_evaluation.sh);
Action-DiT agents add the overrides of their evaluation script.
"""

import hashlib
import os
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
import torch
import yaml
from hydra import compose, initialize_config_module
from hydra.utils import instantiate
from torch.utils.data import DataLoader, Dataset

from navsim.agents.drivoR.score_module.ranking import aggregate_score_logits
from navsim.common.dataclasses import AgentInput, Camera, Cameras, SceneFilter, SensorConfig
from navsim.common.dataloader import SceneLoader

# Ranking weights of the README NAVSIM-v1 evaluation (agent.config.noc, ...).
SCORE_WEIGHTS = {"noc": 1, "dac": 1, "ddc": 0.0, "ttc": 5, "ep": 5, "comfort": 2}
NAV1_OVERRIDES = [
    "agent=drivoR",
    "agent.score_worker=null",
    "agent.config.proposal_num=64",
    "agent.config.refiner_ls_values=0.0",
    "agent.config.image_backbone.focus_front_cam=false",
    "agent.config.one_token_per_traj=true",
    "agent.config.refiner_num_heads=1",
    "agent.config.tf_d_model=256",
    "agent.config.tf_d_ffn=1024",
    "agent.config.area_pred=false",
    "agent.config.agent_pred=false",
    "agent.config.ref_num=4",
] + [f"agent.config.{name}={value}" for name, value in SCORE_WEIGHTS.items()]
# Appended by scripts/evaluation/run_drivor_action_dit_trained_nav1_evaluation.sh after agent=drivoR_action_dit
# (multiview: agent=drivoR_action_dit_multiview agent.config.num_scene_tokens=16).
ACTION_DIT_OVERRIDES = [
    "agent.config.action_dit.pretrained_path=null",
    "agent.config.long_trajectory_additional_poses=0",
    "agent.config.action_dit.num_inference_steps=10",
]

# Scorer heads in the column order of head_probs.
HEADS = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "time_to_collision_within_bound",
    "ego_progress",
    "comfort",
    "driving_direction_compliance",
)

SPLITS = {
    "test": {
        "data_split": "test",
        "oss": "oss://ofasys-multimodal-wlcb-5-toshanghai/anonymous_driving/autonomousvision_navsim/1/"
        "download_test/unzip/test_sensor_blobs/test",
    },
    "train": {
        "data_split": "trainval",
        "oss": "oss://ofasys-multimodal-wlcb-5-toshanghai/anonymous_driving/autonomousvision_navsim/1/"
        "download_trainval/openscene-v1.1/sensor_blobs/trainval",
    },
}


def compose_eval_config(overrides=()):
    """Evaluation config; overrides go after NAV1_OVERRIDES, so e.g. agent=drivoR_action_dit replaces agent=drivoR."""
    with initialize_config_module(config_module="navsim.planning.script.config.pdm_scoring", version_base=None):
        return compose(config_name="default_run_pdm_score_gpu", overrides=NAV1_OVERRIDES + list(overrides))


def load_agent(cfg, checkpoint, device):
    agent = instantiate(cfg.agent, checkpoint_path=str(checkpoint))
    agent.initialize()
    return agent.eval().to(device)


def front_camera_config():
    """Current CAM_F0 only: enough for plotting, no OSS reads for the side cameras."""
    return SensorConfig(cam_f0=[3], cam_l0=[], cam_l1=[], cam_l2=[], cam_r0=[], cam_r1=[], cam_r2=[], cam_b0=[],
                        lidar_pc=[])


def sensor_key(sensor_config):
    return ",".join(f"{name}={value}" for name, value in asdict(sensor_config).items() if value)


def union_sensor_config(agents):
    """Every sensor any of the agents needs; TokenDataset then gives each agent only its own cameras."""
    configs = [asdict(agent.get_sensor_config()) for agent in agents.values()]
    merged = {}
    for name in configs[0]:
        values = [config[name] for config in configs]
        merged[name] = True if any(value is True for value in values) else \
            sorted(set().union(*(value for value in values if value)))
    return SensorConfig(**merged)


def restrict_cameras(agent_input, sensor_config):
    """agent_input without the cameras sensor_config does not request (DrivoRFeatureBuilder stacks every loaded one)."""
    frames = []
    for iteration, cameras in enumerate(agent_input.cameras):
        keep = set(sensor_config.get_sensors_at_iteration(iteration))
        frames.append(Cameras(**{field.name: getattr(cameras, field.name) if field.name in keep else Camera()
                                 for field in fields(Cameras)}))
    return AgentInput(agent_input.ego_statuses, frames, agent_input.lidars)


def log_of(cache_paths, token):
    """Log name of a token; metric cache layout <cache>/<log_name>/unknown/<token>/metric_cache.pkl."""
    return Path(cache_paths[token]).parts[-4]


def sample_navtrain_tokens(cache_paths, num_logs, max_scenes, rng):
    """navtrain scenes of random training logs (the 85k the models train on), capped at max_scenes.

    Sampling whole logs keeps the SceneLoader from reading hundreds of log files.
    :param rng: random.Random instance (reproducible for a given seed)
    :return: (chosen log names, sorted tokens)
    """
    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    config_dir = Path(__file__).resolve().parents[2] / "planning/script/config"
    with open(config_dir / "common/train_test_split/scene_filter/navtrain.yaml") as stream:
        navtrain = yaml.load(stream, Loader=loader)
    with open(config_dir / "training/default_train_val_test_log_split.yaml") as stream:
        train_logs = set(yaml.load(stream, Loader=loader)["train_logs"])
    tokens_by_log = {}
    for token in navtrain["tokens"]:
        if token in cache_paths:
            tokens_by_log.setdefault(log_of(cache_paths, token), []).append(token)
    logs = sorted(log for log in tokens_by_log if log in train_logs)
    chosen_logs = rng.sample(logs, min(num_logs, len(logs)))
    pool = sorted(token for log in chosen_logs for token in tokens_by_log[log])
    return chosen_logs, sorted(rng.sample(pool, min(max_scenes, len(pool))))


def make_scene_loader(split, log_names, tokens, sensor_config):
    data_root = Path(os.environ["OPENSCENE_DATA_ROOT"])
    data_split = SPLITS[split]["data_split"]
    scene_filter = SceneFilter(num_history_frames=4, num_future_frames=10, frame_interval=1, has_route=True,
                               max_scenes=None, log_names=sorted(set(log_names)), tokens=list(tokens))
    return SceneLoader(data_path=data_root / "navsim_logs" / data_split,
                       sensor_blobs_path=data_root / "sensor_blobs" / data_split,
                       scene_filter=scene_filter, sensor_config=sensor_config)


class TokenDataset(Dataset):
    """Mirrors navsim.planning.training.dataset.Dataset (no cache) for an explicit token list.

    Features are keyed by sensor_key: agents with different cameras (e.g. front-view and multiview
    Action-DiT) each get theirs from the same scene; the scene loader must read the union of them.
    """

    def __init__(self, scene_loader, tokens, agents):
        self.scene_loader = scene_loader
        self.tokens = list(tokens)
        self.inputs = {}
        for agent in agents.values():
            sensor_config = agent.get_sensor_config()
            self.inputs.setdefault(sensor_key(sensor_config), (sensor_config, agent.get_feature_builders()))
        self.target_builders = next(iter(agents.values())).get_target_builders()

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        token = self.tokens[idx]
        scene = self.scene_loader.get_scene_from_token(token)
        agent_input = scene.get_agent_input()
        features, targets = {}, {}
        for key, (sensor_config, builders) in self.inputs.items():
            restricted = restrict_cameras(agent_input, sensor_config)
            features[key] = {}
            for builder in builders:
                features[key].update(builder.compute_features(restricted))
        for builder in self.target_builders:
            targets.update(builder.compute_targets(scene))
        return features, targets, token


def predicted_pdms(pdm_score):
    """The scorer's expected PDMS: sigma_NC^noc * sigma_DAC^dac * (5 sigma_TTC + 5 sigma_EP + 2 sigma_C) / 12.

    DrivoRModel's pdm_score is the log of the product before / 12 (logsigmoid + logsumexp), so this
    keeps the model's exact ranking; recomputing from float32 sigmoids can reorder near-ties.
    """
    w = SCORE_WEIGHTS
    return np.exp(np.asarray(pdm_score, dtype=np.float64)) / (w["ttc"] + w["ep"] + w["comfort"])


def flow_noise(tokens, shape, seed):
    """Initial flow-matching noise [B, *shape]: the same for a (seed, token) pair in every model, batch and run."""
    noise = []
    for token in tokens:
        generator = torch.Generator().manual_seed(int(hashlib.sha256(f"{seed}/{token}".encode()).hexdigest()[:15], 16))
        noise.append(torch.randn(shape, generator=generator))
    return torch.stack(noise)


def model_forward(agent, features, tokens, noise_seed, bf16=None):
    """agent.forward(features). Action-DiT agents sample from flow_noise(tokens). bf16=None: BF16 autocast for
    Action-DiT and FP32 for DrivoR, as their default evaluations; True/False forces it (bf16-mixed evaluations)."""
    model = agent._drivor_model
    action_dit = hasattr(model, "sample_normalized")
    bf16 = action_dit if bf16 is None else bf16
    device = next(model.parameters()).device
    autocast = torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16)
    if not action_dit:
        with autocast:
            return agent.forward(features)
    noise = flow_noise(tokens, (model.proposal_num, model.poses_num, 3), noise_seed).to(device)
    # forward() calls self.sample_normalized(scene, ego), or self.sample_joint(scene, ego) when the model denoises
    # future registers with the trajectories; an instance attribute passes it the fixed noise.
    name = "sample_joint" if hasattr(model, "sample_joint") else "sample_normalized"
    sample = getattr(model, name)
    if name == "sample_joint":
        registers = flow_noise(tokens, (model.proposal_num, model.register_count, model.register_dim),
                               f"{noise_seed}/registers").to(device)
        model.sample_joint = lambda scene, ego: sample(scene, ego, action_noise=noise, register_noise=registers)
    else:
        model.sample_normalized = lambda scene, ego: sample(scene, ego, noise=noise)
    try:
        with autocast:
            return agent.forward(features)
    finally:
        delattr(model, name)


def drivor_scene_context(model, features):
    """The scene tokens and ego token DrivoRModel.forward gives its scorer."""
    config = model._config
    ego_status = features["ego_status"].flatten(-2) if config.full_history_status else features["ego_status"][:, -1]
    ego_token = model.hist_encoding(ego_status)[:, None]
    batch = ego_status.shape[0]
    scene = []
    if model.num_cams > 0:
        images = features["image"] if "image" in features else features["camera_feature"]
        scene.append(model.image_backbone(images, model.scene_embeds.repeat(batch, 1, 1, 1)))
    if model.num_lidar > 0:
        scene.append(model.lidar_backbone(features["lidar_feature"], model.lidar_scene_embeds.repeat(batch, 1, 1, 1)))
    return torch.cat(scene, dim=1), ego_token


def drivor_score(model, proposals, scene_features, ego_token):
    """DrivoRModel.forward's ranking score (its scoring branch) for any proposals [B, N, 8, 3]."""
    tokens = model._score_features(proposals, scene_features, ego_token)
    return aggregate_score_logits(model.scorer(proposals, tokens)[0], model._config)


@torch.no_grad()
def iterate_proposals(agents, scene_loader, tokens, device, batch_size=8, workers=8, noise_seed=0, bf16=None):
    """
    Yields (tokens, human [B,8,3], {label: dict}) per batch. Each model dict holds proposals [B,64,8,3],
    head_probs [B,64,6] (HEADS order), pdm_score [B,64] (the model's ranking score), pred_pdms [B,64]
    and selected [B] (= argmax pdm_score, the model output). Action-DiT proposals start from
    flow_noise(token, noise_seed), shared by all models. Precision: BF16 for Action-DiT, FP32 for DrivoR (their
    default evaluations) unless bf16 {label: bool} says otherwise.
    """
    loader = DataLoader(TokenDataset(scene_loader, tokens, agents),
                        batch_size=batch_size, num_workers=workers, shuffle=False)
    for features, targets, batch_tokens in loader:
        features = {key: {name: value.to(device) for name, value in group.items() if torch.is_tensor(value)}
                    for key, group in features.items()}
        outputs = {}
        for label, agent in agents.items():
            out = model_forward(agent, features[sensor_key(agent.get_sensor_config())], batch_tokens, noise_seed,
                                (bf16 or {}).get(label))
            head_probs = torch.stack([torch.sigmoid(out["pred_logit"][name].float()) for name in HEADS], dim=-1)
            pdm_score = out["pdm_score"].float()
            selected = pdm_score.argmax(dim=1)
            proposals = out["proposals"].float()
            if not torch.allclose(proposals[torch.arange(len(selected)), selected], out["trajectory"].float()):
                raise RuntimeError("Selected proposal does not match the model's output trajectory")
            pdm_score = pdm_score.cpu().numpy()
            outputs[label] = {
                "proposals": proposals.cpu().numpy(),
                "head_probs": head_probs.cpu().numpy(),
                "pdm_score": pdm_score,
                "pred_pdms": predicted_pdms(pdm_score),
                "selected": selected.cpu().numpy(),
            }
        yield list(batch_tokens), targets["trajectory"].float().numpy(), outputs


def run_proposals(agents, scene_loader, tokens, device, batch_size=8, workers=8, noise_seed=0, bf16=None):
    """Returns {label: {token: dict}}; per-token version of iterate_proposals (adds 'human')."""
    results = {label: {} for label in agents}
    for batch_tokens, human, outputs in iterate_proposals(agents, scene_loader, tokens, device, batch_size, workers,
                                                          noise_seed, bf16):
        for label, out in outputs.items():
            for row, token in enumerate(batch_tokens):
                results[label][token] = {key: value[row] for key, value in out.items()}
                results[label][token]["selected"] = int(out["selected"][row])
                results[label][token]["human"] = human[row]
    return results


def scorer_order(pdm_score):
    """Proposal indices from the scorer's first choice to its last (ties keep the lower index, like argmax)."""
    return np.argsort(-np.asarray(pdm_score), axis=-1, kind="stable")
