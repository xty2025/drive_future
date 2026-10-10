#!/usr/bin/env python3
"""Re-rank externally generated ChainFlow proposals with the official DrivoR scorer.

ChainFlow remains the trajectory generator.  DrivoR supplies its own image/ego
scene encoding and the complete learned scoring path (trajectory embedding,
cross-attention, and six score heads).
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from navsim.agents.drivoR.drivor_features import DrivoRFeatureBuilder
from navsim.agents.drivoR.drivor_model import DrivoRModel
from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter, SensorConfig, Trajectory


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--chainflow-predictions", required=True, type=Path)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--navsim-log-path", required=True, type=Path)
    p.add_argument("--sensor-blobs-path", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--config", type=Path, default=Path("navsim/planning/script/config/common/agent/drivoR.yaml"))
    return p.parse_args()


def batch_to(features, device):
    return {key: value.unsqueeze(0).to(device) for key, value in features.items()}


def score_external(model, features, proposals):
    if model._config.full_history_status:
        ego_status = features["ego_status"].flatten(-2)
    else:
        ego_status = features["ego_status"][:, -1]
    ego_token = model.hist_encoding(ego_status)[:, None]

    scene_tokens = model.scene_embeds.repeat(ego_status.shape[0], 1, 1, 1)
    scene_features = model.image_backbone(features["image"], scene_tokens)
    embedded = model.pos_embed(proposals.flatten(2))
    scorer_features = model.scorer_attention(embedded, scene_features) + ego_token
    logits = model.scorer(proposals, scorer_features)[0]
    cfg = model._config
    score = (
        cfg.noc * logits["no_at_fault_collisions"].sigmoid().log()
        + cfg.dac * logits["drivable_area_compliance"].sigmoid().log()
        + cfg.ddc * logits["driving_direction_compliance"].sigmoid().log()
        + (
            cfg.ttc * logits["time_to_collision_within_bound"].sigmoid()
            + cfg.ep * logits["ego_progress"].sigmoid()
            + cfg.comfort * logits["comfort"].sigmoid()
        ).log()
    )
    return score, logits


def main():
    args = parse_args()
    source = pickle.load(args.chainflow_predictions.open("rb"))
    tokens = sorted(source)
    agent_cfg = OmegaConf.load(args.config)
    cfg = agent_cfg.config
    device = torch.device(args.device)
    model = DrivoRModel(cfg)
    raw = torch.load(args.checkpoint, map_location="cpu")
    state = raw.get("state_dict", raw)
    prefix = "agent._drivor_model."
    state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    incompatible = model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False).to(device)

    scene_filter = SceneFilter(
        num_history_frames=4, num_future_frames=10, frame_interval=1,
        has_route=True, max_scenes=None, log_names=None, tokens=tokens,
    )
    loader = SceneLoader(
        sensor_blobs_path=args.sensor_blobs_path,
        data_path=args.navsim_log_path,
        scene_filter=scene_filter,
        sensor_config=SensorConfig(
            cam_f0=OmegaConf.to_object(cfg.cam_f0), cam_l0=OmegaConf.to_object(cfg.cam_l0),
            cam_l1=OmegaConf.to_object(cfg.cam_l1), cam_l2=OmegaConf.to_object(cfg.cam_l2),
            cam_r0=OmegaConf.to_object(cfg.cam_r0), cam_r1=OmegaConf.to_object(cfg.cam_r1),
            cam_r2=OmegaConf.to_object(cfg.cam_r2), cam_b0=OmegaConf.to_object(cfg.cam_b0),
            lidar_pc=OmegaConf.to_object(cfg.lidar_pc),
        ),
    )
    builder = DrivoRFeatureBuilder(cfg)
    output = {}
    diagnostics = {
        "checkpoint": str(args.checkpoint), "strict_load": True,
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys), "tokens": {},
    }
    with torch.inference_mode():
        for index, token in enumerate(tokens):
            item = source[token]
            if "diffusion_proposals" not in item:
                raise KeyError(
                    f"{token}: missing diffusion_proposals; regenerate with the patched "
                    "ChainFlow recorder instead of re-ranking the AR proposal_list"
                )
            proposals_np = np.asarray(item["diffusion_proposals"], dtype=np.float32)
            if proposals_np.shape != (64, 8, 3):
                raise ValueError(f"{token}: expected (64,8,3), got {proposals_np.shape}")
            agent_input = loader.get_agent_input_from_token(token)
            features = batch_to(builder.compute_features(agent_input), device)
            proposals = torch.from_numpy(proposals_np).unsqueeze(0).to(device)
            scores, logits = score_external(model, features, proposals)
            selected = int(scores[0].argmax().item())
            trajectory = Trajectory(proposals_np[selected])
            output[token] = dict(item)
            output[token]["trajectory"] = trajectory
            output[token]["drivor_scores"] = scores[0].float().cpu().numpy()
            output[token]["drivor_selected_index"] = selected
            output[token]["chainflow_selected_trajectory"] = item["trajectory"]
            diagnostics["tokens"][token] = {
                "selected_index": selected,
                "score_min": float(scores.min()), "score_max": float(scores.max()),
                "score_std": float(scores.std()),
                "finite": bool(torch.isfinite(scores).all()),
                "heads": {k: float(v[0, selected].sigmoid()) for k, v in logits.items()},
            }
            print(f"[{index + 1:02d}/{len(tokens)}] {token} selected={selected} score={scores[0, selected]:.5f}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        pickle.dump(output, handle)
    args.output.with_suffix(".json").write_text(json.dumps(diagnostics, indent=2))
    print(f"saved {len(output)} trajectories to {args.output}")


if __name__ == "__main__":
    main()
