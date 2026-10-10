"""Compare learned and DrivoR-pretrained score weights on identical NVS inputs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from nvs_v6.modes import ImmutableModeBuffer, StructuredSourceInitializer
from nvs_v6.model import DirectedPairDiT
from nvs_v6.rollout_data import load_paired_rollout_sample
from nvs_v6.scorer import DrivoRSourceScorer
from nvs_v6.selection import select_best_candidate


def _load_official_score_weights(scorer: DrivoRSourceScorer, checkpoint: Path) -> None:
    raw = torch.load(checkpoint, map_location="cpu")
    state = raw.get("state_dict", raw)
    prefix = "agent._drivor_model."
    mapped = {}
    for key, value in state.items():
        if not key.startswith(prefix):
            continue
        name = key[len(prefix) :]
        if name.startswith("pos_embed."):
            name = "trajectory_projection." + name[len("pos_embed.") :]
        if name.startswith("trajectory_projection.") or name.startswith("scorer_attention.") or name.startswith("scorer."):
            mapped[name] = value
    incompatible = scorer.load_state_dict(mapped, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"DrivoR score weight mismatch: {incompatible}")


@torch.inference_mode()
def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = Path(__file__).resolve().parents[1]
    artifact = root / "artifacts/two_scene_overfit_10epoch_drivor_native_cache"
    run_checkpoint = torch.load(artifact / "checkpoint.pt", map_location="cpu")
    model = DirectedPairDiT(
        d_model=256,
        heads=8,
        ffn_dim=1024,
        depth=8,
        dino_checkpoint=run_checkpoint["dino_checkpoint"],
        dino_model_name="vit_large_patch16_dinov3.lvd1689m",
        dino_lora_rank=16,
        dino_lora_alpha=16.0,
    ).to(device)
    model.load_state_dict(run_checkpoint["model"], strict=True)
    own = DrivoRSourceScorer(d_model=256, heads=1, ffn_dim=1024, decoder_layers=4).to(device)
    own.load_state_dict(run_checkpoint["scorer"], strict=True)
    pretrained = DrivoRSourceScorer(d_model=256, heads=1, ffn_dim=1024, decoder_layers=4).to(device)
    _load_official_score_weights(
        pretrained,
        Path("/mnt/zhaozc_workspace/project/checkpoints/DrivoR/nav1_30epochs_with_134k_simscale_bis_103ktrainval.pth"),
    )
    model.eval()
    own.eval()
    pretrained.eval()
    modes = ImmutableModeBuffer().subset(64).to(device)
    initializer = StructuredSourceInitializer(jitter_std=0.0)
    cache_root = Path("/mnt/zhaozc_workspace/official_DrivoR_clean/data_runtime/official_metric_cache_navtrain_20261005/metric_cache")
    scenes = ["8f4a85d43ef351f9", "b00d83c03bbf5b47"]
    raw = np.load(artifact / "official_raw7_scores.npy")
    rows = []
    for scene_index, scene_id in enumerate(scenes):
        sample = load_paired_rollout_sample(root.parent / "multitra2egoview/cosmos3/batch_native10/rollouts/single_gpu_test_gtfallback_v2" / scene_id, cache_root)
        observation = sample.observation
        for key, value in observation.__dict__.items():
            if isinstance(value, torch.Tensor):
                setattr(observation, key, value.to(device))
        source = initializer.clean(observation.ego_history, modes.expand_for_batch(1, device))
        outputs = model.forward_pairs(observation, source, modes.expand_for_batch(1, device))
        own_logits = own(outputs.scene_context, outputs.trajectory)
        pretrained_logits = pretrained(outputs.scene_context, outputs.trajectory)
        own_index = int(select_best_candidate(own_logits)[0].item())
        pretrained_index = int(select_best_candidate(pretrained_logits)[0].item())
        rows.append({
            "scene_id": scene_id,
            "own_index": own_index,
            "own_pdms": float(raw[scene_index, own_index, 6]),
            "drivor_pretrained_index_same_nvs_scene": pretrained_index,
            "drivor_pretrained_pdms_same_nvs_scene": float(raw[scene_index, pretrained_index, 6]),
            "oracle_index": int(raw[scene_index, :, 6].argmax()),
            "oracle_pdms": float(raw[scene_index, :, 6].max()),
            "same_index": own_index == pretrained_index,
        })
    output = {
        "same_candidates": True,
        "same_nvs_scene_context": True,
        "future_register_used_by_either_scorer": False,
        "rows": rows,
    }
    (artifact / "same_nvs_scene_scorer_comparison.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
