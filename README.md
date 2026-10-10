# NVS v6

This directory is the additive v6 implementation for the NAVSIM Multi-GT / Future Register / one-pass Action DiT contract.

## Layout

- `config/config_v6_final.yaml`: v6-only configuration and explicit cache paths.
- `src/nvs_v6/`: observation/targets, fixed target compression API, LoRA scene tower, immutable modes, directed generator, losses, scorer, stages, and oracle adapters.
- `src/nvs_v6/scorer.py`: production `ScratchDrivoRScorer`; it randomly initializes DrivoR's source trajectory projection, four-layer decoder, and six heads, then reads detached NVS DINOv3 Scene K/V and the final two NVS ego-fused tokens. It never loads a DrivoR scorer checkpoint or native DINOv2/ego cache.
- `src/nvs_v6/native_drivor.py`: explicit pretrained/native comparison utility only; default training, real smoke, and epoch search must not construct it.
- `tests/contract_tests_v6.py`: direction, leakage, matching, gradient, scratch-scorer, real-DINO token-path, and shape checks.
- `vendor/`: code-only copies of the DrivoR, Drive-JEPA, and FastWAM source trees.
- `reports/`: audit, migration, and test records.

## Commands

```bash
cd /mnt/zhaozc_workspace/project/NVS
PYTHONPATH="src:vendor/DrivoR" pytest -q tests/contract_tests_v6.py
PYTHONPATH=src python scripts/preprocess_v6.py
PYTHONPATH=src python scripts/run_stage1.py --synthetic --candidates 8
PYTHONPATH=src python scripts/run_stage2.py --synthetic --candidates 8
PYTHONPATH=src python scripts/run_stage2.py --synthetic --candidates 64
```

The synthetic commands are contract smoke tests only. Real Stage1/Stage2 requires audited Observation/PairTargets, fixed DINOv3 Student/Teacher checkpoints, a train-fitted target compressor, split manifests, and an importable NAVSIM/DrivoR runtime. The two-scene scratch-scorer entry is `scripts/run_two_scene_overfit.py`; `scripts/launch_epoch_search.sh` invokes the same path. Both use NVS Scene K/V and no DrivoR scorer checkpoint.

## Audited tensor meanings

- Online DrivoR supervision is per current proposal: raw labels are `[B,64,7]` in `NC,DAC,EP,TTC,C,DDC,Final` order and scorer targets are `[B,64,6]`. It is not a `[64,4]` proposal-by-GT PDMS matrix.
- `[B,64,4,256]` is the predicted future-register tensor: 64 candidates, four ordered two-frame future chunks, and 256 latent channels. If a scene has fewer than four valid GTs, Hungarian matching supervises only those matched candidate/register rows; padding and unmatched candidates contribute no future loss.
- At 224px with patch16, each DINOv3 frame has 201 native tokens (`1 CLS + 4 registers + 196 patches`). Injected scene queries make DINO K/V lengths 228 for current and 204 per history frame; retained visual output is `32 + 4*8 = 64`, then two ego-query outputs make Scene length 66. Patch tokens are not re-patched or pooled after DINO; they participate in DINO attention and are discarded only at the retained-token boundary.
- The real adapters read NAVSIM `driving_command` as an exact four-way vector and form DrivoR order `pose3,velocity2,acceleration2,command4`; one trailing zero preserves the existing NVS 12-wide checkpoint interface. This shared clean feature conditions Scene and is projected once into Action.

## Cache separation

- DrivoR online metric cache (oracle labels only, never scorer Scene/Ego features): `/mnt/zhaozc_workspace/project/DrivoR_multigt_kvcache_final_20261004/data_runtime/train_metric_cache`.
- FastWAM offline trajectory-keyed cache: `/mnt/zhaozc_workspace/project/FastWAM/data/metric_cache_navtrain`.
- `/mnt/zhaozc_workspace/official_DrivoR_clean/data_runtime/train_metric_cache` is a symlink to the FastWAM cache and is rejected for DrivoR online evaluation.

See `reports/repository_audit.md` for missing checkpoints, dependency blockers, and observed data/cache formats. No real NAVSIM score or performance claim is included.
