# NVS — DrivoR scene perception + Wan2.2 Action DiT (joint, end-to-end)

One lineage only: `src/nvs_v6_wan_e2e`. The earlier `src/nvs_v6` and
`src/nvs_v6_wan` packages, their configs, scripts and tests were removed on
2026-10-14; recover them from commit `52a9d64` if you need the history.

The model generates 64 trajectory candidates per scene in a **single forward**
(no sampling loop), scores all of them with DrivoR's scorer, and picks one.

## Layout

| Path | What it is |
|---|---|
| `src/nvs_v6_wan_e2e/` | the package. See the module table below. |
| `config/config_v6_wan_e2e.yaml` | the strict contract loader's input (`config.py` rejects deviations). |
| `tests/contract_tests_v6_wan_e2e.py` | direction, leakage, matching, gradient, flow, scorer, resolution and shape checks. |
| `scripts/dlc_navtrain_wan_e2e.sh` | the two-stage pipeline launcher (Stage 0 → 1 → 2). |
| `scripts/run_stage1_multiview.py` | Stage 1: future-representation training on real data. |
| `scripts/run_navtrain_wan_e2e_ddp.py` | Stage 2: full-navtrain joint generator + scorer training (DDP). Also the single source of truth for model construction, checkpoint restore and eval inference. |
| `scripts/build_future_target_cache.py` | Stage 0: one frozen-Teacher pass per scene → `compressor.pt` + per-scene register targets. |
| `scripts/eval_navtrain_wan_e2e_pdms_bev.py` | navtrain PDMS + ego-BEV visualization. |
| `scripts/eval_navtest_official_pdms_e2e.py` | held-out navtest PDMS via the official navsim scorer. |
| `scripts/official_navtest_oracle.py` | official `MetricCache` + `PDMScorer` + `pdm_score` batch oracle. |
| `scripts/merge_navtest_shards.py` | recombine sharded navtest runs and prove the sharding was disjoint and lossless. |
| `scripts/preprocess_wan_action_dit.py` | build `checkpoints/ActionDiT_Wan22_256hdim_3outdim.pt` from a local Wan2.2-TI2V-5B checkout. |
| `scripts/verify_multiview_correspondence.py`, `scripts/diagnose_target_separability.py`, `scripts/_make_bev_contact_sheet.py` | data-integrity and diagnostic utilities. |
| `vendor/` | third-party source copies. Only `vendor/DrivoR` (1398 files, including `nuplan-devkit`) is force-added to git; `SimWAM`, `FastWAM_...`, `Drive-JEPA` and `DrivoR_official` stay local because `.gitignore` excludes `vendor/`. |

`artifacts/`, `checkpoints/`, `logs/` and `reports/` are gitignored, so no
weights and no run logs are in this repository. The Wan payload must be
regenerated locally with `scripts/preprocess_wan_action_dit.py`; both stages
degrade to random theta initialization (with a logged warning) when it is absent.

SimWAM is not a runtime dependency — the package never imports it. It is only
used by `test_wan_scheduler_matches_simwam_source`, which diffs our flow port
against the upstream scheduler instead of against hard-coded numbers, and which
**skips** (does not fail) on a clone without `vendor/SimWAM`.

### Package modules

`data.py` Observation/PairTargets boundaries · `vision.py` DrivoR-exact DINOv3
Student · `teacher.py` frozen DINOv3 Teacher + ordered 2×2 compressor ·
`multiview_data.py` `/mnt/data/multi_view` readers · `real_data.py` NAVSIM
`CAM_F0` history · `target_cache.py` Stage 0 cache I/O · `modes.py` the immutable
64-anchor grid + rollout · `flow.py` rectified-flow schedule · `wan_dit.py` /
`wan_init.py` Wan2.2 Action DiT and its weight transfer · `model.py`
`DirectedPairDiT` (η/φ/θ) · `scorer.py` `ScratchDrivoRScorer` (ψ) ·
`selection.py` DrivoR's utility + argmax · `losses.py` · `oracle.py` online
NAVSIM labels · `stages.py` the Stage 1 step and optimizer plumbing ·
`config.py` the strict loader · `native_drivor.py` comparison-only, never on a
default training path.

`dino_tokens.py` and `lora.py` are dead code kept only because removing them
needs explicit authorization.

## Architecture

**Scene (η).** DrivoR's exact recipe on DINOv3. Frames arrive at DrivoR's
official `672×1148` (`drivoR.yaml:67 image_size: [1148, 672]`), produced by PIL
BICUBIC resize of the real 1920×1080 `CAM_F0` — never interpolation on our side.
1148 is not divisible by patch16, so `vision.pad_to_patch_multiple()` closes the
12 px remainder with **4 zero columns after normalization** (they land on the
dataset mean colour, not a black bar); the backbone is built at `img_size=(672,
1152)` → grid `42×72`. `_prepare()` refuses to resample and raises on any other
input size. 16 learnable scene queries are prepended per frame (official
`num_scene_tokens: 16`) and the first 16 backbone outputs are kept. Position
encoding is DrivoR's verbatim: DINOv3 has `pos_embed is None`, so
`_pos_embed` returns the tokens unchanged — **RoPE only, no absolute embedding,
no interpolation**. LoRA is rank 32 on q/v, `w_B` zero-initialized, base frozen.

Scene is **81 tokens** = 5 frames × 16 + exactly 1 ego token. The ego token is
11-wide: `pose3, velocity2, acceleration2` from the last history frame plus the
4-way `route_command`. There is no 12-wide pad in this package.

**Trajectories.** DrivoR-native metric `[x, y, heading]` in metres/radians,
8 steps at 0.5 s, `TRAJECTORY_STATE_DIM = 3`. All `/20`, `xy_scale`,
`encode_trajectory` and `decode_trajectory` are gone; nothing is rescaled.

**Action (θ).** SimWAM's Wan2.2 `ActionDiT` at `hidden_dim = 256`, loaded
`strict=True` (226 keys) from `checkpoints/ActionDiT_Wan22_256hdim_3outdim.pt`.
30 Wan blocks → 8 layers via `layer_map = [0,3,7,11,15,18,22,26]`; the
per-dimension 1D linear interpolation and `alpha = sqrt(d_src/d_tgt)` scaling are
SimWAM's own. Wan's convention `num_heads × attn_head_dim == hidden_dim` pins
**heads = 4** at `d_model = 256`. `action_encoder.`, `head.`, `mode_projection.`,
`ego_to_action_projection.` and `time_queries` stay randomly initialized.

**Rectified flow with an anchor prior.** SimWAM's
`WanContinuousFlowMatchScheduler` semantics — `x_σ = (1-σ)·data + σ·noise`,
`v_target = noise - data`, σ = timestep/1000, continuous timesteps, `shift = 5.0`
φ-warping — with one deliberate substitution: the σ = 1 endpoint is **not**
Gaussian noise but the immutable 64-anchor grid, so

```
x_σ      = (1-σ)·norm_odo(gt) + σ·norm_odo(anchor)
v_target = norm_odo(anchor) - norm_odo(gt)
x0       = x_σ - σ · velocity_to_metric_scale(v̂)
```

`norm_odo` is SimWAM's min-max map to `[-1,1]` (`x: 2(x+1.57)/66.74 - 1`,
`y: 2(y+19.68)/42 - 1`, `heading: 2(h+1.67)/3.53 - 1`); the Jacobian
`(33.37, 21.0, 1.765)` converts the velocity loss back to metres. The loss is
weighted by SimWAM's bell `training_weight(t)`, which is exactly 0 at both ends
and peaks at t = 500.

Training samples **one random σ per forward**. Inference pins σ = 1, where the
anchor *is* the noisy action and the recovery subtracts the full velocity, so
`JointStage2Module.inference()` is algebraically the same expression as the
trained path. **`action_sampling_loops = 0`: no rollout at train or eval.** The
64 anchors are velocity `[-4,-2,-1,0,1,2,3,4]` m/s × yaw rate
`[-0.30,-0.18,-0.08,0,0.04,0.08,0.18,0.30]` rad/s (`ImmutableModeBuffer`,
8×8 `torch.cartesian_prod`), rolled out by `StructuredSourceInitializer.clean`
with a small-angle unicycle reading `ego_history[:, -1, 3]` as vx.

**Directed attention.** Token order is `[Scene S | Action A | Register R]` with
`GROUP_ALLOW = ((1,0,0),(1,1,1),(1,0,1))`: `S←S`, `A←[S, A, stopgrad(R)]`,
`R←[S, R]`. Register never reads Action; Scene is never updated by either. Action
reads an 85-token cross-attention context (`S(81) | sg(R)(4)`). Each candidate
carries 8 action tokens and 4 future register tokens; registers of different
candidates never attend to each other.

**Scorer (ψ).** `ScratchDrivoRScorer` randomly initializes the official DrivoR
`Linear(24,1024)-ReLU-Linear(1024,256)` trajectory projection, the four-layer
`TransformerDecoderScorer` and the six-head `Scorer`. It never loads a DrivoR
checkpoint. All 64 candidates become trajectory queries attending to the Scene
tokens as K/V, then the last Scene token is added as the ego term, yielding
`[B, 64, 6]` logits for `NC, DAC, DDC, TTC, EP, comfort`.

Detach semantics are DrivoR's verbatim — **only proposals are detached**:

```python
embedded_traj = self.pos_embed(proposals.reshape(B, N, -1).detach())
tr_out = self.scorer_attention(embedded_traj, scene_features)   # scene_features NOT detached
tr_out = tr_out + ego_token
```

so `loss_score` does shape η. Selection is `aggregate_score_logits` → `argmax`,
returning `([B] pick, [B,64] utility)`.

## Supervision

Multi-GT, up to 4 GT trajectories per scene (`MAX_GT = 4`). The Stage 2 loss is
five independently weighted terms:

```
loss = λ_action·loss_action + λ_velocity·loss_velocity
     + λ_score·source_loss  + λ_rank·rank_loss + λ_future·loss_future
```

| Term | Weight | Note |
|---|---|---|
| `loss_action` | 1.0 | DrivoR's trajectory objective generalized to Multi-GT. Composed inside `multigt_trajectory_loss` of: Hungarian matched L1 at full weight (one assignment per forward over `[K, 64]`, shared with the future loss), a DrivoR dense nearest-GT pull at `dense_weight = 0.1` gated by `dense_gate_m`, and an anti-collapse diversity penalty at `diversity_weight = 1.0` |
| `loss_velocity` | 1.0 | SimWAM's `v_loss` on the rectified-flow velocity in `norm_odo` units, bell-weighted by t, so metres and radians contribute comparably |
| `source_loss` | 1.0 | DrivoR `trajectory_weight = final_score_weight = 1.0` |
| `rank_loss` | 0.1 | DrivoR `preference_weight × preference_rank_weight = 1.0 × 0.1`, weighted **independently** of `λ_score` — nesting it there diluted it 5× |
| `loss_future` | 0.1 | only when a Stage 0 cache exists; otherwise forced to 0 and φ stays frozen |

Because the dense pull covers every candidate against its own nearest valid GT,
the losers of the Hungarian matching are never left unsupervised — which is what
stops the set from degenerating into one good trajectory plus 63 junk ones.

Online score labels are `[B, 64, 7]` (`NC, DAC, EP, TTC, C, DDC, Final`), never
a `64×K` candidate-by-GT matrix.

Reporting separates the two ADE numbers on purpose: **matched-GT ADE** covers
only the Hungarian winners, **all-64 nearest-GT ADE** covers every supervised
candidate. Quoting the first alone is exactly how a collapsed candidate set hides.

## Training

Stage 1 first, then Stage 2 — `scripts/dlc_navtrain_wan_e2e.sh` runs them in order.

**Stage 0** (`build_future_target_cache.py`, once). The generated
`trajectory_k/future.mp4` files are 41 frames at 10 fps; future indices
`range(5,41,5)` give 8 frames at +0.5…+4.0 s, matching DrivoR's `num_poses = 8`.
The frozen Teacher encodes them at their **native `464×832`** (both divisible by
16 → grid `29×52`) with zero resize and zero interpolation — it raises on any
other size. Quadrants are cut by integer division (`29 → 14/15`, `52 → 26/26`),
so each of the 4 register tokens is one 2-frame chunk reduced from
`2 frames × 4 quadrants × 1024 = 8192` dims by a train-fitted PCA to 256. The
Teacher's grid is deliberately coarser than the Student's `42×72`; that is sound
because DINOv3's RoPE normalizes coordinates **per axis**
(`RotaryEmbeddingDinoV3`, `normalize_coords='separate'`), so a token's rotation
depends only on its fractional image position, not on how many patches the frame
was cut into. Reduction is streamed — a `[8,29,52,1024]` float32 grid is 50 MB
and the machine has 50 GB RAM.

**Stage 1** (`run_stage1_multiview.py`). Trains **η + φ only**; θ and ψ are
frozen by `_set_generator_trainability(model, 1)`. The action input is
**noise-free** by construction — `forward_register_only` runs the clean
trajectory, which is the σ = 0 end of the same flow Stage 2 noises. Loss is the
future-register alignment alone. Full-rank compressor fit needs only 16 scenes
(`fit()` rows = `16 × scenes`); below that the target is rank-deficient, so read
`compressor_rank` / `compressor_fit_rows` / `compressor_full_rank` from the audit
instead of guessing.

**Stage 2** (`run_navtrain_wan_e2e_ddp.py`). Trains **η + θ + φ + ψ** jointly
from the Stage 1 checkpoint (`--stage1-checkpoint`). Input is the noised action
at a random σ; the scorer is on. Parameter groups must stay complete, unique and
disjoint — `build_optimizers` asserts it.

## Commands

```bash
cd /mnt/zhaozc_workspace/project/NVS
source /usr/local/PPU_SDK/envsetup.sh ppu

# Contract suite (the wrapper owns the interpreter; see below)
bash scripts/run_contract_tests.sh

# Full two-stage pipeline
bash scripts/dlc_navtrain_wan_e2e.sh
```

Per-stage skips: `NVS_SKIP_STAGE0/1/2=1`. Stage 0 is skipped automatically when
`compressor.pt` already exists, unless `NVS_FORCE_STAGE0=1`. Extra arguments go
through `NVS_STAGE0_EXTRA_ARGS` / `NVS_STAGE1_EXTRA_ARGS` / `NVS_STAGE2_EXTRA_ARGS`.

Two interpreter rules that are easy to get wrong:

- Always `python -m pytest`, never the venv's `pytest` shim — its shebang is
  broken on this image.
- `PYTHONPATH` entries must be **absolute**. The venv's editable `navsim` `.pth`
  points at a nonexistent `/mnt/zczhao_workspace/...` (typo in the source venv),
  so relative paths silently fail to resolve.

```bash
export PYTHONPATH=/mnt/zhaozc_workspace/project/NVS/src:/mnt/zhaozc_workspace/project/NVS/vendor/DrivoR:/mnt/zhaozc_workspace/project/NVS/vendor/DrivoR/nuplan-devkit
```

## Data and caches

| Purpose | Path |
|---|---|
| Multi-GT generated futures (103075 navtrain scenes) | `/mnt/data/multi_future_view/navtrain_full_allcosmos_cuda_v1_gpu64/navtrain_full_allcosmos_cuda_v1_gpu64` |
| Same data via symlink | `/mnt/data/multi_view/...` |
| DrivoR-compatible online metric cache | `/mnt/zhaozc_workspace/project/DrivoR_multigt_kvcache_final_20261004/data_runtime/train_metric_cache` |
| Wan2.2 source weights | `/mnt/LLM_weight/Wan-AI/Wan2.2-TI2V-5B` |

Metric-cache layout: `data_runtime/train_metric_cache/<log>/unknown/<token>/metric_cache.pkl`.
`/mnt/zhaozc_workspace/official_DrivoR_clean/data_runtime/train_metric_cache` is a
symlink to the FastWAM offline cache and is **rejected** for DrivoR online
evaluation.

History frame filenames are **camera tokens, not frame tokens** — resolve them
through `frames[i]["cams"]["CAM_F0"]["data_path"]`. Logs are 2 Hz. `trajectory_k`
directory order does **not** equal candidate index (measured: `folder_slot !=
source_candidate_index` for 193/200 scenes), so pairing must go through the
explicit fields in `metadata.json`. About 19% of multi-GT scenes are degenerate
(duplicate or all-`human_log` labels, inter-GT pair L1 of 0.005–0.044 vs
0.133–0.167 for the rest); `--skip-degenerate` filters them by reading only
`trajectory_k/metadata.json`, so no mp4 is decoded for a scene about to be dropped.

## Current honest status

- The contract suite and the eval-path probe pass, and a 4-scene / 60-step real
  Stage 1 run is verified (`loss_future` 1.1769 → 0.1602,
  `parameter_groups_updated == {eta: true, theta: false, phi: true}`, Wan
  `loaded_keys: 226`). See `AGENTS.md` for the full record.
- **No `compressor.pt` exists anywhere under `/mnt/zhaozc_workspace` yet.** The
  Stage 0 cache has never been built, so the future-supervision path is verified
  against synthetic targets only, not against real Teacher output. Until Stage 0
  runs, the launcher prints a warning and passes `--future-target none`.
- Stage 1's reported ADE/FDE reflect a **randomly initialized** trajectory head
  on top of metric anchors, not trained accuracy — Stage 1 freezes θ by contract.
- DrivoR's `GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1,
  prob=0.7)` augmentation is **not** ported.
- No NAVSIM score in this repository is a performance claim. Oracle numbers
  require the real metric cache; synthetic smoke output never produces one.
