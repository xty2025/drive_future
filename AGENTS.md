# NVS v6 maintenance constraints

- Authoritative contract: `NAVSIM_MultiGT_FutureRegister_ActionDiT_Codex_Spec_v6_FINAL.md`; implementation is in `src/nvs_v6/model.py`, training groups/steps in `src/nvs_v6/stages.py`, and contracts in `tests/contract_tests_v6.py`.
- `SceneEncoder.encode_clean_ego()` encodes observable ego state/history plus the existing route command. `DirectedPairDiT._initialize_action()` applies theta-owned `ego_to_action_projection` and forms `A0 = source_projection(Z) + projected_clean_ego + time_embedding + mode_projection(U)`. This is feature addition, not raw ego/trajectory-coordinate addition; candidate/time sizes come from tensors/config.
- Preserve token order `[Scene S | Action A | Register R]` and directed reads: `S<-S`, `A<-[S,A,stopgrad(R)]`, `R<-[S,R]`. Register never reads Action; Scene is not updated by Action/Register; keep each grouped attention as one shared-softmax block.
- Stage1 freezes theta (including the Action-only ego projection) and scorer. Stage2 unfreezes theta; Action loss may update the projection and shared clean Ego encoder, Future loss must not update the Action-only projection, and Score inputs are detached so only scorer parameters update. Optimizer parameter IDs must be complete, unique, and disjoint.
- Production scoring is `ScratchDrivoRScorer`: randomly initialize the official DrivoR `Linear(24,1024)-ReLU-Linear(1024,256)`, four-layer `TransformerDecoderScorer`, and six-head `Scorer`; never load a DrivoR scorer state dict. Its detached K/V is NVS DINOv3 `scene_context [B,66,256]`, and Ego addition pools the final two ego-fused Scene tokens. Native DINOv2 Scene and native Ego caches are forbidden on default training, real smoke, and epoch-search paths; `NativeDrivoRScorer` is explicit comparison-only. Online labels are `[B,64,7]`, never candidate-by-GT; `[B,64,4,d]` means four future chunks and only valid Hungarian-matched GT rows receive Future loss.
- Preserve the real ego order `pose3,velocity2,acceleration2,command4`; the final zero exists only for the 12-wide NVS checkpoint interface. DINOv3 must retain CLS and all available register tokens, support register-free API outputs, and keep image patches in backbone Q/K/V without changing the downstream 64 visual + 2 ego Scene-token default.
- Validate with `PYTHONPATH="src:vendor/DrivoR" pytest -q tests/contract_tests_v6.py` and a minimal local smoke such as `PYTHONPATH="src:vendor/DrivoR" python scripts/run_two_stage_smoke.py --steps 1 --candidates 8 --output /tmp/nvs_v6_smoke.json`. Do not claim NAVSIM metrics from synthetic smoke tests.

# Model architecture summary (for new agents)

Read this before touching `src/nvs_v6/`. The authoritative spec is `NAVSIM_MultiGT_FutureRegister_ActionDiT_Codex_Spec_v6_FINAL.md`; this section is the orientation layer.

## Parameter groups

- `η` (eta): perception — DINOv3 LoRA adapters, scene queries/projections, shared clean-ego MLP. Base DINOv3 weights stay frozen at every stage.
- `φ` (phi): Future Register branch — per-candidate future tokens and their time queries.
- `θ` (theta): Action branch — action encoder, action time/mode embeddings, `ego_to_action_projection`, endpoint/trajectory decoder.
- `ψ` (psi): `ScratchDrivoRScorer` — official DrivoR trajectory projection + `TransformerDecoderScorer` (4 layers) + six-head `Scorer`, all randomly initialized, never loaded from a DrivoR checkpoint.

## Token layout and directed attention

- Scene tokens: `[B, 66, 256]` = 64 visual (32 current-frame + 4×8 history) + 2 ego-fused.
- Logical order is `[Scene S | Action A | Register R]`; directed reads are `S<-S`, `A<-[S, A, stopgrad(R)]`, `R<-[S, R]`. Register never reads Action; Scene is never updated by Action/Register.
- Action initialization: `A0 = source_projection(Z) + ego_to_action_projection(clean_ego) + action_time_embedding[j] + mode_projection(U[b,m])`. Ego conditioning enters by feature addition, not coordinate concatenation.
- Each candidate has 8 Action tokens and 4 Future Register tokens (future chunks at +0.5/+1.0, +1.5/+2.0, +2.5/+3.0, +3.5/+4.0s). Registers of different candidates never attend to each other.
- Vision input is always 224×224, patch16 → 196 patches per frame; current frame carries 27 injected scene queries (backbone Q/K/V = 228, downstream output = 32), each history frame carries 3 (backbone Q/K/V = 204, downstream output = 8). Do not change these counts without updating the 64-visual-token default.

## Data contract

- Each scene has up to 4 GT trajectories (`K<=4`). Hungarian matching over `[K, 64]` produces one assignment shared by Action loss and Future loss.
- Future Register targets are `[B, K, 4, D]`; the `4` is future chunks, **not** 4 GTs. Only valid Hungarian-matched GT rows receive Future loss.
- Score supervision is online NAVSIM oracle labels `[B, 64, 7]` (NC, DAC, EP, TTC, Comfort, DDC, Final PDMS) for all 64 candidates. It is never a `64×K` candidate-by-GT matrix. Multi-GT does not modify the DrivoR scorer network or loss.

# Training recipe (current, verified on two-scene overfit)

## Stage1 — Future representation

- Train `η + φ`; freeze `θ + ψ`.
- Loss: paired Future L1 only (`loss_future`), one Hungarian assignment per step.
- Entry: `scripts/run_two_scene_overfit.py --stage1-epochs N` (joint mode) or the Stage1 section of `stage2_step`.

## Stage2 — joint generator + scorer (original design)

- Train `η + θ + φ + ψ`.
- Loss: `1.0*action + 0.1*future + 1.0*score`.
- Score inputs (`scene_context`, `trajectory`, `predicted_register`) are detached before the scorer, so `loss_score` only updates `ψ`; `loss_action`/`loss_future` only update generator groups.
- Entry: `scripts/run_two_scene_overfit.py --stage2-epochs N`.

## Stage2-scorer-only — fixed-proposal ranking (added after the 50/50 run)

- Freeze the entire generator (`model.eval().requires_grad_(False)`); generate 64 candidates once; compute oracle labels once; train only `ψ` with the official `DrivoRLoss.score_loss`.
- Why: in joint Stage2 the generator keeps moving the candidate distribution, so the scorer chases a moving target and ranking can regress (observed: extra 50 joint epochs dropped selected PDMS to 0.58/0.88). With fixed proposals the scorer reached 1.0/1.0 in 22 epochs.
- Entry: `scripts/run_two_scene_overfit.py --scorer-only --stop-selected-pdms 0.999999 --resume-checkpoint <ckpt>`. The flag stops early once every scene's selected PDMS meets the threshold; `result.json` records `scorer_only: true` and the actual trained epoch count.
- Verified result (two-scene overfit): Stage1=50, Stage2 joint=50, scorer-only=22 → both scenes selected PDMS 1.0, matched-GT max ADE 0.378m / 0.305m. Generator parameters are byte-identical to the pre-scorer-only checkpoint; only `ψ` changed.

## Environment and validation

- Standard runtime: `source /usr/local/PPU_SDK/envsetup.sh ppu && source /mnt/zhaozc_workspace/envs/driov_dpo/bin/activate`, then `export PYTHONPATH="src:vendor/DrivoR:vendor/DrivoR/nuplan-devkit"` (or the absolute equivalents used by `scripts/launch_epoch_search.sh`). The bare `pytest` shim in that venv has a broken shebang; always invoke `python -m pytest`.
- Contract tests: `python -m pytest -q tests/contract_tests_v6.py` (currently 34 passed).
- Real two-scene overfit: `python -u scripts/run_two_scene_overfit.py --stage1-epochs 50 --stage2-epochs 50 --steps-per-epoch 10 --depth 8 --seed 17 --output-dir artifacts/<name>`.
- Never claim NAVSIM metrics from synthetic smoke tests; oracle numbers require the real metric cache under `DEFAULT_CACHE`.

# TODO: split Stage2 for full navtrain

The two-scene recipe above is verified; scaling it to the full `navtrain` split requires the following work. Each item is independently testable — do them in order and keep contract tests green.

- [ ] **Stage2-A (generator-only, no oracle).** Add a training mode that unfreezes `η + θ + φ`, freezes `ψ`, and skips the online NAVSIM oracle entirely (loss = action + 0.1*future). This removes the most expensive per-step component and should be the default full-dataset generator phase. Wire it into `stages.py` as a first-class step function rather than a script flag so both two-scene and navtrain paths share it.
- [ ] **Offline candidate/oracle cache builder.** After Stage2-A converges, run the frozen generator once over every navtrain scene and persist per-scene records: `scene_context [66,256]` (fp16), `candidate_trajectory [64,8,3]`, `oracle_raw7 [64,7]`, `valid_mask [64]`, `scene_id`. Target a compact on-disk format (e.g. npz shards or a memory-mapped store) sized for ~100k scenes at a few GB total. Include a manifest with generator checkpoint hash so stale caches are detectable.
- [ ] **Stage2-B (scorer-only on cached data).** Train `ψ` from the cache with the official `DrivoRLoss.score_loss`; no DINOv3, no generator forward, no oracle calls per epoch. Support multi-epoch shuffling over scenes, resume from a scorer checkpoint, and log per-epoch selected-PDMS distribution (not just mean) so ranking quality is visible during long runs.
- [ ] **Cache invalidation on generator updates.** Any generator fine-tune after Stage2-B invalidates the cache. Encode the rule: generator change → rebuild cache → re-run Stage2-B calibration. Do not allow joint training to silently reuse a stale cache.
- [ ] **Optional low-LR joint calibration.** If a final short joint phase is desired, cap it (few epochs, reduced LR), checkpoint every epoch, and select by held-out selected-PDMS rather than training loss. The two-scene evidence shows long joint phases regress ranking; treat this as calibration, not training.
- [ ] **Navtrain dataloader integration.** Extend `real_data.py`/`rollout_data.py` to enumerate navtrain scenes, load per-scene multi-GT targets, and feed both Stage2-A and the cache builder. Verify route command remains strictly 4-dim and ego order stays `pose3,velocity2,acceleration2,command4`.
- [ ] **Distributed training hooks.** Full navtrain Stage2-A and Stage2-B should support multi-GPU (DDP or FSDP) with the same parameter-group disjointness asserts that `build_optimizers` already enforces. Scorer-only is small enough to shard by scene; generator-only needs gradient accumulation tuned to the 64-candidate batch.
- [ ] **Evaluation harness.** Add a navtrain-level eval that reports selected-PDMS, oracle-best-PDMS gap, matched-GT ADE, and scorer-oracle rank agreement across a held-out scene subset. Reuse the per-scene evaluation block already in `run_two_scene_overfit.py` but aggregate over the split.

# Packaging note

A weight-free source archive was produced at `/mnt/zhaozc_workspace/NVS_source_no_weights_20261010.zip` (1436 text-source files, ~7 MiB, SHA256 `0f3dff9bb8df1766b3932a29004f74d740f358c4fc9d89a2fee4226eaaf656a7`). It excludes `artifacts/`, `checkpoints/`, `vendor/*/weights/`, all binary tensor/data extensions, `__pycache__`, `.pytest_cache`, and vendor files containing hardcoded third-party API keys. The archive was verified by extracting to a temp dir and re-running `python -m pytest -q tests/contract_tests_v6.py` (34 passed). Regenerate with the same whitelist approach if the tree changes; never zip the project root directly.

# Wan Action DiT variant (nvs_v6_wan, added 2026-10-10)

- `src/nvs_v6_wan/` is a full copy of `src/nvs_v6`; the only difference is the Action branch of `DirectedPairDiT`, which now uses Wan2.2-style blocks (reference: `vendor/FastWAM_dino_all269_peft_qkv_lora_ema_joint/src/fastwam/models/wan22/wan_video_dit.py`). All Wan-style modules are randomly initialized; no Wan checkpoint is ever loaded. The original `src/nvs_v6`, its scripts and its tests remain untouched.
- `WanActionBlock` = self-attn (`A<-A` per candidate, per-head QK RMSNorm + 1D RoPE over the 8 action tokens) → cross-attn (`A<-[S(66) | stopgrad(R)(4)]`, 70-token context, no RoPE) → GELU(tanh) FFN, with Wan's six-parameter adaLN modulation (per-block `modulation(1,6,d)` + shared `t_mod`). The v6 directed reads `S<-S`, `R<-[S,R]`, `A<-[S,A,sg(R)]` are preserved semantically; the Wan variant expresses the Action read as self+cross instead of one shared softmax (that single-softmax contract only binds `src/nvs_v6`).
- adaLN conditioning replaces Wan's diffusion timestep (NVS has no flow time): `cond = theta.mode_projection(U) + theta.ego_to_action_projection(clean_ego)`; `theta.cond_projection` (SiLU + Linear(d,6d)) yields `t_mod [B*M,6,d]`. `WanActionHead` (modulated affine-free LayerNorm + Linear(d,4)) replaces the old trajectory head. The A0 feature-addition formula, register detach, Stage1/Stage2 freezing and optimizer-group contracts are unchanged; all new parameters live under `theta.` so `parameter_groups` needed no edits.
- Single-step output preserved: `endpoint = source + head(action, cond)` → `decode_trajectory`; loss remains `paired_residual_velocity_mse`; zero integration/sampling loops.
- Entry points: `scripts/run_two_scene_overfit_wan.py` (same flags as the v6 script) and `python -m pytest -q tests/contract_tests_v6_wan.py` (35 passed; the KV-length test now asserts action self=8 / cross=70 instead of the shared 78).
- Artifacts: `artifacts/two_scene_overfit_wan_50x50` (joint Stage1=50/Stage2=50) and `artifacts/two_scene_overfit_wan_scorer_only` (fixed-proposal scorer calibration), log at `artifacts/two_scene_overfit_wan_50x50.log`.
- Verified two-scene results (seed 17, depth 8): joint run 815s, all losses finite, all four parameter groups updated, joint selected PDMS 0.9685/0.9722 (v6 baseline 0.964/0.870); scorer-only reached selected PDMS 1.0/1.0 in 20 epochs (16.7s) with the generator state_dict byte-identical to the joint checkpoint and 108/108 scorer tensors changed. Caveat: matched-GT ADE regressed vs the v6 baseline (max 1.81/1.60m vs 0.378/0.305m) and the final action loss was 0.0029 (baseline 0.00023) still trending down, so the Wan action branch was not fully converged at 50 joint epochs; extend Stage2 if trajectory accuracy matters.
