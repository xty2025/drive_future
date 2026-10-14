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

# 方案A — Joint 阶段对齐 DrivoR 感知容量（nvs_v6_wan_e2e，2026-10-13）

## 背景与动机

三阶段协议：Phase A（表征 eta+theta，20ep）→ 冻结 → 缓存 scene_context+proposals+labels → Phase B（scorer-only psi，50ep，带 rank loss）→ Stage2 Joint（eta+theta+psi，scene_context 不 detach，50ep）。

DrivoR 能安全使用 `final_score_weight=1.0`（score loss 直接回传感知）是因为它只训练 LoRA，base backbone 全程冻结，感知容量小。我们之前的 Joint 额外训练了 `scene_blocks`（8 层 1007-token self-attn，6.3M 参数），感知容量远大于 DrivoR。当 source_loss 权重拉到 1.0 时，未 detach 的 score 梯度会把感知冲垮（历史上 sel_pdms 0.978→0.85、ade 0.25→0.56 的回归即源于此）。

方案A：Joint 阶段冻结 `scene_blocks`，把可训练感知容量退回 DrivoR 的 LoRA 量级，于是 source_loss 可以安全地按 DrivoR 原值 1.0 训练；rank loss 按 DrivoR 原值 0.1 独立加权。目标是在不回退的前提下继续涨点、超过 Phase B 的 sel_pdms 0.978。

## 具体改动（3 处，均在 e2e 路径，未触碰 nvs_v6 / nvs_v6_wan）

1. `scripts/run_navtrain_wan_e2e_ddp.py` → `build_optimizers()`：在 phi 冻结之后、scorer 解冻之前，新增冻结 `scene_blocks` 的循环：
   ```python
   for name, parameter in model.named_parameters():
       if name.startswith("scene_blocks."):
           parameter.requires_grad_(False)
   ```
   `parameter_groups` 会跳过 `requires_grad=False` 的参数，因此 L446-448 的「generator 参数集合 == eta+theta 可训练集合」断言仍然成立。

2. `scripts/run_navtrain_wan_e2e_ddp.py` → 训练日志 `trainable` 字典：新增 `"scene_blocks_frozen": sum(p.numel() for n,p in model.named_parameters() if n.startswith("scene_blocks."))`，setup 日志中应 >0（实测 6318080），用于确认冻结生效。

3. `scripts/dlc_navtrain_wan_e2e_allnative_resume.sh` → Stage2 Joint 段：`--lambda-score` 由 0.2 改为 1.0（对齐 DrivoR final_score_weight），`--lambda-rank` 保持 0.1（对齐 DrivoR preference_rank_weight），`--lr 2e-5 --scorer-lr 5e-4`。rank loss 在 `run_navtrain_wan_e2e_ddp.py` 损失块中已「去嵌套」：`loss = lambda_action*loss_action + lambda_score*source_loss + lambda_rank*rank_loss`，rank 有效系数即 0.1（不再被 lambda_score 二次缩放成 0.02）。

## 回退方法（下次想恢复原行为）

- 恢复 scene_blocks 可训：删除 `build_optimizers()` 里那个 `for name, parameter in model.named_parameters(): if name.startswith("scene_blocks."): parameter.requires_grad_(False)` 循环即可（`trainable` 字典里的 `scene_blocks_frozen` 字段可保留，仅作日志）。
- 恢复温和 score 权重：把 resume 脚本 Stage2 段的 `--lambda-score 1.0` 改回 `0.2`。
- 两者独立：只删冻结循环 = 大容量感知 + score 1.0（会回归）；只改 lambda-score = LoRA+scene_blocks 可训但 score 温和。完全回到方案A之前需同时执行这两步。

## 验证

单卡冒烟（`--batch-size 1 --limit-scenes 4 --max-steps 2 --epochs 1`）通过：setup 显示 `scene_blocks_frozen=6318080`、`scene_kv_detached=false`、`checkpoint_loaded=false`；两步 loss_action/loss_score/loss_rank 全部有限；无 build_optimizers 断言失败、无 CPU/CUDA 设备不匹配；exit 0。注意：冒烟的 sel_pdms/ade 仅为合成小样本，不得据此宣称 NAVSIM 指标。

# nvs_v6_wan_e2e 契约改版（2026-10-14）

只影响 `src/nvs_v6_wan_e2e/`、`config/config_v6_wan_e2e.yaml`、`tests/contract_tests_v6_wan_e2e.py` 与 `scripts/run_stage1_multiview.py`。`src/nvs_v6/`、`src/nvs_v6_wan/` 及其脚本、测试**未改动**，上面各段对它们仍然成立。

**配置已拆分（重要）**：`config/config_v6_final.yaml` 是 `NAVSIM_..._Spec_v6_FINAL.md` 指定的 v6 唯一基线，被 `nvs_v6` 与 `nvs_v6_wan` 两个包的严格 loader 同时校验，仍是 66 scene token / 12 宽 ego（含 `checkpoint_pad0`）/ `xy_scale_m: 20.0` / `input_detach: [scene, trajectory, predicted_register]`。**不要**为了 e2e 去改它，那会同时打破两个兄弟包的 `test_strict_config_loader_rejects_direction_changes`。e2e 的契约写在独立的 `config/config_v6_wan_e2e.yaml`，由 `src/nvs_v6_wan_e2e/config.py` 校验。三份 loader 各自只认自己那一份。

## Scene 感知：DrivoR-exact（DINOv3 替换 DINOv2）

- 分辨率**照抄官方** `drivoR.yaml:67 image_size: [1148, 672]`（W,H → `H×W = 672×1148`），loader 用 DrivoR 自己的 `im.resize((1148,672))` 从真实 `CAM_F0` 的 **1920×1080** 做 PIL BICUBIC **降采样**（丢像素，不造像素）。官方是 **patch14**（`1148 = 14×82`，grid `48×82` = 3936），DINOv3 是 **patch16**，`1148` 不能被 16 整除：timm 的 `PatchEmbed` 直接 `AssertionError: Input width (1148) should be divisible by patch size (16)`。
- 所以那 12 px 余数用 **4 列零填充**闭合，**不插值**：`vision.pad_to_patch_multiple()` 走 `F.pad`，DrivoR 的 1148 列像素逐字节原样通过，多出的 4 列不含任何被编造的内容。backbone 以 `img_size=(672,1152)` 构建 → grid `42×72` = 3024 patches/帧。
- **不要用 timm 的 `dynamic_img_pad=True` 绕过那个 assert**：它会跳过检查并 `F.pad` 到 1152，但 `patch_embed.grid_size` 仍停在 `floor(1148/16)=71`，而填充后的 forward 吐 72 列，`forward_features` 里直接 `IndexError: too many indices for tensor of dimension 3`。填充必须在外部做、并声明 `img_size=(672,1152)`，`grid_size` 才和 RoPE 一致。
- 零填充在**归一化之后**施加（`_prepare`），所以那 4 列在归一化空间里恰好是 `0.0` = 数据集均值色，而不是一道会被读成 OOD 内容的硬黑边。
- `DrivoRDinoV3Student._prepare()` **拒绝任何重采样**：传入尺寸不是 `(672,1148)` 就 `ValueError("the scene encoder does not resample: ...")`，连已经填充好的 `1152` 也拒（那意味着 loader 自己插值过）。旧的 `F.interpolate` 兜底分支已删除。
- **图像 resize 必须是 PIL BICUBIC。** DrivoR `drivor_features.py:98-107` 调 `im.resize(image_size)` 不传 `resample`，继承 Pillow 默认（BICUBIC），且是**纯 resize 不裁剪**——内参按轴分别缩放（`cam_K[0,0] *= 1148/1920`，`cam_K[1,1] *= 672/1080`），证明它是宽高比 1.778→1.708 的非均匀拉伸。本包一度写死 BILINEAR，已改。历史帧 JPEG 走 `multiview_data._resize_rgb()` 这**一条**路径；**未来帧不走它**（见下）。
- 每帧前置 `num_scene_tokens = 16` 个可学习 scene query（官方 `num_scene_tokens: 16`），取 backbone 输出的前 16 个 token，再经 `neck: Linear(1024, d_model)`。
- 位置编码**逐字照抄 DrivoR**：`vendor/DrivoR/.../image_encoder/dinov2_lora.py:23-26` 的 `timm_ViT._pos_embed` 第一行就是 `if self.pos_embed is None: return x.view(x.shape[0], -1, x.shape[-1])`。DINOv3 的 `pos_embed is None`，所以 DrivoR 对 DINOv3 **不加任何绝对位置编码，纯靠 RoPE，零插值**；那条 `resample_abs_pos_embed` 分支（28-37 行）只在 `dynamic_img_size` **且** `pos_embed is not None` 时触发，即只有 DINOv2 会走到。本包 `vision.py` 一直就是这个行为，并显式 `raise` 掉 `pos_embed is not None` 的情况。`rope_prefix_tokens = 5 + 16 = 21`。
- **DINOv3 的 RoPE 是按轴归一化的，与网格大小无关**（`timm/layers/pos_embed_sincos.py:1027` `RotaryEmbeddingDinoV3`，`make_coords_dinov3` 默认 `normalize_coords='separate'`：`h_denom=float(height)`、`w_denom=float(width)`，先算 `(i+0.5)/H` 再 `2.0*coords-1.0` 映到 `[-1,1]`）。一个 token 的旋转只取决于它在画面里的**分数位置**，不取决于画面被切成多少块——实测 grid `42×72`、`29×52`、`42×71`、`48×82` 都覆盖约 `[-0.98,0.98]`，角点差异只有 `1/29` vs `1/42` 的离散化步长。**这就是 Teacher 和 Student 允许用不同分辨率的全部依据**，也推翻了本文件此前「两者必须同分辨率」的说法。
- LoRA 保持开启：官方 rank 32、仅包 `attn.qkv` 的 q/v、`w_B` 零初始化（故初始等价于 base）、base 全程冻结。
- **`_widen_rope_prefix()` 只在 `__init__` 调用一次。** 不要在 forward 里临时改 `EvaAttention.num_prefix_tokens` 再还原：`torch.utils.checkpoint` 在反向时重算 block，那时偏移已被还原，RoPE 会静默错位。注意 timm 的 `EvaAttention` 持有**自己独立的**同名属性，改 Eva 模块上的那个没用。

## Token 与 ego

- `scene_tokens = 5 帧 × 16 + 1 ego = 81`（scorer 的 `scene_tokens` 默认值即 81）。
- ego 是**恰好 1 个** token：`hist_encoding = Linear(11, d_model)`，输入 `pose3,velocity2,acceleration2`（只取当前帧 `ego_history[:, -1]`，7 维）拼 `route_command`（4 维）= **11 维**。本包**没有** 12 维 pad；上文第 8 行那条 12 宽接口约束只适用于 `src/nvs_v6/`。
- ego 是 Scene 的**最后一个** slot，scorer 以 `scene_context[:, -1:]` 原样（不过 LayerNorm）取出做 `ego_token` 相加。

## 轨迹：DrivoR 原生米制，无任何归一化

- `[x, y, heading]`，单位米/弧度，`TRAJECTORY_STATE_DIM = 3`。
- 已彻底删除 `/20`、`xy_scale`、`encode_trajectory`、`decode_trajectory`、`gt_encoded`。单步输出仍是 `endpoint = source + delta`。
- `StructuredSourceInitializer.clean` 用 `ego_history[:, -1, 3]`（vx）做小角度 unicycle rollout，输出米制 `[B,M,8,3]`。

## Action 分支：SimWAM 的 Wan2.2 ActionDiT @ hidden_dim=256

- 权重来自 `checkpoints/ActionDiT_Wan22_256hdim_3outdim.pt`，由 `wan_init.build_action_dit_backbone` 从 `/mnt/LLM_weight/Wan-AI/Wan2.2-TI2V-5B` 直接读 safetensors 分片构建（不实例化 5B 视频 DiT）。
- 插值策略与 SimWAM 逐字一致：逐维 1D `F.interpolate(mode="linear", align_corners=True)`，`alpha = sqrt(src_last_dim / target_last_dim)`。
- 30 个 Wan block → 8 层，`select_source_layers` 均匀取步，`layer_map = [0,3,7,11,15,18,22,26]`。
- skip 前缀 `action_encoder.` / `head.` / `mode_projection.` / `ego_to_action_projection.` / `time_queries` 保持随机初始化；其余 226 个 key 以 `strict=True` 加载。
- meta 必须逐项匹配：`{hidden_dim:256, ffn_dim:1024, num_layers:8, num_heads:4, attn_head_dim:64, text_dim:256, freq_dim:256, eps:1e-6}`。构造 `NvsActionDiT` 时 `text_dim` 与 `ego_dim` 都必须显式等于 `hidden_dim`/`d_model`，否则构造期就报错。

## Scorer 的 detach 语义（不要"修回去"）

按官方 `drivor_model.py` 逐字核实：**只有 proposals 被 detach**，scene K/V 不 detach，所以 `loss_score` 会回传塑造 eta（DINOv3 LoRA + scene queries + scene_blocks）。

```python
embedded_traj = self.pos_embed(proposals.reshape(B, N, -1).detach())   # 只 detach proposals
tr_out = self.scorer_attention(embedded_traj, scene_features)          # scene_features 不 detach
tr_out = tr_out + ego_token
```

配置里对应 `scorer.input_detach: [trajectory, predicted_register]`、`scene_kv_detached: false`、`updates_scene_lora: true`。上文第 7 行「detached K/V」与第 48 行「Score inputs are detached」描述的是 `nvs_v6` 的旧约定，**在 e2e 包里是错的**。`config.py` 现在会拒绝旧写法。

## Stage 1 真数据

- 入口 `scripts/run_stage1_multiview.py`（取代已失效的 `run_two_scene_overfit_wan_e2e.py`）。
- Future Register 目标来自 `/mnt/data/multi_view` 的生成多视角 `future.mp4`：41 帧 @10fps，未来帧索引 `range(5,41,5)` → +0.5…+4.0s，正好对上 DrivoR `num_poses = 8`。
- 历史帧是真实 NAVSIM `CAM_F0` sensor blob。**文件名是相机 token 不是帧 token**，必须经 `frames[i]["cams"]["CAM_F0"]["data_path"]` 解析。日志是 2 Hz。
- metric_cache 路径与 DrivoR `train_metric_cache` 完全兼容：`data_runtime/train_metric_cache/<log>/unknown/<token>/metric_cache.pkl`。
- **Teacher 在未来帧的原生分辨率 `464×832` 上编码，零 resize、零插值。** 生成的 `future.mp4` 原生就是 `464×832`（H×W），而 `464 = 16×29`、`832 = 16×52` 都整除 16，patch16 正好铺满 → grid `29×52` = 1508 patches/帧。把 `464×832` 上采样到 Student 的尺寸会凭空造出约 1.4× 于实际存在的像素，那是本轮删掉的做法（更早是 `448×800`）。Teacher 的 grid 比 Student 的 `42×72` **粗**，这是刻意的：见上「RoPE 按轴归一化」，ordered-2×2 的四个象限同样按**画面分数**切分，所以粗网格的目标依然和 scene token 指向同一块图像区域。`config.py` 现在断言 `teacher_image_hw == [464,832]`、`teacher_grid_hw == [29,52]`、每一边 `% 16 == 0`、`teacher_resize_filter == "none_native_resolution_no_interpolation"`、`rope_normalizes_coords_per_axis is True`、`teacher_checkpoint == vision.checkpoint`，并有负向测试 `test_strict_config_rejects_resampling_the_future_frames` 与 `test_strict_config_rejects_resampling_instead_of_zero_padding`。
- `TimmDinoV3Teacher.forward` 收到非原生尺寸直接 `ValueError("Teacher refuses to interpolate: ...")`，不兜底。象限归约统一到 `teacher.ordered_quadrant_vectors()`（脚本里那份重复实现已删）；象限边界用**整除**，所以奇数网格也能用且不丢 token——`29×52` 下行切 `14/15`、列切 `26/26`。旧的「ordered 2×2 要求偶数网格」`ValueError` 已移除。
- Teacher 编码必须**流式归约**：一个 grid 相对它的 pooled 形式极大——原生分辨率下 `[8,29,52,1024]` float32 = **50 MB**，Student 尺寸下 `[8,42,72,1024]` = **103 MB**，而 ordered vectors `[gt,4,8192]` 两种情况都只有 **131 KB**。`_teacher_ordered_vectors()` 产出后立即降到 ordered vectors 并丢弃 grid；目标构建完就 `replace(sample, future_rgb=None)` 释放解码帧。机器只有 50 GB RAM。`TimmDinoV3Teacher` 另有 `frame_chunk=2`，因为 attention 对 patch 数是平方复杂度。
- **满秩拟合只需 16 个 scene**，不是 65 个（旧值算错了）。`Ordered2x2Compressor.fit()` 先 `reshape(-1, last_dim)`，行数 = `scene 数 × 4 GT × 4 chunk = 16 × scene 数`；4 scene → `compressor_fit_rows: 64`，16 scene → `256` = `target_dim`，满秩。
- 数据根目录现在指向 `/mnt/data/multi_future_view/navtrain_full_allcosmos_cuda_v1_gpu64/navtrain_full_allcosmos_cuda_v1_gpu64`（103075 scene）。`/mnt/data/multi_view/...` 是指向同一目录的**符号链接**，两种写法是同一份数据。

## 已知限制

- `Ordered2x2Compressor` 的秩 = `min(target_dim, 拟合行数)`，拟合行数 = `16 × scene 数`（见上「Stage 1 真数据」）。`--scenes` 小于 16 时目标是降秩的，脚本 audit 会输出 `compressor_rank` / `compressor_fit_rows` / `compressor_full_rank`，看这三个字段而不是猜。
- 抽样 200 个 scene 实测 **19%（38/200）** 的 multiGT scene 是退化的（label 重复或全 `human_log`，GT 间 pair L1 仅 0.005–0.044，非退化 scene 为 0.133–0.167）。用 `--skip-degenerate` 过滤；它只读 `trajectory_k/metadata.json`，不会为要丢弃的 scene 付 mp4 解码代价。
- 同一批 200 scene 的对应关系核验：轨迹与日志 `pose_interpolation` 对齐的只有 **241/800**（30%），`folder_slot != source_candidate_index` 的有 **193/200**——即 `trajectory_k` 的目录序号**不等于**候选轨迹序号，配对必须走 metadata 里的显式字段，不能靠目录名下标。
- DrivoR 训练时开的 `GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)`（`dinov2_lora.py:341`，`use_grid_mask = True`）**本包尚未移植**。
- Stage 1 冻结 theta，所以 `scorer_audit.proposals_vs_multigt` 里的 ADE/FDE 反映的是**随机初始化的轨迹头**叠加在米制 mode 锚点上的结果，不是训练后的轨迹精度。60 步真数据跑出的 `min_ade_over_gt_m: 6.10` / `best_pair_ade_m: 1.32` 只说明 Stage 1 没有动 theta（符合契约），轨迹质量要到 Stage 2 才有意义。`min_ade_over_gt_m` 的口径是「64 个 proposal 各自对 4 个 GT 取最小 ADE，再对 proposal 求均值」，衡量的是 proposal 集合对 GT 集合的覆盖度。
- 包内 `dino_tokens.py`、`lora.py` 已是死代码（旧测试对 `split_dinov3_tokens` 的引用已随重写移除），删除需用户明确授权。
- 以下 legacy e2e 脚本本轮**未**重新验证：`run_two_scene_overfit_wan_e2e.py` 已确认损坏（仍 import 已删除的 `model.TimmDinoV3Student`，被 `run_stage1_multiview.py` 取代）；`run_navtrain_wan_e2e_ddp.py`、`run_navtrain_wan_e2e_twophase_ddp.py`、`eval_navtest_official_pdms_e2e.py`、`run_real_two_stage_smoke.py` 的 import 仍能解析（不引用已删除符号），但运行时是否满足新契约（81 Scene token / 11 维 ego / 米制轨迹）尚未跑过，用前先冒烟。

## 验证

```bash
export PYTHONPATH=/mnt/zhaozc_workspace/project/NVS/src:/mnt/zhaozc_workspace/project/NVS/vendor/DrivoR:/mnt/zhaozc_workspace/project/NVS/vendor/DrivoR/nuplan-devkit
/mnt/zhaozc_workspace/envs/driov_dpo/bin/python -m pytest tests/contract_tests_v6_wan_e2e.py -q   # 58 passed
```

PYTHONPATH 必须用绝对路径：该 venv 的 editable navsim `.pth` 指向不存在的 `/mnt/zczhao_workspace/...`（拼写错误）。`pytest` shim 的 shebang 是坏的，永远用 `python -m pytest`。

真数据 Stage 1（4 个非退化 scene、60 步、candidates=64、depth=8，910s）已验证：`loss_future` 1.1769 → 0.1602 单调下降且全部有限；`paired_better_than_shuffled: true`（0.16291 < 0.16593）；`mean_register_std_across_candidates` 0.975 → 0.411（>0，确实在预测彼此不同的 register）；`parameter_groups_updated == {eta: true, theta: false, phi: true}`（Stage 1 冻结 theta，故 theta 组为空是设计使然，`parameter_groups` 会跳过 `requires_grad=False`）；`scorer_audit.logits_finite: true`、`scene_context_shape == [1,81,256]`；Wan backbone `loaded: true` / `loaded_keys: 226`；4 个 scene 的 `history_policy` 全为 `real_CAM_F0_sensor_blob`，`matched_pairs` 全为 4。结果在 `artifacts/stage1_multiview/stage1_multiview_result.json`。
