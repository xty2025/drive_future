# NVS 维护约束（nvs_v6_wan_e2e 唯一谱系）

本仓库现在只有一条实现谱系：`src/nvs_v6_wan_e2e/`，配置 `config/config_v6_wan_e2e.yaml`，
契约测试 `tests/contract_tests_v6_wan_e2e.py`。`src/nvs_v6/`、`src/nvs_v6_wan/`、
`config/config_v6_final.yaml`、`tests/contract_tests_v6.py` 及其脚本已在 2026-10-14
收敛时删除，需要历史版本请从 commit `52a9d64` 取回。本文件此前描述那两个包的段落
（66 scene token / 12 宽 ego / 224×224 视觉 / scorer K/V detach / 单步
`endpoint = source + head(...)`）对现在的代码**全部不成立**，已随包一起删除。

- 权威契约文档：`NAVSIM_MultiGT_FutureRegister_ActionDiT_Codex_Spec_v6_FINAL.md`。
  它是历史 spec，其中面向 `nvs_v6` 的部分（66 token、12 宽 ego、`xy_scale_m`、
  detach 的 scorer 输入）已被 e2e 契约取代；冲突时以 `config/config_v6_wan_e2e.yaml`
  和 `src/nvs_v6_wan_e2e/config.py` 的严格 loader 为准。
- 面向用户的说明在 `README.md`；本文件是给改代码的 agent 看的约束层，不重复架构描述。
- `config.py` 是严格 loader：它会**拒绝**任何偏离契约的配置写法（包括上面那些旧写法）。
  改契约必须同时改 yaml、loader 断言和负向测试，三处一起。

## 硬性规则（不要绕过）

- **不要声称任何 NAVSIM 指标来自合成冒烟。** oracle 数字必须有 `DEFAULT_CACHE` 下的真实
  metric cache；冒烟只能证明 loss 有限、参数在更新、shape 正确。
- **保留真实 ego 顺序 `pose3, velocity2, acceleration2, command4`。** e2e 里可观测 ego
  特征是 **11 维**（取 `ego_history[:, -1]` 的 7 维拼 4 维 command），**没有** 12 宽 pad；
  `ego_history` 张量本身是 7 宽。旧的 12 宽 checkpoint 接口只属于已删除的 `nvs_v6`。
- **轨迹永远是米制 `[x, y, heading]`，任何地方都不做 rescale。** `/20`、`xy_scale`、
  `encode_trajectory`、`decode_trajectory`、`gt_encoded` 都已删除，不要加回来。
- `native_drivor.py`（原生 DINOv2 Scene + 原生 Ego cache）**只用于显式对比**，禁止出现在
  默认训练路径、真数据冒烟或任何 launcher 里。
- 永远用 `python -m pytest`，不要用 venv 的 `pytest` shim（shebang 在本镜像上是坏的）。
  `PYTHONPATH` 必须写绝对路径：该 venv 的 editable `navsim` `.pth` 指向不存在的
  `/mnt/zczhao_workspace/...`（源头拼写错误），相对路径会静默解析失败。
- `torch.load` 一律 `weights_only=True`（仓库有 PostToolUse 安全钩子会标记 `False`）。
  只有在 payload 确实含非张量对象、且已确认来源可信时才例外，并要在代码里写明原因。
- 本镜像没有 `/usr/bin/time`，计时用 bash 的 `$SECONDS`。机器只有 **50 GB RAM**，
  任何持有 per-scene 大张量的循环都要算一遍驻留量（Stage 1 约 195 MB/scene，
  见 `dlc_navtrain_wan_e2e.sh` 的 preflight）。
- 提交前用 `git status` 复核暂存内容；**永远不要** `git add -A` / `git add .`。
- `dino_tokens.py` 和 `lora.py` 是死代码，但删除需要用户明确授权，不要顺手清掉。
  `manifest.py` 和 `rollout_data.py` 已在本次收敛中删除（同样无人 import，且
  `rollout_data` 只是 `multiview_data.load_multiview_sample` 的透传壳）。

## 契约要点（改动时最容易破坏的几处）

- Token 顺序 `[Scene S | Action A | Register R]`，定向读
  `GROUP_ALLOW = ((1,0,0),(1,1,1),(1,0,1))`：`S←S`、`A←[S, A, stopgrad(R)]`、`R←[S, R]`。
  Register 永不读 Action；Scene 永不被 Action/Register 更新。Action 的 cross-attn 上下文是
  **85 token**（`S(81) | sg(R)(4)`）。不同 candidate 的 register 之间也永不互相 attend。
- `scene_tokens = 5 帧 × 16 + 1 ego = 81`，scorer 的默认值同为 81。ego 是 Scene 的
  **最后一个** slot，scorer 用 `scene_context[:, -1:]` 原样取出（不过 LayerNorm）相加。
- Scorer 的 detach 语义**逐字照抄 DrivoR**：只有 proposals 被 detach，scene K/V **不**
  detach，所以 `loss_score` 会回传塑造 η。对应配置
  `input_detach: [trajectory, predicted_register]`、`scene_kv_detached: false`、
  `updates_scene_lora: true`。**不要"修回去"。**
- 分辨率：Student 收 DrivoR 官方的 `672×1148`，内部用 4 列**零填充**（归一化之后施加）
  闭合成 `672×1152` → patch16 grid `42×72`。`_prepare()` **拒绝任何重采样**，连已经
  填好的 `1152` 也拒（那意味着 loader 自己插值过）。**不要用 timm 的
  `dynamic_img_pad=True` 绕过 assert**——它会跳过检查但 `patch_embed.grid_size` 仍停在 71，
  forward 吐 72 列直接 `IndexError`。填充必须在外部做并声明 `img_size=(672,1152)`。
- 位置编码逐字照抄 DrivoR：DINOv3 的 `pos_embed is None`，所以 `_pos_embed` 直接返回原张量
  ——**纯 RoPE，无绝对位置编码，零插值**。`rope_prefix_tokens = 5 + 16 = 21`。
  `_widen_rope_prefix()` **只在 `__init__` 调一次**：在 forward 里临时改
  `EvaAttention.num_prefix_tokens` 再还原会让 `torch.utils.checkpoint` 反向重算时 RoPE
  静默错位（注意 timm 的 `EvaAttention` 持有自己独立的同名属性，改 Eva 模块上那个没用）。
- Teacher 在未来帧的**原生 `464×832`** 上编码，零 resize 零插值，收到别的尺寸直接
  `ValueError`。Teacher grid `29×52` 比 Student 的 `42×72` **粗**，这是刻意的且合法：
  DINOv3 的 RoPE 按轴归一化（`normalize_coords='separate'`），token 的旋转只取决于它在
  画面里的**分数位置**，与被切成多少块无关。象限边界用**整除**（`29 → 14/15`，
  `52 → 26/26`），所以奇数网格可用且不丢 token。
- Teacher 编码必须**流式归约**：`[8,29,52,1024]` float32 是 50 MB，Student 尺寸下 103 MB，
  而 ordered vectors `[gt,4,8192]` 两种情况都只有 131 KB。产出 ordered vectors 后立即丢弃
  grid；目标构建完就 `replace(sample, future_rgb=None)`。`TimmDinoV3Teacher` 另有
  `frame_chunk=2`，因为 attention 对 patch 数是平方复杂度。
- Wan backbone：30 block → 8 层，`layer_map = [0,3,7,11,15,18,22,26]`，226 个 key
  `strict=True`。meta 必须逐项匹配
  `{hidden_dim:256, ffn_dim:1024, num_layers:8, num_heads:4, attn_head_dim:64,
  text_dim:256, freq_dim:256, eps:1e-6}`。Wan 的约定 `num_heads × attn_head_dim == hidden_dim`
  把 **heads 钉死在 4**（`d_model=256` 时）；构造 `NvsActionDiT` 时 `text_dim` 与 `ego_dim`
  都必须显式等于 `hidden_dim`。skip 前缀 `action_encoder.` / `head.` / `mode_projection.` /
  `ego_to_action_projection.` / `time_queries` 保持随机初始化。
- **单次 forward 出轨迹，训练和推理都没有 rollout**（`action_sampling_loops: 0`）。
  训练每次 forward 随机采一个 σ；推理把 σ 钉在 1（`INFERENCE_SIGMA`），此时
  `add_noise` 原样返回 anchor、`recover` 减掉整条 velocity，所以
  `JointStage2Module.inference()` 与训练路径是**同一个代数表达式**。任何引入多步采样的
  改动都会让评测偏离训练路径。
- 优化器参数组必须**完整、唯一、互斥**，`build_optimizers` 会断言。`parameter_groups`
  跳过 `requires_grad=False` 的参数，所以 Stage 1 的 theta 组为空是设计使然，不是 bug。

## 模型结构与数据契约速查

- 四个参数组：`η`(eta) = DINOv3 LoRA + scene queries + neck + hist_encoding + scene_blocks；
  `φ`(phi) = Future Register 分支；`θ`(theta) = Wan2.2 ActionDiT；`ψ`(psi) =
  `ScratchDrivoRScorer`（官方 `Linear(24,1024)-ReLU-Linear(1024,256)` + 四层
  `TransformerDecoderScorer` + 六头 `Scorer`，全部随机初始化，**永不**加载 DrivoR
  checkpoint）。
- Stage 1 冻结 `θ + ψ`，只训 `η + φ`，输入是**不加噪**的 clean 轨迹（同一条 flow 的 σ=0 端）。
  Stage 2 四组全开，输入是加噪 action，σ 随机。
- 每个 scene 最多 4 条 GT（`MAX_GT = 4`）。Hungarian 在 `[K, 64]` 上产出**一份**
  assignment，被 action loss 与 future loss 共享。
- Future Register 目标是 `[B, K, 4, D]`，那个 `4` 是**未来时间块**（每块 2 帧）不是 4 条 GT。
  8 个未来帧来自 41 帧 @10fps 的 `future.mp4`，索引 `range(5,41,5)` → +0.5…+4.0s，
  正好对上 DrivoR `num_poses = 8`。每块由 `2 帧 × 4 象限 × 1024 = 8192` 维经 train-fitted
  PCA 降到 256。
- Score 监督是在线 NAVSIM oracle 标签 `[B, 64, 7]`（NC, DAC, EP, TTC, C, DDC, Final），
  scorer 目标是 `[B, 64, 6]`。**永远不是** `64×K` 的 candidate-by-GT 矩阵。Multi-GT 不改动
  DrivoR 的 scorer 网络或 loss。
- 数据根目录 `/mnt/data/multi_future_view/navtrain_full_allcosmos_cuda_v1_gpu64/...`
  （103075 scene）；`/mnt/data/multi_view/...` 是指向同一目录的**符号链接**。
  历史帧文件名是**相机 token 不是帧 token**，必须经
  `frames[i]["cams"]["CAM_F0"]["data_path"]` 解析；日志是 2 Hz。
  `trajectory_k` 的**目录序号不等于候选轨迹序号**（实测 193/200 不等），配对必须走
  metadata 里的显式字段。
- metric_cache 路径与 DrivoR `train_metric_cache` 完全兼容：
  `data_runtime/train_metric_cache/<log>/unknown/<token>/metric_cache.pkl`。
  `/mnt/zhaozc_workspace/official_DrivoR_clean/data_runtime/train_metric_cache` 是指向
  FastWAM 离线 cache 的符号链接，**被 DrivoR 在线评测拒绝**。

## 方案A — Joint 阶段对齐 DrivoR 感知容量（2026-10-13）

DrivoR 能安全使用 `final_score_weight=1.0`（score loss 直接回传感知）是因为它只训练 LoRA，
base backbone 全程冻结，感知容量小。我们之前的 Joint 额外训练了 `scene_blocks`
（8 层 1007-token self-attn，6.3M 参数），感知容量远大于 DrivoR。当 source_loss 权重拉到
1.0 时，未 detach 的 score 梯度会把感知冲垮（历史上 sel_pdms 0.978→0.85、ade 0.25→0.56
的回归即源于此）。

方案A：Joint 阶段冻结 `scene_blocks`，把可训练感知容量退回 DrivoR 的 LoRA 量级，于是
source_loss 可以安全地按 DrivoR 原值 1.0 训练；rank loss 按 DrivoR 原值 0.1 独立加权。

改动（3 处，都在 e2e 路径）：

1. `scripts/run_navtrain_wan_e2e_ddp.py` → `build_optimizers()`：在 phi 冻结之后、scorer
   解冻之前冻结 `scene_blocks.`：

   ```python
   for name, parameter in model.named_parameters():
       if name.startswith("scene_blocks."):
           parameter.requires_grad_(False)
   ```

   `parameter_groups` 会跳过 `requires_grad=False` 的参数，因此「generator 参数集合 ==
   eta+theta 可训练集合」的断言仍然成立。

2. 同文件训练日志的 `trainable` 字典新增 `"scene_blocks_frozen"`（实测 6318080），
   setup 日志中应 >0，用于确认冻结生效。

3. `scripts/dlc_navtrain_wan_e2e.sh` 的 Stage 2 段：`--lambda-score 1.0`（对齐 DrivoR
   `final_score_weight`）、`--lambda-rank 0.1`（对齐 `preference_rank_weight`）、
   `--lr 2e-4 --scorer-lr 5e-4`。rank loss 在损失块中已「去嵌套」：
   `loss = λ_action·loss_action + λ_velocity·loss_velocity + λ_score·source_loss
   + λ_rank·rank_loss + λ_future·loss_future`，rank 有效系数即 0.1
   （不再被 lambda_score 二次缩放成 0.02）。

回退方法：删掉 `build_optimizers()` 里那个冻结循环即恢复 `scene_blocks` 可训
（`scene_blocks_frozen` 日志字段可保留）；把 launcher 的 `--lambda-score` 改回 `0.2`
即恢复温和 score 权重。两者独立，完全回到方案A之前需同时执行这两步。

## 训练入口与流水线

`scripts/dlc_navtrain_wan_e2e.sh` 是**唯一**的流水线 launcher，按序跑：

- **Stage 0** `build_future_target_cache.py`（torchrun）→ `compressor.pt` + per-scene
  targets。`compressor.pt` 已存在时自动跳过，`NVS_FORCE_STAGE0=1` 强制重跑。
  一次冻结 Teacher 过全量数据，之后训练不再解码 mp4。
- **Stage 1** `run_stage1_multiview.py`（单进程）→ `stage1_checkpoint.pt`。
- **Stage 2** `run_navtrain_wan_e2e_ddp.py --stage1-checkpoint "$STAGE1_CKPT"`。
  `compressor.pt` 存在则 `--future-target cache --lambda-future 0.1`，否则
  `--future-target none` 并打印显著警告（此时 φ 冻结、`lambda_future` 强制为 0）。

跳过某阶段用 `NVS_SKIP_STAGE0/1/2=1`；追加参数用 `NVS_STAGE{0,1,2}_EXTRA_ARGS`。
preflight 会：跳过 Stage 1 但 `STAGE1_CKPT` 不存在时**中止**；Wan payload 缺失时
**非致命**警告（两阶段都会退化成随机 θ 初始化）；按 195 MB/scene 核算 Stage 1 的
RAM 需求，超过物理内存一半就 `exit 2`。

Stage 2 的模型构造 / checkpoint 恢复 / 评测推理**只有一处实现**：
`run_navtrain_wan_e2e_ddp.py` 里的 `build_generator_and_scorer()`、
`load_joint_checkpoint()` 和 `JointStage2Module.inference()`。两个评测脚本
（`eval_navtrain_wan_e2e_pdms_bev.py`、`eval_navtest_official_pdms_e2e.py`）都 import 它们，
**不要在评测脚本里重新拼一遍模型构造**——它们曾经各自写死 `heads=8`、
`dino_lora_rank=16`、以及不存在的 `dino_lora_alpha` / `use_all_native_tokens` 参数，
加上一个过时的 3 参 `net(...)` 调用，于是**静默坏掉**（构造期就 TypeError，但没人跑过）。
`load_joint_checkpoint` 的 `--heads` / `--target-dim` / `--depth` 默认 0 表示「读 checkpoint
里的 config」，只有显式传非零值才覆盖。

`stages.py` **不含** Stage 2：它需要梯度累积、LR schedule 和中间 flow 张量
（`velocity`、`predicted_register`、`logits`）用于日志，塞不进一个自包含的
`zero_grad/backward/step` helper；再放一份只会和生产路径漂移。

## 评测

- 两个 ADE 口径必须**同时**报：**matched-GT ADE**（只覆盖 Hungarian 赢家）和
  **all-64 nearest-GT ADE**（覆盖所有被监督的 candidate）。只报前者正是「个别收到 GT
  监督的很高、总体质量偏烂」被掩盖的原因。两者共用 `nearest_gt_ade()` 这一个实现，
  所以 batch 汇总和 per-scene 记录不可能各说一套。
- `eval_navtest_official_pdms_e2e.py --shard-count N` 分片后必须用
  `merge_navtest_shards.py` 合并；它会顺带证明分片正确（各分片 scene 集合两两不相交、
  并集等于 `scenes_indexed_total`，且带 `--reference` 时每条 per-scene 记录与不分片跑
  **逐字相同**）。合并后由同一份代码重算 summary，避免 sharded 与 baseline 用不同归约。
- `official_navtest_oracle.py` 走官方 navsim 评测路径：官方 `MetricCache` +
  官方 `PDMScorer` + `pdm_score` 聚合。

## 验证

```bash
export PYTHONPATH=/mnt/zhaozc_workspace/project/NVS/src:/mnt/zhaozc_workspace/project/NVS/vendor/DrivoR:/mnt/zhaozc_workspace/project/NVS/vendor/DrivoR/nuplan-devkit
bash scripts/run_contract_tests.sh   # 68 passed（有 vendor/SimWAM 时）
```

`vendor/` 被 gitignore，只有 `vendor/DrivoR`（1398 个文件，含 `nuplan-devkit`）是
force-add 进仓库的；`SimWAM`、`FastWAM_...`、`Drive-JEPA`、`DrivoR_official` 只在本地。
SimWAM **不是运行时依赖**（包内从不 import 它），只被
`test_wan_scheduler_matches_simwam_source` 用来把我们的 flow 移植和上游 scheduler 源码
逐行对齐；缺 SimWAM 时该测试用 `pytest.importorskip` **skip 而不是 fail**，
所以新克隆上是 67 passed + 1 skipped。改这条测试时不要把 importorskip 换回硬 import。

真数据 Stage 1（4 个非退化 scene、60 步、candidates=64、depth=8，910s）已验证：
`loss_future` 1.1769 → 0.1602 单调下降且全部有限；`paired_better_than_shuffled: true`
（0.16291 < 0.16593）；`mean_register_std_across_candidates` 0.975 → 0.411（>0，确实在预测
彼此不同的 register）；`parameter_groups_updated == {eta: true, theta: false, phi: true}`；
`scorer_audit.logits_finite: true`、`scene_context_shape == [1,81,256]`；Wan backbone
`loaded: true` / `loaded_keys: 226`；4 个 scene 的 `history_policy` 全为
`real_CAM_F0_sensor_blob`，`matched_pairs` 全为 4。结果在
`artifacts/stage1_multiview/stage1_multiview_result.json`。

方案A 单卡冒烟（`--batch-size 1 --limit-scenes 4 --max-steps 2 --epochs 1`）通过：
`scene_blocks_frozen=6318080`、`scene_kv_detached=false`、两步 loss 全部有限、
无 `build_optimizers` 断言失败、无 CPU/CUDA 设备不匹配、exit 0。**冒烟的 sel_pdms/ade
仅为合成小样本，不得据此宣称 NAVSIM 指标。**

Stage1→Stage2 链路探针（12 项全过）：853 vs 853 key 全部对齐、输出逐位相同、
anchor 在 38.4 m 展布上 endpoint 只偏 0.0108 m（0.028%）。评测路径探针（36 项全过）：
真实 DINOv3-large Student + 真实 226-key Wan payload 下，
`build_generator_and_scorer` / `load_joint_checkpoint`（`weights_only=True` + `strict=True`）
/ 错误 geometry 被 Wan meta 检查拦住（`target_dim=128`、`heads=8` 都抛 `ValueError`）/
`inference()` 与 `forward_flow(data=anchor, σ=1)` 逐位相同 / σ=1 时 noisy action 就是 anchor
（无 GT 泄漏，`max|diff|=3.81e-06`）且 `v_target` 恰为 0 / 跨调用确定性 / 无 GT 的 scene
不监督任何 candidate。

## 已知限制（不要粉饰）

- **`/mnt/zhaozc_workspace` 下没有任何 `compressor.pt`。** Stage 0 从未真正跑过，
  所以未来监督路径只在合成目标上验证过，**没有**在真实 Teacher 输出上验证过。
  `build_future_target_cache.py` 是唯一生产者（每 scene 一次冻结 Teacher，
  `[K,4,256]` float32，约 66 KB/scene，103075 scene 语料约 6.8 GB）。
- `checkpoints/`、`artifacts/`、`logs/`、`reports/` 都在 `.gitignore` 里，所以 36 MB 的
  Wan payload 和所有跑批记录**不会**进仓库；换机器必须先用
  `preprocess_wan_action_dit.py` 从 `/mnt/LLM_weight/Wan-AI/Wan2.2-TI2V-5B` 重建。
- `Ordered2x2Compressor` 的秩 = `min(target_dim, 拟合行数)`，拟合行数 = `16 × scene 数`
  （`fit()` 先 `reshape(-1, last_dim)`，行数 = `scene 数 × 4 GT × 4 chunk`）。
  **满秩只需 16 个 scene**，不是 65 个（旧值算错了）。`--scenes` 小于 16 时目标是降秩的；
  看 audit 里的 `compressor_rank` / `compressor_fit_rows` / `compressor_full_rank` 三个字段，
  不要猜。
- 抽样 200 个 scene 实测 **19%（38/200）** 的 multiGT scene 是退化的（label 重复或全
  `human_log`，GT 间 pair L1 仅 0.005–0.044，非退化 scene 为 0.133–0.167）。用
  `--skip-degenerate` 过滤；它只读 `trajectory_k/metadata.json`，不会为要丢弃的 scene
  付 mp4 解码代价。同一批的对应关系核验：轨迹与日志 `pose_interpolation` 对齐的只有
  **241/800**（30%）。
- DrivoR 训练时开的 `GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1,
  prob=0.7)`（`dinov2_lora.py:341`，`use_grid_mask = True`）**本包尚未移植**。
- Stage 1 冻结 θ，所以 `scorer_audit.proposals_vs_multigt` 里的 ADE/FDE 反映的是
  **随机初始化的轨迹头**叠加在米制 mode 锚点上的结果，不是训练后的轨迹精度。60 步真数据
  跑出的 `min_ade_over_gt_m: 6.10` / `best_pair_ade_m: 1.32` 只说明 Stage 1 没有动 θ
  （符合契约）。`min_ade_over_gt_m` 的口径是「64 个 proposal 各自对 4 个 GT 取最小 ADE，
  再对 proposal 求均值」，衡量 proposal 集合对 GT 集合的覆盖度。
- `dino_tokens.py`、`lora.py` 是死代码，删除需用户明确授权。
