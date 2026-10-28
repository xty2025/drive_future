# Human＋diverse8 的未来 register 实验

本次新增六个独立配置。分别基于 human vMSE＋diverse8 L1 的 Action-DiT、human X-L1＋diverse8 L1 的 Action-DiT，以及 human L1＋diverse8 L1 的普通 Transformer；每组各有“预测 register，但不输入 Scorer”和“预测 register，并输入 Scorer”两个版本。

另有基于 A27/A28 的 [仅 human 未来监督 register 对照](DRIVOR_HUMAN_ONLY_REGISTER.md)：轨迹仍为 human＋D8，register loss 只对齐 human，使用独立配置和启动脚本。本页六组原配置保持 human＋D8 未来监督。

## 配方

`H` 为 NAVSIM 原始 human GT，`D8` 为有效的 diverse8 轨迹，最多八条。没有旧 top4、多余长 GT 或 CLOVER pseudo-expert。所有配置保留 **64 个 proposal**，每条输出八个轨迹点，同时预测 **8 帧×4 个＝32 个 register token**，register 原始宽度 1024。

| 简称 | 轨迹监督（保持原基线） | 未来 register 对齐 | register 进 Scorer | 配置 |
|---|---|---|---|---|
| DiT human-v / R | `V(H) + 0.5 L1(D8)` | `Rv(H,D8)` | 否 | [human_diverse8_l1_register](navsim/planning/script/config/common/agent/drivoR_action_dit_dinov3_vitl_human_diverse8_l1_register.yaml) |
| DiT human-v / R+Scorer | 同上 | 同上 | 是，detach 后融合 | [human_diverse8_l1_register_scorer](navsim/planning/script/config/common/agent/drivoR_action_dit_dinov3_vitl_human_diverse8_l1_register_scorer.yaml) |
| DiT human-X-L1 / R | `L1(H) + 0.5 L1(D8)` | `Rv(H,D8)` | 否 | [human_xl1_diverse8_l1_register](navsim/planning/script/config/common/agent/drivoR_action_dit_dinov3_vitl_human_xl1_diverse8_l1_register.yaml) |
| DiT human-X-L1 / R+Scorer | 同上 | 同上 | 是，detach 后融合 | [human_xl1_diverse8_l1_register_scorer](navsim/planning/script/config/common/agent/drivoR_action_dit_dinov3_vitl_human_xl1_diverse8_l1_register_scorer.yaml) |
| Transformer / R | `L1(H) + 0.5 L1(D8)` | `R1(H,D8)` | 否 | [query_human_diverse8_l1_register](navsim/planning/script/config/common/agent/drivoR_query_dinov3_vitl_human_diverse8_l1_register.yaml) |
| Transformer / R+Scorer | 同上 | 同上 | 是，detach 后融合 | [query_human_diverse8_l1_register_scorer](navsim/planning/script/config/common/agent/drivoR_query_dinov3_vitl_human_diverse8_l1_register_scorer.yaml) |

总 loss 为表内轨迹项＋`1.0×未来对齐项＋1.0×原 DrivoR 评分 loss`。`L1(D8)` 在组内按有效 diverse8 数平均，外层乘 0.5；未来项则把 **H 和所有具有有效未来的 D8 一起等权平均**，外层系数 1，不再单独给 D8 未来乘 0.5。

## GT、proposal 与未来视频如何配对

1. 每条 H/D8 GT 独立从全部 64 个输出中选择最近的 proposal，可以重复选中同一 proposal，不用 Hungarian。
2. H 的匹配使用该基线自己的 loss：human-v 版为归一化 X→v MSE；human-X-L1 和普通 Transformer 为物理坐标 L1。D8 始终使用原 coverage 的物理 L1，`sum(|dx|,|dy|,|dheading|)` 后平均八个时刻。
3. GT 选中候选 `m` 后，同一 batch 的 `register[m]` 对齐该 GT 的八帧未来目标。不会重新按 register 特征匹配，不会把所有 GT 都对齐到 human 的未来。
4. human 与 D8 的轨迹损失分别汇总为 `human + 0.5×mean(D8)`。无效/padding 的 D8 不贡献轨迹或未来 loss。

Action-DiT 仍以原 human 轨迹为全部 64 个加噪输入的锚点；register 加噪输入的锚点仍来自 **真实 human 未来帧的 EMA register**。每个候选独立采样 action/register Gaussian 噪声，同场景共享 t。一次 DiT 前向产生全部轨迹与 register，并复用给轨迹 loss、未来 loss、Scorer；训练不增加 Euler 循环。推理沿用十步联合 Euler。

`V` 和 `Rv` 沿用现有 X-prediction 转 vMSE、`sigma_min=0.1` 和 timestep 权重。human-X-L1 版只把 human 主项保留为物理 X-L1，新增的 register 仍为参考 Action/register 版的 vMSE。普通 Transformer 无扩散加噪或采样循环，用可学习的 Action/future query 在同一次前向中直接输出轨迹和 register，未来项 `R1` 为直接特征 L1。

## 网络和 Scorer

- 当前图像：仅前视当前帧，672×1152；DINOv3 ViT-L 原生 3024 个 patch K/V，不注入 scene query。预训练基座冻结，Q/V LoRA rank 32。
- Action-DiT：沿用预训练 30 层、hidden 256、FFN 1024、24×128 attention；初始化来自原 FastWAM ActionDiT 256 权重。普通 Transformer：沿用随机初始化的四层 query decoder、256/1024 维。
- 未来目标：真实 H 未来图像及每条 D8 的 Cosmos 未来图像，384×384；通过在线视觉编码器的 EMA teacher 得到四个原生 register。EMA decay 0.999，仅在实际 optimizer step 后更新，teacher 不接收梯度。
- 注意力方向不变：Action 看不到未来 register，register 可以读取对应 Action；当前视觉 K/V 参与两者交互。
- 原 Scorer 版：只输入 detach 的轨迹及原当前视觉/ego 条件。未来 register 只用于辅助训练。
- 融合 Scorer 版：每条预测 register 展平后 `32768→256` 压缩，与轨迹 embedding 拼接，经 `512→1024→256` MLP，再进入原四层、六评分头 Scorer。轨迹和预测 register 都 detach；视觉 K/V 的原梯度路径保留。
- 所有 Scorer 仍对同一组 64 条轨迹计算原在线 NAVSIM 标签。推理只需当前图像/ego 和本版本 checkpoint，不读取 GT、未来视频或 teacher 目标。

## 数据与抽帧

轨迹根目录：

```text
/cpfs29jb/data/zhaozc/project/Cosmos3forMultiGT/runtime/multigt_raw64/outputs/navtrain_diverse8_dense_0p1s_f64
```

对应视频根目录：

```text
/cpfs29jb/data/zhaozc/project/Cosmos3forMultiGT/runtime/rollouts/navtrain_diverse8_full_allcosmos_cuda_v1_gpu256
```

读取 `<token>/trajectory_<slot>/future.mp4`。第 0 帧是当前帧，取零基索引 **5、10、15、20、25、30、35、40**，对应 **+0.5～+4.0 秒**。通过 slot、selected candidate index、标签和 `pose_times_s.npy` 检查配对，不做 SHA/哈希检验。

六个实验共用 RGB 缓存：`DrivoR/exp/world4scorer_data/frames_diverse8/<token前2位>/<token>.zip`。缓存保存原始槽位、轨迹和帧索引；GT 少于八条或存在 padding 空洞时不压缩错位。采用每场景一个 ZIP，原子发布，不产生数百万个零散 JPEG 文件。缓存的是 RGB，EMA 特征每次在线计算。

**默认可直接训练：**缓存缺失时由 DataLoader 解码对应视频并写入缓存，后续直接读取。首轮可能受解码/I/O 影响。存在有效 D8 但缺少对应视频时会报出具体 token/slot，不会静默取消其未来监督；没有有效 D8 的场景沿基线保留 H。

如需提前生成全部缓存，在项目根目录运行（CPU 抽帧，不占 GPU）：

```bash
cd /cpfs29jb/data/zhaozc/project/drive_future
/cpfs29jb/data/zhaozc/miniconda3/envs/drive_future/bin/python \
  DrivoR/tools/extract_diverse8_future_frames.py --workers 8
```

可用 `--tokens-json` 做小样本检查，或用 `--num-shards N --shard-id i` 分片预处理。`DRIVOR_DIVERSE8_GT_ROOT`、`DRIVOR_DIVERSE8_ROLLOUTS_ROOT`、`DRIVOR_DIVERSE8_FRAMES_ROOT` 分别覆盖训练时的三个根目录；离线抽帧脚本对应 `--gt-root`、`--rollouts`、`--output`。若已完成预抽帧，可加 `agent.config.future_register.build_missing_cache=false` 强制只读缓存。

## DLC 启动命令：六个独立实验，训练后自动 FP32 测试

每个命令单独用于一个 DLC PyTorch 任务，默认 **8 机×8 卡、每卡 batch 1、累积 1、总 batch 64、BF16 训练、25 epochs、seed 2**。沿用每 epoch 保存 checkpoint、FP32 val PDMS；训练完成后仅主节点执行 FP32 navtest。六组使用独立的新脚本与输出目录。

```bash
cd /cpfs29jb/data/zhaozc/project/drive_future

# 1. Action-DiT：human vMSE + diverse8 L1；预测 R，不进 Scorer
bash DrivoR/scripts/training/run_drivor_action_dit_dinov3_vitl_human_diverse8_l1_register_train_eval.sh

# 2. 同上；预测 R 进入 Scorer
bash DrivoR/scripts/training/run_drivor_action_dit_dinov3_vitl_human_diverse8_l1_register_scorer_train_eval.sh

# 3. Action-DiT：human X-L1 + diverse8 L1；预测 R，不进 Scorer
bash DrivoR/scripts/training/run_drivor_action_dit_dinov3_vitl_human_xl1_diverse8_l1_register_train_eval.sh

# 4. 同上；预测 R 进入 Scorer
bash DrivoR/scripts/training/run_drivor_action_dit_dinov3_vitl_human_xl1_diverse8_l1_register_scorer_train_eval.sh

# 5. 普通 Transformer：human L1 + diverse8 L1；预测 R，不进 Scorer
bash DrivoR/scripts/training/run_drivor_query_human_diverse8_l1_register_train_eval.sh

# 6. 同上；预测 R 进入 Scorer
bash DrivoR/scripts/training/run_drivor_query_human_diverse8_l1_register_scorer_train_eval.sh
```

只训练时将相应入口的 `_train_eval.sh` 换成 `_training.sh`。可显式设置 `DRIVOR_RUN_ID` 区分重复实验；初始化方式沿用各自基线的预训练配置，新 future/scorer 融合参数随机初始化。

单独测试本版本完整 checkpoint，例如：

```bash
cd /cpfs29jb/data/zhaozc/project/drive_future
NPROC_PER_NODE=8 bash DrivoR/scripts/evaluation/run_drivor_diverse8_register_evaluation.sh \
  action_dit_human_xl1_diverse8_l1_register_scorer /absolute/path/to/last.ckpt
```

测试 variant 与训练匹配：`action_dit_human_diverse8_l1_register[_scorer]`、`action_dit_human_xl1_diverse8_l1_register[_scorer]`、`query_human_diverse8_l1_register[_scorer]`。测试保留预测 register 的网络结构，只省去未来 GT/视频读取。

## 实现入口

- [视频抽帧与缓存](navsim/common/diverse_future_frames.py)、[保留 GT 槽位的数据读取](navsim/agents/drivoR_action_dit_register/diverse_targets.py)。
- [Action-DiT human/diverse8/register 模型与 agent](navsim/agents/drivoR_action_dit_register/diverse_gt.py)。
- [普通 Transformer human/diverse8/register 模型与 agent](navsim/agents/drivoR_query_register/diverse_gt.py)。
- [共同训练流水线](scripts/training/run_drivor_diverse8_register_train_eval.sh)、[FP32 测试入口](scripts/evaluation/run_drivor_diverse8_register_evaluation.sh)。

## 本地验证

- 45 项针对性检查通过：涵盖不同轨迹 cost 的匹配、重复 winner、GT/padding 掩码、未来帧索引、缓存复用、Scorer detach、EMA 更新、完整 checkpoint 恢复和自动 FP32 测试入口；包含旧 GT/register 配方的兼容检查。
- 六份配置均通过实际启动脚本的 Hydra 解析，确认前视 672×1152、64 proposals、D8 L1 权重 0.5、无旧 M4/长 GT，以及三种主 loss/未来 loss 的选择。
- 三次真实数据双卡 smoke 均通过（GPU 0/1）：分别覆盖 human-v Action-DiT、human-X-L1 Action-DiT、普通 Transformer 的 register 融合 Scorer 版；每次一个训练 step 加一个验证 batch。训练使用 BF16，验证使用 FP32，读取了每场景 8 条 D8 和 9 组 H＋D8 未来目标。
- 已从实际视频抽取并验证 6 个 smoke 场景、384 张未来帧。没有执行全量预抽帧；首次训练按需补齐缓存，或先执行上面的预处理命令。smoke 用于验证流程，不是训练完成的 navtest 性能。
- 验证记录：[validation.json](../artifacts/diverse8_register_20261022/validation.json)。
