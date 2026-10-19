# DINOv3 ViT-L Action-DiT：合并 poor GT、单 GT、单 GT＋长 GT

三组都以 `training_drivoR_action_dit_dinov3_vitl_navtrain_multigt` 的现有配置为基线，只调整轨迹监督数据。保留当前单前视 672×1152、DINOv3 ViT-L 原生 3024 patches、冻结基座与 Q/V LoRA、256 维 Action-DiT、64 个候选及原 Scorer。默认沿用原 DINOv3 / Action-DiT 预训练初始化，新开独立实验。

| 版本 | 轨迹监督 | 每次 DiT 训练调用中的候选数 | Scorer 的生成候选数 |
|---|---|---:|---:|
| `multigt_poor` | 原有效 top4 GT＋有效 poor4 GT，最多 8 条，按实际 GT 数等权平均 | 64 | 64 |
| `single_gt` | 原始 NAVSIM human GT，best-of-64 v loss | 64 | 64 |
| `single_gt_long` | 原始 human GT＋原 DrivoR 长 GT，各自 best-of-64 v loss 后相加 | 128（两个 64 合批） | 普通组 64 |

三组都只调用一次 DiT 训练前向。它们使用基础 Action-DiT agent，不额外启用 Bank 或 pseudo-expert coverage。

## 1. 原 multi-GT 加入 poor GT

配置：[`drivoR_action_dit_dinov3_vitl_multigt_poor.yaml`](navsim/planning/script/config/common/agent/drivoR_action_dit_dinov3_vitl_multigt_poor.yaml)。

来源：

```text
multi_traj_gt_data/navtrain_multigt_top4_dense_0p1s_f64_8gpu
multi_traj_gt_data/navtrain_poor4_multigt_dense_0p1s_f64_16gpu
```

按场景 token 读取两个目录的 `gt_trajectories.npy`，使用 ego 相对坐标下的 `[K,8,3]` 轨迹（0.5～4.0 s）。两个来源均通过 `gt_valid_mask & ~gt_padded` 取有效行，再直接拼接。`max_gt` 从 4 改为 8，batch 内其余位置仅做张量 padding，不计入 loss。

poor 数据每场景实际有 0～4 条。其发布汇总包含 103,075 个场景、291,599 条候选，其中 26,006 个场景没有 poor GT；空集合或缺失 token 贡献 0 条，不补另一份 human GT。原 top4 缺失时，保留基线的 human 回退，再追加存在的 poor GT。

这组把 poor GT 同样作为轨迹监督目标，所有有效行权重相同。保留原 Action-DiT 匹配与 loss：

```text
每条有效 GT 各自从同一组 64 个 X-predictions 中找最小 v MSE
L_trajectory(scene) = 所有有效 GT 的最佳 v loss 之和 / 有效 GT 数
L_total = mean_batch(L_trajectory) + L_score
```

不同 GT 可以匹配同一 proposal。v 误差仍使用原轨迹归一化、时间权重和 `sigma_min=0.1` 下界。加噪输入继续全部由原始 human GT 构造，使用独立的 64 份噪声和每场景共享的 t。

新增日志：`multi_gt_primary_count`、`multi_gt_additional_count`；`multi_gt_count` 为合并后的有效总数。数据路径可分别通过 `DRIVOR_MULTI_GT_ROOT`、`DRIVOR_POOR_GT_ROOT` 设置。合并目标使用独立缓存名，避免读取旧的 4-GT target cache。

## 2. 原始 human 单 GT

配置：[`drivoR_action_dit_dinov3_vitl_single_gt.yaml`](navsim/planning/script/config/common/agent/drivoR_action_dit_dinov3_vitl_single_gt.yaml)。

```yaml
multi_gt:
  enabled: false
long_trajectory_additional_poses: 0
```

使用当前 NAVSIM 场景的原始 4 s human trajectory。仍生成 64 个 proposals，仅选择与这一条 human GT 对应 v loss 最小的 proposal 计算轨迹主 loss：

```text
L_total = best-of-64 v loss(human GT) + L_score
```

该配置无需加载 top4 或 poor GT 导出目录。

## 3. Human 单 GT＋长 GT

配置：[`drivoR_action_dit_dinov3_vitl_single_gt_long.yaml`](navsim/planning/script/config/common/agent/drivoR_action_dit_dinov3_vitl_single_gt_long.yaml)。

在第 2 组上设置 `long_trajectory_additional_poses=2`。复用原 DrivoR target builder：多取两个 0.5 s 点，得到 5 s human GT，再用原 CubicSpline 和非均匀递增采样规则压成 8 个轨迹点。场景缺少足够未来数据时沿用原逻辑，回退普通 GT。

这组正是 [`run_drivor_action_dit_training.sh`](scripts/training/run_drivor_action_dit_training.sh) 的原始轨迹监督配方：普通 human GT＋长 GT，两项 best-of-64 v loss；视觉部分继续使用本次 DINOv3 ViT-L 基线。需要与该脚本对齐的“单 GT 基线”时，选择 `single_gt_long`；`single_gt` 保留为仅 human 的额外消融。

普通 GT 和长 GT 分别构造 64 份加噪输入，共享同一组 noise/t，合并成 `[B,128,8,3]` 进行一次 DiT 前向；Action self-attention 仍在各候选内部，候选之间不互相读取。两组分别选 best-of-64：

```text
L_total = best-of-64 v loss(human GT)
        + best-of-64 v loss(long GT)
        + L_score(普通组的 64 条预测)
```

两项轨迹 loss 权重均为 1，取和。日志分别为 `flow_normal_loss` 和 `flow_long_loss`。Scorer 只接收普通组 64 条轨迹，不额外评分长 GT 组。

## 4. DLC 训练后自动 FP32 测试

每组单独提交 **8 节点×每节点 8 GPU** 任务。训练 BF16、每卡 batch 1、累积 1、总 batch 64、25 epochs、seed 2、LR 2e-4。训练成功后仅节点 0 自动查找本次 run 最新 `version_*` 下的 `last.ckpt`，用本机 8 卡、**FP32 (`32-true`)** 进行 navtest，32 个 CPU 进程算分。

原 top4＋poor4：

```bash
cd /cpfs29jb/data/zhaozc/project/drive_future
DRIVOR_RUN_ID=action_dit_dinov3_vitl_multigt_poor_seed2_train_eval_run1 \
DRIVOR_SCENE_QUERIES=0 DRIVOR_PRECISION=bf16-mixed DRIVOR_EVAL_PRECISION=32-true \
bash DrivoR/scripts/training/run_drivor_action_dit_dinov3_vitl_gt_train_eval.sh multigt_poor
```

原始 human 单 GT：

```bash
cd /cpfs29jb/data/zhaozc/project/drive_future
DRIVOR_RUN_ID=action_dit_dinov3_vitl_single_gt_seed2_train_eval_run1 \
DRIVOR_SCENE_QUERIES=0 DRIVOR_PRECISION=bf16-mixed DRIVOR_EVAL_PRECISION=32-true \
bash DrivoR/scripts/training/run_drivor_action_dit_dinov3_vitl_gt_train_eval.sh single_gt
```

原始 human＋长 GT：

```bash
cd /cpfs29jb/data/zhaozc/project/drive_future
DRIVOR_RUN_ID=action_dit_dinov3_vitl_single_gt_long_seed2_train_eval_run1 \
DRIVOR_SCENE_QUERIES=0 DRIVOR_PRECISION=bf16-mixed DRIVOR_EVAL_PRECISION=32-true \
bash DrivoR/scripts/training/run_drivor_action_dit_dinov3_vitl_gt_train_eval.sh single_gt_long
```

训练实验名分别为 `training_drivoR_action_dit_dinov3_vitl_navtrain_<版本名>`，评测实验名为 `eval_navtest_action_dit_dinov3_vitl_<版本名>`。可用 `DRIVOR_TRAIN_EXPERIMENT`、`DRIVOR_EVAL_EXPERIMENT` 修改；组合入口追加的 Hydra 参数用于训练。需要直接改 Hydra `output_dir` 或实验/run 名时，使用独立训练和测试命令。

只训练的入口分别为：

- `scripts/training/run_drivor_action_dit_dinov3_vitl_multigt_poor_training.sh`
- `scripts/training/run_drivor_action_dit_dinov3_vitl_single_gt_training.sh`
- `scripts/training/run_drivor_action_dit_dinov3_vitl_single_gt_long_training.sh`

三组模型参数结构均与原 DINOv3-L Action-DiT 一致，测试加载完整 checkpoint，仍为 64 份纯噪声、10 步 Euler 和原 Scorer。推理不读取监督 GT，原有训练配置保持兼容。

## 5. 验证

108 项模型/数据/loss 检查、15 项启动参数检查和 20 项训练后测试流程检查通过。三组分别在本地两张 A100 上使用真实图像与目标数据、BF16/DDP 完成每卡 2 个训练 step 和 1 个验证 batch，loss 有限，正常反传。该 smoke 只验证运行链路，不代表全量训练效果。

配置对比、三个 smoke 路径与验证记录位于 `../artifacts/action_dit_dinov3_gt_variants_20261017/validation.json`。
