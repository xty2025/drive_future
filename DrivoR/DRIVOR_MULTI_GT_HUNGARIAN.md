# DrivoR / Action-DiT：匈牙利匹配与有效 GT 等权监督

更新：2026-10-15。此变体新增独立配置和入口；之前的最近邻等权方案、原版 DrivoR 的 human/辅助加权方案保持可用。

## 1. 匹配与 loss

每个场景读取实际有效的 `K` 条 GT（1–4），预测 `M=64` 条 proposal。先计算 `[K,64]` 的 cost，再用匈牙利算法求最小总 cost 的一对一分配：

```text
π* = argmin_π sum_k cost[k, π(k)]
约束：不同 GT 的 π(k) 不能相同

L_scene = sum_k cost[k, π*(k)] / K
L_batch = mean_scene L_scene
```

**4 条有效 GT 匹配 4 条不同的 proposal，3 条匹配 3 条，1 条时退化为 best-of-64。** 每个场景除以自己的有效 GT 数；不除以 64，也不把 padding 当作 GT。

匈牙利匹配优化的是整组匹配的总误差。某条 GT 可能让出自己单独最近的 proposal，以得到更低的全局总 cost；这与按顺序挑选“尚未占用的最近 proposal”的贪心算法不同。

导出集合中的 human GT 与其他有效 GT 等权，不再单独加入 human loss，也没有 `1.0 / 0.5` 分组系数。额外长 GT 仍关闭。匹配到的输出计算直接轨迹回归 loss，其他输出没有直接轨迹回归监督；网络共享参数仍可能间接影响其他候选。

## 2. 两种网络保留各自的 cost

原版 DrivoR 使用物理局部坐标 L1：

```text
cost[k,m] = mean_8点 (|Δx| + |Δy| + |Δheading|)
```

Action-DiT 使用原有归一化 v MSE：

```text
sigma_safe = max(sigma, 0.1)
cost[k,m] = mean_8×3 ((normalize(GT[k]) - X_prediction[m]) / sigma_safe)^2
```

Action-DiT 先按该 cost 做匈牙利匹配，再施加原有 `training_weight(t)`，按有效 GT 数平均。同一场景的 64 条候选共享 t，所以 sigma 分母和时间权重不会改变候选之间的分配偏好。归一化与小 t 下界保持原实现。

Action-DiT 仍用**原始 human GT 构造全部 64 个加噪输入**，64 份独立噪声、场景内共享 t；一次 DiT 前向得到用于匹配和 Scorer 的同一组 64 条 X-prediction。训练不追加 Euler，验证/推理仍使用 10 步 Euler。

## 3. 数据、Scorer 与计算位置

数据仍来自 `multi_traj_gt_data/navtrain_multigt_top4_dense_0p1s_f64_8gpu`：读取局部 8 点 `gt_trajectories.npy`，有效条件是 `gt_valid_mask & ~gt_padded`。因此不足 4 条时只使用真实条数，补齐行不会占用 proposal。

`missing_policy=original` 保持现有行为：缺少该场景导出数据时，回退为原始 human GT 一条，并记录 fallback 比例。设置 `missing_policy=error` 可要求完整覆盖。

所有 64 条候选继续进入原 Scorer，在线标签保持 `[B,64,7]`。Scorer、视觉 K/V、轨迹到评分分支的 detach、优化器、随机流和 batch 配置保持各自版本的设置。

匈牙利分配使用 SciPy，在 CPU 上处理 detach 后的小型 cost 矩阵；仅复制 cost 和有效 mask，不传图像、特征或完整轨迹网络。得到索引后，从原设备上的 cost 张量 `gather`，保留匹配误差的梯度。相比纯 GPU 的逐行 min，这增加少量数据传输及 CPU 求解，不能据矩阵大小承诺训练零开销。

## 4. 新配置和启动入口

| 模型 | 配置 | 启动脚本 |
|---|---|---|
| 原版 DrivoR | [drivoR_multigt_hungarian.yaml](navsim/planning/script/config/common/agent/drivoR_multigt_hungarian.yaml) | [run_drivor_multigt_hungarian_training.sh](scripts/training/run_drivor_multigt_hungarian_training.sh) |
| Action-DiT 单前视 | [drivoR_action_dit_multigt_hungarian.yaml](navsim/planning/script/config/common/agent/drivoR_action_dit_multigt_hungarian.yaml) | [run_drivor_action_dit_multigt_hungarian_training.sh](scripts/training/run_drivor_action_dit_multigt_hungarian_training.sh) |
| Action-DiT 多视图 query | [drivoR_action_dit_multiview_multigt_hungarian.yaml](navsim/planning/script/config/common/agent/drivoR_action_dit_multiview_multigt_hungarian.yaml) | [run_drivor_action_dit_multiview_multigt_hungarian_training.sh](scripts/training/run_drivor_action_dit_multiview_multigt_hungarian_training.sh) |

共同配置是：

```yaml
config:
  multi_gt:
    matching: hungarian
    weighting: equal
```

PAI-DLC 仍选择 PyTorch 分布式任务、8 节点 × 每节点 8 GPU；每卡 batch 1、累积 1、总 batch 64、BF16。下面每段用于单独提交一个新实验。

原版 DrivoR：

```bash
cd /cpfs29jb/data/zhaozc/project/drive_future
DRIVOR_EXPERIMENT=training_drivoR_Nav1_navtrain_multigt_hungarian \
DRIVOR_RUN_ID=drivor_multigt_hungarian_seed2_run1 \
DRIVOR_PRECISION=bf16-mixed \
bash DrivoR/scripts/training/run_drivor_multigt_hungarian_training.sh \
  auto_resume=false train_ckpt_path=null runtime.require_resume=false \
  runtime.console_log_interval=25
```

Action-DiT 单前视：

```bash
cd /cpfs29jb/data/zhaozc/project/drive_future
DRIVOR_EXPERIMENT=training_drivoR_action_dit_navtrain_multigt_hungarian \
DRIVOR_RUN_ID=action_dit_multigt_hungarian_seed2_run1 \
DRIVOR_PRECISION=bf16-mixed \
bash DrivoR/scripts/training/run_drivor_action_dit_multigt_hungarian_training.sh \
  auto_resume=false train_ckpt_path=null runtime.require_resume=false \
  runtime.console_log_interval=25
```

Action-DiT 多视图：

```bash
cd /cpfs29jb/data/zhaozc/project/drive_future
DRIVOR_EXPERIMENT=training_drivoR_action_dit_multiview_navtrain_multigt_hungarian \
DRIVOR_RUN_ID=action_dit_multiview_multigt_hungarian_seed2_run1 \
DRIVOR_PRECISION=bf16-mixed \
bash DrivoR/scripts/training/run_drivor_action_dit_multiview_multigt_hungarian_training.sh \
  auto_resume=false train_ckpt_path=null runtime.require_resume=false \
  runtime.console_log_interval=25
```

## 5. 原版训练完成后自动测试

将原版训练命令最后的 `runtime.console_log_interval=25` 替换成下面整段。只有训练成功后的 DLC 主节点执行测试，使用最后保存的 checkpoint：

```bash
  runtime.console_log_interval=25 && \
if [ "${NODE_RANK:-${RANK:-0}}" = "0" ]; then
  DRIVOR_EXPERIMENT=eval_navtest_drivor_multigt_hungarian \
  DRIVOR_RUN_ID=drivor_multigt_hungarian_seed2_run1 \
  DRIVOR_EVAL_WORKER=single_machine_thread_pool \
  DRIVOR_SCORE_WORKERS=32 NPROC_PER_NODE=8 \
  bash DrivoR/scripts/evaluation/run_drivor_pdm_score_evaluation.sh \
    DrivoR/exp/ke/training_drivoR_Nav1_navtrain_multigt_hungarian/drivor_multigt_hungarian_seed2_run1/lightning_logs/version_0/checkpoints/last.ckpt
fi
```

checkpoint 路径对应上述新 run 的 `version_0`，更换 run 或恢复到其他 version 时需同步更新。推理网络未变，普通 `agent=drivoR` 即可评测；navtest 不需要多 GT 数据。

## 6. 兼容与诊断

原来的三个多 GT 入口保持其最近邻匹配和既有权重。新入口使用独立实验目录，不自动接续旧实验。未指定 `matching` 的旧配置仍解释为 `nearest`；匈牙利配置必须使用 `weighting=equal`，不会静默沿用 human/辅助加权。

同一模型、同一视觉变体的参数结构和 checkpoint 键不变；仅训练目标改变。数据缓存继续使用已有等权目标名 `drivor_multi_gt_v1`，无需重新生成 GT 文件。原版加权方案的 `drivor_multi_gt_human_v1` 是另一种目标缓存。

匈牙利模式下，`multi_gt_matched_proposals` 应与 `multi_gt_count` 相等；前者统计不同 proposal 数，后者统计有效 GT 数。可用它们核对一对一分配是否生效。

已通过 **124 项检查**，覆盖全局最优分配、有效条数 1–4、padding 排除、所选候选互不重复、未匹配输出的直接回归梯度为零、每场景按 GT 数平均、原有模式兼容以及三个新启动入口。原版 DrivoR、Action-DiT 单前视和多视图版均完成真实双 A100/BF16/NCCL 的两步训练和每卡一个验证 batch；平均有效 GT 数与不同匹配 proposal 数均为 2.25，fallback 为 0。

实现见 [multi_gt.py](navsim/agents/drivoR/multi_gt.py)，验证记录见 [validation.json](../artifacts/drivor_multigt_hungarian_20261015/validation.json)。短 smoke 仅验证训练链路，不代表收敛、正式吞吐或 NAVSIM 指标。
