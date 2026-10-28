# DrivoR / Action-DiT：多 GT 轨迹监督

更新：2026-10-15。适用于顶层 `DrivoR/` 的原版 DrivoR、Action-DiT 单前视版和 Action-DiT 多视图 query 版。

另有新增的[匈牙利匹配＋有效 GT 等权版本](DRIVOR_MULTI_GT_HUNGARIAN.md)：每条有效 GT 占用不同的 proposal，按实际 GT 数平均，使用独立配置和启动入口。本文下面继续描述原来的最近邻方案及 human/辅助加权方案。

Action-DiT 现也有独立的[human / 非 human 加权版本](DRIVOR_ACTION_DIT_MULTI_GT_WEIGHTED.md)，使用 `human v loss + 0.5 * mean(非 human v loss)`；本文中 Action-DiT 等权的说明仍指原有两个多 GT 配置。

新配置使用有效多 GT，关闭额外长 GT 监督。原版 DrivoR 默认采用 **human GT loss × 1.0＋其他有效 GT 的平均 loss × 0.5**，与 Drive-JEPA 的主/辅助目标配比对齐；Action-DiT 继续对所有有效 GT 等权平均。每条 GT 独立从同一组 64 个 proposal 中选择最近者，未被选中的输出不施加直接轨迹回归 loss。特征提取、轨迹网络参数结构、Scorer、在线评分标签和推理流程沿用各自版本。

按用户确认，**Action-DiT 的全部 64 个加噪输入仍由原始 human GT 构造**，不按多 GT 分配输入。多 GT 在预测之后参与匹配和 loss。

## 1. 数据读取与有效条数

默认数据目录：

```text
/cpfs29jb/data/zhaozc/project/drive_future/multi_traj_gt_data/navtrain_multigt_top4_dense_0p1s_f64_8gpu
```

通过 NAVSIM scene 的 `initial_token` 查找对应子目录。

| 文件 | 内容 | 本次用途 |
|---|---|---|
| `gt_trajectories.npy` | `[4,8,3]`，当前 ego 后轴局部坐标，`x,y,heading` | 轨迹监督，读取后转 FP32 |
| `gt_valid_mask.npy` | `[4]` 有效标记 | 与 padding 标记共同决定有效行 |
| `trajectory.json` | `gt_padded`、`target_times`、`gt_labels` 等 | 排除补齐行，检查时刻；原版加权监督按 `human_log` 识别 human 行 |
| `selected_ego_poses_global.npy` | `[4,41,3]`，0–4s、0.1s 间隔的全局坐标 | 不直接用于当前 8 点回归 |
| `gt_scores.npy` | 导出 GT 的分数 | 不作为 Scorer 标签或 GT loss 权重 |

`gt_trajectories.npy` 已经包含 0.5、1.0、…、4.0s 的 8 个局部轨迹点，直接与现有网络输出对应，不再对全局 41 点数据做重复归一化或坐标转换。

该导出数据中，`gt_valid_mask.npy` **可能把复制补齐的行也标为 true**，因此真正的有效条件是：

```python
valid = gt_valid_mask & ~np.asarray(metadata["gt_padded"])
```

例如 `1aa44d46e4ab5bc7` 的 npy mask 四项都为 true，但 `gt_padded=[false,true,true,true]`，只使用 1 条 GT；`654ad3d9062c5e32` 使用 4 条。有效的静止轨迹即使坐标全零也会保留；不能用“轨迹是否全零”判断有效性。

批处理打包为 `trajectory_multi_gt [B,4,8,3]` 和 `trajectory_multi_gt_valid [B,4]`，真实有效条数为 1–4；无效行只用于对齐张量形状，不参与匹配结果或 loss 汇总。

原版 `weighting=drive_jepa` 还生成 `trajectory_multi_gt_is_human [B,4]`。human GT 可能在导出数组第 0、1、2 或 3 行，按 `gt_labels == "human_log"` 识别，不按位置或浮点坐标相似性猜测。主项读取原始 NAVSIM `targets["trajectory"]`，导出的 human 行从辅助项中排除。

新配置的 `missing_policy: original` 表示：若某个 token 没有导出文件或没有有效行，使用其原始 human GT 作为唯一监督，并记录 `multi_gt_fallback_fraction`。这用于保持未覆盖的训练/验证场景可用。目录不存在、已有文件形状错误或有效轨迹含非有限值仍直接报错。需要严格要求每个场景都有导出 GT 时，启动命令末尾加：

```bash
agent.config.multi_gt.missing_policy=error
```

## 2. 每条 GT 独立选最近 proposal

设第 b 个场景有效 GT 数为 `K_b`，候选数 `M=64`。匹配方向如下，其中等权汇总仍用于 Action-DiT 和显式设置 `weighting=equal` 的 DrivoR：

```text
cost[b,k,m] = distance(proposal[b,m], GT[b,k])
winner[b,k] = argmin_m cost[b,k,m]
L_scene[b] = sum_k cost[b,k,winner[b,k]] / K_b
L_traj = mean_b L_scene[b]
```

先在每个场景内按有效 GT 数平均，再对 batch 平均；不会因为某个场景有 4 条 GT，就把该场景整体权重放大到 1 条 GT 场景的四倍。

该方向对应 Drive-JEPA 的 `drive_jepa_agent.py` 中“每条 GT 在 proposal 维度取 min，再对 GT 取 mean”的做法。它是独立最近邻匹配，**允许多个 GT 选到同一个 proposal**，不做 Hungarian 一对一分配；精确并列时只选第一条，避免把直接回归梯度分给所有并列项。

原版 DrivoR 默认改用 Drive-JEPA 的权重配方：

```text
L_human[b] = min_m d(proposal[b,m], original_human_GT[b])
L_other[b] = mean_有效非human_GT min_m d(proposal[b,m], GT[b,k])
L_traj = mean_b (1.0 * L_human[b] + 0.5 * L_other[b])
```

若某场景只有 human GT，`L_other[b]=0`，保留完整的 human loss；辅助项仍对整个 batch 平均，不只平均有其他 GT 的场景。两组加权结果直接相加，不再除以 1.5 或总 GT 数。若有 1 条 human＋3 条其他 GT，每条辅助 GT 的系数为 `0.5/3`，human 系数为 `1.0`。没有重复把导出的 human 行纳入辅助平均，也没有恢复额外的长 GT loss。

配置仅在 [drivoR_multigt.yaml](navsim/planning/script/config/common/agent/drivoR_multigt.yaml) 中启用：

```yaml
config:
  multi_gt:
    weighting: drive_jepa
    human_weight: 1.0
    other_weight: 0.5
```

Action-DiT 的两种视觉配置保持原来的等权 v loss。这里仅对齐原版 DrivoR 的 human/辅助 loss 配比，不引入 Drive-JEPA 的动态 anchor 重采样或中间层权重设置。

原版 DrivoR 的距离保持原有定义：

```text
d(g,p) = mean_8_points (|gx-px| + |gy-py| + |gheading-pheading|)
```

使用物理局部坐标的 L1，未增加新的坐标尺度或 heading wrap。默认启动配置保留 `prev_weight=0`、`inter_weight=0`，最终层的每条 GT 只监督其最近候选。共享参数和 attention 仍可能间接影响其他候选；“不监督其他 proposal”指没有对其输出施加直接回归 loss。

## 3. Action-DiT：保留原始 GT 加噪与 X→V loss

两个 Action-DiT 多 GT 配置都指定：

```yaml
config:
  long_trajectory_additional_poses: 0
  multi_gt:
    enabled: true
    max_gt: 4
    missing_policy: original
    input_gt_assignment: original
```

训练过程：

1. 原始 `targets["trajectory"] [B,8,3]` 经过现有 Action-DiT 归一化，展开成 64 份。
2. 每个场景采样一个 t，在其 64 个候选间共享；每个候选使用独立高斯噪声。时间分布、rank 随机流规则保持不变。
3. `x_sigma=(1-sigma)*original_GT + sigma*noise`，形成 `[B,64,8,3]`。
4. 一次 DiT 前向得到这 64 条干净轨迹的 X-prediction。每条有效 GT 都可以从全部 64 条输出中选择最近者。
5. 同一组 64 条 X-prediction 经过 `detach` 和反归一化送入现有 Scorer；训练不为评分追加 Euler 前向。

原始 human GT 不一定排在导出数组第一行，因此输入明确读取原来的 `targets["trajectory"]`，不把多 GT 的第 0 行当作原始 GT。

归一化、时间权重和 `sigma_min=0.1` 沿用原模型：

```text
x' = 2 * (x + 1.57) / 66.74 - 1
y' = 2 * (y + 19.68) / 42.00 - 1
h' = 2 * (h + 1.67) / 3.53 - 1
sigma_safe = max(sigma, 0.1)
```

对候选 m 及待匹配的归一化 GT g，正确的 X→V 转换是：

```text
v_pred[m] = (x_sigma[m] - pred_x[m]) / sigma_safe
v_gt[k,m] = (x_sigma[m] - GT[k]) / sigma_safe
v_pred[m] - v_gt[k,m] = (GT[k] - pred_x[m]) / sigma_safe
cost[k,m] = mean_8x3 ((GT[k] - pred_x[m]) / sigma_safe)^2
```

多 GT 不是构造该输入时的同一条轨迹，不能把所有 `v_gt` 都写成 `noise-original_GT`。代码直接计算上面的等价残差，避免两次大数相减，同时保留小 t 的分母下界。

每条 GT 按该距离选最近候选后，乘原 `training_weight(t)`，再按有效 GT 数平均。t 在同一场景的候选间共享，因此时间权重不改变最近邻关系。默认只有一个有效 GT 时，退化为原普通 GT 分支的 best-of-64 v loss；不恢复已被替代的额外长 GT 项。

验证/推理继续从 64 份纯噪声出发，执行原有 10 步 X→V Euler，随后由 Scorer 选出一条轨迹。没有把训练用 GT 输入引入推理接口。

## 4. Scorer 与特征保持各自原有设置

| 版本 | 视觉输入及 Scorer K/V | 轨迹回归 |
|---|---|---|
| 原版 DrivoR 多 GT | 四个当前视角，原有 16 queries/视角，64 个视觉 tokens | human 最近 L1＋0.5×其他 GT 最近 L1 的平均 |
| Action-DiT 单前视多 GT | 当前前视原生 3936 个 patch tokens | 每个有效 GT 的最近 proposal v MSE |
| Action-DiT 多视图多 GT | 四个当前视角，16 queries/视角，64 个视觉 tokens | 同上 |

Scorer 仍接收全部 64 条候选，六个评分头及 loss 不变；在线 NAVSIM oracle 标签仍为 `[B,64,7]`，没有变成 `[B,4,64,7]`，也不读取 `gt_scores.npy` 替代在线标签。评分分支对轨迹坐标的梯度截断保持原样，视觉/ego 的梯度路径也沿用各自版本。

多 GT 只增加小型 GT 文件读取和 GT 与 64 个候选的配对距离计算；原版加权方式另算一项原始 human GT 的距离，不增加网络或 oracle 调用。验证中已有的 human-GT L2 和 NAVSIM 评分指标继续使用原定义。

## 5. 独立配置和启动命令

| 模型 | agent 配置 | 启动脚本 |
|---|---|---|
| 原版 DrivoR | [drivoR_multigt.yaml](navsim/planning/script/config/common/agent/drivoR_multigt.yaml) | [run_drivor_multigt_training.sh](scripts/training/run_drivor_multigt_training.sh) |
| Action-DiT 单前视 | [drivoR_action_dit_multigt.yaml](navsim/planning/script/config/common/agent/drivoR_action_dit_multigt.yaml) | [run_drivor_action_dit_multigt_training.sh](scripts/training/run_drivor_action_dit_multigt_training.sh) |
| Action-DiT 多视图 query | [drivoR_action_dit_multiview_multigt.yaml](navsim/planning/script/config/common/agent/drivoR_action_dit_multiview_multigt.yaml) | [run_drivor_action_dit_multiview_multigt_training.sh](scripts/training/run_drivor_action_dit_multiview_multigt_training.sh) |

在 PAI-DLC 分别提交所需模型，每个任务仍配置 8 节点 × 8 GPU。每卡 batch 1、梯度累积 1、总 batch 64、BF16、LR 2e-4、25 epochs 及已有加速设置不变。DLC 分发命令，无需逐台机器手动启动。

原版 DrivoR：

```bash
cd /cpfs29jb/data/zhaozc/project/drive_future
DRIVOR_EXPERIMENT=training_drivoR_Nav1_navtrain_multigt_jepa_weighted \
DRIVOR_RUN_ID=drivor_multigt_jepa_seed2_run1 \
DRIVOR_PRECISION=bf16-mixed \
bash DrivoR/scripts/training/run_drivor_multigt_training.sh \
  auto_resume=false train_ckpt_path=null runtime.require_resume=false \
  runtime.console_log_interval=25
```

原版脚本默认也使用上述独立的 `multigt_jepa_weighted` 实验目录，避免默认恢复旧的等权实验。若要进一步降低辅助目标权重，在命令末尾追加 `agent.config.multi_gt.other_weight=0.25`；复现先前的等权方案可追加 `agent.config.multi_gt.weighting=equal`，并使用独立的 run ID。

Action-DiT 单前视：

```bash
cd /cpfs29jb/data/zhaozc/project/drive_future
DRIVOR_EXPERIMENT=training_drivoR_action_dit_navtrain_multigt \
DRIVOR_RUN_ID=action_dit_multigt_seed2_run1 \
bash DrivoR/scripts/training/run_drivor_action_dit_multigt_training.sh \
  auto_resume=false train_ckpt_path=null runtime.require_resume=false \
  runtime.console_log_interval=25
```

Action-DiT 多视图、每视角 16 queries：

```bash
cd /cpfs29jb/data/zhaozc/project/drive_future
DRIVOR_EXPERIMENT=training_drivoR_action_dit_multiview_navtrain_multigt \
DRIVOR_RUN_ID=action_dit_multiview_multigt_seed2_run1 \
bash DrivoR/scripts/training/run_drivor_action_dit_multiview_multigt_training.sh \
  auto_resume=false train_ckpt_path=null runtime.require_resume=false \
  runtime.console_log_interval=25
```

脚本自动进入现有 Conda 环境；数据仍为 `data/navsim`，metric cache 仍为 `data/metric_cache_navtrain`。多 GT 目录可通过 `DRIVOR_MULTI_GT_ROOT` 覆盖。`auto_resume=false` 不关闭原有 DINO/Action-DiT 预训练初始化。

只检查配置、不训练：

```bash
bash DrivoR/scripts/training/run_drivor_action_dit_multigt_training.sh \
  --cfg job --package agent --resolve
```

## 6. 兼容性和观测项

原来的三个 agent 配置及启动脚本继续走原来的普通/长 GT 逻辑；只有新配置打开 `multi_gt.enabled=true`。每个新脚本有独立实验目录，上面的命令显式指定目录以避免继承 shell 中其他实验的环境变量。

本次不增加模型参数，原/新监督方式在**同一模型、同一视觉变体**内的 state_dict 结构兼容。若显式从旧训练 checkpoint 续训，会恢复其 optimizer、scheduler 和训练进度，同时使用新的 loss；从头实验使用上述关闭恢复命令。完整 checkpoint 仍不能在单前视 patch 与多视图 query 两种视觉结构间直接互换。

等权 target builder（包括 Action-DiT）继续使用 `drivor_multi_gt_v1`；原版加权配置使用 `drivor_multi_gt_human_v1`，额外保存 human 身份 mask。启用 feature/target 缓存训练时，加权配置需要包含新目标，旧缓存不能提供该 mask。更换多 GT 数据目录或内容后，应重建对应 target 缓存；当前默认在线数据路径直接读取新目标，不增加哈希或 SHA 检验。

训练日志增加：

- `multi_gt_count`：每场景有效 GT 条数的平均值。
- `multi_gt_matched_proposals`：每场景被选中的不同 proposal 数，允许小于 GT 数。
- `multi_gt_fallback_fraction`：回退到原始单 GT 的场景比例。
- 原版加权配置的 `multi_gt_human_loss`、`multi_gt_other_loss`：乘权重之前的主/辅助 L1；总轨迹 loss 为 `1.0*human + 0.5*other`。
- 原版加权配置的 `multi_gt_other_count`：每场景有效非 human GT 条数的平均值。
- Action-DiT 的 `flow_multi_gt_loss`：新的多 GT v loss，替代旧的 normal/long 分项。

## 7. 验证

最初的等权多 GT 版本通过 **94 项针对性检查**（模型、loss、数据及旧路径回归 64 项，启动脚本 30 项），覆盖有效条数 1–4、排除 padding、独立多对一匹配、未匹配输出没有直接回归梯度、小 t 的 X→V 数值稳定性、原始 human GT 加噪、一次共享 64 候选前向，以及旧监督路径和评分 loss 的回归。

原版、Action-DiT 单前视和多视图三个配置均已通过真实双 GPU smoke，使用含 1、1、3、4 条有效 GT 的四个训练场景，以及两个各含 2 条 GT 的验证场景，每种配置只执行两步训练和每卡一个验证 batch。训练全局平均有效 GT 数均为 2.25，fallback 比例为 0，两卡各完成两次 optimizer step。两个 Action-DiT 使用最终的 `input_gt_assignment=original` 配置。检查覆盖 BF16、NCCL、真实图像/预训练权重与在线 oracle 的训练链路，不用于推断收敛、吞吐或全量 NAVSIM 成绩。完成记录见 [validation.json](../artifacts/drivor_multigt_20261015/validation.json)。

原版 DrivoR 的 human/辅助权重更新另通过 **92 项针对性检查**，覆盖辅助条数 0–3、human 位置变化与去重、不同 GT 数的 batch 平均、每条辅助轨迹的实际梯度系数、评分 loss 不变及 Action-DiT/旧等权路径回归。真实双 A100/BF16/NCCL 两步训练和每卡一个验证 batch 已完成，平均有效 GT 数为 2.25、非 human GT 数为 1.25、fallback 为 0。记录见[加权版本验证](../artifacts/drivor_multigt_jepa_weighting_20261015/validation.json)。

主要实现：[multi_gt.py](navsim/agents/drivoR/multi_gt.py)、[DrivoR loss](navsim/agents/drivoR/layers/losses/drivor_loss.py)、[Action-DiT model](navsim/agents/drivoR_action_dit/model.py)、[target builder 接入](navsim/agents/drivoR/drivor_agent.py)。
