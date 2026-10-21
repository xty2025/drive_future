# Action-DiT DINOv3-L：human GT + diverse8 辅助轨迹

基于原 `eval_navtest_action_dit_dinov3_vitl_multigt_pseudo_expert` 的普通 Action-DiT/DINOv3-L/DrivoR Scorer 架构，新增两组实验。主监督改为原始human GT，额外监督改用新生成目录中已选出的有效轨迹。

| 实验 | 主轨迹 loss | 新增轨迹 loss | 新增权重 |
|---|---|---|---|
| `human_diverse8_l1` | human GT best-of-64 v MSE | 物理坐标 L1 coverage | 0.5 |
| `human_diverse8_vmse` | human GT best-of-64 v MSE | 归一化 X-prediction→v MSE | 0.5 |

这两版不使用原top4 multi-GT、CLOVER专家缓存、长GT或未来register分支。有效监督目标为1条human加最多8条生成轨迹，最多9条；human与新增轨迹分组计算loss。

## 数据读取

默认目录与 `Cosmos3forMultiGT/MultiGT_generation/env.sh` 中的 `OUTPUT_ROOT` 加 `navtrain_diverse8_dense_0p1s_f64` 一致：

```text
/cpfs29jb/data/zhaozc/project/Cosmos3forMultiGT/runtime/multigt_raw64/outputs/navtrain_diverse8_dense_0p1s_f64
```

DSW/DLC使用同一个CPFS路径。可通过 `DRIVOR_DIVERSE8_GT_ROOT` 覆盖。

1. 每个token的 `complete.json` 存在后才读取；只检查完成标记，不做SHA/哈希校验。
2. 使用 `gt_trajectories.npy` 中自车后轴坐标的8个点，对应 `+0.5,+1.0,...,+4.0s`，float64转为训练用float32。
3. 有效mask为 `gt_valid_mask.npy & ~trajectory.json["gt_padded"]`。有效停止轨迹即使坐标全零也保留。
4. `gt_like_mask.npy` 只用于统计：它标记接近human的生成轨迹，不是human身份，不用于剔除或加权。新数据注明 `human_in_candidate_pool=false`。

直接使用已选出的有效轨迹，不再运行FPS、不按旧PDMS≥0.8规则重筛、不根据分数加权。`candidates.npz`、`selected_ego_poses_global.npy`、`gt_scores.npy` 不参与训练目标加载。

默认 `missing_policy=ignore`：缺少完成标记时，该场景只有主human监督，新增loss为0，不向额外集合重复补入human。每次访问重新检查完成标记，后续完成的场景可以被下一次训练访问读到。若要求所有场景都已完成，可追加 `agent.config.pseudo_expert.missing_policy=error`。

## Loss、匹配与网络

```text
L_main = min_64 vMSE(normalize(human_GT), X_pred)

L_extra_L1 = mean_有效生成GT min_64 mean_8点 sum_xy_heading |GT - denormalize(X_pred)|
L_extra_v  = mean_有效生成GT min_64 [时间权重 * mean_8点×3维 ((normalize(GT)-X_pred)/sigma_safe)^2]
sigma_safe = max(sigma, 0.1)

L_total = L_main + 0.5 * L_extra + L_score
```

先在每场景内部平均有效额外GT，再对batch平均；没有额外GT的场景该项取0。两组不是合并成9条后一起平均。每条GT独立选择最近proposal，允许不同GT选择同一个proposal，不使用匈牙利匹配。

全部64份加噪轨迹仍来自原始human GT，各份独立噪声，沿用原time采样和Time+Ego条件。一次DiT前向产生同组64条X-prediction，主loss、新增loss和Scorer共用这些输出，不额外跑8组前向。

L1版使用未detach的反归一化物理轨迹；vMSE版直接复用未detach的归一化X-prediction与实际time。额外loss可回传DiT和当前视觉LoRA；Scorer继续接收detach后的64条轨迹，通过原在线Oracle评分监督训练。

输入仍为当前单前视 `672×1152`、DINOv3 ViT-L/16的3024个原生patches、无额外scene query。DINOv3基座冻结并训练Q/V LoRA，Action-DiT保留原Action256预训练初始化。推理仍为10步Euler，Scorer结构不变。

## 代码与配置

- 新读取器及agent：[`diverse_gt.py`](navsim/agents/drivoR_action_dit/diverse_gt.py)。内部复用已有 `pseudo_experts` loss字段，但数据来自diverse8，不调用CLOVER读取器。
- L1配置：[`drivoR_action_dit_dinov3_vitl_human_diverse8_l1.yaml`](navsim/planning/script/config/common/agent/drivoR_action_dit_dinov3_vitl_human_diverse8_l1.yaml)。
- vMSE配置：[`drivoR_action_dit_dinov3_vitl_human_diverse8_vmse.yaml`](navsim/planning/script/config/common/agent/drivoR_action_dit_dinov3_vitl_human_diverse8_vmse.yaml)。
- 日志：`human_trajectory_loss`、`diverse_gt_loss`（未乘0.5）、`diverse_gt_count`、`diverse_gt_scene_coverage`、`diverse_gt_missing_fraction`、`diverse_gt_like_count`。

## DLC：训练结束后自动 FP32 navtest

分别提交两个8机×8卡任务，总batch64、每卡batch1、累积1，25epochs、seed2、LR=2e-4。训练BF16；每epoch保存checkpoint、FP32验证val PDMS并写CSV；训练结束后节点0用本次last.ckpt自动进行FP32 navtest。

L1 coverage版：

```bash
cd /cpfs29jb/data/zhaozc/project/drive_future

DRIVOR_TRAIN_EXPERIMENT=training_drivoR_action_dit_dinov3_vitl_navtrain_human_diverse8_l1 \
DRIVOR_RUN_ID=action_dit_human_diverse8_l1_seed2_train_eval_run1 \
DRIVOR_SCENE_QUERIES=0 \
DRIVOR_PRECISION=bf16-mixed DRIVOR_EVAL_PRECISION=32-true \
bash DrivoR/scripts/training/run_drivor_action_dit_dinov3_vitl_aux_train_eval.sh \
  human_diverse8_l1
```

vMSE版：

```bash
cd /cpfs29jb/data/zhaozc/project/drive_future

DRIVOR_TRAIN_EXPERIMENT=training_drivoR_action_dit_dinov3_vitl_navtrain_human_diverse8_vmse \
DRIVOR_RUN_ID=action_dit_human_diverse8_vmse_seed2_train_eval_run1 \
DRIVOR_SCENE_QUERIES=0 \
DRIVOR_PRECISION=bf16-mixed DRIVOR_EVAL_PRECISION=32-true \
bash DrivoR/scripts/training/run_drivor_action_dit_dinov3_vitl_aux_train_eval.sh \
  human_diverse8_vmse
```

只训练的独立入口：

```bash
bash DrivoR/scripts/training/run_drivor_action_dit_dinov3_vitl_human_diverse8_l1_training.sh \
  auto_resume=false train_ckpt_path=null runtime.require_resume=false runtime.console_log_interval=25

bash DrivoR/scripts/training/run_drivor_action_dit_dinov3_vitl_human_diverse8_vmse_training.sh \
  auto_resume=false train_ckpt_path=null runtime.require_resume=false runtime.console_log_interval=25
```

两个新实验使用原预训练初始化。原multi-GT、CLOVER、register等配置继续可用。新版本没有新增网络参数，测试用原 `drivoR_action_dit_dinov3_vitl` agent严格恢复checkpoint，测试不需要diverse8目录。

## 本地验证

本地18项模型/loss/目标读取/训练后测试衔接检查、10项启动参数检查及两套Hydra配置合并通过。检查了6个真实场景，均有8条有效生成GT。

两版均完成本地两卡smoke：每卡2个训练step、1个val batch，使用真实DINOv3-L/Action256权重和真实diverse8目标，报告的loss均有限，额外GT数量为8，并完成FP32 val PDMS计算。仅验证运行链路，没有进行全量训练或全量navtest。

记录：[validation.json](../artifacts/action_dit_human_diverse8_20261020/validation.json)。
