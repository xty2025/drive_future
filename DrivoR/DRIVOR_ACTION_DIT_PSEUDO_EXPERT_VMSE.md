# Action-DiT DINOv3-L：pseudo-expert 使用 X-prediction→v MSE

基于本版新增的两组未来 register 实验、DLC 训练及 FP32 测试命令见 [pseudo-expert + registers](DRIVOR_ACTION_DIT_PSEUDO_EXPERT_VMSE_REGISTERS.md)。

这是基于 `run_drivor_action_dit_dinov3_vitl_pseudo_expert_training.sh` 新增的独立实验。旧脚本与旧配置继续使用物理坐标 L1 coverage；新入口是：

```text
scripts/training/run_drivor_action_dit_dinov3_vitl_pseudo_expert_vmse_training.sh
navsim/planning/script/config/common/agent/drivoR_action_dit_dinov3_vitl_pseudo_expert_vmse.yaml
```

## Loss

Pseudo-expert 直接复用 `multi_gt_velocity_loss()`，与原 multi-GT 相同：

1. 将原始 pseudo-expert 轨迹按 Action-DiT 的相同规则归一化。
2. 使用**同一次 DiT 前向的归一化 X-prediction**及其实际采样时间。
3. 对每条有效 pseudo-expert，从 64 个候选中选 v MSE 最小者。
4. 对有效 pseudo-expert 平均，再对 batch 平均，沿用原系数 0.5。

```text
v_pred   = (x_t - x_pred) / sigma_safe
v_target = (x_t - normalize(pseudo_GT)) / sigma_safe
sigma_safe = max(sigma, 0.1)

因此：
v_pred - v_target = (normalize(pseudo_GT) - x_pred) / sigma_safe

L_PE = mean_batch mean_有效PE min_64候选 [时间权重 × mean_8点×3维(v误差²)]
L_total = L_原multiGT_v + 0.5 × L_PE_v + L_原Scorer
```

沿用原 4 条以内 multi-GT 主 loss；pseudo-expert 作为最多 8 条额外监督单独平均、乘 0.5 后相加。两组的回归误差和匹配函数一致，汇总权重保持原实验配方。不同 GT/专家仍可匹配同一 proposal。

新分支直接使用未 detach 的归一化预测，不经过反归一化再归一化的往返。Scorer 继续接收 detach 后的物理轨迹。GT 目标无梯度，pseudo-expert v loss 可回传到 DiT、视觉和 ego 路径。

## 保留的输入与网络

- 当前单前视 DINOv3 ViT-L、原生 patches、Q/V LoRA，以及原 Action-DiT/Scorer 网络均不变。
- 64 个加噪输入仍全部由原始 human GT 构造，pseudo-expert 只改变监督目标。
- 原 noise/t 采样、shift=5、时间权重和 `sigma_min=0.1` 保留。
- 一次 DiT 训练前向；原 Scorer 仍评分同组 64 条轨迹。
- CLOVER 数据筛选继续使用 PDMS≥0.8、最远点采样、最多 8 条；复用现有缓存，无需重建。
- 没有新增网络参数；测试使用原 `drivoR_action_dit_dinov3_vitl` 模型，FP32、10 步 Euler。

## DLC 启动

分别新开 **8 机×8 卡**任务，总 batch64，训练 BF16，25 epochs，seed2。只训练：

```bash
cd /cpfs29jb/data/zhaozc/project/drive_future

DRIVOR_EXPERIMENT=training_drivoR_action_dit_dinov3_vitl_navtrain_multigt_pseudo_expert_vmse \
DRIVOR_RUN_ID=action_dit_dinov3_vitl_pseudo_expert_vmse_seed2_run1 \
DRIVOR_SCENE_QUERIES=0 DRIVOR_PRECISION=bf16-mixed \
bash DrivoR/scripts/training/run_drivor_action_dit_dinov3_vitl_pseudo_expert_vmse_training.sh \
  auto_resume=false train_ckpt_path=null runtime.require_resume=false \
  runtime.console_log_interval=25
```

训练成功后自动用本次 `last.ckpt` 做 **FP32 navtest**：

```bash
cd /cpfs29jb/data/zhaozc/project/drive_future

DRIVOR_RUN_ID=action_dit_dinov3_vitl_pseudo_expert_vmse_seed2_train_eval_run1 \
DRIVOR_SCENE_QUERIES=0 DRIVOR_PRECISION=bf16-mixed DRIVOR_EVAL_PRECISION=32-true \
bash DrivoR/scripts/training/run_drivor_action_dit_dinov3_vitl_aux_train_eval.sh pseudo_expert_vmse
```

仅节点 0 使用本机 8 卡测试，默认每卡推理 batch1、32 个 CPU 评分进程。训练结果目录带 `_multigt_pseudo_expert_vmse`，不复用旧 L1 实验目录。已有 L1 权重的训练历史不会因新代码而改变。

原来的 `pseudo_expert.weight=0.5` 保留，可通过 `agent.config.pseudo_expert.weight` 调整。原 `flow_multi_gt_loss`、新 `pseudo_expert_loss` 分别记录，并增加 `pseudo_expert_matched_proposals` 以观察覆盖的候选数量。

## 验证

104 项模型/loss/兼容性检查、5 项启动参数检查及 4 项训练后测试衔接检查通过。已完成真实 DINOv3-L/Action-DiT 权重、真实 GT 数据的两张 A100 smoke：每卡 2 个训练 step、1 个验证 batch，pseudo-expert v loss 实际参与反传，loss 有限。仅验证运行链路，不代表完整训练效果。

记录：`../artifacts/action_dit_pseudo_expert_vmse_20261019/validation.json`。
