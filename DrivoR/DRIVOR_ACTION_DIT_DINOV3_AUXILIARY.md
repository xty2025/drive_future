# DINOv3 ViT-L Action-DiT multi-GT：Bank / pseudo-expert 两组实验

本文的 `pseudo_expert` 指原 L1 coverage 版本。新增的独立 `pseudo_expert_vmse` 实验改为与 multi-GT 相同的 X-prediction→v MSE，配置和命令见 [pseudo-expert v MSE 说明](DRIVOR_ACTION_DIT_PSEUDO_EXPERT_VMSE.md)。

两组实验基于以下已完成训练保存的配置：

```text
DrivoR/exp/ke/training_drivoR_action_dit_dinov3_vitl_navtrain_multigt/
  action_dit_dinov3_vitl_seed2_train_eval_run1/code/hydra/config.yaml
```

已逐项核对：新启动脚本解析后的模型配置（去掉新增辅助项）、原 loss 配置和优化器参数，与上述记录一致。

## 保留的基线

- 当前单前视，672×1152；冻结 DINOv3 ViT-L/16，Q/V LoRA rank 32；无 Scene queries，3024 个原生 patches 同时供 DiT 和 Scorer 使用。
- 原 Action-DiT 256 维、30 层；训练一次 X-prediction，产生 64 个 proposals。当前 human GT 构造所有加噪输入，64 份独立噪声、每场景共享采样时间。
- 原始 multi-GT v loss 保留：最多 4 条有效 GT，各自匹配最近 proposal，允许重复匹配，等权平均；`input_gt_assignment=original`、`matching=nearest`、`weighting=equal`。
- 不使用原长 GT 项（沿用此 multi-GT 基线的 `long_trajectory_additional_poses=0`）。
- 原六头 Scorer 对生成的 64 条轨迹计算在线 NAVSIM 标签和 score loss，输入轨迹坐标 detach。
- 8 机×8 卡、每卡 batch 1、累积 1、总 batch 64、BF16、25 epochs、seed 2、LR 2e-4。

默认新开独立训练，仍加载原 DINOv3 和 Action-DiT 的预训练初始化文件。下面的新训练命令不会自动续训已完成的基线 `last.ckpt`；如要做断点续训，按原训练入口显式指定 `train_ckpt_path` 和相应 resume 参数。

## 实验 A：Scorer 额外加 Bank loss

配置：[`drivoR_action_dit_dinov3_vitl_bank.yaml`](navsim/planning/script/config/common/agent/drivoR_action_dit_dinov3_vitl_bank.yaml)。

每个场景从现有 CLOVER Bank 选 16 条候选：先抽最多 8 条不安全轨迹（NC<0.5 或 DAC<0.5），其余从剩余有效候选抽取，不足则有放回补齐。同一套 Scorer 分别处理生成的 64 条和 Bank 的 16 条，两组各自执行候选 self-attention。两次调用复用同一份 DINOv3 特征和 ego embedding。

Bank 使用已存的真实预计算子分，不增加在线 Oracle 计算。仍只进行一次图像编码、一次 DiT 前向；新增一次 16 候选的 Scorer 前向。

```text
L_Bank = mean(5 个非 EP 子分的 BCE) + 0.1 × L1(sigmoid(EP), EP_label)
L_total = L_multiGT_v + L_generated_score + 0.5 × L_Bank
```

Bank loss 更新 Scorer、当前视觉和 ego 特征路径；Bank 轨迹坐标 detach，不通过 Bank 分支更新 DiT。

默认缓存已存在：`DrivoR/exp/world4scorer_data/bank_clover`，可用 `DRIVOR_BANK_PACK` 覆盖。权重参数为 `agent.config.bank.weight`，默认 0.5。

DLC 单独提交一个 **8 机×8 卡**任务：

```bash
cd /cpfs29jb/data/zhaozc/project/drive_future

DRIVOR_EXPERIMENT=training_drivoR_action_dit_dinov3_vitl_navtrain_multigt_bank \
DRIVOR_RUN_ID=action_dit_dinov3_vitl_multigt_bank_seed2_run1 \
DRIVOR_SCENE_QUERIES=0 DRIVOR_PRECISION=bf16-mixed \
bash DrivoR/scripts/training/run_drivor_action_dit_dinov3_vitl_bank_training.sh \
  auto_resume=false train_ckpt_path=null runtime.require_resume=false \
  runtime.console_log_interval=25
```

## 实验 B：轨迹额外加 pseudo-expert coverage

配置：[`drivoR_action_dit_dinov3_vitl_pseudo_expert.yaml`](navsim/planning/script/config/common/agent/drivoR_action_dit_dinov3_vitl_pseudo_expert.yaml)。

沿用已经接入原 DrivoR 的 CLOVER Stage-1 筛选及 loss：从同场景原始候选中筛 PDMS≥0.8，贪心最远点采样最多 8 条专家；不足且未覆盖 human 时按 CLOVER 规则补入 human，没有合格候选时回退 human。原始来源仍是 `dataset_decoupled_v2_clean.pkl`。

对同一次 X-prediction 的 64 条干净预测进行反归一化，再计算物理坐标下的 coverage：

```text
d(i,k) = mean_8时刻(|Δx| + |Δy| + |Δheading|)
L_PE = mean_batch[sum_有效专家 k(min_64个proposal i d(i,k)) / 有效专家数]
L_total = L_multiGT_v + L_generated_score + 0.5 × L_PE
```

这里新增的是 CLOVER L1 coverage，原 multi-GT v loss 继续按原逻辑计算。Pseudo-experts 仅参与额外监督；human 加噪输入、noise/t 采样、原 multi-GT 匹配均保持基线设置。

带梯度的反归一化预测单独存于 `_pseudo_expert_proposals`，数值与 Scorer 的 `proposals` 完全相同。这样 coverage 梯度可以更新 DiT、视觉和 ego 路径，Scorer 仍接收 detach 后的轨迹。仅一次图像编码、一次 DiT、一次生成候选 Scorer 调用。

默认缓存已存在：`DrivoR/exp/clover_data/pseudo_experts_k8_thr0p8`，可用 `DRIVOR_PSEUDO_EXPERT_PACK` 覆盖。权重参数为 `agent.config.pseudo_expert.weight`，默认 0.5。

DLC 另提交一个 **8 机×8 卡**任务：

```bash
cd /cpfs29jb/data/zhaozc/project/drive_future

DRIVOR_EXPERIMENT=training_drivoR_action_dit_dinov3_vitl_navtrain_multigt_pseudo_expert \
DRIVOR_RUN_ID=action_dit_dinov3_vitl_multigt_pseudo_expert_seed2_run1 \
DRIVOR_SCENE_QUERIES=0 DRIVOR_PRECISION=bf16-mixed \
bash DrivoR/scripts/training/run_drivor_action_dit_dinov3_vitl_pseudo_expert_training.sh \
  auto_resume=false train_ckpt_path=null runtime.require_resume=false \
  runtime.console_log_interval=25
```

两个配置分别只启用自己的辅助 loss。日志分别增加 `bank_loss / bank_scene_coverage` 或 `pseudo_expert_loss / pseudo_expert_count / pseudo_expert_fallback_fraction`；原 `flow_multi_gt_loss` 单独保留，便于判断轨迹主项是否变化。

## 测试与兼容性

上面的两个 `*_training.sh` 入口只训练。需要同一个 DLC 任务训完自动测试时，分别使用以下命令，每组各提交一个 8 机×8 卡任务。训练成功后，仅节点 0 自动读取本次 run 最新 `version_*` 下的 `last.ckpt`，用本机 8 卡、FP32 测试 navtest。

Bank：

```bash
cd /cpfs29jb/data/zhaozc/project/drive_future
DRIVOR_RUN_ID=action_dit_dinov3_vitl_multigt_bank_seed2_train_eval_run1 \
DRIVOR_SCENE_QUERIES=0 DRIVOR_EVAL_PRECISION=32-true \
bash DrivoR/scripts/training/run_drivor_action_dit_dinov3_vitl_aux_train_eval.sh bank
```

Pseudo-expert：

```bash
cd /cpfs29jb/data/zhaozc/project/drive_future
DRIVOR_RUN_ID=action_dit_dinov3_vitl_multigt_pseudo_expert_seed2_train_eval_run1 \
DRIVOR_SCENE_QUERIES=0 DRIVOR_EVAL_PRECISION=32-true \
bash DrivoR/scripts/training/run_drivor_action_dit_dinov3_vitl_aux_train_eval.sh pseudo_expert
```

组合入口后面的 Hydra 参数用于训练。训练目录沿用本页的实验名（可用 `DRIVOR_TRAIN_EXPERIMENT` 修改）；自定义 `output_dir` 或直接用 Hydra 修改实验/run 名时，应使用分开的训练与评测命令。测试结果位于 `DrivoR/exp/ke/eval_navtest_action_dit_dinov3_vitl_multigt_<bank或pseudo_expert>/<DRIVOR_RUN_ID>/`，含逐场景 CSV 与 `pdms_summary.txt`。

两组均没有新增网络参数，checkpoint 与原 `drivoR_action_dit_dinov3_vitl` 网络结构兼容。推理使用原始 10 步 Euler，测试无需 Bank 或 pseudo-expert 缓存。

独立测试时，DLC 使用 **1 机×8 卡、FP32**，将 checkpoint 路径和实验/run 名替换为对应训练：

```bash
cd /cpfs29jb/data/zhaozc/project/drive_future

NPROC_PER_NODE=8 DRIVOR_EVAL_WORKER=single_machine_thread_pool \
DRIVOR_SCORE_WORKERS=32 \
DRIVOR_EXPERIMENT=eval_navtest_action_dit_dinov3_vitl_auxiliary \
DRIVOR_RUN_ID=填写对应训练的run_id \
bash DrivoR/scripts/evaluation/run_drivor_pdm_score_evaluation.sh \
  /实际路径/last.ckpt \
  agent=drivoR_action_dit_dinov3_vitl \
  agent.config.multi_gt.enabled=false \
  agent.config.action_dit.pretrained_path=null \
  agent.config.action_dit.num_inference_steps=10 \
  trainer.params.precision=32-true dataloader.params.batch_size=1
```

已通过 95 项模型/loss/兼容性检查和 10 项 DLC 启动参数检查，覆盖单次 DiT 前向、Scorer 调用次数、梯度边界、基线输出与 v loss 一致性，以及 FP32 推理 state_dict 兼容。验证记录保存于 `../artifacts/action_dit_dinov3_auxiliary_20261017/`。

两组均完成真实数据、真实预训练 DINOv3-L / Action-DiT 权重、两张 A100、BF16/DDP smoke：每卡 2 个训练 step 和 1 个验证 batch，辅助 loss 有限并参与训练。该检查只验证运行和反传链路，不代表完整训练效果。记录分别为：

- Bank：`../artifacts/drivor_two_gpu_smoke/action_dit_dinov3_vitl_bank_r5g7bx0u/`。
- Pseudo-expert：`../artifacts/drivor_two_gpu_smoke/action_dit_dinov3_vitl_pseudo_expert_1103dmzs/`。
