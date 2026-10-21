# DINOv3 ViT-L Action-DiT：human＋long GT＋CLOVER pseudo-expert

基线是 `exp/ke/eval_navtest_action_dit_dinov3_vitl_single_gt_long/action_dit_dinov3_vitl_single_gt_long_seed2_train_eval_run1` 对应的训练配置。新增实验保留普通 human GT 和长 human GT 的 v loss，再增加权重为 **0.5** 的原 CLOVER **L1 coverage**。

## 网络与监督

配置：`navsim/planning/script/config/common/agent/drivoR_action_dit_dinov3_vitl_single_gt_long_pseudo_expert.yaml`。

沿用 `DrivoRActionDiTAuxiliaryAgent` 已有实现，视觉编码器、Action-DiT、Scorer 和参数形状与基线一致。当前帧单前视、672×1152、DINOv3 ViT-L、原生 3024 个 patch、LoRA rank 32；Action-DiT 仍采用原 256 维预训练权重和原轨迹归一化。

- 普通 human GT 和长 GT 分别构造 64 条加噪输入，共 128 条打包成一次 DiT 前向，各自计算 best-of-64 v loss。长 GT 沿用 `long_trajectory_additional_poses=2`。
- 新增 pseudo-expert 使用普通组的 64 条 X-prediction，反归一化到自车坐标系后计算 L1 coverage。每条有效 expert 独立选最近的 proposal，再按有效 expert 数平均；不同 expert 可以选择同一 proposal。
- Scorer 继续使用这组普通分支的 64 条轨迹，保留轨迹梯度截断和原在线 NAVSIM 评分监督。
- pseudo-expert loss 回传到轨迹生成器及视觉特征；GT 本身 detach。长分支保持原来的监督。
- 原 top4 `multi_gt.enabled=false`。新增标签从 CLOVER 缓存读取。

设普通组反归一化输出为 `P[b,m,t,d]`，有效 pseudo-expert 为 `G[b,k,t,d]`：

```text
distance[b,m,k] = mean_t sum_(x,y,heading) |P[b,m,t,d] - G[b,k,t,d]|
L_PE = mean_b mean_(valid k) min_m distance[b,m,k]

L_total = L_human_v + L_long_v + 0.5 * L_PE + L_score
```

坐标误差使用米，heading 使用弧度，直接沿用现有 CLOVER L1 实现。普通/长 GT 的 v loss 继续采用原时间权重和 sigma 下界 0.1。精确并列最小值沿用 `amin` 的梯度分配行为。

## pseudo-expert 数据

默认目录：

```text
/cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/clover_data/pseudo_experts_k8_thr0p8
```

可用 `DRIVOR_PSEUDO_EXPERT_PACK` 覆盖。读取现有 `tokens.npy`、`pseudo_experts.npy` 和 `pseudo_expert_mask.npy`；每场景最多 8 条，分数筛选阈值 0.8。其来源是现有 CLOVER/World4Scorer 的 `dataset_decoupled_v2_clean.pkl`。继续使用现有 reader 的 human 补入/回退规则，未重新生成或重选标签。

## DLC 训练后自动测试

在 DLC 创建 8 节点、每节点 8 GPU 的 PyTorch 分布式任务，启动命令填写：

```bash
cd /cpfs29jb/data/zhaozc/project/drive_future

export DRIVOR_RUN_ID=action_dit_dinov3_vitl_single_gt_long_pseudo_expert_seed2_train_eval_run1
export DRIVOR_PRECISION=bf16-mixed
export DRIVOR_SCENE_QUERIES=0

bash DrivoR/scripts/training/run_drivor_action_dit_dinov3_vitl_single_gt_long_pseudo_expert_train_eval.sh
```

每卡 batch 1、累积 1、有效总 batch 64、25 epochs。沿用当前每 epoch 保存权重和 FP32 验证 PDMS 的设置；训练成功后，由主节点自动测试最后的 `last.ckpt`，navtest 使用 FP32、64 proposals、10 步 Euler。评测使用本机 CPU 进程池。

默认训练目录：

```text
DrivoR/exp/ke/training_drivoR_action_dit_dinov3_vitl_navtrain_single_gt_long_pseudo_expert/<DRIVOR_RUN_ID>/
```

默认测试目录：

```text
DrivoR/exp/ke/eval_navtest_action_dit_dinov3_vitl_single_gt_long_pseudo_expert/<DRIVOR_RUN_ID>/
```

`pdms_summary.txt` 保存最终 navtest 结果。默认沿用基线的 DINOv3 和 Action-DiT 预训练初始化，启动一个新的对照实验。

## 只训练

```bash
cd /cpfs29jb/data/zhaozc/project/drive_future
export DRIVOR_RUN_ID=action_dit_dinov3_vitl_single_gt_long_pseudo_expert_seed2_run1
export DRIVOR_PRECISION=bf16-mixed
export DRIVOR_SCENE_QUERIES=0

bash DrivoR/scripts/training/run_drivor_action_dit_dinov3_vitl_single_gt_long_pseudo_expert_training.sh \
  auto_resume=false train_ckpt_path=null runtime.require_resume=false
```

两个入口都允许在末尾追加 Hydra 参数。新增文件复用现有 loss 和训练流程，没有修改正在运行的公共启动脚本。

## 本地验证

对普通/长 GT loss、预测及 Scorer 结果做基线对照，验证新增项精确等于 `0.5 × L_PE`；检查单次 128 条 DiT 输入、64 条 Scorer 输入、变长有效 expert 的平均，以及 pseudo-expert 梯度只从普通组输出回传。启动流程覆盖主节点、其他节点、配置查询和训练失败的情况。另通过原 vMSE 回归测试确认已有版本兼容。

```bash
PYTHONPATH=DrivoR:DrivoR/nuplan-devkit \
  /cpfs29jb/data/zhaozc/miniconda3/envs/drive_future/bin/python -m pytest -q \
  DrivoR/tests/test_action_dit_single_gt_long_pseudo_expert.py \
  DrivoR/tests/test_pseudo_expert_velocity.py
```

9 项测试通过；上述验证不代表新实验已经取得 NAVSIM 指标。
