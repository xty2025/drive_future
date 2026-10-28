# DINOv3-L Action/Register Scorer：单 human GT 对照

对应多 GT 基线是 `training_drivoR_action_dit_dinov3_vitl_navtrain_register_scorer/dinov3_register_scorer_seed2_run1`，其评测目录为 `eval_navtest_action_dit_dinov3_vitl_register_scorer/dinov3_register_scorer_seed2_run1`。已核对训练配置：原版 `multi_gt.enabled=true`、最多4条GT、register Scorer 为 `project` 融合，未启用 pseudo-expert。

新配置：[`drivoR_action_dit_dinov3_vitl_register_scorer_single_gt.yaml`](navsim/planning/script/config/common/agent/drivoR_action_dit_dinov3_vitl_register_scorer_single_gt.yaml)。

新训练脚本：[`run_drivor_action_dit_dinov3_vitl_register_scorer_single_gt_training.sh`](scripts/training/run_drivor_action_dit_dinov3_vitl_register_scorer_single_gt_training.sh)。

## 监督方式

| 项目 | 原多 GT register Scorer | 新单 GT register Scorer |
|---|---|---|
| 轨迹 GT | 最多4条有效 GT，各自选最近 proposal | 原始 human GT 选最近 proposal |
| 未来目标 | 各 GT 对应的 Cosmos 未来帧 | 原始 human 对应的真实未来帧 |
| Register 对齐索引 | 每个 GT 匹配到的 proposal | human GT 匹配到的同一 proposal |
| 候选输出 | 64条轨迹，各配32个 registers | 相同 |
| Scorer | 压缩 registers，与轨迹特征 concat→MLP | 相同 |

单 GT 表示每场景只有一个监督目标，仍生成并评分64条候选。普通 human GT 不再搭配长 GT、其他 multi-GT 或 pseudo-expert。

```text
i* = argmin_i vMSE(Action_i, human_GT)

L_total = vMSE(Action_i*, human_GT)
        + vMSE(Register_i*, EMA(真实human未来8帧))
        + L_score(全部64候选)
```

未来时间仍为 `+0.5,+1.0,...,+4.0s`，每帧4个原生 DINOv3 registers。全部64份 Action/Register 加噪基准仍分别来自 human 轨迹和真实未来特征；human 的8张未来图像只经过 teacher 编码一次，结果同时用于 register 加噪基准及 loss 目标。

DINOv3 ViT-L/16、当前单前视 `672×1152`、3024个原生视觉 patches、Q/V LoRA rank32、Action256预训练初始化、EMA=0.999、一次联合训练前向、sigma下界0.1、10步Euler推理和评分 detach 路径沿用基线。未来 teacher 输入仍为 `384×384`。

代码现在让 `multi_gt.enabled=false` 在 register 模型中真正选择 human GT；数据使用独立的 `HumanFutureRegisterTargetBuilder`，只读取 NAVSIM 日志中的真实未来前视图，不要求 multi-GT 文件或 Cosmos 帧缓存。原多 GT 训练配置继续使用原目标读取分支。

## DLC：训练后自动 FP32 测试

DLC 配置 **8个节点、每节点8张GPU**。总batch64，每卡batch1，梯度累积1，25epochs，seed2，LR=2e-4；训练BF16，测试FP32。

```bash
cd /cpfs29jb/data/zhaozc/project/drive_future

DRIVOR_TRAIN_EXPERIMENT=training_drivoR_action_dit_dinov3_vitl_navtrain_register_scorer_single_gt \
DRIVOR_RUN_ID=dinov3_register_scorer_single_gt_seed2_train_eval_run1 \
DRIVOR_SCENE_QUERIES=0 \
DRIVOR_PRECISION=bf16-mixed DRIVOR_EVAL_PRECISION=32-true \
bash DrivoR/scripts/training/run_drivor_action_dit_dinov3_vitl_register_train_eval.sh \
  register_scorer_single_gt
```

该入口从预训练初始化开始新实验；训练完成后，由节点0使用本次 `last.ckpt` 自动运行 FP32 navtest。测试目录默认 `DrivoR/exp/ke/eval_navtest_action_dit_dinov3_vitl_register_scorer_single_gt/<run_id>`。

如果只训练，使用：

```bash
cd /cpfs29jb/data/zhaozc/project/drive_future

DRIVOR_EXPERIMENT=training_drivoR_action_dit_dinov3_vitl_navtrain_register_scorer_single_gt \
DRIVOR_RUN_ID=dinov3_register_scorer_single_gt_seed2_run1 \
DRIVOR_SCENE_QUERIES=0 DRIVOR_PRECISION=bf16-mixed \
bash DrivoR/scripts/training/run_drivor_action_dit_dinov3_vitl_register_scorer_single_gt_training.sh \
  auto_resume=false train_ckpt_path=null runtime.require_resume=false runtime.console_log_interval=25
```

已有单 GT 权重单独测试时，在单节点 DLC 任务运行：

```bash
NPROC_PER_NODE=8 DRIVOR_EVAL_PRECISION=32-true \
bash DrivoR/scripts/evaluation/run_drivor_action_dit_dinov3_vitl_register_evaluation.sh \
  register_scorer_single_gt /绝对路径/last.ckpt
```

单 GT 与原多 GT 的模型参数结构完全一致，测试使用原 `register_scorer` 网络严格恢复 checkpoint；推理不需要 GT 轨迹或未来图像。

## 验证

45 项模型、目标读取、旧配置兼容性和训练后测试衔接检查，以及5项新增DLC启动参数检查通过。已完成两张A100真实数据smoke：每卡2个训练step和1个验证batch；刻意将multi-GT和Cosmos路径设为不存在的目录，仍正常完成。实际日志中的GT数量和未来GT数量均为1，真实未来帧覆盖率为1，EMA更新正常，报告的loss有限。

记录：[validation.json](../artifacts/action_dit_register_scorer_single_gt_20261019/validation.json)。仅验证运行链路，没有进行全量训练或全量navtest。
