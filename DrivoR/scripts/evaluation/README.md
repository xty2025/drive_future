# 已训练 Action-DiT 的 navtest 评测

新的 DINOv3-L Action/Register 联合去噪模型使用 `run_drivor_action_dit_dinov3_vitl_register_evaluation.sh <register|register_scorer|register_staged> <checkpoint>`，默认 FP32、每卡 batch1、32 个 CPU 评分进程。三个训练后自动测试命令见 [未来 register 说明](../../DRIVOR_ACTION_DIT_DINOV3_REGISTERS.md)。

其单 human GT 对照使用参数 `register_scorer_single_gt`，测试模型结构仍为 register Scorer、默认 FP32。训练与自动测试命令见 [单 GT 对照](../../DRIVOR_ACTION_DIT_REGISTER_SINGLE_GT.md)。

pseudo-expert v MSE + registers 两版使用同一测试脚本，第一个参数分别为 `pseudo_expert_vmse_register` / `pseudo_expert_vmse_register_scorer`，第二个参数为完整 checkpoint，默认 FP32。测试只恢复模型，无需 pseudo-expert 或未来帧数据；完整命令见 [pseudo-expert + registers](../../DRIVOR_ACTION_DIT_PSEUDO_EXPERT_VMSE_REGISTERS.md)。

DrivoR + CLOVER pseudo-expert coverage 使用 [run_drivor_pseudo_expert_evaluation.sh](run_drivor_pseudo_expert_evaluation.sh) 加完整 checkpoint 路径，内部使用原 `drivoR` 模型、FP32（`32-true`）、每卡 batch 1 和 32 个 CPU 评分进程；训练后自动测试同样默认 FP32。单独测试及训练后自动测试命令见 [pseudo-expert 说明](../../DRIVOR_PSEUDO_EXPERT.md#4-pai-dlc-启动命令)。

DINOv3 和 V-JEPA 2 使用各自的 agent 配置。对应的“训练成功后自动测试”命令见 [DINOv3 说明](../../DRIVOR_ACTION_DIT_DINOV3.md#6-训练成功后自动测试-navtest)和 [V-JEPA 2 说明](../../DRIVOR_ACTION_DIT_VJEPA2.md#训练成功后自动测试-navtest)。下面默认脚本对应原有 DINOv2 checkpoint。

V-JEPA 2.1 使用 `agent=drivoR_action_dit_vjepa2_1_vitl`，完整训练后测试命令见 [V-JEPA 2.1 说明](../../DRIVOR_ACTION_DIT_VJEPA2_1.md#训练成功后自动测试-navtest)。

原版 DrivoR 的 Bank / V-JEPA / 独立未来分支实验使用 [run_drivor_world_evaluation.sh](run_drivor_world_evaluation.sh)，第一个参数选择 `bank`、`vjepa2_vitl`、`vjepa2_1_vitl`、`vjepa2_world` 或 `vjepa2_world_multigt`，第二个参数是完整训练checkpoint。配置与训练后自动测试见[四组实验说明](../../DRIVOR_WORLD4SCORER_VARIANTS.md)。

入口：[run_drivor_action_dit_trained_nav1_evaluation.sh](run_drivor_action_dit_trained_nav1_evaluation.sh)。默认评测指定的完整训练权重：

```text
DrivoR/exp/ke/training_drivoR_action_dit_navtrain/multinode/lightning_logs/version_3/checkpoints/epoch=24-step=33250.ckpt
```

脚本使用与该 checkpoint 匹配的 `agent=drivoR_action_dit`：当前单前视原生 DINOv2 patches、672×1148 图像、64 条候选、10 步 Euler、原 DrivoR Scorer。完整 checkpoint 严格恢复全部网络参数，评测无需先加载 Action-DiT 初始化文件。

## DLC 单节点启动

配置 **1 个节点、8 张 GPU**，启动命令填写：

```bash
cd /cpfs29jb/data/zhaozc/project/drive_future

NPROC_PER_NODE=8 \
DRIVOR_EXPERIMENT=eval_navtest_drivor_action_dit_navtrain_v3_epoch24 \
DRIVOR_RUN_ID=action_dit_v3_navtest_run1 \
bash DrivoR/scripts/evaluation/run_drivor_action_dit_trained_nav1_evaluation.sh
```

脚本自动激活 `/cpfs29jb/data/zhaozc/miniconda3/envs/drive_future`，内部使用 `torchrun --standalone --nnodes=1`。推理默认 BF16、每卡 batch 1；收齐预测后，rank 0 使用 32 个 CPU 进程计算 PDM 分数，不启动 Ray。

结果目录为：

```text
DrivoR/exp/ke/eval_navtest_drivor_action_dit_navtrain_v3_epoch24/action_dit_v3_navtest_run1/
```

逐场景结果在带时间戳的 CSV 中，汇总成绩在 `pdms_summary.txt`。更换 `DRIVOR_RUN_ID` 可区分重复评测。

## 数据与可覆盖参数

测试读取 `data/navsim/navsim_logs/test`、`data/navsim/sensor_blobs/test` 和 `data/metric_cache_navtest`。相机读取沿用现有本地/OSS 流程。

| 环境变量 | 默认值 / 用途 |
|---|---|
| `NPROC_PER_NODE` | 可见 GPU 数；DLC 命令中显式设为 8 |
| `DRIVOR_CHECKPOINT` | 上述 version_3 / epoch 24 checkpoint；可换为同一模型结构的完整权重 |
| `DRIVOR_EVAL_PRECISION` | `bf16-mixed` |
| `DRIVOR_EVAL_BATCH_SIZE` | `1`，每卡推理 batch |
| `DRIVOR_NUM_WORKERS` | `4`，每卡数据加载 workers |
| `DRIVOR_SCORE_WORKERS` | `32`，主进程评分池的 CPU 进程数 |
| `DRIVOR_EVAL_WORKER` | `single_machine_thread_pool`，启用其 `use_process_pool=true` |
| `NAVSIM_TEST_METRIC_CACHE_PATH` | `drive_future/data/metric_cache_navtest` |

命令末尾可追加 Hydra 参数；例如 `train_test_split.scene_filter.max_scenes=64` 限制评测规模。默认 DLC 命令不限制场景数，按完整 navtest 配置评测。

仅查看模型配置：

```bash
bash DrivoR/scripts/evaluation/run_drivor_action_dit_trained_nav1_evaluation.sh \
  --cfg job --package agent --resolve
```

## 本地验证

已核对权重的 epoch=24、global_step=33250，完成脚本语法和配置检查，并用该 checkpoint 跑通真实测试数据的双 A100/BF16 推理与 CPU 进程池评分。Smoke 场景上限为 2，实际与 metric cache 交集为 1 个场景，成功评分 1 个；该检查不代表全量 navtest 成绩。

验证记录：[validation.json](../../../artifacts/drivor_action_dit_v3_evaluation_20261015/validation.json)。
