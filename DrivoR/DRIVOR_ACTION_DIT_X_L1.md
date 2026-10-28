# Action-DiT：human / long GT 的直接 X-prediction L1

这两组新实验基于现有 `human_diverse8_l1` 和 `single_gt_long`，将普通 human GT 和长 human GT 的轨迹监督改为**物理坐标 L1**。旧 vMSE 配置和启动入口继续保留。

| 新配置后缀 | 总 loss |
|---|---|
| `human_xl1_diverse8_l1` | `human X-L1 + 0.5 × diverse8 L1 coverage + score loss` |
| `single_gt_long_xl1` | `human X-L1 + long GT X-L1 + score loss` |

普通 human 与长 GT 两项各自权重为 1，第二组直接相加，不再除以 2。第一组 diverse8 监督仍使用当前生成的有效轨迹，最多 8 条；其来源、mask、匹配方式和 0.5 权重不变。

## 直接 X L1 的计算

DiT 仍输出归一化空间中的干净轨迹 `x_hat`。计算 loss 时，先转换为自车物理坐标：

```text
P = denormalize(x_hat)
  = (x_hat + 1) * (66.74, 42.0, 3.53) / 2 - (1.57, 19.68, 1.67)

cost[b,m] = mean_t (|P_x - GT_x| + |P_y - GT_y| + |P_heading - GT_heading|)
L_human  = mean_b min_m cost[b,m]
```

`x,y` 使用米，`heading` 使用弧度；沿用现有 DrivoR / coverage 的“坐标维求和、8 个时刻平均”定义。预测反归一化后不裁剪。只对每个场景匹配到的最佳 proposal 回传该 GT 的轨迹梯度。

长 GT 使用相同公式，独立从其 64 个候选中选择最佳者。新 loss **不调用 X→v 转换，不除以 sigma，也不乘原 vMSE 的时间权重**。时间仍作为模型输入和加噪条件；固定预测误差的 L1 不因 t 很小而放大。

diverse8 的 coverage 保持原样：每条有效 GT 从普通 human 组的 64 个 proposal 中独立选择最近者，允许重复选择同一 proposal；先平均有效 GT，再平均 batch。整组 coverage 乘 0.5。缺失或没有有效 diverse8 的场景，该辅助项为 0，human 监督继续有效。

## 保持的训练流程

- 视觉：当前单前视 `672×1152`；DINOv3 ViT-L/16，冻结基座、Q/V LoRA rank 32；3024 个原生 patch 作为视觉 K/V。
- 轨迹网络：沿用当前 256 维 Action-DiT 及其预训练初始化；没有增加网络参数。
- 加噪：普通 human GT 仍先归一化，再与 64 份独立高斯噪声插值；每个场景采样一个 t，候选共享该 t。
- `human_xl1_diverse8_l1`：一次 DiT 前向产生 64 条轨迹；同组输出用于 human L1、diverse8 L1 和 Scorer。
- `single_gt_long_xl1`：普通／长 GT 仍复用同组噪声和 t，64＋64 合批为一次 DiT 前向；两组分别算 best-of-64 L1，Scorer 继续评分普通组的 64 条。
- Scorer：保留原网络、原 NAVSIM 标签和评分 loss，输入预测轨迹继续 detach。
- 推理：保留原 10 步 Euler。推理中的积分仍根据 X prediction 构造更新量；此次改动针对训练的轨迹 loss。

实现使用新的模型子类覆盖主轨迹监督。`_flow_loss` 这个内部接口键继续复用，其值在这两版中是上述 L1；新增日志 `human_x_l1_loss` 和 `long_x_l1_loss` 可直接核对。主训练仍使用 `trajectory_loss` / `final_score_loss` / `loss` 汇总。

新旧模型的 `state_dict` 参数名和形状一致，完整 checkpoint 可以由普通 `drivoR_action_dit_dinov3_vitl` 测试 agent 严格加载。测试无需读取 diverse8 或长 GT。

## DLC：训练后自动 FP32 测试

每个实验分别配置 **8 节点×8 GPU**，平台分发一次填写的启动命令。默认每卡 batch 1、累积 1、全局 batch 64、BF16 训练、25 epochs、seed 2、峰值 LR 2e-4。每 epoch 保留 checkpoint 并进行 FP32 val PDMS；最终自动 FP32 navtest。

Human L1＋diverse8 L1：

```bash
cd /cpfs29jb/data/zhaozc/project/drive_future
export DRIVOR_RUN_ID=action_dit_dinov3_vitl_human_xl1_diverse8_l1_seed2_train_eval_run1
export DRIVOR_PRECISION=bf16-mixed
bash DrivoR/scripts/training/run_drivor_action_dit_dinov3_vitl_human_xl1_diverse8_l1_train_eval.sh
```

Human L1＋长 GT L1：

```bash
cd /cpfs29jb/data/zhaozc/project/drive_future
export DRIVOR_RUN_ID=action_dit_dinov3_vitl_single_gt_long_xl1_seed2_train_eval_run1
export DRIVOR_PRECISION=bf16-mixed
bash DrivoR/scripts/training/run_drivor_action_dit_dinov3_vitl_single_gt_long_xl1_train_eval.sh
```

仅训练时，把入口换成对应的 `*_training.sh`。训练后测试入口默认新训练；需要从 checkpoint 恢复时可明确传入 `train_ckpt_path=/完整路径/last.ckpt`。

输出路径：

```text
DrivoR/exp/ke/training_drivoR_action_dit_dinov3_vitl_navtrain_<新配置后缀>/<RunID>/
DrivoR/exp/ke/eval_navtest_action_dit_dinov3_vitl_<新配置后缀>/<RunID>/
```

权重位于训练目录的 `lightning_logs/version_*/checkpoints/`，最新权重为 `last.ckpt`。两个新入口复用已有训练/测试流程，没有修改正在使用的旧公共 Bash 文件。

## 文件

- [x_l1.py](navsim/agents/drivoR_action_dit/x_l1.py)：物理坐标 L1、两种新 agent。
- [human_xl1_diverse8_l1 配置](navsim/planning/script/config/common/agent/drivoR_action_dit_dinov3_vitl_human_xl1_diverse8_l1.yaml)。
- [single_gt_long_xl1 配置](navsim/planning/script/config/common/agent/drivoR_action_dit_dinov3_vitl_single_gt_long_xl1.yaml)。

## 验证

- 24 项测试通过，覆盖物理坐标匹配、仅最佳候选梯度、极小 t 下不调用 v/时间权重、普通/长 GT 独立匹配、一次前向、旧 checkpoint 与推理兼容，以及训练后 FP32 测试入口。
- 两份启动配置解析通过；配置检查使用简化 scene filter，避免展开完整数据 token 列表。四个新 shell 入口通过语法检查。
- 两组均在本地 GPU 0、1 上完成真实数据、真实 NAVSIM oracle 的 smoke：每卡 1 个训练 step＋1 个验证 batch，BF16 训练、FP32 验证，loss 均有限。第一组实际读到每场景 8 条有效 diverse8；第二组日志包含普通和长 GT 两项 X L1。

记录见 [validation.json](../artifacts/action_dit_x_l1_20261021/validation.json)。本地 smoke 只用于确认流程，不代表完整训练后的 NAVSIM 成绩。
