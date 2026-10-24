# 普通 Transformer：GT / Future Register 对照

这些实验统一使用当前单前视 DINOv3-L 和普通 Transformer，比较长 GT、未来 register 辅助监督，以及新的 diverse8 GT。新增的 **T10 `single_gt`** 从 T05 去掉长 GT，只保留原始 human GT，作为单 human、无未来 register 的基线。T10 已通过本地验证并提交 DLC `dlc1q4uo0bszf0gl`，见[部署记录](../artifacts/dlc_submissions/query_single_gt_20261022_221845/README.md)。

| 配置后缀 | 普通 human | 长 GT | diverse8 | 未来 register | 总 loss |
|---|---|---|---|---|---|
| `single_gt_long` | L1×1 | L1×1 | — | 无 | `L_human + L_long + L_score` |
| `single_gt_long_register` | L1×1 | L1×1 | — | 仅普通 human 匹配者对齐，L1×1 | `L_human + L_long + L_register + L_score` |
| `human_diverse8_l1` | L1×1 | — | coverage L1×0.5 | 无 | `L_human + 0.5 L_diverse8 + L_score` |
| **`single_gt`（T10）** | **L1×1** | **—** | **—** | **无** | **`L_human + L_score`** |

## 共用的网络与输入

- 当前单前视图，输入 `[B,1,3,672,1152]`。
- DINOv3 ViT-L/16 加载已有预训练权重，冻结基座、训练 Q/V LoRA rank 32。
- 当前图像的原生 3024 个 patch 经投影得到 `[B,3024,256]`，作为轨迹预测器及 Scorer 的视觉 K/V；不加额外 scene queries。
- 64 个可学习 Action queries，加上当前 11 维 Ego 经 Linear 得到的 256 维 embedding。
- 4 层普通 DrivoR blocks：Self-Attention → Cross-Attention → FFN，宽度 256、FFN 1024；最后一个 MLP 输出 `[B,64,8,3]` 物理坐标轨迹。
- 原轨迹 Scorer：24→1024→256 的轨迹投影、4 层评分 Transformer、六个评分头。本表四组都不向 Scorer 输入预测 registers；轨迹输入继续 detach。
- 不使用 Action-DiT、扩散时间、加噪轨迹或 vMSE。训练与推理均为一次普通 Transformer 前向。

本表四个配置先按相同顺序初始化所有共用模块，再构造可选 Future 模块，因此**相同 seed 下，共享网络参数初值完全一致**。这已通过参数逐项比较与同输入推理检查。旧 query-register 配置的初始化和运行逻辑不改动。

## 一组 64 个 proposals，分别匹配 human 与长 GT

四组都直接在米／弧度坐标上计算 L1：

```text
C(P_m, GT) = mean_8_times (|Δx| + |Δy| + |Δheading|)
L_human   = mean_batch min_m C(P_m, human_GT)
L_long    = mean_batch min_m C(P_m, long_GT)
```

第一、二组对**同一组 64 个输出**分别匹配普通 human GT 和长 GT，两个 winner 可以相同，也可以不同。两项 loss 直接相加，不除以 2。模型输入不包含 GT，因此不会为长 GT 再生成一组候选。

长 GT 沿用原 `DrivoRTargetBuilder`：请求额外 2 个未来位姿，经现有样条重采样生成 8 个监督点。其时间对应关系与普通 `+0.5,...,+4s` GT 不同，因此第二组的长 GT 只提供轨迹监督。

## 第二组：register 只配对普通 human winner

启用后每个候选增加 32 个可学习 Future queries（8 个未来时刻×4 个 register），另加时刻和槽位 embedding。逻辑序列为 64 个 Action＋2048 个 Future，共 2112 tokens。

读关系保持原设计：Action 读取全部 Action；每组 Future 读取本候选 Action 和本组 Future；Action 不读取 Future。两路的 Cross-Attention 都使用同一份当前图像 patch 特征。

```text
m_h = argmin_m C(P_m, human_GT)
L_register = L1(predicted_registers[:, m_h], EMA_teacher(human_future_RGB))
```

未来目标来自真实 human 未来前视图的 `+0.5,+1.0,...,+4.0s`，每帧 4×1024 维原生 DINOv3 registers。Teacher 为当前 DINOv3 有效 LoRA 权重的 EMA（0.999），在真实 optimizer step 后更新；teacher 本身不接收梯度。先平均 register / 通道，再平均有效未来帧及 batch。长 GT winner 不参与 register 对齐。

第一、三组的 `future_register.enabled=false` 会**不构造** Future queries、register head、EMA teacher，并且不读取未来 RGB。其 decoder 序列只有 64 个 Action tokens。DINOv3 编码器内部原生 register tokens 保持原预训练结构；它们与新增的未来预测分支是两回事。

## 第三组：human＋新的 diverse8

数据路径沿用原 `human_diverse8_l1` 实验：

```text
/cpfs29jb/data/zhaozc/project/Cosmos3forMultiGT/runtime/multigt_raw64/outputs/navtrain_diverse8_dense_0p1s_f64
```

读取各场景 `gt_trajectories.npy`、有效 mask、padding 标记及 `complete.json`。每条有效 diverse8 GT 独立选择最近 proposal，允许复用同一候选，按实际有效 GT 数平均后乘 0.5。普通 human 始终单独保留权重 1；不将所有 GT 混在一起平均。

这组不读取原 top4 multi-GT、CLOVER 专家包或未来视频，也不使用长 GT。缺失或无有效 diverse8 时，辅助项为 0，human L1 和 Scorer 正常计算。

## T10：仅原始 human GT，无未来 register

[T10 配置](navsim/planning/script/config/common/agent/drivoR_query_dinov3_vitl_single_gt.yaml) 直接继承 T05，只将 `long_trajectory_additional_poses` 从 2 改为 0；启动脚本也显式传入 0，覆盖底层原版训练入口的长 GT 默认值。

每场景仍一次输出 64 条 `[8,3]` 轨迹。唯一 GT 是 NAVSIM human log 的未来 0.5～4 秒、2 Hz、8 点轨迹；按上面的物理 L1 选择一个最近 proposal，只有该候选获得轨迹回归监督。全部 64 条仍进入原 Scorer，在线计算评分标签并训练六个评分头。总损失为 `L_human + L_score`，两项权重均为 1。

本配置不构造 Future queries、register head 或 EMA teacher，不读取长 GT、multi-GT、diverse8、pseudo-expert 或未来视频。DINOv3 内部的原生 register 保持其预训练结构；它们不等于未来 register 预测分支。

| 对照 | 需要比较的因素 |
|---|---|
| T10 → T05 | 加入长 GT L1 的效果；两者网络和相同 seed 下的参数初值一致。 |
| T10 → T01 | 单 human 条件下增加未来 register L1 辅助对齐，保持原 Scorer。 |
| T10 → T03 | 单 human 条件下同时增加未来预测和 register 融合 Scorer 的整体效果。 |
| T01 → T03 | 已预测 register 后，是否将其输入 Scorer。 |
| T10 → T07 | 单 human 条件下增加 `0.5×diverse8 L1 coverage`。 |

历史 T01/T03 使用较早的模块构造顺序，同一个 seed 不保证与 T10 的共享参数初值逐项相同；T10/T05 则是只改长 GT 开关的直接对照。

## DLC 启动命令：训练后自动 FP32 测试

每组分别配置 **8 节点×8 GPU**，平台分发启动命令。默认每卡 batch 1、累积 1、总 batch 64、BF16 训练、seed 2、25 epochs、峰值 LR 2e-4。沿用每 epoch 保存 checkpoint 和 FP32 val PDMS；训完后自动 FP32 navtest。

**新增 T10，仅原始 human GT，无 register：**

```bash
cd /cpfs29jb/data/zhaozc/project/drive_future
export DRIVOR_RUN_ID=query_dinov3_vitl_single_gt_seed2_train_eval_run1
bash DrivoR/scripts/training/run_drivor_query_single_gt_train_eval.sh
```

第一组，human＋长 GT，无 register：

```bash
cd /cpfs29jb/data/zhaozc/project/drive_future
export DRIVOR_RUN_ID=query_dinov3_vitl_single_gt_long_seed2_train_eval_run1
bash DrivoR/scripts/training/run_drivor_query_single_gt_long_train_eval.sh
```

第二组，human＋长 GT，加 register 对齐：

```bash
cd /cpfs29jb/data/zhaozc/project/drive_future
export DRIVOR_RUN_ID=query_dinov3_vitl_single_gt_long_register_seed2_train_eval_run1
bash DrivoR/scripts/training/run_drivor_query_single_gt_long_register_train_eval.sh
```

第三组，human＋diverse8，无 register：

```bash
cd /cpfs29jb/data/zhaozc/project/drive_future
export DRIVOR_RUN_ID=query_dinov3_vitl_human_diverse8_l1_seed2_train_eval_run1
bash DrivoR/scripts/training/run_drivor_query_human_diverse8_l1_train_eval.sh
```

仅训练时使用同名 `*_training.sh`。四组分别使用 `drivoR_query_dinov3_vitl_<配置后缀>` agent 配置。

输出目录：

```text
DrivoR/exp/ke/training_drivoR_query_dinov3_vitl_navtrain_<配置后缀>/<RunID>/
DrivoR/exp/ke/eval_navtest_query_dinov3_vitl_<配置后缀>/<RunID>/
```

权重在 `lightning_logs/version_*/checkpoints/`。单独测试时：

```bash
bash DrivoR/scripts/evaluation/run_drivor_query_gt_evaluation.sh \
  single_gt_long /完整路径/checkpoint.ckpt
```

将第一个参数换成对应配置后缀即可。测试会保留相应的 Future 模块开关，关闭长 GT target，且不加载 diverse8 或未来 RGB。Teacher 不在推理时执行。

## 实现

- [gt_variants.py](navsim/agents/drivoR_query_register/gt_variants.py)：共享网络、可选未来分支与 human / 长 GT / diverse8 监督。
- [公共配置](navsim/planning/script/config/common/agent/drivoR_query_dinov3_vitl.yaml)。
- [第一组配置](navsim/planning/script/config/common/agent/drivoR_query_dinov3_vitl_single_gt_long.yaml)。
- [第二组配置](navsim/planning/script/config/common/agent/drivoR_query_dinov3_vitl_single_gt_long_register.yaml)。
- [第三组配置](navsim/planning/script/config/common/agent/drivoR_query_dinov3_vitl_human_diverse8_l1.yaml)。
- [T10 仅 human 配置](navsim/planning/script/config/common/agent/drivoR_query_dinov3_vitl_single_gt.yaml)。

## 本地验证

- 42 项检查通过：共享参数初值一致、关闭 Future 时无预测模块/teacher/RGB 读取、普通和长 GT 共用候选、仅普通 human winner 对齐 future、diverse8 权重和 mask、Scorer 梯度截断、EMA 与 optimizer、checkpoint 加载及三组自动 FP32 测试入口。
- 三份训练配置解析通过，9 个新 shell 入口通过语法检查。
- 三组均在本地 GPU 0、1 跑通真实数据与真实 NAVSIM oracle：每卡 1 个训练 step＋1 个验证 batch，BF16 训练、FP32 验证，loss 有限。第二组读到一套有效 human 未来帧；第三组每场景读到 8 条有效 diverse8 GT。

以上是原三组的验证，记录见 [validation.json](../artifacts/query_gt_controls_20261021/validation.json)。Smoke 用于确认数据、梯度及分布式流程，不代表完整训练后的 NAVSIM 成绩。

**T10 增补验证：**35 项模型与启动流程检查通过，覆盖四组共享参数初值、human-only 匹配梯度、Scorer 截断、无未来数据读取以及自动 FP32 测试。两份实际启动入口解析出的 agent 配置，唯一差异是 T10 的 `long_trajectory_additional_poses=0` 与 T05 的 `2`。本地 GPU 0、1 各跑通 1 个真实数据训练 step 和 1 个验证 batch，BF16 训练、FP32 验证；轨迹项只有 human L1，未来 loss 为 0。记录见 [T10 验证](../artifacts/query_single_gt_20261022/validation.json)。T10 已提交 8 机×8 卡 DLC，navtest 成绩待完整训练与测试后补入总表。
