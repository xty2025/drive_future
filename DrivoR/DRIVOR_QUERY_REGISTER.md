# 普通 Transformer：可学习 Action/Future queries

本版本把 Action-DiT 换成随机初始化的普通 DrivoR Transformer blocks。按确认的方案，每个场景有 **64 个候选**，每个候选有 **1 个 Action query＋32 个独立的 Future queries**。两组 query 都是模型参数，初始化一次后由训练更新。

提供两种监督和两种 Scorer，共四组配置。单 GT 使用真实 human 轨迹和未来帧；多 GT 使用现有 top4 有效 GT 及其对应未来视频。两种 Scorer 分别只接收轨迹，或接收轨迹与预测 registers。相同 Scorer 下，单/多 GT 网络参数形状一致。

| 配置后缀 | 轨迹与未来监督 | Scorer 的候选输入 |
|---|---|---|
| `single_gt` | human GT | 预测轨迹 |
| `multigt` | top4 有效 GT，各自最近匹配、等权平均 | 预测轨迹 |
| `scorer_single_gt` | human GT | 预测轨迹＋预测 registers |
| `scorer_multigt` | top4 有效 GT，各自最近匹配、等权平均 | 预测轨迹＋预测 registers |

完整 agent 配置名为 `drivoR_query_register_dinov3_vitl_<配置后缀>`。这些实验使用新的模型类；原 Action-DiT 版本及其启动入口保留。

## 输入和初始化

| 项目 | 默认形状/设置 |
|---|---|
| 当前图像 | 单前视、当前帧，`[B,1,3,672,1152]` |
| 当前视觉编码器 | DINOv3 ViT-L/16，加载现有预训练权重；基础权重冻结，Q/V LoRA rank 32 |
| Cross-Attention K/V | 原生 patch 经投影得到 `[B,3024,256]`，无额外 scene queries |
| Ego | 当前 `pose3, velocity2, acceleration2, command4`，11 维经 Linear 得到 256 维 |
| Action 参数 | `nn.Embedding(64,256)`，与原 DrivoR 一样使用可学习 embedding |
| Future 参数 | 独立的 `nn.Parameter[64,32,256]`，截断正态初始化，std=0.02 |
| Future 槽位 | 32＝未来 8 个时刻 × 每帧 4 个 register；另加可学习的时刻和 register 槽位编码 |

```text
A[b,m]     = ActionEmbedding[m] + EgoEmbedding[b]
R[b,m,t,s] = FutureEmbedding[m,4*t+s]
             + FutureTimeEmbedding[t] + RegisterSlotEmbedding[s] + EgoEmbedding[b]
```

Future 参数在不同候选间独立，在不同样本间共享。它们借鉴 V-JEPA2 的预测 query 用途；这里按要求随机初始化，V-JEPA2 官方默认的共享 mask embedding 则为零初始化。

图像特征中不注入这 32 个 Future queries，它们属于下游预测器。模型没有轨迹/未来特征加噪、扩散时间采样、AdaLN 时间调制或 Euler 循环，也不构造或加载 Action-DiT 权重。

## 四层普通 Transformer

默认 4 层、宽度 256、1 个 attention head、FFN 宽度 1024。复用原 DrivoR `Block` 的权重结构、LayerNorm、GELU、残差、projection dropout=0.1 和 DropPath=0.2：

```text
Pre-LN Self-Attention → residual
Pre-LN Cross-Attention(current DINO patches) → residual
Pre-LN FFN → residual
```

物理序列布局为 `[A0...A63 | R0的32个 | ... | R63的32个]`，长度 **2112**。默认 self-attention 读关系：

- 每个 Action 读取全部 64 个 Action，沿用 DrivoR 的候选间交互。
- `R_m` 读取本候选的 `A_m` 和本组 32 个 Future queries。
- Action 不读取 Future queries；不同候选的 Future queries 不直接互读。
- 两类 query 的 Cross-Attention 都读取当前前视图的 patch 特征。

保留可选 `agent.config.query_register.attention_mode=bidirectional`，允许 Action 额外读取本候选的 Future queries。默认使用已有 register 版本的单向约束。

每次训练和推理都只进行一次上述四层 decoder 前向。训练仍有普通 Dropout/图像增强的随机性；推理时不重新采样 query 或噪声。

## 输出

| 分支 | 预测头 | 输出 |
|---|---|---|
| Action | 原 DrivoR 形式的三层 MLP：256→1024→1024→24，中间 LayerNorm＋ReLU | `[B,64,8,3]` |
| Future | LayerNorm(256)＋Linear(256,1024) | `[B,64,8,4,1024]` |
| Scorer | 原 DrivoR 轨迹投影、4 层评分 decoder、6 个 score heads | 64 个候选的评分，选最高者作为最终轨迹 |

Action 输出直接是自车坐标系下的 `(x,y,heading)`，单位为米/弧度，使用原 DrivoR 的 L1 误差，不再使用 Action-DiT 的归一化或 X→v loss。

`single_gt` / `multigt` 的 Scorer 只使用预测轨迹；`scorer_single_gt` / `scorer_multigt` 启用 register 融合：先将预测的 `32×1024` registers 压缩到 256 维，与轨迹投影的 256 维特征 concat，再经 MLP 得到评分 token。两种模式下送入 Scorer 的轨迹均 detach；融合版的预测 registers 也 detach，评分 loss 不经过这两路回传生成器。视觉 K/V 与 Ego 保持原 DrivoR 的梯度路径。

```text
轨迹评分：detach(trajectory)[24] → Linear(24,1024) → ReLU → Linear(1024,256)
融合评分：
  detach(registers)[32×1024] → Linear(32768,256) → ReLU
  concat(trajectory_embedding[256], register_embedding[256])
    → Linear(512,1024) → ReLU → Linear(1024,256)
两者随后都进入原 4 层 TransformerDecoderScorer 与六头 Scorer。
```

## GT 匹配与 loss

```text
cost[b,k,m] = mean_t sum_(x,y,heading) |P[b,m,t] - GT[b,k,t]|
matched[b,k] = argmin_m cost[b,k,m]
L_action = mean_b mean_(valid k) cost[b,k,matched[b,k]]
```

单 GT 的 K=1。多 GT 使用各场景实际有效 GT 数，独立最近匹配，允许多个 GT 选择同一个 proposal；不使用匈牙利匹配，也不区分 human/其他 GT 的权重。

未来监督使用**同一个轨迹匹配索引**：`predicted_registers[b,matched[b,k]]` 对齐该 GT 对应未来视频的 EMA 特征。先在 register 槽位和特征维度平均，再平均有效时刻、具有未来帧的 GT，最后平均 batch。

```text
L_future = mean_b mean_(GTs with valid frames)
             mean_(valid future times) mean_(4 registers, 1024 channels)
             |PredictedRegister[b,matched[b,k]] - EMATarget[b,k]|

L_total = 1.0 * L_action + 1.0 * L_future + 1.0 * L_score
```

未来默认直接 L1 对齐，也支持 `agent.config.future_register.loss_type=mse`。不存在 v 转换或小 t 放大。没有有效未来帧时，该项贡献 0，轨迹/评分监督正常计算。

Scorer 仍对全部 64 个候选计算原 NAVSIM oracle 标签与原六头评分 loss，标签形状为 `[B,64,7]`，不是 candidate×GT。

## EMA 与未来帧

沿用 `EmaDinoRegisters`：teacher 复制当前 DINOv3，并对有效 Q/V 权重 `W_base + B@A` 做 EMA（decay=0.999）。仅在真实 optimizer step 后更新，teacher 全程冻结且处于 eval 模式；当前视觉编码器继续训练 LoRA。

- 单 GT：读取原日志中 `+0.5,+1.0,...,+4.0s` 的 8 张真实前视图。
- 多 GT：默认读取 `multi_traj_gt_data/navtrain_multigt_top4_dense_0p1s_f64_8gpu` 和 `DrivoR/exp/world4scorer_data/frames_cosmos`。Cosmos 第一帧为当前帧，使用索引 `5,10,...,40` 的未来帧，与 GT 槽位一一对应。
- 未来图像保持 384×384，teacher 每次处理 4 帧，输出每帧 4 个原生 register 特征。
- 多 GT 路径不再额外编码 human-noise anchor；若 GT 导出缺失而回退到 human GT，会使用真实 human 未来帧配对。
- 推理只使用当前图像和 Ego，不读取未来 RGB/GT、不运行 teacher。加载 checkpoint 中训练后的在线 DINOv3/LoRA 进行视觉编码。

未来 loss 回传到 Future queries、相关 Action queries、共享 decoder 和当前视觉 LoRA；不回传到 EMA teacher 或 Scorer。轨迹 MLP 不位于 future-loss 的直接计算路径中，仍由轨迹 loss 训练。

## 四个 DLC 入口：训练后自动 FP32 测试

在 DLC 配置 **8 节点 × 每节点 8 卡**；每卡 batch 1、累积 1、全局 batch 64。平台自动提供节点地址和 rank，只需填写一次启动命令。

单 GT、原 Scorer：

```bash
cd /cpfs29jb/data/zhaozc/project/drive_future
export DRIVOR_RUN_ID=query_register_dinov3_vitl_single_gt_seed2_train_eval_run1
export DRIVOR_PRECISION=bf16-mixed
bash DrivoR/scripts/training/run_drivor_query_register_single_gt_train_eval.sh
```

多 GT、原 Scorer：

```bash
cd /cpfs29jb/data/zhaozc/project/drive_future
export DRIVOR_RUN_ID=query_register_dinov3_vitl_multigt_seed2_train_eval_run1
export DRIVOR_PRECISION=bf16-mixed
bash DrivoR/scripts/training/run_drivor_query_register_multigt_train_eval.sh
```

单 GT、轨迹＋registers 融合 Scorer：

```bash
cd /cpfs29jb/data/zhaozc/project/drive_future
export DRIVOR_RUN_ID=query_register_dinov3_vitl_scorer_single_gt_seed2_train_eval_run1
export DRIVOR_PRECISION=bf16-mixed
bash DrivoR/scripts/training/run_drivor_query_register_scorer_single_gt_train_eval.sh
```

多 GT、轨迹＋registers 融合 Scorer：

```bash
cd /cpfs29jb/data/zhaozc/project/drive_future
export DRIVOR_RUN_ID=query_register_dinov3_vitl_scorer_multigt_seed2_train_eval_run1
export DRIVOR_PRECISION=bf16-mixed
bash DrivoR/scripts/training/run_drivor_query_register_scorer_multigt_train_eval.sh
```

上面每个代码块对应一个独立实验。融合版本的自动测试会选择相同的融合 agent 配置，保证权重与 Scorer 结构一致。

默认 25 epochs，LR=2e-4；保留当前每 epoch 保存权重、FP32 验证 PDMS 的流程。训练成功后仅主节点执行 FP32 navtest，评分使用本机 CPU 进程池。

只训练时，使用各入口对应的 `run_drivor_query_register_<配置后缀>_training.sh`，并按需要追加 `auto_resume=false train_ckpt_path=null runtime.require_resume=false`。

`DRIVOR_QUERY_LAYERS=6` 可统一更改训练和测试的 decoder 层数。也可在训练入口末尾追加 Hydra 参数；自动测试会转发相关网络结构参数（含 register-scorer 开关）。

训练目录：`DrivoR/exp/ke/training_drivoR_query_register_dinov3_vitl_navtrain_<配置后缀>/<RunID>/`。

测试目录：`DrivoR/exp/ke/eval_navtest_query_register_dinov3_vitl_<配置后缀>/<RunID>/`，最终指标见 `pdms_summary.txt`。

单独评测同架构 checkpoint：

```bash
cd /cpfs29jb/data/zhaozc/project/drive_future
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NPROC_PER_NODE=8
bash DrivoR/scripts/evaluation/run_drivor_query_register_evaluation.sh /完整路径/checkpoint.ckpt
```

若训练时使用融合 Scorer，单独评测时追加 `agent.config.future_register.scorer.enabled=true`，或明确传入对应的 `agent=drivoR_query_register_dinov3_vitl_scorer_single_gt` / `agent=drivoR_query_register_dinov3_vitl_scorer_multigt`。评测入口统一关闭多 GT target 读取，不需要未来帧或 GT registers。

checkpoint 应来自本 query-register 架构；旧 Action-DiT checkpoint 不能直接加载。轨迹 Scorer 与融合 Scorer 的参数不同，应保持训练/评测配置一致。

## 实现位置

- `navsim/agents/drivoR_query_register/decoder.py`：普通 Transformer 与定向注意力。
- `model.py`：可学习 queries、预测头、Scorer、EMA teacher。
- `losses.py`：一次匹配共享给轨迹与未来 loss、原评分 loss。
- `targets.py`：多 GT 视频与 human 回退配对。
- `agent.py`：复用训练运行接口、optimizer/EMA 生命周期和 checkpoint。
- `navsim/planning/script/config/common/agent/drivoR_query_register_dinov3_vitl*.yaml`：公共配置和四个实验配置。

## 本地验证

- 36 项测试通过：包含与原 DrivoR Action 路径等价、attention 方向、重复 GT 匹配、loss 平均、梯度截断、EMA 更新、推理无 GT 依赖，以及四组训练到 FP32 测试的衔接。
- 四组生产启动配置解析通过；融合版测试配置保留融合 Scorer，并关闭 GT/未来帧 target 读取。11 个新 shell 入口通过语法检查。
- 四组均在本地 GPU 0、1 上跑通真实数据与真实 NAVSIM oracle 的 smoke：每卡 1 个训练 step＋1 个验证 batch，训练 BF16，验证 FP32。多 GT 样本分别含 3、4 条有效 GT，loss 有限。

验证记录见 [validation.json](../artifacts/query_register_20261021/validation.json)，其中链接到各组完整日志。本地 smoke 用于检查数据、前后向和分布式流程，不代表训练完成后的 NAVSIM 成绩；本次未提交 DLC 任务。
