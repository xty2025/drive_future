# A28 半量实验：MoT 与 Wan2.1 1.3B

新增两组独立 A28 改动，沿用 `run_training.py` 和 `navtrain_half`：42,555 个 train 场景、完整 18,179 个 val 场景。默认 8 机 × 8 卡，总 batch64、25 epochs、64 proposals。输出统一在 `DrivoR/exp/navtrain_half_ablations/`。

| 实验 | 轨迹生成器 | 未来 register 生成器 | self-attention | 配置 |
|---|---|---|---|---|
| MoT / 改动 11 | 原 A28 Action-DiT 256，加载原 Action256 权重 | Wan2.2 5B：hidden3072 / FFN14336，加载原生 block/time 权重 | Register 可读 Action；Action 不可读 Register | [half_a28_mot.yaml](navsim/planning/script/config/training/half_a28_mot.yaml) |
| Wan2.1 / 改动 12 | 与 register 共用 Wan2.1 1.3B：hidden1536 / FFN8960 | 同一个 joint DiT | 原 A28 单向 mask | [half_a28_wan21_1p3b.yaml](navsim/planning/script/config/training/half_a28_wan21_1p3b.yaml) |

两者都直接继承原 A28，彼此不叠加，也不继承前面的 pooled patch、VAE、分组噪声、冻结 ViT 等改动。

## MoT

参考 `FastWam_re/src/fastwam/models/wan22/mot.py` 的逐层 expert 流程；仅移植 DiT/MoT 结构。

- Action expert：30 层，hidden256、FFN1024。
- Register/video expert：30 层，hidden3072、FFN14336。
- 两路都使用 24 个 attention heads、每 head128 维，所以投影后的 Q/K/V 宽度都为 3072；残差通道宽度和 FFN 保持各自独立。
- 每层分别进行各自的 norm、time＋ego adaLN、Q/K/V 投影和 RoPE，然后合并 Q/K/V 做一次带 mask 的 attention；拆分结果，经过各自的 output projection、cross-attention 和 FFN。
- 实际 attention 打包为 `[R32 | A8]`。为保持 A28 的位置语义，RoPE 的位置仍是 A=0～7、R=8～39；打包顺序不重新编号。未来 time/slot embedding 加在 register expert 输入上。
- 每个场景有 64 对 proposal；self-attention 的 batch 展开为 `B×64`，不同候选不互相读取。Register 的 32 个查询可读自己这组的 R32＋A8；Action 的 8 个查询只读自己这组的 A8。

| Query | Action K/V | Register K/V |
|---|---|---|
| Action | 可读 | 不可读 |
| Register | 可读 | 可读 |

两路 cross-attention 都使用当前前视 DINOv3 的特征。原特征为 3024 个 256 维 patch tokens；Action 路直接使用，register 路通过新建 256→3072 projection 适配。Ego 同样由共享的 256 维特征分别进入各自时间调制。Cross-attention 不把另一分支的输出作为 K/V。

Register 能读 Action，因此未来 loss 可通过这条 attention 链路更新 Action expert 的隐藏特征/K/V 参数；没有额外 detach 两路之间的特征。Scorer 输入的最终轨迹和预测 register 仍按 A28 detach。

初始化：

```text
Action:
/cpfs29jb/data/zhaozc/project/FastWam_re/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_256hdim_1024ffn_3outdim.pt

Register/video:
/cpfs29jb/data/zhaozc/project/FastWam_re/checkpoints/Wan-AI/Wan2.2-TI2V-5B
```

视频 backbone 的 block/time 权重直接载入，不插值；新的 register 输入/输出线性层、scene/ego 适配层从随机初始化开始。这里的 video expert 输出 DINO register 特征，不生成像素；没有加载 FastWAM 的 text encoder、text embedding 或 VAE。

## Wan2.1 1.3B

加载用户给定目录的 `diffusion_pytorch_model.safetensors`。与 Wan2.2 的分片文件相比，这是单文件，加载器已兼容两种格式。原生配置为 hidden1536、FFN8960、12 heads×128、30 层、freq256；816 个 block/time 张量的实际形状已核对。

```text
/cpfs29jb/data/zhaozc/project/FastWam_re/checkpoints/Wan-AI/Wan2.1-T2V-1.3B
```

与改动 8 的适配方式相同：轨迹和 register 使用一个 joint DiT，输入 `[A8,R32]`，Action 不能读 Register，Register 可读 Action。替换视频 patch I/O 和 text embedding，以新的轨迹/register 输入输出及 256→1536 scene/ego projection 接入 A28。Scorer 保持 256 维。

## 沿用的监督与运行方式

- 当前图像：单前视 1152×672、DINOv3 ViT-L，基础权重冻结、Q/V LoRA rank32 更新，不加 scene queries。
- 未来目标：EMA DINOv3 编码对应的 8 张未来帧，每帧 4 个 register，仍为每条候选 `[8,4,1024]`，共 64 组。
- 所有 64 组加噪输入仍锚定 human 轨迹与 human 未来。各 GT 的输出匹配可选择全部 64 条候选，允许多个 GT 选中同一 proposal。
- 轨迹 loss 保持 `human 物理坐标 L1 + 0.5 × diverse8 L1 coverage`；未来仍是匹配位置的 X→v MSE、sigma 下界0.1、有效 GT/帧/token/通道取平均，外部权重1。
- 轨迹和对应预测 registers 均 detach 后进入原 A28 融合 Scorer；原评分监督不变。
- 训练一次前向，推理保留原10步 Euler；仅训练使用 GT 未来特征，推理不读取未来 GT。
- 两组默认使用 ZeRO-2，FP32 参数/优化器状态、BF16 模型训练前向，每 epoch 进行 FP32 val。保留所有 epoch 的分片 checkpoint；训练结束自动导出最后一轮的 FP32 单文件，再进行 navtest。

## 命令

```bash
cd /cpfs29jb/data/zhaozc/project/drive_future

# MoT：Wan5B register expert + 原 Action256，Register 能看见 Action
bash DrivoR/scripts/training/run_drivor_half_a28_mot_train_eval.sh

# 原生 Wan2.1 1.3B joint DiT
bash DrivoR/scripts/training/run_drivor_half_a28_wan21_1p3b_train_eval.sh
```

只训练时把 `_train_eval.sh` 换为 `_training.sh`。单独测试使用通用入口，第二个参数支持普通 checkpoint 文件或 ZeRO-2 checkpoint 目录：

```bash
bash DrivoR/scripts/evaluation/run_drivor_half_ablation_evaluation.sh a28_mot /path/to/last.ckpt
bash DrivoR/scripts/evaluation/run_drivor_half_ablation_evaluation.sh a28_wan21_1p3b /path/to/last.ckpt
```

训练与测试输出分别为：

```text
DrivoR/exp/navtrain_half_ablations/ke/training_drivoR_half_a28_mot/<RunID>
DrivoR/exp/navtrain_half_ablations/ke/eval_navtest_drivoR_half_a28_mot/<RunID>

DrivoR/exp/navtrain_half_ablations/ke/training_drivoR_half_a28_wan21_1p3b/<RunID>
DrivoR/exp/navtrain_half_ablations/ke/eval_navtest_drivoR_half_a28_wan21_1p3b/<RunID>
```

## 验证

定向测试覆盖单向 attention 的数值和梯度、跨候选隔离、与 FastWAM 原 MoT/DiT 方法的数值对照、单次 forward、Scorer 梯度截断、strict checkpoint 往返及训练/评测入口。另进行小模型双卡 ZeRO-2 smoke。完整大模型吞吐和 NAVSIM 成绩不能由这些合成数据检查推断。

验证记录：`exp/navtrain_half_ablations/validation_mot_wan21_20261025/`。

14 项新增检查通过；MoT 小模型双卡训练＋FP32 验证完成，EMA 更新一次。实际 Wan2.1 单文件中 816 个待加载张量的形状全部匹配。补充原配置回归和 Wan2.1 双卡 smoke 在导入 torch/pytest 时被 DSW 共享文件系统 `Too many open files` 阻断，未进入模型执行；没有进行全尺寸模型训练或真实 NAVSIM 指标验证。详见 [validation.json](exp/navtrain_half_ablations/validation_mot_wan21_20261025/validation.json)。

## DLC 提交

2026-10-25 已提交两个新 8 机 × 8 卡任务，回读时均为 `Queuing`：

| 实验 | DLC ID |
|---|---|
| a28_mot | `dlc1qmkeffdojqos` |
| a28_wan21_1p3b | `dlc1qwjzt7r6p1oj` |

实际启动命令、RunID 和训练/测试路径见 [提交记录](exp/navtrain_half_ablations/dlc_deployment_mot_wan21_20261025_004807/README.md)。
