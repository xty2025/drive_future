# Action-DiT：DrivoR 多视图 Scene-query 特征版

更新：2026-10-15。此变体只替换 Action-DiT 的视觉特征来源，保留原单前视版本。

新增 `drivoR_action_dit_multiview_multigt_hungarian` 提供[匈牙利一对一匹配与等权多 GT v loss](DRIVOR_MULTI_GT_HUNGARIAN.md)，视觉结构及 Scorer 沿用本文配置。

`drivoR_action_dit_multiview_multigt_weighted` 另提供 [human / 非 human 加权 v loss](DRIVOR_ACTION_DIT_MULTI_GT_WEIGHTED.md)，默认 human 权重 1.0、非 human 平均项权重 0.5，使用独立启动脚本。

现另有 `drivoR_action_dit_multiview_multigt` 和 `run_drivor_action_dit_multiview_multigt_training.sh`：视觉和 Scorer 沿用本文配置，用有效多 GT 替代普通/长 GT 回归。全部 64 个输入仍使用原始 human GT 加噪，训练一次 DiT 前向；每条有效 GT 从该组输出中取最近者计算 v loss。详见[多 GT 说明与启动命令](DRIVOR_MULTI_GT.md)。以下双 GT、128 条训练输入的说明仅指原 `drivoR_action_dit_multiview` 配置。

## 1. 和单前视版本的区别

| 项目 | 原单前视版 | 新多视图版 |
|---|---|---|
| agent 配置 | `drivoR_action_dit` | `drivoR_action_dit_multiview` |
| `visual_feature_mode` | `native_patches` | `drivor_queries` |
| 当前相机 | F0 | F0、B0、L0、R0 |
| 每张图分辨率 | 672×1148 | 672×1148 |
| 每张图额外 Scene queries | 0 | 16 |
| 每张图 DINO 内部 token 数 | 3941 | 3957 |
| 下游视觉特征 | 3936 个前视 patch tokens | 四视角各 16 个 query 输出，合计 64 个 tokens |
| DiT 与 Scorer 的共同视觉输入 | `[B,3936,256]` | `[B,64,256]` |

原配置和原启动脚本继续使用单前视原生 patches。新变体继承它的 Action-DiT 结构、预训练文件、ego 编码、归一化、时间/噪声采样、双 GT v loss、单次 X-prediction 训练、10 步 Euler 推理、评分网络/损失、rank seed、优化器及总 batch 设置。

## 2. 视觉网络复用原版 DrivoR

新适配器 `DrivoRSceneQueryImageEncoder` 直接使用原版 `ImgEncoder`，按 `DrivoRModel` 的方式创建查询参数：

```text
scene_embeds: [1,4,16,384]
初始化: randn * 1e-6，可训练
```

图像仍来自当前帧，四个启用相机的历史索引都为 `[3]`；其余相机和 LiDAR 禁用。特征构建器实际堆叠顺序是 **F0、B0、L0、R0**，四组查询按同一顺序对应四路图像。

```text
images [B,4,3,672,1148]
  → 按相机展平 [B*4,3,672,1148]
  → 原版 ImgEncoder / DINOv2 ViT-S/14 reg4

每个视角的 DINO 输入序列:
  [16 Scene queries | 1 CLS | 4 registers | 3936 patches]
  总共 3957 个 tokens，特征维度 384

  → DINO 输出前 16 个 Scene-query tokens
  → 原版 neck: Linear(384,256)
  → 合并四路 query 输出
  → scene_context [B,64,256]
```

这些查询是在 DINO 内部参加 attention 的，不是在 DINO 输出之后新增 pooling。原生 CLS、registers 和图像 patches 仍参加 DINO 的 Q/K/V；下游只保留每个视角的 16 个 Scene-query 输出。

四路图像共用一个 DINO 和 neck，各视角有自己的 16 个可学习查询。DINO 图像编码阶段各视角独立处理，DiT/Scorer 随后读取拼接后的 64 个视觉 tokens。默认 `focus_front_cam=false`、`use_feature_pooling=false`、`compress_fc=false`，与原版 DrivoR 的 query 特征选择一致。

保留 DINO Q/V LoRA rank 32、原始 DINO 参数冻结、GridMask 和现有 gradient checkpointing 设置。共享的是场景特征，各个网络层的 K/V 投影参数仍各自独立。

## 3. DiT 和 Scorer 的接入

```text
scene_context [B,64,256]
  ├─ Action-DiT cross-attention 的 K/V 输入
  └─ 原版四层 TransformerDecoderScorer 的 K/V 输入
```

64 个视觉 tokens 和 64 条候选轨迹是两个不同的维度。Scorer 的 query 仍来自 64 条轨迹，视觉 key/value 则来自四视角的 Scene queries。

Action-DiT 仍是 30 层、hidden 256、FFN 1024、24 heads×128；Scorer 仍是轨迹 `24→1024→256` 投影、4 层/1-head decoder、ego 加法与六个 `256→1024→1` 评分头。本次没有改变这些网络或评分公式，也没有加载 DrivoR Release 的 Scorer 权重。

训练仍将普通/长 GT 合成 128 条输入，只做一次 DiT 前向；普通分支的 64 条 X-prediction 反归一化并 detach 后评分。两组独立 best-of-64 v loss、`sigma_min=0.1`、64 份噪声及共享 t 的关系保持不变。验证/推理仍从纯噪声执行 10 步 Euler。

| Attention | 训练 Q | 视觉 K/V |
|---|---|---|
| Action-DiT，每层 | `[B,24,1024,128]` | `[B,24,64,128]` |
| Action-DiT，推理每步 | `[B,24,512,128]` | `[B,24,64,128]` |
| Scorer，每层投影前 | `[B,64,256]` | `[B,64,256]` |

query 参数可接收轨迹 loss 和评分 loss 的梯度；评分 loss 仍不通过轨迹坐标更新 Action-DiT。视觉 K/V 长度缩短，但图像从一路增加为四路，实际整体吞吐需要实测，不能按 token 减少比例直接换算加速倍数。

## 4. 独立配置与启动命令

配置：[drivoR_action_dit_multiview.yaml](navsim/planning/script/config/common/agent/drivoR_action_dit_multiview.yaml)。

入口：[run_drivor_action_dit_multiview_training.sh](scripts/training/run_drivor_action_dit_multiview_training.sh)。它复用现有 Action-DiT launcher，只覆盖 agent 配置和每视角 query 数。默认实验目录独立为：

```text
DrivoR/exp/ke/training_drivoR_action_dit_multiview_navtrain/<DRIVOR_RUN_ID>/
```

在 PAI-DLC 配置 8 节点、每节点 8 GPU，填一次以下命令：

```bash
cd /cpfs29jb/data/zhaozc/project/drive_future

DRIVOR_EXPERIMENT=training_drivoR_action_dit_multiview_navtrain \
DRIVOR_RUN_ID=action_dit_multiview_q16_seed2_run1 \
DRIVOR_PERF_ENABLED=true DRIVOR_PERF_SAMPLE_EVERY=10 \
bash DrivoR/scripts/training/run_drivor_action_dit_multiview_training.sh \
  auto_resume=false train_ckpt_path=null runtime.require_resume=false \
  runtime.console_log_interval=25
```

默认仍为 BF16、每卡 batch 1、累积 1、global batch 64、fused AdamW、数据 workers 4、Oracle 进程 2、CPU threads 1、prefetch 2。数据和 metric cache 路径沿用现有脚本。

新实验从相同 DINOv2 和 Action-DiT 256 预训练文件初始化，并新建 Scene-query 参数。它的视觉参数结构与单前视原生 patch 版不同，应使用独立完整 checkpoint；不要把单前视版的 `multinode` 目录作为该变体的完整续训来源。

只检查配置：

```bash
bash DrivoR/scripts/training/run_drivor_action_dit_multiview_training.sh \
  --cfg job --package agent --resolve
```

以后恢复该新变体自身的实验，保留其 `DRIVOR_RUN_ID`，改用 `auto_resume=true runtime.require_resume=true`。

## 5. 验证范围

针对性测试覆盖：配置只改变视觉输入；视觉输出与原版 DrivoR 数值对齐；查询进入 DINO 内部；各视角保留独立的 16 个输出；DiT/Scorer 共用同一组 64-token 特征；单次训练前向；query 与 Scorer 的梯度边界；原单前视版本与三个 launcher 的回归。

完整训练/loss/归一化细节仍见 [Action-DiT 主文档](DRIVOR_ACTION_DIT.md)。本变体的验证产物位于 `artifacts/drivor_action_dit_multiview_20261015/`，其中缩小网络的数值测试不作为 NAVSIM 指标或吞吐结论。

已通过 **49 项检查**，并使用真实四视图图像、DINO/Action-DiT 权重和在线 Oracle 完成双 A100、BF16、NCCL 的两步训练及每卡一个验证 batch，正常退出。检查包含反向异常检测和冷启动，不用来推断正式训练速度。详见[验证记录](../artifacts/drivor_action_dit_multiview_20261015/validation.json)。
