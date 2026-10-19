# Action-DiT：V-JEPA 2.1 ViT-L

更新：2026-10-16。新增独立的 **V-JEPA 2.1 ViT-L/16** 视觉骨干入口，默认当前单前视、672×1152、原生 patches、有效多 GT 等权监督。Action-DiT、Scorer、轨迹归一化、加噪和优化器设置沿用现有版本。

## 1. 官方权重与模型

代码来自 [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2)，本地使用 `third_party/vjepa2/app/vjepa_2_1/models/` 中的官方 2.1 实现。权重使用官方 README 的 [vjepa2_1_vitl_dist_vitG_384.pt](https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt)，对应官方 Hub 名称 `vjepa2_1_vit_large_384`。

按官方规则从发布包读取 **`ema_encoder`**。提取工具为 [prepare_vjepa2_1_vitl_weights.py](../environment/prepare_vjepa2_1_vitl_weights.py)，只去掉包装前缀并保存视觉编码器；加载时严格匹配全部参数的键和形状。

权重已下载并提取完成：官方发布包为 5,151,198,524 bytes；独立编码器为 1,218,752,608 bytes，包含 **302 个 tensors、304,680,960 个参数**。训练时每个 rank 只加载独立编码器文件。

| 文件 | 用途 |
|---|---|
| `weights/vjepa2_1/vitl/vjepa2_1_vitl_dist_vitG_384.pt` | 官方完整发布包 |
| `weights/vjepa2_1/vitl/ema_encoder.safetensors` | 默认训练读取的独立编码器 |
| `weights/vjepa2_1/vitl/source.json` | 下载来源 |
| `weights/vjepa2_1/vitl/ema_encoder.json` | 提取结果和参数数量 |

覆盖权重路径使用 `DRIVOR_VJEPA2_1_VITL_WEIGHTS`，指向提取后的 safetensors。

ViT-L 为 **24 层、1024 hidden、16 heads、4096 FFN、GELU**，LayerNorm eps 为 `1e-6`。加载后冻结原有参数，只训练每层 Q/V 的 rank-32 LoRA、`Linear(1024,256)` 和启用时的 Scene queries。LoRA 的 B 初始为零，K 和 FFN 原参数冻结。默认视觉可训练参数为 3,408,128 个。

使用最后一层输出及官方 `norms_block[-1]`，向下游提供 1024 维特征再投影到 256 维。中间层的层级特征用于官方预训练，不拼接到本模型的下游输入中。

## 2. 单图分支与 RoPE

2.1 带有预训练的单图 patch embedding 和图像 modality embedding，本版本直接使用该分支：

```text
当前前视图 [B,1,3,672,1152]
  → [B,3,1,672,1152]
  → patch_embed_img，kernel/stride=(1,16,16)
  → 加冻结的 img_mod_embed
  → 原生 ViT-L，时间网格1 × 空间网格42×72
  → 最后一层 + 最后一个输出 LayerNorm
  → [B,3024,1024]
  → Linear(1024,256)
  → scene_context [B,3024,256]
  → Action-DiT 和 DrivoR Scorer 共用
```

每个相机输入一张当前帧，其他相机启用后按 `[B*C,3,1,H,W]` 独立编码。数据侧按配置 `image_size: [1152,672]`（W,H）直接 resize，沿用 ImageNet mean/std 和训练 GridMask。

位置编码使用 **2.1 官方时空 RoPE，`interpolate_rope=True`**。空间坐标按当前网格缩放到官方参考范围；patch16 的公式为：

```text
t_rope = temporal_index             # 当前仅一个时间位置，值为0
h_rope = height_index * 15 / (H_patches - 1)
w_rope = width_index  * 15 / (W_patches - 1)
```

这里的 15 来自官方 `pretrained_grid_size=16`。`384` 是发布模型的预训练尺寸标识；当前输入仍使用配置中的 672×1152。没有绝对位置表需要插值，也不对图像做边缘 padding。改变配置尺寸时保持空间维度为 16 的整数倍且每个方向至少两个 patches，同一 batch 使用统一大小。

与已有 V-JEPA 2 入口的区别集中在视觉骨干：

| 项目 | V-JEPA 2 | V-JEPA 2.1 |
|---|---|---|
| 发布包编码器键 | `target_encoder` | `ema_encoder` |
| 当前图像的入口 | 同图复制两次，时间核2 | 专用单图分支，时间核1 |
| Modality embedding | 无 | 加载并使用 `img_mod_embed` |
| 输出归一化 | `norm` | `norms_block[-1]` |
| RoPE | 原生网格位置 | 官方空间坐标缩放；使用2.1的旋转实现 |
| 默认下游视觉特征 | 3024×256 | 3024×256 |

## 3. 可选 Scene queries 与监督

默认 `DRIVOR_SCENE_QUERIES=0` 使用全部 3024 个 patches。设置为 `16` 时，每个相机使用 `[16 Scene queries | 3024 patches]` 参加骨干 Self-Attention，RoPE 只作用于 patches，输出取 16 个 query 特征。

单前视加 queries 的下游长度为 16，四视角加 queries 的长度为 64；DiT 与 Scorer 共用同一份特征。Query 参数形状为 `[1,C,16,1024]`，初始化为 `randn*1e-6`。只开启 query 开关时仍为单前视，相机数量由相机配置独立控制。

训练配方与当前等权多 GT 版本一致：

- 有效 GT 为真实的 1–4 条，human 和非 human 等权；每场景的匹配轨迹 loss 除以实际有效 GT 数。
- 默认最近邻匹配，允许多个 GT 选择同一 proposal；Scorer 仍监督全部 64 条候选。
- 全部 64 个输入由原始 human GT 加独立噪声构造，场景内共享 t；训练只执行一次 DiT 前向。
- X-prediction 转换为 v loss，保持原时间权重和小 t 下界；同一组干净轨迹用于评分，轨迹坐标梯度截断保持原样。
- Action-DiT 继续从原来的 256 维初始化文件加载；验证和推理使用 10 步 Euler。

## 4. DLC 启动

配置：[drivoR_action_dit_vjepa2_1_vitl.yaml](navsim/planning/script/config/common/agent/drivoR_action_dit_vjepa2_1_vitl.yaml)。入口：[run_drivor_action_dit_vjepa2_1_vitl_training.sh](scripts/training/run_drivor_action_dit_vjepa2_1_vitl_training.sh)。

PAI-DLC 配置 **8 节点 × 8 GPU**。默认每卡 batch 1、累积 1、总 batch 64、BF16、25 epochs、seed 2、LR 2e-4。

当前单前视原生 patches：

```bash
cd /cpfs29jb/data/zhaozc/project/drive_future
DRIVOR_EXPERIMENT=training_drivoR_action_dit_vjepa2_1_vitl_navtrain_multigt \
DRIVOR_RUN_ID=action_dit_vjepa2_1_vitl_seed2_run1 \
DRIVOR_SCENE_QUERIES=0 DRIVOR_PRECISION=bf16-mixed \
bash DrivoR/scripts/training/run_drivor_action_dit_vjepa2_1_vitl_training.sh \
  auto_resume=false train_ckpt_path=null runtime.require_resume=false \
  runtime.console_log_interval=25
```

四视角加每视角 16 个 queries：

```bash
cd /cpfs29jb/data/zhaozc/project/drive_future
DRIVOR_EXPERIMENT=training_drivoR_action_dit_vjepa2_1_vitl_navtrain_multigt_q16_multiview \
DRIVOR_RUN_ID=action_dit_vjepa2_1_vitl_q16_multiview_seed2_run1 \
DRIVOR_SCENE_QUERIES=16 DRIVOR_PRECISION=bf16-mixed \
bash DrivoR/scripts/training/run_drivor_action_dit_vjepa2_1_vitl_training.sh \
  'agent.config.cam_b0=[3]' 'agent.config.cam_l0=[3]' 'agent.config.cam_r0=[3]' \
  auto_resume=false train_ckpt_path=null runtime.require_resume=false \
  runtime.console_log_interval=25
```

单前视加 queries：使用第一条命令，将 `DRIVOR_SCENE_QUERIES` 改为 `16`，并更换实验名 / run ID。

### 训练成功后自动测试 navtest

下面整段用于新的 DLC 8×8 任务，默认单前视、不加 queries。训练成功后，主节点使用本机 8 GPU 和 32 个 CPU 评分进程评测最终 `last.ckpt`。

```bash
set -e
cd /cpfs29jb/data/zhaozc/project/drive_future

DRIVOR_TRAIN_EXPERIMENT=training_drivoR_action_dit_vjepa2_1_vitl_navtrain_multigt
export DRIVOR_RUN_ID=action_dit_vjepa2_1_vitl_seed2_train_eval_run1
export DRIVOR_SCENE_QUERIES=0
export DRIVOR_PRECISION=bf16-mixed

DRIVOR_EXPERIMENT="${DRIVOR_TRAIN_EXPERIMENT}" \
bash DrivoR/scripts/training/run_drivor_action_dit_vjepa2_1_vitl_training.sh \
  auto_resume=false train_ckpt_path=null runtime.require_resume=false \
  runtime.console_log_interval=25

if [ "${NODE_RANK:-${RANK:-0}}" = "0" ]; then
  DRIVOR_EVAL_CHECKPOINT="${NAVSIM_EXP_ROOT:-${PWD}/DrivoR/exp}/ke/${DRIVOR_TRAIN_EXPERIMENT}/${DRIVOR_RUN_ID}/lightning_logs/version_0/checkpoints/last.ckpt"
  DRIVOR_EXPERIMENT=eval_navtest_action_dit_vjepa2_1_vitl \
  DRIVOR_EVAL_WORKER=single_machine_thread_pool \
  DRIVOR_SCORE_WORKERS=32 NPROC_PER_NODE=8 \
  bash DrivoR/scripts/evaluation/run_drivor_pdm_score_evaluation.sh "${DRIVOR_EVAL_CHECKPOINT}" \
    agent=drivoR_action_dit_vjepa2_1_vitl \
    agent.config.multi_gt.enabled=false \
    agent.config.action_dit.pretrained_path=null \
    agent.config.action_dit.num_inference_steps=10 \
    trainer.params.precision=bf16-mixed \
    dataloader.params.batch_size=1
fi
```

新 run 对应 `version_0`，已有其他 version 时改成实际 checkpoint 路径。评测读取 `data/navsim` 的测试集和 `data/metric_cache_navtest`。结果位于 `DrivoR/exp/ke/eval_navtest_action_dit_vjepa2_1_vitl/<DRIVOR_RUN_ID>/`，汇总为 `pdms_summary.txt`。

若训练启用了 queries 或其他相机，评测也要显式传入同样的 `agent.config.visual_feature_mode`、`agent.config.num_scene_tokens` 和相机配置；通用评测入口不读取训练脚本的 `DRIVOR_SCENE_QUERIES` 环境开关。

实现：[vjepa2_1_image_encoder.py](navsim/agents/drivoR_action_dit/vjepa2_1_image_encoder.py)。V-JEPA 2.1、V-JEPA 2 和 DINO 的完整训练 checkpoint 分别使用各自的 agent 配置。

## 5. 验证

已通过 **189 项针对性检查**：包括单图分支的 FP32/BF16 原生前向对齐、2.1 RoPE 数值与坐标范围、Scene queries 不旋转、多相机隔离、冻结参数与 LoRA/query 梯度、gradient checkpointing 一致性，以及现有视觉骨干、Action-DiT、多 GT loss 和 DLC 启动脚本回归。

使用已下载的官方 ViT-L 权重完成两种真实双 A100/BF16/NCCL smoke：默认单前视原生 patches，以及四视角每视角 16 queries。两种均完成每卡两步训练和一个验证 batch，完整加载 302 个编码器 tensors，Scorer/在线评分/反向传播正常，loss 有限，GT fallback 为 0。

这里只验证链路，不代表收敛、正式吞吐或全量 NAVSIM 指标。配置展开和本文三个 Bash 命令的语法检查也已通过。详细记录：[validation.json](../artifacts/drivor_action_dit_vjepa2_1_20261016/validation.json)。
