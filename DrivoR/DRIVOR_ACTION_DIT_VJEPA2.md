# Action-DiT：V-JEPA 2 ViT-L

更新：2026-10-16。新增独立的 V-JEPA 2 ViT-L 视觉骨干配置，沿用当前 DINOv3 版本的图像尺寸、Q/V LoRA 和可选 Scene queries。默认仍是当前单前视、有效多 GT 等权监督；Action-DiT、Scorer、轨迹归一化、时间采样和训练流程沿用现有实现。

另有独立的 [V-JEPA 2.1 ViT-L 版本](DRIVOR_ACTION_DIT_VJEPA2_1.md)，使用 2.1 的 `ema_encoder`、原生单图分支和 RoPE 坐标缩放。

## 1. 官方代码与预训练权重

使用 [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2) 的 **V-JEPA 2 ViT-L/16**。官方 README 的权重链接为 [vitl.pt](https://dl.fbaipublicfiles.com/vjepa2/vitl.pt)，已经下载到本项目。按官方 Hub 默认规则选择 `target_encoder`，即预训练的 EMA 视觉编码器。

| 文件 | 用途 / 大小 |
|---|---|
| `third_party/vjepa2/` | 官方源码，骨干实现文件保持原样；许可证见其 `LICENSE` |
| `weights/vjepa2/vitl/vitl.pt` | 官方完整发布包，5,127,726,842 bytes |
| `weights/vjepa2/vitl/target_encoder.safetensors` | 默认训练加载的视觉编码器，1,215,569,056 bytes |
| `weights/vjepa2/vitl/source.json` | 下载来源记录 |
| `weights/vjepa2/vitl/target_encoder.json` | 提取结果，292 个 tensors、303,885,312 个参数 |

编码器的权重键和形状严格匹配官方 ViT-L。提取过程只去掉包装前缀，不改动参数值。使用独立 safetensors 后，各训练 rank 只读取约 1.22 GB 的编码器文件；模型只实例化视觉编码器。

提取工具是 [prepare_vjepa2_vitl_weights.py](../environment/prepare_vjepa2_vitl_weights.py)，在项目根目录使用 `drive_future` 环境运行即可从已有 `vitl.pt` 重新生成。覆盖训练权重路径可设置 `DRIVOR_VJEPA2_VITL_WEIGHTS`，指向该编码器格式的 safetensors。

## 2. ViT-L 架构和冻结范围

| 项目 | 当前实现 |
|---|---|
| Transformer | 24 层，hidden 1024，16 heads，每 head 64 维 |
| 每层结构 | LayerNorm → Self-Attention → 残差；LayerNorm → GELU FFN → 残差 |
| FFN | 1024 → 4096 → 1024 |
| LayerNorm | eps `1e-6`；最后有官方输出 LayerNorm |
| 图像 patch | 16×16；原生 `Conv3d` 的时间 tubelet 为 2 |
| 位置编码 | 时间、高度、宽度三个轴的 RoPE，无绝对位置表 |
| 原生 CLS / register | 均无 |
| 下游投影 | 可训练 `Linear(1024,256)` |
| LoRA | 每层 Q、V，rank 32；K 和 FFN 沿用冻结参数 |

加载预训练权重后冻结全部原有视觉参数，随后添加与现有 DrivoR 一致的 Q/V LoRA：

```text
Q = Q_base + Bq(Aq(x))
K = K_base
V = V_base + Bv(Av(x))
```

A 使用 Kaiming 初始化，B 初始化为零，初始低秩增量为零。冻结骨干保持 eval 模式，仍保留向 LoRA、输出投影和可选 Scene queries 传播的梯度。训练时 GridMask、gradient checkpointing 沿用当前配置。

默认视觉分支可训练参数为 3,408,128 个，包含 Q/V LoRA 和输出投影；启用 queries 时再增加 `相机数 × query数 × 1024` 个参数。

## 3. 当前单帧如何输入视频 ViT

默认只从数据中读取当前 `CAM_F0` 一张图。V-JEPA 2 的预训练 patch embedding 是时间核大小为 2 的 `Conv3d`，因此参照官方 [单图评测包装器](third_party/vjepa2/evals/image_classification_frozen/modelcustom/vit_encoder.py)，将同一张图复制成两份，构成一个时间 tubelet：

```text
当前前视图 [B,1,3,672,1152]
  → 同一张图复制两次 [B,3,2,672,1152]
  → 原生 PatchEmbed3D，kernel/stride = (2,16,16)
  → 一个时间位置 × 42×72 空间网格
  → ViT-L 输出 [B,3024,1024]
  → Linear(1024,256)
  → scene_context [B,3024,256]
  → Action-DiT 和 DrivoR Scorer 共用
```

两份图像内容完全相同，实际传感器输入仍为当前一帧。开启多个相机时，每个相机独立复制，输入合并为 `[B*C,3,2,H,W]`；不同相机作为独立样本通过骨干。

配置设置 `image_size: [1152,672]`，顺序为 W,H。数据侧直接 resize 到 **672×1152** 并沿用 ImageNet mean/std，骨干入口没有 padding。

RoPE 按实际 `T=1,H=42,W=72` 网格生成，保留官方的频率布局，不需要绝对位置 embedding 插值。可以在配置中选用其他长宽比和分辨率；使用 16 的整数倍完整划分 patches，同一 batch 保持统一图像尺寸，显存和计算量随 patch 数增加。

## 4. 可选 DrivoR Scene queries

默认 `DRIVOR_SCENE_QUERIES=0`，下游使用全部 3024 个原生 patch 特征。设置为 `16` 时，每个相机加入 16 个可学习 Scene queries：

```text
骨干输入：[16 Scene queries | 3024 image patches]
所有 token 参与同一 Self-Attention；RoPE 只作用于图像 patch 的 Q/K。
骨干输出：取前 16 个 queries → Linear(1024,256)。
```

Query 参数形状为 `[1,C,16,1024]`，初始化 `randn*1e-6`。单前视时下游为 `[B,16,256]`，四视角时为 `[B,64,256]`。Action-DiT 和 Scorer 都使用同一份 query 特征。

Query 开关与相机配置相互独立：只设 `DRIVOR_SCENE_QUERIES=16` 时仍只有当前前视图。

## 5. 轨迹监督和训练配方

继承 `drivoR_action_dit_multigt` 的当前设置：

- 每场景使用实际有效的 1–4 条 GT，human 与非 human 等权；padding 不参与监督。
- 默认独立最近邻匹配，每条 GT 从全部 64 个 proposal 选最小 v MSE，允许不同 GT 选中同一个 proposal。
- 每场景将这些匹配对的 v loss 相加，除以实际有效 GT 数，再对 batch 平均。时间权重和 `sigma_min=0.1` 沿用原设置；Scorer loss 单独计算。
- 全部 64 个加噪输入仍由原始 human GT 构造，候选噪声独立、场景内共享 t。训练只执行一次 DiT 前向，X-prediction 同时用于轨迹监督和 Scorer 的 64 条评分候选；轨迹坐标到评分分支的梯度截断保持原样。
- Action-DiT 继续加载指定的 Action-DiT 256 初始化权重，验证和推理继续使用 10 步 X→V Euler。

默认 DLC 配方为 **8 节点 × 8 GPU、每卡 batch 1、梯度累积 1、总 batch 64、BF16、25 epochs、seed 2、LR 2e-4**。数据、OSS 读取、在线 CPU oracle、优化器和 rank 随机种子策略沿用已有实现。

## 6. 配置与 DLC 启动命令

配置：[drivoR_action_dit_vjepa2_vitl.yaml](navsim/planning/script/config/common/agent/drivoR_action_dit_vjepa2_vitl.yaml)。入口：[run_drivor_action_dit_vjepa2_vitl_training.sh](scripts/training/run_drivor_action_dit_vjepa2_vitl_training.sh)。

在 PAI-DLC 配置 **8 节点、每节点 8 GPU**，当前单前视原生 patches：

```bash
cd /cpfs29jb/data/zhaozc/project/drive_future
DRIVOR_EXPERIMENT=training_drivoR_action_dit_vjepa2_vitl_navtrain_multigt \
DRIVOR_RUN_ID=action_dit_vjepa2_vitl_seed2_run1 \
DRIVOR_SCENE_QUERIES=0 DRIVOR_PRECISION=bf16-mixed \
bash DrivoR/scripts/training/run_drivor_action_dit_vjepa2_vitl_training.sh \
  auto_resume=false train_ckpt_path=null runtime.require_resume=false \
  runtime.console_log_interval=25
```

四视角加每视角 16 个 Scene queries：

```bash
cd /cpfs29jb/data/zhaozc/project/drive_future
DRIVOR_EXPERIMENT=training_drivoR_action_dit_vjepa2_vitl_navtrain_multigt_q16_multiview \
DRIVOR_RUN_ID=action_dit_vjepa2_vitl_q16_multiview_seed2_run1 \
DRIVOR_SCENE_QUERIES=16 DRIVOR_PRECISION=bf16-mixed \
bash DrivoR/scripts/training/run_drivor_action_dit_vjepa2_vitl_training.sh \
  'agent.config.cam_b0=[3]' 'agent.config.cam_l0=[3]' 'agent.config.cam_r0=[3]' \
  auto_resume=false train_ckpt_path=null runtime.require_resume=false \
  runtime.console_log_interval=25
```

单前视加 queries 只需将第一条命令的 `DRIVOR_SCENE_QUERIES` 改为 `16`，并使用独立实验名 / run ID。

### 训练成功后自动测试 navtest

下面整段用于新的 DLC 8×8 任务，默认单前视、不加 Scene queries。训练成功后，由主节点使用本机 8 GPU 推理和 32 个 CPU 评分进程测试 `last.ckpt`：

```bash
set -e
cd /cpfs29jb/data/zhaozc/project/drive_future

DRIVOR_TRAIN_EXPERIMENT=training_drivoR_action_dit_vjepa2_vitl_navtrain_multigt
export DRIVOR_RUN_ID=action_dit_vjepa2_vitl_seed2_train_eval_run1
export DRIVOR_SCENE_QUERIES=0
export DRIVOR_PRECISION=bf16-mixed

DRIVOR_EXPERIMENT="${DRIVOR_TRAIN_EXPERIMENT}" \
bash DrivoR/scripts/training/run_drivor_action_dit_vjepa2_vitl_training.sh \
  auto_resume=false train_ckpt_path=null runtime.require_resume=false \
  runtime.console_log_interval=25

if [ "${NODE_RANK:-${RANK:-0}}" = "0" ]; then
  DRIVOR_EVAL_CHECKPOINT="${NAVSIM_EXP_ROOT:-${PWD}/DrivoR/exp}/ke/${DRIVOR_TRAIN_EXPERIMENT}/${DRIVOR_RUN_ID}/lightning_logs/version_0/checkpoints/last.ckpt"
  DRIVOR_EXPERIMENT=eval_navtest_action_dit_vjepa2_vitl \
  DRIVOR_EVAL_WORKER=single_machine_thread_pool \
  DRIVOR_SCORE_WORKERS=32 NPROC_PER_NODE=8 \
  bash DrivoR/scripts/evaluation/run_drivor_pdm_score_evaluation.sh "${DRIVOR_EVAL_CHECKPOINT}" \
    agent=drivoR_action_dit_vjepa2_vitl \
    agent.config.multi_gt.enabled=false \
    agent.config.action_dit.pretrained_path=null \
    agent.config.action_dit.num_inference_steps=10 \
    trainer.params.precision=bf16-mixed \
    dataloader.params.batch_size=1
fi
```

新 run 对应 `version_0`；重复新实验时换 run ID，已有其他 Lightning version 则修改 checkpoint 路径。评测读取 `data/navsim` 的测试集和 `data/metric_cache_navtest`。结果写入 `DrivoR/exp/ke/eval_navtest_action_dit_vjepa2_vitl/<DRIVOR_RUN_ID>/`，汇总为 `pdms_summary.txt`。`set -e` 保证训练失败时不继续测试。

若训练时使用 queries 或其他相机，评测命令也需显式设置同样的 `agent.config.visual_feature_mode`、`agent.config.num_scene_tokens` 和相机配置。评测入口不读取训练脚本的 `DRIVOR_SCENE_QUERIES` 开关。

## 7. 兼容性和验证

V-JEPA 2 使用独立配置和启动脚本。原 DINOv2、DINOv3 入口继续使用各自的骨干；环境中的 timm 等依赖保持现有版本。不同骨干、相机数或 query 配置需要匹配的完整训练 checkpoint。

已通过 **176 项针对性检查**，覆盖原生单图复制前向、RoPE 数值对齐、patch/query 的位置编码范围、多相机隔离、冻结边界、LoRA/query 梯度、checkpointing 数值与梯度一致性，以及已有 Action-DiT / 多 GT / DINOv3 / 启动脚本回归。

使用已下载的真实 ViT-L 权重完成两种模式的真实双 A100/BF16/NCCL smoke：单前视原生 patches、四视角加 16 queries。每种均完成每卡两步训练和一个验证 batch，两个 rank 均正常退出，训练 loss 有限，GT fallback 为 0。这里只验证训练链路，不代表收敛、正式吞吐或全量 NAVSIM 指标。

实现：[vjepa2_image_encoder.py](navsim/agents/drivoR_action_dit/vjepa2_image_encoder.py)。验证记录：[validation.json](../artifacts/drivor_action_dit_vjepa2_20261016/validation.json)。
