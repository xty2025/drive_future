# Action-DiT：DINOv3 ViT-S / ViT-L

更新：2026-10-16。新增两个视觉骨干版本，保留原有 DINOv2 入口。Action-DiT、ego 编码、轨迹归一化、时间采样、64 个候选、Scorer、评分标签和优化器沿用现有实现；DINOv3 图像尺寸在配置中指定为 672×1152，由数据读取阶段直接 resize。

## 1. 预训练权重与网络

权重已经从 ModelScope 下载到本项目：

| 项目 | ViT-S | ViT-L |
|---|---|---|
| ModelScope 仓库 | [facebook/dinov3-vits16-pretrain-lvd1689m](https://modelscope.cn/models/facebook/dinov3-vits16-pretrain-lvd1689m) | [facebook/dinov3-vitl16-pretrain-lvd1689m](https://modelscope.cn/models/facebook/dinov3-vitl16-pretrain-lvd1689m) |
| 本地文件 | `weights/dinov3/vits16/model.safetensors` | `weights/dinov3/vitl16/model.safetensors` |
| 文件大小 | 86,406,384 bytes | 1,212,559,808 bytes |
| 层数 / heads | 12 / 6 | 24 / 16 |
| 原生特征维度 | 384 | 1024 |
| Patch size | 16×16 | 16×16 |
| 原生前缀 | 1 CLS＋4 storage/register tokens | 同左 |
| 下游投影 | `Linear(384,256)` | `Linear(1024,256)` |
| Q/V LoRA rank | 32 | 32 |

使用 [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3) 的官方骨干代码，源码位于 `third_party/dinov3`，许可证保留在其 `LICENSE.md`。原环境 `timm==1.0.15` 保持不变，DINOv2 仍走原 timm 路径；DINOv3 按需导入本地官方源码。

ModelScope 提供的是 HF 格式 safetensors。适配器将拆开的 Q/K/V 权重合并为官方实现的 QKV，将 FFN、LayerScale、token 等键映射到官方格式。HF 没有 K bias，官方模型使用屏蔽 K bias 的 Linear，因此补零 K bias 并保留固定 mask；RoPE 周期也由官方公式生成。ViT-S 的 211 个源 tensors、ViT-L 的 415 个源 tensors 均完整读取，随后严格检查官方模型的键名和形状，不增加文件哈希校验。

## 2. 冻结与 LoRA

加载预训练权重后，冻结 DINOv3 原有的全部参数。每层 attention 的 Q/V 加上与 DrivoR 一致的低秩更新：

```text
Q = Q_base + Bq(Aq(x))
K = K_base
V = V_base + Bv(Av(x))
```

A 使用 Kaiming 初始化，B 初始化为零，初始输出保持预训练骨干的行为。K 的原有 bias mask 保留，不给 K 或 FFN 额外加 LoRA。可训练的视觉参数为 Q/V LoRA、输出投影，以及启用时的 Scene queries。

冻结骨干保持 eval 模式，关闭其预训练阶段的随机 RoPE 坐标缩放；这不会阻止 LoRA 或 Scene queries 的梯度。外层 GridMask 仍在训练时启用，gradient checkpointing 沿用原设置。没有对整个视觉前向使用 `no_grad`，因此这些适配参数可以训练。

无 Scene queries 时，视觉分支可训练参数约为 ViT-S 的 688,384 个、ViT-L 的 3,408,128 个，包含 LoRA 与 256 维投影。

## 3. 分辨率、RoPE 与 Scene queries

ViT-S 和 ViT-L 配置均设置 `image_size: [1152, 672]`（顺序为 W,H）。数据读取阶段直接 resize 到 **672×1152**，沿用 ImageNet mean/std，骨干入口不做 padding。

```text
patch 网格 = 672/16 × 1152/16 = 42×72
原生 patch 数 = 3024
不加 Scene queries 时，骨干内部共 3024 + 1 CLS + 4 registers = 3029 tokens
每视角加 16 个 Scene queries 时，骨干内部共 3045 tokens
```

DINOv3 使用二维 RoPE，没有 DINOv2 那种需要插值的绝对位置表。位置编码按实际 patch 网格生成，可在配置中选择不同分辨率和长宽比；使用 patch16 整数倍可完整划分图像。同一 batch 仍需按配置统一尺寸，更高分辨率会增加计算量和显存。

两个版本默认沿用**当前单前视、不加 Scene queries**：

```text
image [B,1,3,672,1152]
  → DINOv3
  → 3024 个原生 patch 特征
  → Linear(D,256)
  → scene_context [B,3024,256]
```

通过 `DRIVOR_SCENE_QUERIES=16` 可启用每视角 16 个可学习查询，参数形状为 `[1,C,16,D]`，初始化 `randn*1e-6`，与 DrivoR 对应：

```text
每个视角的骨干输入：
[16 Scene queries | 1 CLS | 4 storage tokens | 3024 patches]

全部 token 参与骨干 attention，只有图像 patches 施加二维 RoPE。
下游取前 16 个 query 输出，投影到 256 维。
```

单前视加 queries 时，下游为 `[B,16,256]`；启用 F0、B0、L0、R0 四视角时，下游为 `[B,64,256]`。图像 patches、CLS 和所有 registers 始终参加骨干 attention。轨迹 DiT 和 Scorer 共用同一份下游视觉特征。

## 4. 训练配方

默认继承等权多 GT Action-DiT 配置，每个场景按实际有效 GT 数平均：

```text
L_scene = sum_有效GT(L_该GT与所选proposal的v_loss) / 有效GT数
L_trajectory = mean_scene(L_scene)
L_total = L_trajectory + L_scorer
```

保持独立最近邻匹配，允许不同 GT 选同一 proposal。human 与非 human GT 等权，padding 不参与计算，不额外加入 human 或长 GT loss。全部 64 个加噪输入仍由原始 human GT 构造，候选噪声独立、场景内共享 t；训练只做一次 DiT 前向，X-prediction 用于 v loss 与 Scorer，轨迹到评分分支的梯度截断保持原样。时间权重和小 t 分母下界沿用原设置，Scorer loss 单独计算，不除以 GT 数。

Action-DiT 仍从原有 Action-DiT 256 初始化文件加载，验证/推理仍为 10 步 Euler。默认仍为 BF16、8 机×8 卡、每卡 batch 1、累积 1、总 batch 64、25 epochs、seed 2、LR 2e-4。

## 5. 配置与 DLC 启动命令

| 版本 | 配置 | 入口 |
|---|---|---|
| ViT-S | [drivoR_action_dit_dinov3_vits.yaml](navsim/planning/script/config/common/agent/drivoR_action_dit_dinov3_vits.yaml) | [run_drivor_action_dit_dinov3_vits_training.sh](scripts/training/run_drivor_action_dit_dinov3_vits_training.sh) |
| ViT-L | [drivoR_action_dit_dinov3_vitl.yaml](navsim/planning/script/config/common/agent/drivoR_action_dit_dinov3_vitl.yaml) | [run_drivor_action_dit_dinov3_vitl_training.sh](scripts/training/run_drivor_action_dit_dinov3_vitl_training.sh) |

在 PAI-DLC 配置 8 节点、每节点 8 GPU，两个模型分别提交任务。

ViT-S，单前视原生 patches：

```bash
cd /cpfs29jb/data/zhaozc/project/drive_future
DRIVOR_EXPERIMENT=training_drivoR_action_dit_dinov3_vits_navtrain_multigt \
DRIVOR_RUN_ID=action_dit_dinov3_vits_seed2_run1 \
DRIVOR_SCENE_QUERIES=0 DRIVOR_PRECISION=bf16-mixed \
bash DrivoR/scripts/training/run_drivor_action_dit_dinov3_vits_training.sh \
  auto_resume=false train_ckpt_path=null runtime.require_resume=false \
  runtime.console_log_interval=25
```

ViT-L，单前视原生 patches：

```bash
cd /cpfs29jb/data/zhaozc/project/drive_future
DRIVOR_EXPERIMENT=training_drivoR_action_dit_dinov3_vitl_navtrain_multigt \
DRIVOR_RUN_ID=action_dit_dinov3_vitl_seed2_run1 \
DRIVOR_SCENE_QUERIES=0 DRIVOR_PRECISION=bf16-mixed \
bash DrivoR/scripts/training/run_drivor_action_dit_dinov3_vitl_training.sh \
  auto_resume=false train_ckpt_path=null runtime.require_resume=false \
  runtime.console_log_interval=25
```

开启 Scene queries，将 `DRIVOR_SCENE_QUERIES=0` 改为 `16`。若还要恢复原 DrivoR 四视图输入，追加三个相机配置；例如 ViT-S：

```bash
cd /cpfs29jb/data/zhaozc/project/drive_future
DRIVOR_EXPERIMENT=training_drivoR_action_dit_dinov3_vits_navtrain_multigt_q16_multiview \
DRIVOR_RUN_ID=action_dit_dinov3_vits_q16_multiview_seed2_run1 \
DRIVOR_SCENE_QUERIES=16 DRIVOR_PRECISION=bf16-mixed \
bash DrivoR/scripts/training/run_drivor_action_dit_dinov3_vits_training.sh \
  'agent.config.cam_b0=[3]' 'agent.config.cam_l0=[3]' 'agent.config.cam_r0=[3]' \
  auto_resume=false train_ckpt_path=null runtime.require_resume=false \
  runtime.console_log_interval=25
```

ViT-L 同样支持上述开关，替换模型对应的脚本、实验名和 run ID 即可。相机数量与 query 开关相互独立；只打开 queries 不会自动添加其他视角。

也可以直接用 Hydra 设置 `agent.config.visual_feature_mode=drivor_queries agent.config.num_scene_tokens=16`，命令末尾参数优先。单独覆盖权重目录用 `DRIVOR_DINOV3_VITS_WEIGHTS` / `DRIVOR_DINOV3_VITL_WEIGHTS`。

## 6. 训练成功后自动测试 navtest

DLC 配置 8 节点 × 8 GPU，下面整段直接作为启动命令。`DRIVOR_DINO_SIZE=vits` 使用 ViT-S，改为 `vitl` 使用 ViT-L。两者都保持单前视、不加 Scene queries、672×1152 和有效 GT 等权平均。

```bash
set -e
cd /cpfs29jb/data/zhaozc/project/drive_future

DRIVOR_DINO_SIZE=vits
DRIVOR_TRAIN_EXPERIMENT="training_drivoR_action_dit_dinov3_${DRIVOR_DINO_SIZE}_navtrain_multigt"
export DRIVOR_RUN_ID="action_dit_dinov3_${DRIVOR_DINO_SIZE}_seed2_train_eval_run1"
export DRIVOR_SCENE_QUERIES=0
export DRIVOR_PRECISION=bf16-mixed

DRIVOR_EXPERIMENT="${DRIVOR_TRAIN_EXPERIMENT}" \
bash "DrivoR/scripts/training/run_drivor_action_dit_dinov3_${DRIVOR_DINO_SIZE}_training.sh" \
  auto_resume=false train_ckpt_path=null runtime.require_resume=false \
  runtime.console_log_interval=25

if [ "${NODE_RANK:-${RANK:-0}}" = "0" ]; then
  DRIVOR_EVAL_CHECKPOINT="${NAVSIM_EXP_ROOT:-${PWD}/DrivoR/exp}/ke/${DRIVOR_TRAIN_EXPERIMENT}/${DRIVOR_RUN_ID}/lightning_logs/version_0/checkpoints/last.ckpt"
  DRIVOR_EXPERIMENT="eval_navtest_action_dit_dinov3_${DRIVOR_DINO_SIZE}" \
  DRIVOR_EVAL_WORKER=single_machine_thread_pool \
  DRIVOR_SCORE_WORKERS=32 NPROC_PER_NODE=8 \
  bash DrivoR/scripts/evaluation/run_drivor_pdm_score_evaluation.sh "${DRIVOR_EVAL_CHECKPOINT}" \
    "agent=drivoR_action_dit_dinov3_${DRIVOR_DINO_SIZE}" \
    agent.config.multi_gt.enabled=false \
    agent.config.action_dit.pretrained_path=null \
    agent.config.action_dit.num_inference_steps=10 \
    trainer.params.precision=bf16-mixed \
    dataloader.params.batch_size=1
fi
```

这是新建 run 的命令，checkpoint 路径对应其 `version_0`。重复新实验时更换 run ID；若使用已有的其他 Lightning version，则相应修改 checkpoint 路径。

训练成功后，只有 DLC 主节点使用本机 8 卡测试最后保存的 checkpoint，PDM 评分使用 32 个 CPU 进程。评测显式选择对应的 DINOv3 S/L agent，关闭训练用多 GT 目标读取，并直接恢复完整训练 checkpoint；单前视、图像尺寸和 10 步 Euler 与训练验证配置一致。

结果写入 `DrivoR/exp/ke/eval_navtest_action_dit_dinov3_<vits或vitl>/<DRIVOR_RUN_ID>/`，汇总成绩为 `pdms_summary.txt`。`set -e` 保证训练失败时不继续测试。

## 7. 兼容性与验证

旧 DINOv2 配置和入口继续走原代码；未升级环境中的 timm 或 transformers。两个 DINOv3 入口现继承原 Action-DiT 等权多 GT 配置，默认实验目录已去掉 `_weighted` 后缀。

不同骨干、不同 query 开关或不同相机数量应使用各自的完整训练 checkpoint。DINOv2、DINOv3-S、DINOv3-L 的完整 checkpoint 不直接混用；新实验依然使用原 Action-DiT 初始化文件和对应 DINOv3 视觉预训练文件。

初版已通过 **164 项针对性检查**：包括 HF→官方权重映射、原生前向对齐、CLS/register/patch/query 的位置与 RoPE 范围、冻结原参数、Q/V LoRA 与 queries 的梯度、checkpointing 数值/梯度一致性，以及旧模型/损失/启动脚本回归。该历史记录使用当时的边缘 padding 和加权 loss；2026-10-16 已按要求改为配置直接 resize 到 672×1152，并对有效 GT 等权平均。

ViT-S、ViT-L 的单前视原生 patch 和四视图 16-query 模式均已完成真实双 A100/BF16/NCCL smoke：每种模式两步训练和每卡一个验证 batch，正常退出。只用于验证链路，不代表收敛、正式吞吐或全量 NAVSIM 指标。

2026-10-16 的直接 resize＋等权配置更新已通过 **130 项针对性检查**，确认两个启动脚本解析为 `image_size=[1152,672]`、`weighting=equal`，图像直接进入原生 patch embedding，其他 GT loss 与启动参数回归通过。记录见[当前配置验证](../artifacts/drivor_dinov3_resize_equal_20261016/validation.json)。

实现：[dinov3_image_encoder.py](navsim/agents/drivoR_action_dit/dinov3_image_encoder.py)。初版真实训练 smoke 记录：[validation.json](../artifacts/drivor_action_dit_dinov3_20261015/validation.json)。
