# `drive_future` `zhaozc-1019` 与当前 NVS 架构对比

> **版本 v2（2026-10-20，已逐项对照源码验证）**：本文全部技术声明已逐文件核对
> `drive_future` 侧（`action_dit.py`、`model.py`、`dit.py`、`teacher.py`、
> `multi_gt.py`、`drivor_features.py`、`image_encoder.py`、
> `dinov3_image_encoder.py` 及全部相关 YAML）与 NVS 侧
> （`wan_dit.py`、`model.py`、`flow.py`、`scorer.py`、`data.py`、
> `vision.py`/`teacher.py`、`config_v6_wan_e2e.yaml`）。验证结论：核心声明
> 全部与源码一致；v2 补充了 register 变体 EMA 教师、scorer `project` 融合
> 完整结构、`future_only` 阶段冻结清单与多 GT 噪声端点等实现细节。

## 0. 审阅范围与版本身份

本文是一次静态代码审阅，目标是把两个当前实现的网络结构、端到端数据流、训练/推理路径以及 tensor shape 说清楚。本文没有运行训练、真实数据缓存构建或 NAVSIM 评测。

### 代码版本

| 对象 | 路径 | 版本/状态 |
|---|---|---|
| `drive_future` | `/mnt/zhaozc_workspace/project/drive_future_zhaozc_1019` | 分支 `zhaozc-1019`，`a5e70d4`，远端 `origin/zhaozc-1019` |
| 当前 NVS | `/mnt/zhaozc_workspace/project/NVS` | 分支 `snapshot/navtrain-wan-20261011`；工作树有未提交修改，本文以工作树当前代码和 `config/config_v6_wan_e2e.yaml` 为准 |
| NVS 唯一实现 | `NVS/src/nvs_v6_wan_e2e/` | 当前维护谱系；不要把已经删除的 `nvs_v6` / `nvs_v6_wan` 旧实现混入结论 |

### `AGENTS.md` 结论

目标仓库根目录的 `AGENTS.md` 实际上是一份 NVS 维护约束快照，并不是 `DrivoR` 源代码的专属架构说明。它明确要求以当前严格 loader、YAML 和实现为准，并强调当前 e2e 契约的关键点：11 维可观测 ego、米制 `[x,y,heading]` 轨迹、Wan 4-head/64 head-dim、以及当前未来分支的 mask 约束。

目标仓库中另外存在 `DrivoR/docs/agents.md`，它是 NAVSIM agent 开发接口说明，规定 `AbstractAgent`、feature builder、target builder、`forward`、`compute_loss`、optimizer 和最终 `[B,T,3]` trajectory 接口。它不改变 ActionDiT 的具体 tensor geometry。

需要特别注意：目标根 `AGENTS.md` 的部分历史段落与当前 NVS 工作树已经不一致。例如旧段落可能出现 `Register(8)`、Stage 1/Stage 2 两张 mask、zero padding 或 `Register -> Action` 等表述；当前 NVS 的权威约束是 `NVS/AGENTS.md`、当前 YAML、`config.py` 和实际源码。本文的 NVS 描述采用后者。

---

## 1. 先给结论

两套代码不是同一个模型的小改版，而是两条不同的工程路线：

1. **目标仓库的主 ActionDiT 路径**是 DrivoR scorer 外面包一个 FastWAM/Wan 风格的连续 flow Action-DiT：输入当前图像、当前 ego 和一组 noisy trajectory，生成 `M=64` 个候选轨迹，再用 DrivoR scorer/PDM logits 选出一个候选。
2. **目标仓库的 register 变体**在上述 Action-DiT 上拼接未来 register token，并用 DINOv3 自身的 register token 作为 EMA teacher target；它是目标仓库中与 NVS 最接近的实验路径，但它仍然保留了目标仓库的 `30-layer / 24-head / 128-head-dim` Action-DiT 和原有 scorer/Lightning 接口。
3. **当前 NVS**是一个严格收敛后的 e2e `DirectedPairDiT`：先产生 81 个 Scene token，再进入 8 层、`d_model=256`、4-head、64-head-dim 的 Wan ActionDiT。其序列是 `[Scene(81) | Action(8) | Register(32)]`，共 121 token。
4. **当前 NVS 的未来不是 Action 的输入旁路**。自注意力只有一个下三角 segment mask：Scene 只读 Scene，Action 读 Scene+Action，Register 读 Scene+Action+Register。因此 Action 永远读不到 Register；未来预测是共享 trunk 上的辅助 readout/监督，不是“未来表征直接辅助动作生成”的 `Register -> Action` 通路。
5. 目标仓库的 register 变体与当前 NVS 的主要结构差异是：目标仓库把 register 和 action 放进一个 ActionRegisterDiT，但 mask 只禁止 Action 读 Register；它没有当前 NVS 的 Scene 进入 self-attention、121-token geometry、严格 single-mask contract、DINOv3 future-query cosine readout 和 1024 维标准化缓存目标这一整套收敛后的契约。

---

## 2. 目标仓库 `drive_future/zhaozc-1019` 的实际结构

### 2.1 代码入口

目标仓库把大量 NAVSIM/DrivoR 源码以普通文件形式放在 `DrivoR/` 下。主要入口如下：

| 功能 | 文件 |
|---|---|
| 普通 Action-DiT agent | `DrivoR/navsim/agents/drivoR_action_dit/agent.py` |
| 普通 Action-DiT 外壳模型 | `DrivoR/navsim/agents/drivoR_action_dit/model.py` |
| Action-DiT block、attention、flow head | `DrivoR/navsim/agents/drivoR_action_dit/action_dit.py` |
| 图像 encoder | `DrivoR/navsim/agents/drivoR_action_dit/image_encoder.py`、`dinov3_image_encoder.py` |
| register agent | `DrivoR/navsim/agents/drivoR_action_dit_register/agent.py` |
| register 外壳模型 | `DrivoR/navsim/agents/drivoR_action_dit_register/model.py` |
| register DiT | `DrivoR/navsim/agents/drivoR_action_dit_register/dit.py` |
| register teacher | `DrivoR/navsim/agents/drivoR_action_dit_register/teacher.py` |
| register target builder | `DrivoR/navsim/agents/drivoR_action_dit_register/targets.py` |
| 基础配置 | `DrivoR/navsim/planning/script/config/common/agent/drivoR.yaml` |
| Action-DiT 配置 | `DrivoR/navsim/planning/script/config/common/agent/drivoR_action_dit.yaml` |
| DINOv3 配置 | `DrivoR/navsim/planning/script/config/common/agent/drivoR_action_dit_dinov3_vitl.yaml` |
| register 配置 | `DrivoR/navsim/planning/script/config/common/agent/drivoR_action_dit_dinov3_vitl_register*.yaml` |

目标仓库存在多个 image family 和多个实验配置。下面将默认主路径、DINOv3 主路径、多视角路径和 register 路径分开描述，避免把它们误认为一个固定模型。

### 2.2 NAVSIM feature 输入

`DrivoRFeatureBuilder.compute_features()` 将每个时刻的 ego pose、velocity、acceleration、driving command 拼接为 11 维向量，见 `DrivoR/navsim/agents/drivoR/drivor_features.py:50-64`：

```text
ego_feature = [pose(3), velocity(2), acceleration(2), driving_command(4)]
             = 11 dimensions
```

在 Action-DiT 模型中只使用当前帧：`features["ego_status"][:, -1]`，见 `drivoR_action_dit/model.py:91-96`。因此标准形状可以记为：

```text
ego_status history: [B, T_history, 11]
current ego:       [B, 11]
hist_encoding:     Linear(11, 256)
ego embedding:     [B, 256]
```

普通 `drivoR.yaml` 的基础传感器配置是四个相机 `cam_f0/cam_l0/cam_r0/cam_b0`，但 `drivoR_action_dit.yaml` 将主 Action-DiT 路径收敛到只取当前 front camera：`cam_f0: [3]`，其余相机为空，且 `model.py:30-35` 强制每个启用相机只能请求 `[3]`。

### 2.3 目标仓库图像编码路径

#### A. 默认 `native_patches` + DINOv2

默认 Action-DiT 配置见 `drivoR_action_dit.yaml:8-27`：

```text
input image: W,H = 1148,672
backbone:    timm/vit_small_patch14_reg4_dinov2
patch size:  14 x 14
patch grid:  672/14 x 1148/14 = 48 x 82
patch count: 3936
native prefix: CLS + 4 register = 5
```

`NativePatchImageEncoder.forward()` 在 `image_encoder.py:45-56` 中：

```text
images                  [B, Cams, 3, 672, 1148]
flatten camera           [B*Cams, 3, 672, 1148]
DINOv2 forward_features [B*Cams, 5+3936, D_backbone]
drop native prefix      [B*Cams, 3936, D_backbone]
neck Linear(D_backbone,256)
scene context           [B, Cams*3936, 256]
```

对于 DINOv2 ViT-S，`D_backbone` 由 timm backbone 决定，通常为 384；代码最终不依赖该具体值，因为 neck 将其投影到 `tf_d_model=256`。

#### B. DINOv3 ViT-L 主路径

`drivoR_action_dit_dinov3_vitl.yaml:5-16` 将图像改成 `[1152,672]`、DINOv3 ViT-L/16：

```text
input image: W,H = 1152,672
patch size:  16 x 16
patch grid:  42 x 72
patch count: 3024
native prefix: 1 CLS + 4 storage (register) tokens = 5
feature width: 1024 for ViT-L
neck:          Linear(1024,256)
```

没有 scene query 时，`DinoV3ImageEncoder.forward()`（`dinov3_image_encoder.py:159-185`）返回：

```text
[B, Cams*3024, 256]
```

DINOv3 的 base 权重冻结，只保留 Q/V LoRA、neck，以及在 scene-query 模式下的 learned scene embeddings 训练；其图像 encoder 的构造和冻结逻辑见 `dinov3_image_encoder.py:108-157`。

#### C. DrivoR scene-query 模式

如果配置 `visual_feature_mode: drivor_queries`，每个相机使用 `num_scene_tokens=16` 个 learned query。每个相机的输出不再是全部 patch，而是：

```text
per camera: [B, 16, 256]
C cameras:  [B, C*16, 256]
```

目标仓库的多视角配置 `drivoR_action_dit_multiview.yaml:7-24` 打开 front/left/right/back 四个相机，并得到：

```text
visual scene tokens = 4 * 16 = 64
scene context       = [B,64,256]
```

这和当前 NVS 的 5 帧单 front camera、80 visual tokens + 1 ego token 不同。

### 2.4 普通 Action-DiT 的逐层结构与 shape

普通 Action-DiT 的基础配置见 `drivoR_action_dit.yaml:26-44`：

```text
hidden_dim       = 256
ffn_dim          = 1024
freq_dim         = 256
num_heads        = 24
attn_head_dim    = 128
num_layers       = 30
action_dim       = 3
num_poses       = 8
proposal_num     = 64
```

这里 `24*128=3072`，而 hidden width 是 256。代码的 attention projection width 是 `num_heads * attn_head_dim`，所以这是目标仓库 FastWAM/Wan 权重接口的 attention channel geometry，而不是当前 NVS 的 `4*64=256` geometry。

对一批输入，令 `M=64`、`T=8`、`A=3`、`D=256`：

#### 1. flow source/noise

训练时 `model.py:186-205` 产生：

```text
noise       [B, M, T, A] = [B,64,8,3]
clean GT    [B, M, T, A]
noisy       [B, M, T, A]
timesteps   [B, M]
```

多 GT 版本仍然使用 `proposal_num=64` 个 action candidates，并通过 nearest 或 Hungarian matching 将输出与最多 4 个 GT 对齐，见 `model.py:215-259` 和 `multi_gt.py`（Hungarian 用 `scipy.optimize.linear_sum_assignment`，要求 proposals 数不少于有效 GT 数）。注意两边的噪声端点不同：目标仓库的 flow 是 **GT ↔ 纯高斯噪声**（`anchors = original human GT` 或 `balanced` GT，`noise = torch.randn`），而当前 NVS 的 flow 是 **GT ↔ 64-anchor 网格**（网格是 σ=1 端的结构化先验）。

#### 2. action encoder

`ActionDiT.action_encoder = Linear(3,256)`，见 `action_dit.py:132`：

```text
[B,M,8,3] -> [B,M,8,256]
```

#### 3. time condition

`timesteps` 经一维 sinusoidal embedding，再经：

```text
sinusoidal(freq_dim=256)
  -> Linear(256,256)
  -> SiLU
  -> Linear(256,256)
```

然后加上 ego embedding，并通过 `time_projection: Linear(256,6*256)` 产生每个 candidate 的 6 组 AdaLN 调制量：

```text
condition: [B,M,256]
t_mod:     [B,M,6,256]
```

#### 4. 30 个 ActionBlock

`ActionBlock` 在 `action_dit.py:99-116`，每层包括：

```text
x: [B,M,8,256]

SelfAttention:
  q/k/v linear width = 24*128 = 3072
  q,k RMSNorm over 3072
  split -> [B*M,24,8,128]
  SDPA -> [B*M,24,8,128]
  merge -> [B,M,8,3072]
  output Linear(3072,256)

SharedSceneCrossAttention:
  query from x:     [B,M,8,256]
  flatten queries:  [B, M*8, 24, 128]
  scene K/V:        [B, S, 24, 128]
  output:            [B,M,8,256]

FFN:
  Linear(256,1024) -> GELU(tanh) -> Linear(1024,256)
```

目标实现的 self-attention 只在每个 candidate 的 8 个 action token 内进行；candidate 维度 `M` 被 reshape 到 batch，candidate 之间不互相 attend。Scene 只作为 cross-attention 的共享 K/V，并没有进入 ActionDiT 的 self-attention 序列，见 `action_dit.py:61-96`。

#### 5. flow head 与 endpoint

`head = Linear(256,3)`，见 `action_dit.py:138-140`：

```text
velocity prediction: [B,M,8,3]
```

训练时用连续 flow matching loss；推理时从 noise `[B,64,8,3]` 开始，用 `num_inference_steps=10` 的 Euler schedule 迭代 `ActionDiT -> velocity -> current`，见 `model.py:98-115`。最后一个 Euler 步直接返回 X 预测本身（最后一步 `delta=-sigma` 时该步是精确的，不走训练 loss 的下限）。因此普通目标路径是**多步采样**，不是 NVS 当前的单次 algebraic recovery。

#### 6. trajectory denormalization 与 scorer

Action-DiT 输出归一化轨迹，随后 `denormalize_trajectory()` 得到物理坐标：

```text
proposals: [B,64,8,3]
```

score 分支在 `model.py:117-141`：

```text
proposals.detach().reshape(B,64,8*3)
 -> pos_embed Linear(24,1024) -> ReLU -> Linear(1024,256)
 -> TransformerDecoderScorer，scene K/V + ego embedding
 -> DrivoR Scorer 六类 score heads
 -> PDM weighted score [B,64]
 -> argmax
 -> trajectory [B,8,3]
```

普通 Action-DiT 的 scorer 对 proposal 输入 detach；score loss 不将轨迹梯度反传回 generator 的 proposal 输入。这一点与当前 NVS scorer 的 detach 约束要分别核对，不能只看类名。

### 2.5 目标仓库 register 变体

目标 register 配置由 `drivoR_action_dit_dinov3_vitl_register.yaml` 打开：

```text
future_stage       = joint 或 future_only
register_frames    = 8
registers_per_frame= 4
register_count     = 32
register_dim       = image_backbone.num_features = 1024 (DINOv3-L)
```

代码见 `drivoR_action_dit_register/model.py:15-58`。

#### Register DiT 的输入

`ActionRegisterDiT` 复用普通 ActionDiT 的：

```text
action_encoder, time_embedding, time_projection, blocks, head
```

然后增加：

```text
register_encoder = Linear(1024,256)
register_head    = Linear(256,1024)
future_time_embedding   [8,256]
register_slot_embedding [4,256]
```

见 `drivoR_action_dit_register/dit.py:33-55`。

输入输出 shape 为：

```text
noisy_action:    [B,M,8,3]
noisy_registers: [B,M,32,1024]

action tokens:   Linear(3,256)      -> [B,M,8,256]
register tokens: Linear(1024,256)   -> [B,M,32,256]
position:        [8,4,256] broadcast -> [B,M,32,256]
concat:          [B,M,40,256]
```

注意：目标 register 变体的 self-attention sequence 是 **Action(8)+Register(32)=40 token**，并没有把 Scene token 拼入 self-attention；Scene 仍通过复用的 cross-attention 作为 K/V。

#### Register mask 与信息流

`dit.py:53-55` 构造 40x40 bool mask：Action query 不能读 Register key，其余方向开放。即：

```text
query \ key | Action | Register
------------|--------|---------
Action      | yes    | no
Register    | yes    | yes
```

因此 target register 变体和当前 NVS 一样，都避免了 `Register -> Action` 对动作 velocity 的直接信息泄漏；但两者仍有重要差异：NVS 的 Scene 也在 self-attention 中，NVS 有 81+8+32=121 token，NVS 用 one shared mask 覆盖所有 entry point，target register 代码只在 Action+Register 的 40 token 上做 mask。

#### Register teacher/target

`drivoR_action_dit_register/targets.py:49-108` 从 paired future frame cache 读取：

```text
future_register_images: [K,8,3,H,W]
future_register_valid:  [K,8]
```

其中每个未来时刻有 4 个 register，最终 target 是：

```text
[B,K,32,1024]
```

`EmaDinoRegisters` 位于 `drivoR_action_dit_register/teacher.py`，以当前 DINOv3 encoder 的 native ViT 为 teacher，EMA 更新 teacher 权重。已验证的实现细节：

```text
teacher 构造:  deepcopy(学生 native ViT)，Q/V LoRA 合并为 native QKV Linear
EMA 更新对象:  合并后的实际前向权重 W = W_base + B @ A（不是分别平均 A、B）
教师输出:      encoder.forward_features(image)["x_storage_tokens"]
               即 DINOv3 原生 4 个 register token 逐字（verbatim），无 readout
输入/输出:     pixels [B,K,8,3,H,W] uint8 -> targets [B,K,8,4,1024]
像素归一化:    /255 后减 ImageNet mean / std
anchor 选择:   全部 64 个 noisy register 输入统一用 original logged-human
               未来帧的教师特征作 anchor（Cosmos GT 只影响输出监督）
register 加噪: noisy_registers = (1-sigma)*anchor + sigma*randn
```

目标仓库的 register 训练阶段支持：

```text
future_only: 只训练 register denoising；action 输入是 balanced GT 且
             clean_action=True（action 段 modulation 用零 timestep），
             冻结 action_dit.head、pos_embed、scorer_attention、scorer、
             register_score_projection、score_fusion
joint:       Action velocity loss + register loss 联合训练，register 的
             matching 复用 action 的 nearest matched indices
```

推理时 `sample_joint` 对 action 与 register **双流同时** 做 10-step Euler 积分，见 `drivoR_action_dit_register/model.py:150-167`。

配置中的 `future_register.loss_type: velocity` 指 action 侧仍可使用 flow velocity loss；register 本身的 supervision 在 `register/losses.py` 中实现。

#### Register-conditioned scorer

当 `future_register.scorer.enabled=true` 时，预测 register 会和 proposal 一起进入 scorer：

```text
proposal: [B,M,8,3] -> detach -> flatten [B,M,24]
register: [B,M,32,1024] -> detach -> flatten [B,M,32768]
```

`fusion=flatten` 时将二者拼接后进入替换过的 `pos_embed = Linear(24+32768, 1024) -> ReLU -> Linear(1024,256)`；`fusion=project` 时 register 先经 `register_score_projection = Linear(32768,256) -> ReLU`，与轨迹 embedding 拼接后再经 `score_fusion = Linear(512,1024) -> ReLU -> Linear(1024,256)` 融合，见 `drivoR_action_dit_register/model.py:40-72`。这属于目标仓库 register scorer 实验，不是普通 Action-DiT 默认路径。

---

## 3. 当前 NVS 的数据和网络 pipeline

当前 NVS 的权威配置是 `NVS/config/config_v6_wan_e2e.yaml`。核心实现集中在 `NVS/src/nvs_v6_wan_e2e/`。

### 3.1 数据输入与 observation

NVS 的 `Observation` 定义在 `data.py:43-70`：

```text
rgb_history       [B,5,V,3,H,W]
history_valid     [B,5,V]
ego_history       [B,5,7]
route_command    [B,4]
relative_ego_poses ...
```

其中 ego 的 7 维是：

```text
pose(3) + velocity(2) + acceleration(2) = 7
```

`SceneEncoder.encode_clean_ego()` 取 `ego_history[:, -1]` 的 7 维，再拼 `route_command` 的 4 维，得到严格的 11 维 observable ego，见 `model.py:224-240`：

```text
current ego [B,7] + command [B,4]
    -> [B,11]
    -> hist_encoding Linear(11,256)
    -> clean_ego [B,256]
```

NVS 不使用旧版 12 维 ego，也不对轨迹做 `/20` 或 `xy_scale`。

### 3.2 历史图像 Student

当前配置是单 front camera、5 个时刻、`num_scene_tokens=16`：

```text
历史时间: -2.0,-1.5,-1.0,-0.5,0.0
frames   = 5
views    = 1
visual tokens = 5*16 = 80
ego tokens    = 1
scene tokens  = 81
```

配置见 `config_v6_wan_e2e.yaml:152-181`。

图像先保持 DrivoR 原始输入 `672x1148 (H,W)`，在 DINOv3 Student 内部 bicubic resample 到 `672x1152`，而不是 zero padding：

```text
Student image [B*5,3,672,1152]
patch size   16
patch grid   42*72
patch count  3024
DINOv3-L dim 1024
neck         Linear(1024,256)
```

DINOv3 base 冻结，只训练 DrivoR 风格 Q/V LoRA、scene query、neck 和外围模块。Student 输出：

```text
per history frame: [B,16,256]
5 frames:          [B,80,256]
```

每帧的 16 个 token 加一个可学习 frame time embedding，然后拼接一个 `[B,1,256]` ego token，得到：

```text
SceneEncoder output: [B,81,256]
```

代码见 `model.py:171-258`；实际 DINOv3 实现见 `vision.py`。

如果没有 DINO checkpoint，`SceneEncoder` 会使用 `SpatialStudent` fallback：16x16 conv patch projection + adaptive pooling 到 16 个 query + `Linear(256,256)` neck，见 `model.py:112-164`。这只是结构测试/无 checkpoint fallback，不等同于真实 DINOv3 训练路径。

### 3.3 Scene refinement

`DirectedPairDiT` 在 `model.py:321-359` 建立 8 个 `SceneBlock`。每个 SceneBlock 是：

```text
input: [B,81,256]
LayerNorm(256)
MultiheadAttention(embed_dim=256, heads=4, batch_first=True)
residual
LayerNorm(256)
Linear(256,1024) -> GELU -> Linear(1024,256)
residual
output: [B,81,256]
```

8 层后仍为 `[B,81,256]`。Scene token 只读 Scene 自身，故后续 scorer 取到的 `scene_context` 不会被 Action/Register token 写回。

### 3.4 ActionFlow 数据

NVS 的轨迹状态固定为：

```text
[x, y, heading] = 3 dimensions
单位: x/y 米，heading 弧度
```

每个 scene 有最多 `K=4` 条 valid Multi-GT，64 个 immutable 8x8 mode code：

```text
modes: [B, M=64, 2]
source/anchor: [B,64,8,3]
GT targets:     [B,K,8,3]
```

NVS 的 ActionFlow 在训练时对 source 和 target 进行单步 flow interpolation，得到：

```text
noisy_action:    [B,64,8,3]
velocity_target: [B,64,8,3]
timestep:        [B]
```

推理时固定 `sigma=1`，source 直接是 flow anchor；`recover()` 用一次代数表达式恢复 endpoint，不进行多步 rollout。入口见 `model.py:434-483`、`flow.py:183-242`。

### 3.5 NVS Wan ActionDiT 输入编码

当前 NVS ActionDiT 配置见 `config_v6_wan_e2e.yaml:206-235` 和 `wan_dit.py:463-543`：

```text
hidden_dim     = 256
ffn_dim        = 1024
num_heads      = 4
attn_head_dim  = 64
num_layers     = 8
action_tokens  = 8
register_tokens= 32
action_dim     = 3
```

`4*64=256`，所以 q/k/v projection 的总 attention width 与 hidden width 相同。

#### Action encoder

NVS 的 action encoder 输入不是单纯 3 维，而是两路 3 维拼接：

```text
norm_odo(source):                    [B,M,8,3]
detrend(source)/SHAPE_RANGE:         [B,M,8,3]
concat:                               [B,M,8,6]
action_encoder Linear(6,256):        [B,M,8,256]
```

之后加 mode projection、ego projection 和 8 个 `time_queries`：

```text
mode code [B,M,2] -> Linear(2,256) -> broadcast over 8 action tokens
ego [B,256]       -> Linear(256,256) -> broadcast over M and 8
time_queries      [8,256]
action tokens     [B,M,8,256]
```

相关实现见 `wan_dit.py:697-731`。

#### Register encoder

当前 NVS future register 不是把 GT 轨迹直接喂给 register。`encode_future()` 只用 learned future queries 和 projected ego：

```text
future_queries: [32,256]
ego projection: [B,1,1,256]
register input: [B,M,32,256]
```

它没有 trajectory/mode 输入；Register 之所以能区分同一 scene 的不同 candidate，是因为它在 self-attention 中读取旁边的 Action tokens。见 `wan_dit.py:733-750`。

### 3.6 NVS 121-token block

NVS 每个 candidate 展开为 `[B*M, ...]`，然后把 Scene、Action、Register 拼成：

```text
[Scene(81) | Action(8) | Register(32)] = 121 tokens
```

Scene embedding 只做一次：

```text
context Scene [B*M,81,256]
 -> text_embedding
 -> embedded Scene [B*M,81,256]
```

这个 embedded Scene 同时用于：

1. 拼入 self-attention sequence；
2. 作为 cross-attention 的 K/V，保持 Wan 预训练 cross-attention 的输入分布。

代码见 `wan_dit.py:752-803`。

#### Segment mask

唯一的 self-attention visibility table 在 `wan_dit.py:429-461`：

| query \\ key | Scene | Action | Register |
|---|---:|---:|---:|
| Scene | 1 | 0 | 0 |
| Action | 1 | 1 | 0 |
| Register | 1 | 1 | 1 |

这里行是 query，列是 key。换成数据流箭头：

```text
Scene   -> Scene
Scene   -> Action
Action  -> Action
Scene   -> Register
Action  -> Register
Register -X-> Action
```

该 mask 在 Stage 1、Stage 2 和部署 inference 共享，没有 stage 参数。`wan_dit.py:774-782` 明确说明未来 target 不进入 sequence；register 起点只是 learned future queries + ego，Teacher target 只用于和 `future_head` 输出比较。

#### 每个 NVS DiT block

`DiTBlock` 在 `wan_dit.py:325-361`，每层：

```text
x: [B*M,121,256]

AdaLN modulation:
  condition [B*M,256]
  time_projection -> [B*M,6,256]

SelfAttention:
  q/k/v width = 4*64 = 256
  RoPE 分段计算：Scene 81、Action 8、Register 每帧 4 个 query 共享同一时间相位
  mask: [121,121]
  output: [B*M,121,256]

CrossAttention:
  query: [B*M,121,256]
  context K/V: [B*M,81,256]
  output: [B*M,121,256]

FFN:
  Linear(256,1024) -> GELU(tanh) -> Linear(1024,256)
```

NVS 还有 per-block `TrajectoryGate`，只写 Register slice：它读取 register 和 action，输出加回 register；Scene 和 Action slice 不被 gate 写回。见 `wan_dit.py:810-819`。

#### 两个输出 head

```text
action readout:   x[:,81:89]   [B*M,8,256]
Head(256,3):      velocity     [B*M,8,3]

register readout: x[:,89:121]  [B*M,32,256]
future_head       LayerNorm(256) + Linear(256,1024)
predicted_register [B,M,32,1024]
```

Head 采用接近零的初始化，使部署初始 endpoint 保持在 anchor 附近；见 `wan_dit.py:364-397`。最终 `_flow_pass()` 将 velocity reshape 回 `[B,M,8,3]`，register reshape 回 `[B,M,32,1024]`，见 `model.py:418-432`。

### 3.7 NVS future Teacher 和 target cache

NVS 的 future video 只用于 target，不是 observation，也不是模型输入。当前配置 future 时间为 `+0.5,+1.0,...,+4.0s`，共 8 帧。

`multiview_data.py` 形成：

```text
future_rgb: [1,K,8,V,3,H,W]
```

具体 shape 约束见 `teacher.py:560-606`。Teacher 使用原生 `464x832` future resolution：

```text
Teacher image: [1,8,3,464,832]
patch grid:    29*52 = 1508
feature dim:   1024
```

Teacher 与 Student 使用同一个 DINOv3 checkpoint，但 future frame 不 resize、不插值。`FutureQueryExtractor` 使用每帧 DINOv3 自身的 4 个 prefix register 作为 cosine-attention query，对 patch grid 做 readout：

```text
per future frame: 4 query outputs x 1024
8 frames:         8*4 = 32 tokens
target per GT:    [32,1024]
all GT:            [K,32,1024]
```

目标读取是 parameter-free 的 cosine attention，temperature 为 0.005、query heads 为 8，之后做 train-only frozen diagonal mean/std standardization。相关契约由 `config.py:155-225` 固定。

因此 NVS 的 PairTargets future target：

```text
future_register: [B,K,32,1024]
future_chunk_valid: [B,K,32]
future_quality: ...
```

每个未来 frame 的 4 个 token 是 frame-major：`token = q + 4*f`。NVS 当前没有时间 pooling，没有旧版 quadrant pooling，也没有 PCA rank-deficient target。

### 3.8 NVS Stage 1

Stage 1 的目标只有 future register representation，不训练 velocity head。入口和 trainability 见 `stages.py:35-86`。

输入：

```text
observation: history [B,5,1,3,H,W], ego [B,5,7], command [B,4]
source clean trajectory: [B,64,8,3]
modes: zero code [B,64,2]
timestep: zeros [B]
```

`forward_register_only()` 使用 `sigma=0` 的 clean trajectory，经过与正式模型相同的 `_flow_pass()`，取：

```text
predicted_register: [B,64,32,1024]
```

之后用 dense nearest-GT future L1/paired contrast 等 future loss。每条 candidate 会配到最近的有效 future GT，future loss 不复用 action Hungarian assignment。Stage 1 冻结 `theta.head.*`、scorer 和不使用的模块，但训练 Scene encoder、scene blocks、Wan trunk、future queries 和 future head。

### 3.9 NVS Stage 2

Stage 2 是 joint flow + future + scorer 训练，生产入口是 `scripts/run_navtrain_wan_e2e_ddp.py`。

```text
source anchor: [B,64,8,3]
GT trajectory: [B,64,8,3] after candidate expansion/assignment policy
sample sigma/timestep
noisy_action: [B,64,8,3]
forward NVS once
velocity: [B,64,8,3]
endpoint: [B,64,8,3]
predicted register: [B,64,32,1024]
```

endpoint 通过 flow `recover()` 直接得到，没有 Euler rollout。scorer 使用 endpoint proposals、refined Scene K/V 和 ego embedding，输出六类 DrivoR score logits，再产生 `[B,64]` PDM score 和 selected `[B,8,3]` trajectory。

NVS scorer 是 `ScratchDrivoRScorer`，它的参数随机初始化，不加载 DrivoR scorer checkpoint；结构包括 proposal embedding、Transformer decoder 和六个 scorer heads，见 `scorer.py:49-186`。当前 scorer 的关键约束是 proposals detach，但 scene K/V 不 detach，因此 score loss 可以塑造 eta/Scene LoRA。

### 3.10 NVS deployment inference

部署入口 `DirectedPairDiT.forward_pairs()`，见 `model.py:434-451`：

```text
source anchor: [B,64,8,3]
timestep:      sigma=1, shape [B]
Scene encoder + 8 scene blocks
NVS DiT one forward
velocity:      [B,64,8,3]
endpoint:      [B,64,8,3]
```

不做多步 sampling。`endpoint` 即 scorer 输入的 64 个 candidate proposals。

---

## 4. 逐项差异对照

### 4.1 视觉输入与 Scene token

| 项目 | `drive_future` 主 Action-DiT | `drive_future` register 变体 | 当前 NVS |
|---|---|---|---|
| 默认历史 | 当前时刻 | 当前时刻 | 5 帧历史：-2.0 到 0.0s |
| 相机 | 主配置只 front；另有四相机 variant | DINOv3 配置可 front/多视角 | 1 个 front camera |
| 原始尺寸 | 默认 `1148x672`；DINOv3 variant `1152x672` | DINOv3 `1152x672` | 输入 `672x1148`，模型内部 bicubic 到 `672x1152` |
| visual token | native patch 可为 3936 或 DINOv3 3024；query variant 为 C*16 | 同左 | 5*16=80 |
| ego token | ego 作为 ActionDiT condition，不拼进 visual scene | 同左 | Scene 最后一个 token，Scene 总数 81 |
| backbone | DINOv2、DINOv3、V-JEPA2 等可切换 | 要求 DINOv3 | DINOv3 student + fixed future teacher |
| 图像增强 | `GridMask` 可开 | 同左 | 当前 NVS AGENT 明确未移植 DrivoR 的 GridMask |

最大的结构差异是：目标普通路径把视觉 token作为 cross-attention K/V，NVS 把 80 visual + 1 ego token 先 refine 成 Scene，再把 Scene 同时放入 self-attention 和 cross-attention。

### 4.2 Ego 输入

两套当前代码都最终使用 11 维 DrivoR observable ego：

```text
pose3 + velocity2 + acceleration2 + command4 = 11
```

但使用位置不同：

- 目标仓库：`Linear(11,256)` 后作为 ActionDiT 的 `ego_embedding`，加到时间 condition；原始 Scene context 不包含 ego token。
- NVS：同样 `Linear(11,256)`，但 ego encoding 一方面成为 Scene 的第 81 个 token，另一方面又经过 `ego_to_action_projection` 作为 Action/Register condition。

### 4.3 Action token geometry

| 项目 | 目标普通 Action-DiT | 目标 register ActionRegisterDiT | 当前 NVS |
|---|---:|---:|---:|
| hidden dim | 256 | 256 | 256 |
| FFN dim | 1024 | 1024 | 1024 |
| layers | 30 | 30 | 8 |
| heads | 24 | 24 | 4 |
| head dim | 128 | 128 | 64 |
| attention width | 3072 | 3072 | 256 |
| action tokens | 8 | 8 | 8 |
| action input | 3 | 3 | 6（absolute + detrended） |
| inference | 10-step Euler flow | 10-step Euler flow | one forward + algebraic recover |

这不是单纯的“层数不同”：目标仓库的 pretrained ActionDiT attention projection width 是 3072，而 NVS 为了严格匹配当前 Wan payload 使用 `4*64=256`。因此目标仓库的 pretrained checkpoint 不能直接当作 NVS 的同形状 backbone 使用。

### 4.4 Register geometry 与未来监督

| 项目 | 目标 register 变体 | 当前 NVS |
|---|---|---|
| register token count | 8 future frames * 4 = 32 | 8 future frames * 4 = 32 |
| register hidden input | `[B,M,32,1024] -> Linear(1024,256)` | learned `[B,M,32,256]`，不把 target 喂入 |
| target width | DINOv3 feature width 1024 | DINOv3 feature width 1024 |
| teacher | EMA DINOv3 registers | fixed/frozen DINOv3 + parameter-free query readout |
| target readout | DINOv3 native register teacher path | 4 prefix query 对 1508 patch 做 cosine attention |
| temporal pooling | 目标 register builder 按 8 帧读取 | 明确 none；32 token frame-major |
| target entering model | noisy register 是 denoising input | Teacher target 永不进入 model sequence |
| action reads register | mask 禁止 | mask 禁止，且由 one shared contract 固定 |
| register reads action | 允许 | 允许，且是 register 区分 candidate 的必要路径 |

两边虽然都出现 32 个未来 register 和 1024 宽 target，但这不能说明它们是同一个未来分支。目标变体的 register 是一个显式 noisy register denoising 输入；NVS 的 register slot 是 learned query，Teacher target 只在 loss 端比较。

### 4.5 Multi-GT 与 candidate 组织

目标仓库：

```text
M=64 candidates
最多 K=4 original multi-GT
所有 candidate 通常从 original human GT 作为 noise anchor
action loss 使用 nearest 或 Hungarian 等配置策略
```

NVS：

```text
M=64 immutable 8x8 mode codes
最多 K=4 valid GT
action assignment 和 future assignment 分开
future 每条 candidate 独立 nearest valid GT
```

NVS 这样拆开是为了保证 64 个 predicted register 行都有有效 future gradient，不能直接把 action Hungarian assignment 当作 future assignment。

### 4.6 Scorer 差异

目标仓库普通路径沿用 DrivoR scorer：

```text
proposal [B,64,8,3] detach
pos_embed [B,64,256]
TransformerDecoderScorer with scene K/V + ego
six score heads
PDM weighted score -> argmax
```

目标 register scorer variant 可把 `[B,64,32,1024]` flatten/project 后并入 proposal embedding，属于额外实验配置。

NVS 使用独立 `ScratchDrivoRScorer`，不加载 DrivoR checkpoint；默认的 NVS scorer 语义是 proposal detach、scene K/V 不 detach。这意味着 NVS score loss 能够回传到 Scene encoder/LoRA，目标普通 Action-DiT 的 proposal detach 则不应被描述为同一种 scorer gradient path。

### 4.7 训练阶段差异

| 阶段 | 目标仓库普通路径 | 目标 register 变体 | 当前 NVS |
|---|---|---|---|
| future-only Stage 1 | 无 | `future_only`，action head/scorer 冻结，训练 register denoising | 只训练 future register readout，velocity head 冻结 |
| joint Stage 2 | flow trajectory + scorer | action velocity + register loss，可选 register-conditioned scorer | flow velocity + future L1/contrast + scorer |
| action input | noisy candidate trajectory | noisy action + noisy register | noisy action；register 从 learned queries 开始 |
| inference | 10 Euler steps | 10 Euler steps | one forward，sigma=1，algebraic recovery |
| target use | trajectory GT；register variant 另有 RGB future target | RGB future frames -> EMA DINOv3 register target | cached future target `[B,K,32,1024]` 只做监督 |

### 4.8 信息流的根本差异

目标 register 变体：

```text
Scene --cross-attention--> Action/Register
Action <-> Action
Action --> Register
Register -X-> Action
```

当前 NVS：

```text
Scene --self-attention--> Scene
Scene --> Action
Scene --> Register
Action --> Action
Action --> Register
Register -X-> Action
```

NVS 额外保证 Scene token 在 DiT 中是 read-only context，不能被 candidate action 或 future register 污染；这就是 NVS scorer 可以使用 refined Scene 作为稳定 K/V 的原因。

---

## 5. 容易误读的地方

### 5.1 不能把目标仓库根 `AGENTS.md` 当成目标模型说明

目标根 `AGENTS.md` 是 NVS 约束快照，仓库实际源代码则是 DrivoR/ActionDiT。阅读目标仓库时必须同时看：

```text
root AGENTS.md                 -> agent 约束/历史契约
DrivoR/docs/agents.md          -> NAVSIM agent 接口
DrivoR/.../drivoR_action_dit   -> 普通 Action-DiT 实现
DrivoR/.../drivoR_action_dit_register -> register 实验实现
```

### 5.2 不能把 `Register(32)` 等同于 NVS 当前未来条件

两边都有 32 个 register token，但：

- 目标 register 的 32 个 token 有 noisy register 输入，并做 register denoising；
- NVS 的 32 个 token 由 learned future queries + ego 初始化，Teacher target 仅在 loss 端使用；
- 目标仓库没有 NVS 的 81 Scene self-attention tokens；
- NVS 的 target 是 frame-major `8*4`，每 4 个 query 共享该 frame 的 RoPE phase。

### 5.3 当前 NVS 不是未来条件直接生成动作

当前 NVS 的设计目标曾经考虑过 future-conditioned planning，但最终实现删除了 `future_fuse` 和所有 `Register -> Action` bypass。当前可确认的事实是：

```text
future target -> future loss -> shared DiT trunk gradient
future register -> action velocity: forbidden
```

所以应称为“未来表征作为共享 trunk 的辅助监督/预测 readout”，不能称为“Register 直接辅助动作生成”。

### 5.4 不能用旧 NVS 文档的 geometry 对账

当前 NVS 已从旧版 `Register(8)/97 token` 收敛到 `Register(32)/121 token`；图像也从旧 zero padding 约定改为内部 bicubic resampling 到 `672x1152`。如果某段历史 spec、root AGENTS 旧段落或实验日志与当前 `config.py`/YAML/代码冲突，应以当前严格 loader 和契约测试为准。

---

## 6. 结论与后续实现含义

如果目标是把 `drive_future/zhaozc-1019` 的做法迁移或融合到当前 NVS，不能只复制 `register_dim=1024` 或 `register_count=32`。至少需要重新决定：

1. 是否保留 NVS 的 5 帧 Scene encoder 和 81-token Scene self-attention；
2. 是否采用 NVS 的 `4 heads * 64 = 256` Wan checkpoint geometry，而不是目标仓库的 `24 heads * 128 = 3072`；
3. 是否让 future target 只在 loss 侧出现，还是引入显式 noisy register denoising；
4. 是否继续维持 NVS 的单一 mask 和 `Register -> Action` 禁止约束；
5. 是否继续使用 one-shot flow recovery，而不是目标仓库的 10-step Euler sampling；
6. scorer 是否继续使用 NVS 的 scratch scorer 和 scene-K/V gradient，还是迁移目标仓库的 DrivoR scorer/Lightning 路径。

从当前代码看，最小风险的兼容方向是：保留 NVS 的 81/8/32 token contract、4x64 Wan geometry、one-shot flow 和 future target cache，只借鉴目标 register 变体的 teacher/register loss 组织；不应直接把目标仓库的 30 层/24-head ActionDiT 或 noisy register 输入拷贝进 NVS，否则会同时破坏 checkpoint geometry、mask contract、Stage 1 输入语义和 inference algebra。
