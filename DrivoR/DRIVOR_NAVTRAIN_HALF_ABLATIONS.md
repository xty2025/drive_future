# 半量 navtrain：A23 / A27 / A28 及 A28 的 10 组独立改动

共 **13 个实验**。三个基线和十个改动均使用 `run_training.py`、同一份 42,555 场景的 `navtrain_half`；原 val 18,179 场景保留用于每轮验证。十个改动的 agent 配置都直接继承原 A28，互不叠加。

新的输出根目录是 `DrivoR/exp/navtrain_half_ablations/`；下面每个实验有独立训练、权重和 navtest 目录。RGB/OSS 缓存继续复用原目录，不会因切换输出根目录而重建视频缓存。

2026-10-24 已安排到 DLC：3 个基线＋改动 1～3 续接到 6 个正在运行的任务，改动 4～10 分别新建任务。48 个续接节点已核对，具体 Job ID、前后实验关系和输出路径见 [DLC 部署记录](exp/navtrain_half_ablations/dlc_deployment_20261024_231341/README.md)。

| 实验 | 变更 | 每候选未来 token 数 | 未来目标 | 未来 token 进 Scorer | 配置 |
|---|---|---:|---|---|---|
| A23-half | 基线：human L1 + 0.5 D8 L1；不预测未来特征 | 0 | 无 | 不进入 | [half_a23.yaml](navsim/planning/script/config/training/half_a23.yaml) |
| A27-half | 基线：human + D8 未来 register；原 Scorer | 32 | EMA DINOv3 register | 不进入 | [half_a27.yaml](navsim/planning/script/config/training/half_a27.yaml) |
| A28-half | 基线：human + D8 未来 register；融合 Scorer | 32 | EMA DINOv3 register | 进入 | [half_a28.yaml](navsim/planning/script/config/training/half_a28.yaml) |
| 改动 1 | 只将 Action-DiT 改为随机初始化，不读取给定 DiT 权重 | 32 | EMA DINOv3 register | 进入 | [half_a28_random_dit.yaml](navsim/planning/script/config/training/half_a28_random_dit.yaml) |
| 改动 2 | 未来 DINO patch 空间池化为每帧 16 个 token | 128 | EMA DINOv3 patch，4×4 pooling | 进入 | [half_a28_patch16.yaml](navsim/planning/script/config/training/half_a28_patch16.yaml) |
| 改动 3 | 顺序改为 [R32,A8]；R 能读 A，A 不能读 R | 32 | EMA DINOv3 register | 进入 | [half_a28_register_first.yaml](navsim/planning/script/config/training/half_a28_register_first.yaml) |
| 改动 4 | 保持 [A8,R32]；A/R 双向 self-attention | 32 | EMA DINOv3 register | 进入 | [half_a28_bidirectional.yaml](navsim/planning/script/config/training/half_a28_bidirectional.yaml) |
| 改动 5 | 未来使用当前 ViT＋LoRA 的即时权重；未来分支 no-grad，无 EMA | 32 | 共享当前 DINOv3，detach | 进入 | [half_a28_shared_vit.yaml](navsim/planning/script/config/training/half_a28_shared_vit.yaml) |
| 改动 6 | 当前和未来共享冻结 ViT，连 LoRA 也不更新 | 32 | 共享冻结 DINOv3，detach | 进入 | [half_a28_frozen_vit.yaml](navsim/planning/script/config/training/half_a28_frozen_vit.yaml) |
| 改动 7 | DiT 1024/FFN4096，加载指定 FastWAM Action1024 权重 | 32 | EMA DINOv3 register | 进入 | [half_a28_dit1024.yaml](navsim/planning/script/config/training/half_a28_dit1024.yaml) |
| 改动 8 | DiT 3072/FFN14336，直接加载原生 Wan2.2 5B block/time 权重 | 32 | EMA DINOv3 register | 进入 | [half_a28_wan5b.yaml](navsim/planning/script/config/training/half_a28_wan5b.yaml) |
| 改动 9 | 每张未来帧独立经冻结 Wan2.2 VAE，再池化为 16 个 token | 128 | 冻结 Wan2.2 VAE，48 维 | 进入 | [half_a28_vae16.yaml](navsim/planning/script/config/training/half_a28_vae16.yaml) |
| 改动 10 | human 和有效 D8 分组初始化 action/future 噪声锚点；loss 不变 | 32 | EMA DINOv3 register | 进入 | [half_a28_grouped_noise.yaml](navsim/planning/script/config/training/half_a28_grouped_noise.yaml) |

## 共同训练设置

- 默认 8 机 × 8 卡，单卡 batch 1，梯度累积 1，总 batch 64，seed 2，25 epochs。半量数据每轮约 665 steps，25 轮约 16,625 steps；scheduler 使用实际训练数据长度。
- 输入当前单前视，DINOv3 ViT-L，1152×672，3024 个原生 patch K/V；不注入 scene queries。除改动 6 外，DINO 基础权重冻结，Q/V LoRA rank32 更新。
- 64 个 proposal，每条 4 秒、2 Hz、8 个 `(x,y,heading)`。默认每个场景一个采样时间 `t`，64 份独立噪声；训练只执行一次联合 DiT forward，测试使用原 10 步 Euler。
- 轨迹 GT 为真实 human + 有效 D8，D8 来自 `Cosmos3forMultiGT/runtime/multigt_raw64/outputs/navtrain_diverse8_dense_0p1s_f64`（含现有 hist3 软链接）；没有旧 top4 GT，没有长 GT，没有 CLOVER pseudo-expert。
- 轨迹 loss：`L1(H) + 0.5 × mean_valid_D8 L1(D8)`。每条 GT 从全部 64 条预测中独立选择物理坐标 L1 最近者，可重复选择同一 proposal，不用匈牙利；L1 是每点三个坐标绝对误差之和，再平均 8 个点。
- 未来 loss：沿用 A28 的 X-prediction 转 v MSE，`sigma` 下界 0.1，保留原时间权重。使用轨迹匹配的同一 proposal、同一 batch 的未来预测，与该 GT 的未来目标对齐；平均有效 GT、有效时刻、槽位和通道，外部权重 1.0。D8 的 0.5 仅用于轨迹 coverage，不用于未来 loss。
- A27/A28 及其改动：未来时刻仍为 +0.5、+1.0……+4.0 秒，视频取零基帧 5、10……40。human 用真实未来前视，D8 用各自生成视频；不改变既有抽帧与配对。
- A28 的 Scorer 始终对同一组 64 条预测评分：轨迹特征和压缩后的预测未来特征 concat→MLP，后接原 256 维 DrivoR Scorer。输入的轨迹和未来预测都 detach；原视觉/ego 分支的梯度路径保留。
- 每个 epoch 保存权重并在完整 val 上进行 FP32 PDMS 验证，训练结束自动对 `last.ckpt` 进行 FP32 navtest。该 val 不参加本组训练。

## 各改动的边界

**改动 1：**只把 `action_dit.pretrained_path` 设为 null。DINOv3 仍加载预训练并训练 LoRA，Scorer 和新增适配层仍按原规则初始化。

**改动 2：**EMA DINOv3 对每张 384×384 未来帧输出 24×24 个归一化 patch；按二维网格 adaptive average pooling 到 4×4。每帧 16 个 1024 维目标，8 帧共 128 个，每条 proposal 独立一组，共 64 组。attention 序列变为 `[A8,R128]`；Scorer 的未来压缩输入随之扩展，输出仍为 256 维。

**改动 3：**实际 token 顺序是 `[R32,A8]`，RoPE 按这个实际序列从 0 计数，R 在 0～31，A 在 32～39。mask 同步移动，A→R 的读取仍被禁止，R→A 允许；输出再按类型正确切回。

**改动 4：**序列仍 `[A8,R32]`，候选内部 A/R 全互通；不同 proposal 的 self-attention 仍隔离。Cross-attention 仍以当前前视 patch 为共享 K/V。

**改动 5：**当前与未来引用同一个 ViT 对象，没有第二套 EMA 参数。当前分支照常建立梯度图，未来分支分块 `no_grad` 编码并 detach，不把目标侧 loss 直接回传给 ViT；当前视觉→DiT 的梯度链路保留，Scorer 原有梯度规则也保留。

**改动 6：**在共享 ViT 的基础上，将 native ViT 的所有参数（包含 LoRA）冻结。DINO→256 的下游 neck、ego、DiT 和 Scorer 仍训练；“冻结 ViT”不冻结整个规划器。

**改动 7：**读取指定 `ActionDiT_linear_interp_Wan22_alphascale_1024hdim_3outdim.pt`，30 层、hidden1024、FFN4096、24×128 attention 通道。该文件省略 action 输入/输出层；这两层及新增的 scene/ego 256→1024 适配层随机初始化，其余匹配的 816 个非 text 张量全部加载。Scorer 宽度保持 256。

**改动 8：**读取 `Wan-AI/Wan2.2-TI2V-5B` 原始分片，直接加载 30 个 full-width block 和 time 模块，共 816 个张量、4,976,572,416 个参数，没有降维或权重插值。视频 patch 输入/输出和 text embedding 不适配轨迹任务，因此替换为新 action/future 输入输出及 DINO/ego 256→3072 适配层；Action/R 的序列、1D RoPE、mask、X-prediction 和 loss 仍沿用 A28。

5B 的训练入口默认 ZeRO-2 分摊优化器与梯度，FP32 参数/优化器状态＋模型 forward 的 BF16 autocast；这样保留与其他实验相同的 FP32 每轮验证。EMA 在外层优化器完成更新后执行。每轮权重是 DeepSpeed checkpoint 目录；训练结束自动导出最后一轮的 FP32 单文件，再进入普通 navtest。不要把 checkpoint 目录直接交给只接收文件的原评测脚本。

**改动 9（按最终确认）：**8 张未来帧独立作为长度为 1 的视频送入冻结 Wan2.2 VAE，像素映射到 `[-1,1]`。取 VAE 原生按 mean/std 缩放的 posterior mean，不采样 posterior；每张 384×384 得到 `[48,1,24,24]`，空间池化成 4×4，最终 `[8,16,48]`。每个 proposal 的未来噪声为 `[128,48]`，总共 64 组；保留原 vMSE 和 human 噪声锚点，未来目标不反传。当前图像仍由可训练 LoRA 的 DINOv3 提取，其他实验不会切换 VAE。

**改动 10：**按实际有效的 human＋D8 初始化成对的轨迹与未来噪声输入。若共有 G 条有效 GT，每条先分 `floor(64/G)` 个，余数全部加给 human。G=9 时分配 `[8,7,7,7,7,7,7,7,7]`；只有 human 时分配 64 个。无效 D8 不占候选。action 和未来特征始终取相同 GT；噪声仍各自独立，时间采样规则、输出匹配及所有 loss 完全不变，输出仍可跨初始化组匹配。

## 训练并自动测试的命令

在 DLC 的 PyTorch 任务中设置 8 节点、每节点 8 卡，平台按既有方式注入节点环境。以下每条命令对应一个独立实验；不要把全部命令同时放在同一个节点资源内并发执行。

```bash
cd /cpfs29jb/data/zhaozc/project/drive_future

# A23-half：基线：human L1 + 0.5 D8 L1；不预测未来特征
bash DrivoR/scripts/training/run_drivor_half_a23_train_eval.sh

# A27-half：基线：human + D8 未来 register；原 Scorer
bash DrivoR/scripts/training/run_drivor_half_a27_train_eval.sh

# A28-half：基线：human + D8 未来 register；融合 Scorer
bash DrivoR/scripts/training/run_drivor_half_a28_train_eval.sh

# 改动 1：只将 Action-DiT 改为随机初始化，不读取给定 DiT 权重
bash DrivoR/scripts/training/run_drivor_half_a28_random_dit_train_eval.sh

# 改动 2：未来 DINO patch 空间池化为每帧 16 个 token
bash DrivoR/scripts/training/run_drivor_half_a28_patch16_train_eval.sh

# 改动 3：顺序改为 [R32,A8]；R 能读 A，A 不能读 R
bash DrivoR/scripts/training/run_drivor_half_a28_register_first_train_eval.sh

# 改动 4：保持 [A8,R32]；A/R 双向 self-attention
bash DrivoR/scripts/training/run_drivor_half_a28_bidirectional_train_eval.sh

# 改动 5：未来使用当前 ViT＋LoRA 的即时权重；未来分支 no-grad，无 EMA
bash DrivoR/scripts/training/run_drivor_half_a28_shared_vit_train_eval.sh

# 改动 6：当前和未来共享冻结 ViT，连 LoRA 也不更新
bash DrivoR/scripts/training/run_drivor_half_a28_frozen_vit_train_eval.sh

# 改动 7：DiT 1024/FFN4096，加载指定 FastWAM Action1024 权重
bash DrivoR/scripts/training/run_drivor_half_a28_dit1024_train_eval.sh

# 改动 8：DiT 3072/FFN14336，直接加载原生 Wan2.2 5B block/time 权重
bash DrivoR/scripts/training/run_drivor_half_a28_wan5b_train_eval.sh

# 改动 9：每张未来帧独立经冻结 Wan2.2 VAE，再池化为 16 个 token
bash DrivoR/scripts/training/run_drivor_half_a28_vae16_train_eval.sh

# 改动 10：human 和有效 D8 分组初始化 action/future 噪声锚点；loss 不变
bash DrivoR/scripts/training/run_drivor_half_a28_grouped_noise_train_eval.sh

```

只训练时，将 `_train_eval.sh` 换为 `_training.sh`，各版本都已提供。默认 RunID 使用 DLC Job ID，本地默认 `half_<variant>_seed2_run1`；重复实验可显式设置 `DRIVOR_RUN_ID`。这些脚本只在当前任务内启动，不创建 DLC 任务。

单独测试任一版本：

```bash
bash DrivoR/scripts/evaluation/run_drivor_half_ablation_evaluation.sh \
  a28 /absolute/path/to/last.ckpt

# Wan 5B 支持直接传其 DeepSpeed checkpoint 目录，脚本会先导出 FP32 文件
bash DrivoR/scripts/evaluation/run_drivor_half_ablation_evaluation.sh \
  a28_wan5b /absolute/path/to/last.ckpt
```

## 输出与复核

```text
DrivoR/exp/navtrain_half_ablations/
  experiments.json
  ke/training_drivoR_half_<variant>/<RunID>/
    run_training.log
    val_pdms.csv
    lightning_logs/version_*/checkpoints/
  ke/eval_navtest_drivoR_half_<variant>/<RunID>/
    pdms_summary.txt
    *.csv
```

所有绝对路径、配置和入口映射见 [experiments.json](exp/navtrain_half_ablations/experiments.json)。数据采样及分布验证见 [半量数据说明](DRIVOR_NAVTRAIN_HALF.md)。默认复用 `exp/world4scorer_data/frames_diverse8` 的 RGB ZIP，不缓存随 LoRA/EMA 变化的目标特征。

## 本地验证

- 75 项新检查通过：13 份配置、26 个训练入口和 13 个独立 FP32 测试入口路由、13 个缩小网络的单次训练 forward/反传/FP32推理及 checkpoint 往返，以及注意力方向、二维池化、目标 detach、分组身份和原匹配/loss 不变性。
- 原有 register/D8/普通 Transformer 路径 46 项检查通过，默认网络参数键和旧 checkpoint 路径兼容。
- 已逐张量核对真实 FastWAM1024、Wan5B 和 WanVAE 权重结构，不进行文件哈希检查。实际大模型训练吞吐、显存峰值和 NAVSIM 成绩不由这些 smoke 推断。
- 缩小网络、64 proposals 的双 GPU ZeRO-2 smoke 已完成一次训练、FP32 验证和 checkpoint 保存，EMA 更新次数为 1；不代表原生 5B 的完整显存/吞吐验证。
- ZeRO-2 最后一轮分片已导出为 FP32 单文件，183 个 state 项与原保存状态完全一致，并通过原 Agent 的 strict 加载与 `[1,64,8,3]` 轨迹推理检查。导出器兼容当前 PyTorch 2.7 与 Lightning 2.4。
- 真实 Wan2.2 VAE 权重已完成单张 384×384 随机图像的 GPU 编码，输出 `[1,1,8,16,48]`（仅一个有效时刻），其余无效时刻为零、所有数值有限且没有目标侧梯度。
- 其他 GPU 的占卡/任务进程保持原样；本地轻量 GPU 验证仅使用空闲 GPU 0、1。

详细验证记录位于 `exp/navtrain_half_ablations/validation_20261024/`。
