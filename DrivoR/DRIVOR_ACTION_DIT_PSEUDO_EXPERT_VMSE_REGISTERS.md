# Action-DiT：pseudo-expert v MSE + 未来 registers

基于 `run_drivor_action_dit_dinov3_vitl_pseudo_expert_vmse_training.sh`，新增两组独立实验：

| 版本 | 训练脚本 | Scorer 输入 |
|---|---|---|
| `pseudo_expert_vmse_register` | `scripts/training/run_drivor_action_dit_dinov3_vitl_pseudo_expert_vmse_register_training.sh` | 原 DrivoR 轨迹编码 + 当前视觉/ego |
| `pseudo_expert_vmse_register_scorer` | `scripts/training/run_drivor_action_dit_dinov3_vitl_pseudo_expert_vmse_register_scorer_training.sh` | 压缩预测 registers，与轨迹特征 concat → MLP，再使用原评分网络 |

对应配置位于 `navsim/planning/script/config/common/agent/`，名称与版本一致，前缀为 `drivoR_action_dit_dinov3_vitl_`。第一版同时继承原 pseudo-expert v MSE 和 register 配置，第二版只开启 register Scorer。

## 网络与配对

- 当前单前视，输入 `672×1152`；DINOv3 ViT-L/16 的 3024 个原生 patches 经 neck 变为 256 维 K/V。默认不加 scene query。
- 加载原 DINOv3 预训练权重，基座冻结，Q/V LoRA rank32 可训练；Action-DiT 继续加载原 Action256 初始化权重。
- 每条候选含 8 个 Action tokens、32 个 Register tokens，隐维 256，沿用 30 层 Action-DiT。32 个 registers 对应未来 8 帧，每帧 4 个 DINOv3 原生 register/storage tokens。
- 自注意力中，Action 只读 Action，Register 可以读同候选的 Action 和 Register。各候选之间不互相做自注意力；两个分支都 cross-attend 到当前视觉特征。
- 训练一次联合 DiT 前向，输出轨迹 `[B,64,8,3]` 和 registers `[B,64,8,4,1024]`，没有新增 Euler 训练循环或独立的 pseudo-expert 前向。
- 全部 64 份 Action 输入仍以原始 human GT 为干净基准加独立噪声；Register 输入以原始真实未来帧的 EMA features 为干净基准加独立噪声。time/shift、Time+Ego adaLN 条件沿用原实现。

未来目标使用 `+0.5,+1.0,...,+4.0s` 的图像，teacher 输入分辨率 `384×384`。Cosmos 视频第 0 帧为当前帧，提取的零基帧下标是 `5,10,...,40`。已有帧缓存直接复用：

```text
DrivoR/exp/world4scorer_data/frames_cosmos/<token前两位>/<token>.zip
```

当前 student 的有效 Q/V 权重在每次 optimizer step 后以 decay=0.999 更新 EMA teacher；teacher 无梯度、始终 eval，输出每帧四个原生 1024 维 registers。推理使用 checkpoint 中的 student 编码当前帧，不需要 GT 未来图像，也不调用 EMA teacher。

## 轨迹监督与未来监督的范围

原有效 top4 multi-GT 对应已有 Cosmos 视频。CLOVER pseudo-expert 来自另一套离线候选，目前没有对应视频，因此两组监督分别处理：

1. 原 multi-GT：每条有效 GT 从同组 64 条轨迹中选择 v MSE 最小的 proposal；允许多条 GT 选中同一 proposal，不用匈牙利匹配。
2. Pseudo-expert：每条有效专家也从这 64 条轨迹中独立选择 v MSE 最小者，只增加轨迹 loss。
3. Register：沿用第 1 步的 proposal 索引，将同 batch、同候选位置的 registers 与该 GT 的视频特征对齐。不会给 pseudo-expert 绑定其他 GT 的视频。

所以轨迹监督目标最多为原 4 条 + 8 条 pseudo-expert；有视频的 register 监督仍对应原 top4。轨迹匹配和未来帧位置严格一一对应，缺失的未来帧通过有效 mask 排除。

Pseudo-expert 继续使用原缓存：

```text
DrivoR/exp/clover_data/pseudo_experts_k8_thr0p8
```

其筛选和 human 补入规则保持原训练行为：PDMS≥0.8、最高分起始的 FPS、最多 8 条；不满足覆盖条件且有空位时补入 human，无候选时回退 human。此处不根据 `pseudo_expert_is_human` 自动剔除 human。最近导出的固定轨迹集合是独立数据产物，本次没有把训练路径切换过去。

## Loss 与梯度

```text
L_total = L_multiGT_v + 0.5 * L_pseudoExpert_v + 1.0 * L_register_v + L_score
```

- 原 multi-GT 与 pseudo-expert 分别按自己的有效 GT 数求平均，再对 batch 平均。保留原 pseudo-expert 系数 0.5，并非合并全部 GT 后统一平均。
- Action/pseudo-expert 使用同一归一化 X-prediction、同一次采样 time。X→v 的回归误差为 `(normalize(GT)-X_pred)/sigma_safe`，`sigma_safe=max(sigma,0.1)`，沿用原时间权重。
- Register 同样使用 X-prediction 转 v MSE，在对应 GT 的有效未来帧上求平均。使用相同 time 和 sigma 下界，register 特征本身不套用轨迹坐标归一化。
- 轨迹和 register 去噪 loss 可更新共享 DiT 及当前视觉 LoRA；teacher 始终无梯度。
- 第一版 Scorer 接收 detach 后的轨迹。第二版额外将预测 registers detach，展平 `32×1024` 后压缩至 256 维，与 256 维轨迹特征拼接，再过 `512→1024→256` MLP。评分损失不能通过这两类预测反传到 DiT；当前视觉/ego 的评分梯度保持原实现。
- 验证/推理仍为 10 步 Euler，同时生成轨迹与 registers；Scorer 评分这同组 64 条候选。

## DLC：训练后自动 FP32 navtest

每个实验单独提交一个 **8 节点×每节点 8 GPU** 的 DLC 任务。DLC 分发同一启动命令并提供 rendezvous 环境；默认总 batch64、每卡 batch1、累积1、BF16训练、25 epochs、seed2。

第一版：原 Scorer。

```bash
cd /cpfs29jb/data/zhaozc/project/drive_future

DRIVOR_TRAIN_EXPERIMENT=training_drivoR_action_dit_dinov3_vitl_navtrain_multigt_pseudo_expert_vmse_register \
DRIVOR_RUN_ID=action_dit_pe_vmse_register_seed2_train_eval_run1 \
DRIVOR_SCENE_QUERIES=0 DRIVOR_PRECISION=bf16-mixed DRIVOR_EVAL_PRECISION=32-true \
bash DrivoR/scripts/training/run_drivor_action_dit_dinov3_vitl_register_train_eval.sh \
  pseudo_expert_vmse_register
```

第二版：registers 融入 Scorer。

```bash
cd /cpfs29jb/data/zhaozc/project/drive_future

DRIVOR_TRAIN_EXPERIMENT=training_drivoR_action_dit_dinov3_vitl_navtrain_multigt_pseudo_expert_vmse_register_scorer \
DRIVOR_RUN_ID=action_dit_pe_vmse_register_scorer_seed2_train_eval_run1 \
DRIVOR_SCENE_QUERIES=0 DRIVOR_PRECISION=bf16-mixed DRIVOR_EVAL_PRECISION=32-true \
bash DrivoR/scripts/training/run_drivor_action_dit_dinov3_vitl_register_train_eval.sh \
  pseudo_expert_vmse_register_scorer
```

两个入口都从头训练新的实验。训练成功后，仅节点0使用本机8卡读取本次 `last.ckpt` 做 FP32 navtest；每卡推理 batch1，CPU 评分使用 `single_machine_thread_pool`，无需 Ray。训练数据为 `data/navsim`，metric cache 为 `data/metric_cache_navtrain`；测试使用 `data/metric_cache_navtest`。

只训练时，使用表中的两个独立脚本，例如：

```bash
bash DrivoR/scripts/training/run_drivor_action_dit_dinov3_vitl_pseudo_expert_vmse_register_training.sh \
  auto_resume=false train_ckpt_path=null runtime.require_resume=false runtime.console_log_interval=25

bash DrivoR/scripts/training/run_drivor_action_dit_dinov3_vitl_pseudo_expert_vmse_register_scorer_training.sh \
  auto_resume=false train_ckpt_path=null runtime.require_resume=false runtime.console_log_interval=25
```

已有完整权重单独测试时，在单节点 DLC 任务中运行：

```bash
NPROC_PER_NODE=8 DRIVOR_EVAL_PRECISION=32-true \
bash DrivoR/scripts/evaluation/run_drivor_action_dit_dinov3_vitl_register_evaluation.sh \
  pseudo_expert_vmse_register /绝对路径/last.ckpt

NPROC_PER_NODE=8 DRIVOR_EVAL_PRECISION=32-true \
bash DrivoR/scripts/evaluation/run_drivor_action_dit_dinov3_vitl_register_evaluation.sh \
  pseudo_expert_vmse_register_scorer /绝对路径/last.ckpt
```

测试分别使用结构相同的 `register` / `register_scorer` agent 严格加载完整 checkpoint；pseudo-expert 没有新增模型参数，因此测试不加载专家缓存或未来帧。

## 验证

43 项模型、梯度、loss、旧 register 兼容性及训练后测试衔接检查通过；新增两个脚本的 10 项 DLC 启动参数检查、Hydra 配置合并及 shell 语法检查通过。

两版均完成两张 A100 的真实数据 smoke，加载真实 DINOv3-L / Action256 权重：每卡 2 个训练 step、1 个验证 batch，包含实际 CPU NAVSIM 评分、pseudo-expert loss、未来帧 teacher 编码和 EMA 更新，报告的 loss 均有限。仅验证运行链路，没有完成全量训练或全量 navtest。

记录：[validation.json](../artifacts/action_dit_pseudo_expert_vmse_register_20261019/validation.json)。
