# DrivoR 与 Action-DiT：当前启动命令

各脚本均使用 `run_training.py`。在 **PAI-DLC 配置 8 个节点、每节点 8 张 GPU**，每个模型分别提交一个任务，启动命令各填写一次。DLC 自动分配节点、分发命令并注入 `WORLD_SIZE/RANK/MASTER_ADDR/MASTER_PORT`，不需要 hostfile、手写 IP 或逐节点启动。

默认均为 **8×8 卡、每卡 batch 1、累积 1、总 batch 64、BF16、25 epochs、seed 2、峰值 LR 2e-4**。各入口都启用 fused AdamW、每卡 4 个数据 workers、2 个 Oracle 进程、CPU threads 1、prefetch 2，以及训练模型的 `seed + global_rank` 随机流。

| 模型 | 脚本 | 当前行为 |
|---|---|---|
| 原版 DrivoR | `run_drivor_training.sh` | 四路当前图像，原版轨迹生成器与 Scorer |
| DrivoR + CLOVER Stage-1 coverage | `run_drivor_pseudo_expert_training.sh` | 保留原普通/长 GT 与评分 loss；最终 64 proposals 额外覆盖最多 8 条高分 pseudo-expert，权重 0.5 |
| Action-DiT | `run_drivor_action_dit_training.sh` | 当前单前视；训练一次 X-prediction，普通/长 GT 合批；验证/推理 10 步 Euler |
| Action-DiT 多视图 query 版 | `run_drivor_action_dit_multiview_training.sh` | 原 DrivoR 四视角、每视角 16 个 query；DiT 和 Scorer 的视觉 K/V 均为 64 tokens |
| 原版 DrivoR 多 GT | `run_drivor_multigt_training.sh` | 原视觉/轨迹网络；human 最近 L1 ×1.0＋其他有效 GT 平均最近 L1 ×0.5 |
| Action-DiT 单前视多 GT | `run_drivor_action_dit_multigt_training.sh` | 原始 human GT 构造全部 64 个输入；各有效 GT 从同组输出匹配 v loss |
| Action-DiT 多视图多 GT | `run_drivor_action_dit_multiview_multigt_training.sh` | 四视角 query 特征；与单前视多 GT 相同的输入和监督规则 |
| 原版 DrivoR 匈牙利多 GT | `run_drivor_multigt_hungarian_training.sh` | 一对一分配，按实际有效 GT 数平均 L1 |
| Action-DiT 单前视匈牙利多 GT | `run_drivor_action_dit_multigt_hungarian_training.sh` | 原始 human GT 加噪；一对一分配后平均 v MSE |
| Action-DiT 多视图匈牙利多 GT | `run_drivor_action_dit_multiview_multigt_hungarian_training.sh` | 四视角 query 特征；同样的一对一等权监督 |
| Action-DiT 单前视加权多 GT | `run_drivor_action_dit_multigt_weighted_training.sh` | human v loss ×1.0＋非 human GT 平均 v loss ×0.5 |
| Action-DiT 多视图加权多 GT | `run_drivor_action_dit_multiview_multigt_weighted_training.sh` | 四视角 query 特征；同样的 human/非 human 分组权重 |
| Action-DiT + DINOv3 ViT-S | `run_drivor_action_dit_dinov3_vits_training.sh` | 冻结 DINOv3-S，Q/V LoRA；默认单前视 patches，Scene queries 可选 |
| Action-DiT + DINOv3 ViT-L | `run_drivor_action_dit_dinov3_vitl_training.sh` | 冻结 DINOv3-L，Q/V LoRA；有效 GT 等权平均 |
| Action-DiT + DINOv3-L 未来 registers | `run_drivor_action_dit_dinov3_vitl_register_training.sh` | 64 对轨迹/32-register；EMA teacher，多 GT 同索引监督，原 Scorer |
| Action-DiT + DINOv3-L register Scorer | `run_drivor_action_dit_dinov3_vitl_register_scorer_training.sh` | Registers 压缩到256，与轨迹特征 concat→MLP 评分，两类预测 detach |
| Action-DiT + DINOv3-L register Scorer 单 GT | `run_drivor_action_dit_dinov3_vitl_register_scorer_single_gt_training.sh` | Human GT best-of-64，同候选 registers 对齐真实 human 未来帧；不读取 Cosmos 多 GT |
| Action-DiT + DINOv3-L register 分阶段 | `run_drivor_action_dit_dinov3_vitl_register_staged_training.sh` | 先 GT 条件未来去噪，再联合 Action/Register/Scorer；延续 EMA |
| Action-DiT + DINOv3-L 原 top4＋poor4 | `run_drivor_action_dit_dinov3_vitl_multigt_poor_training.sh` | 合并最多 8 条有效 GT，原最近邻 v loss 等权平均 |
| Action-DiT + DINOv3-L 原始单 GT | `run_drivor_action_dit_dinov3_vitl_single_gt_training.sh` | 仅原始 human GT，best-of-64 v loss |
| Action-DiT + DINOv3-L human＋长 GT | `run_drivor_action_dit_dinov3_vitl_single_gt_long_training.sh` | 普通/长 GT 分别 best-of-64，128 输入合批一次 DiT 前向 |
| Action-DiT + DINOv3-L 多 GT + Bank | `run_drivor_action_dit_dinov3_vitl_bank_training.sh` | 保留原 multi-GT v loss，Scorer 额外处理 16 条 Bank 候选，L_Bank ×0.5 |
| Action-DiT + DINOv3-L 多 GT + pseudo-expert | `run_drivor_action_dit_dinov3_vitl_pseudo_expert_training.sh` | 保留原 multi-GT v loss，同一次 X-prediction 额外计算 CLOVER L1 coverage ×0.5 |
| Action-DiT + DINOv3-L 多 GT + pseudo-expert v MSE | `run_drivor_action_dit_dinov3_vitl_pseudo_expert_vmse_training.sh` | Pseudo-expert 也复用原 multi-GT X→v MSE，独立平均后 ×0.5；旧 L1 版保留 |
| Action-DiT + pseudo-expert v MSE + registers | `run_drivor_action_dit_dinov3_vitl_pseudo_expert_vmse_register_training.sh` | 同次前向联合生成 64 对轨迹/registers；原 Scorer；未来监督对应原 top4 视频 |
| Action-DiT + pseudo-expert v MSE + register Scorer | `run_drivor_action_dit_dinov3_vitl_pseudo_expert_vmse_register_scorer_training.sh` | 上一版基础上，预测 registers 压缩后与轨迹特征 concat→MLP 评分，两者 detach |
| Action-DiT + V-JEPA 2 ViT-L | `run_drivor_action_dit_vjepa2_vitl_training.sh` | 冻结官方预训练 ViT-L，Q/V LoRA；单前视 patches、等权多 GT，queries 可选 |
| Action-DiT + V-JEPA 2.1 ViT-L | `run_drivor_action_dit_vjepa2_1_vitl_training.sh` | 官方 EMA 编码器的原生单图分支，Q/V LoRA；等权多 GT，queries 可选 |
| 原版 DrivoR + Bank | `run_drivor_bank_training.sh` | 原四视角DINOv2/query；新增World4Scorer的L_Bank |
| 原版 DrivoR + V-JEPA 2-L | `run_drivor_vjepa2_vitl_training.sh` | 当前单前视3024原生patches；原轨迹生成器与Scorer |
| 原版 DrivoR + V-JEPA 2.1-L | `run_drivor_vjepa2_1_vitl_training.sh` | 同上，使用2.1原生单图分支 |
| 原版 DrivoR + V-JEPA 2 + 未来预测 | `run_drivor_vjepa2_world_training.sh` | Bank＋独立预测器，预测GT最近proposal的8时刻特征 |
| 原版 DrivoR + V-JEPA 2 + 多GT未来预测 | `run_drivor_vjepa2_world_multigt_training.sh` | 每GT独立最近邻，允许重复proposal，等权平均；Cosmos对应未来帧 |

以下命令显式指定实验目录，避免继承 shell 中其他模型的 `DRIVOR_EXPERIMENT`。需要分段计时时，可设置 `DRIVOR_PERF_ENABLED=true DRIVOR_PERF_SAMPLE_EVERY=10`；不添加额外的跨 rank 统计同步。

## 新增：DINOv3-L Action-DiT 未来 registers 与 EMA teacher

当前视觉基座仍冻结、Q/V LoRA rank32 可训练。每条候选生成 8 个轨迹点和 8×4 个原生 DINOv3 register 特征；Action 不能读 Register，Register 可以读 Action。所有 64 份 Action/Register 加噪基准来自原始 human GT/真实未来帧；各 GT 选择最近轨迹后，以同 batch、同候选索引的 registers 对齐对应 Cosmos 视频。

三个训练后自动 FP32 测试入口统一为 `run_drivor_action_dit_dinov3_vitl_register_train_eval.sh register|register_scorer|register_staged`。第三版默认 Stage1=5、Stage2=25 epochs。完整结构、loss、EMA、阶段切换和 DLC 命令见 [未来 register 说明](../../DRIVOR_ACTION_DIT_DINOV3_REGISTERS.md)。

register Scorer 的单 human GT 对照使用同一训练后测试入口，参数为 `register_scorer_single_gt`；完整命令见 [单 GT 对照](../../DRIVOR_ACTION_DIT_REGISTER_SINGLE_GT.md)。仍输出和评分64条轨迹，每条配32个未来 registers。

pseudo-expert v MSE 的两个联合 register 版本也使用该入口，参数分别为 `pseudo_expert_vmse_register` / `pseudo_expert_vmse_register_scorer`。两版默认 `L_multiGT_v + 0.5 L_PE_v + L_register_v + L_score`；配套视频只对应原 top4 GT。完整训练、FP32 测试命令见 [pseudo-expert + registers](../../DRIVOR_ACTION_DIT_PSEUDO_EXPERT_VMSE_REGISTERS.md)。

## 新增：DINOv3-L Action-DiT 的 GT 对照实验

基于现有 DINOv3-L Action-DiT 配置，增加原 top4＋poor4、原始 human 单 GT、human＋长 GT 三组监督。训练成功后自动 FP32 navtest 的入口为 `run_drivor_action_dit_dinov3_vitl_gt_train_eval.sh`，版本参数分别是 `multigt_poor`、`single_gt`、`single_gt_long`。

三组均保持 8 机×8 卡、总 batch 64，图像与 Scorer 架构不变。完整命令及 loss 说明见 [GT 对照实验](../../DRIVOR_ACTION_DIT_DINOV3_GT_VARIANTS.md)。

## 新增：DINOv3-L Action-DiT 多 GT 的两个独立辅助 loss 实验

新增的 pseudo-expert v MSE 对照使用 `run_drivor_action_dit_dinov3_vitl_pseudo_expert_vmse_training.sh`；训练后自动 FP32 测试使用 `run_drivor_action_dit_dinov3_vitl_aux_train_eval.sh pseudo_expert_vmse`。它复用原 multi-GT 的归一化、时间权重、sigma 下界和最近邻 v MSE，原 L1 脚本不变。完整命令见 [pseudo-expert v MSE](../../DRIVOR_ACTION_DIT_PSEUDO_EXPERT_VMSE.md)。

基于已训练的 `training_drivoR_action_dit_dinov3_vitl_navtrain_multigt` 配置，分别只加 Bank loss 或 pseudo-expert coverage。两个启动脚本各提交一个 DLC 8 机×8 卡任务；默认保持当前单前视原生 patches、原多 GT 等权最近邻 v loss、一次训练 DiT 前向和原预训练初始化。数据缓存已准备完成。

两个 `*_training.sh` 只训练；需要训完自动测试时，使用 `run_drivor_action_dit_dinov3_vitl_aux_train_eval.sh bank` 或 `run_drivor_action_dit_dinov3_vitl_aux_train_eval.sh pseudo_expert`，训练成功后节点 0 自动用本次 `last.ckpt` 进行 FP32 navtest。完整命令、loss、梯度边界见 [两个辅助 loss 实验](../../DRIVOR_ACTION_DIT_DINOV3_AUXILIARY.md)。

## 新增：DrivoR + CLOVER pseudo-expert coverage

沿用当前四视角 DINOv2 DrivoR，仅新增最终 proposals 的专家 coverage loss。专家来自已下载的 CLOVER 官方轨迹包，经 PDMS≥0.8 和最远点采样选择；全量紧凑缓存已准备完成。每专家独立选择最近 proposal，按有效专家数均分，默认权重 0.5。

```bash
cd /cpfs29jb/data/zhaozc/project/drive_future
DRIVOR_EXPERIMENT=training_drivoR_pseudo_expert_navtrain \
DRIVOR_RUN_ID=drivor_pseudo_expert_seed2_run1 \
bash DrivoR/scripts/training/run_drivor_pseudo_expert_training.sh \
  auto_resume=false train_ckpt_path=null runtime.require_resume=false \
  runtime.console_log_interval=25
```

8 机 × 8 卡，总 batch 64。同一个任务训完自动测试时，将入口换成 `run_drivor_pseudo_expert_train_eval.sh`；仅节点 0 在训练成功后评测本次 run 的 `last.ckpt`。独立测试入口为 `scripts/evaluation/run_drivor_pseudo_expert_evaluation.sh <checkpoint>`。训练保持 BF16，两种测试入口均默认 FP32（`32-true`）。数据来源、选择规则、loss、缓存重建及完整命令见 [DrivoR + pseudo-expert 说明](../../DRIVOR_PSEUDO_EXPERT.md)。

## 新增：原版 DrivoR 的四组 World4Scorer / V-JEPA 实验

这些入口使用原版 DrivoR 轨迹生成器。第1组仅增加CLOVER候选Bank监督；第2组包含V-JEPA2和2.1两个单前视骨干；第3、4组增加独立未来预测器。未来分支只处理与GT匹配的预测proposal，当前场景K/V直接detach，轨迹路径正常反传。第4组最近邻匹配允许不同GT选中同一proposal，不使用匈牙利或human/其他GT的不同权重。

完整配置、Bank来源、8时刻抽帧、梯度边界和五条DLC命令见 [四组实验说明](../../DRIVOR_WORLD4SCORER_VARIANTS.md)。例如第4组的一条命令完成未来特征准备、训练及主节点测试：

```bash
cd /cpfs29jb/data/zhaozc/project/drive_future
DRIVOR_RUN_ID=drivor_vjepa2_world_multigt_seed2_run1 \
bash DrivoR/scripts/training/run_drivor_world_train_eval.sh vjepa2_world_multigt
```

其他版本名为 `bank`、`vjepa2_vitl`、`vjepa2_1_vitl`、`vjepa2_world`。使用8节点×8GPU，总batch64；第3、4组自动补齐冻结teacher特征后再训练。Bank在这些实验之间共用，只需准备一次。

## 新增：V-JEPA 2.1 ViT-L

使用官方 2.1 `ema_encoder`，原参数冻结、Q/V LoRA rank 32。默认当前单前视 672×1152，走预训练的单图 patch embedding 和 image modality embedding，输出 3024 个 patches。位置编码保留 2.1 原生的 RoPE 坐标缩放。

```bash
cd /cpfs29jb/data/zhaozc/project/drive_future
DRIVOR_EXPERIMENT=training_drivoR_action_dit_vjepa2_1_vitl_navtrain_multigt \
DRIVOR_RUN_ID=action_dit_vjepa2_1_vitl_seed2_run1 \
DRIVOR_SCENE_QUERIES=0 DRIVOR_PRECISION=bf16-mixed \
bash DrivoR/scripts/training/run_drivor_action_dit_vjepa2_1_vitl_training.sh \
  auto_resume=false train_ckpt_path=null runtime.require_resume=false \
  runtime.console_log_interval=25
```

默认沿用有效 GT 等权平均、一次 DiT 训练前向和 10 步 Euler 推理。`DRIVOR_SCENE_QUERIES=16` 开启每视角 16 queries，相机仍由配置独立控制。权重路径、架构、四视角及训练后自动测试的命令见 [V-JEPA 2.1 说明](../../DRIVOR_ACTION_DIT_VJEPA2_1.md)。

## 新增：V-JEPA 2 ViT-L

官方 `vitl.pt` 已下载，默认加载其中的 `target_encoder`，原有参数冻结、Q/V LoRA rank 32。当前前视图直接 resize 到 672×1152，将同一张图复制两次适配预训练的时间 tubelet，得到 3024 个原生 patches；实际仍只读取当前一帧。位置编码使用官方时空 RoPE，按实际网格生成。

```bash
cd /cpfs29jb/data/zhaozc/project/drive_future
DRIVOR_EXPERIMENT=training_drivoR_action_dit_vjepa2_vitl_navtrain_multigt \
DRIVOR_RUN_ID=action_dit_vjepa2_vitl_seed2_run1 \
DRIVOR_SCENE_QUERIES=0 DRIVOR_PRECISION=bf16-mixed \
bash DrivoR/scripts/training/run_drivor_action_dit_vjepa2_vitl_training.sh \
  auto_resume=false train_ckpt_path=null runtime.require_resume=false \
  runtime.console_log_interval=25
```

默认单前视、有效 GT 等权平均，Action-DiT 仍训练一次前向，Scorer 和其余配方沿用。`DRIVOR_SCENE_QUERIES=16` 开启每视角 16 个 query；四视角再追加 `'agent.config.cam_b0=[3]' 'agent.config.cam_l0=[3]' 'agent.config.cam_r0=[3]'` 并换独立实验名。完整架构、配置、训练后自动测试命令和双卡 smoke 记录见 [V-JEPA 2 说明](../../DRIVOR_ACTION_DIT_VJEPA2.md)。

## 新增：DINOv3 ViT-S / ViT-L

两份 ModelScope 权重已下载，使用本地官方 DINOv3 代码；预训练原参数冻结，只训练 Q/V LoRA、256 维投影与启用时的 Scene queries。默认使用单前视原生 patches，轨迹 loss 按实际有效 GT 数等权平均，human GT 不单独加权；匹配仍为最近邻，Scorer 与其他训练设置沿用。

ViT-S：

```bash
cd /cpfs29jb/data/zhaozc/project/drive_future
DRIVOR_EXPERIMENT=training_drivoR_action_dit_dinov3_vits_navtrain_multigt \
DRIVOR_RUN_ID=action_dit_dinov3_vits_seed2_run1 \
DRIVOR_SCENE_QUERIES=0 DRIVOR_PRECISION=bf16-mixed \
bash DrivoR/scripts/training/run_drivor_action_dit_dinov3_vits_training.sh \
  auto_resume=false train_ckpt_path=null runtime.require_resume=false \
  runtime.console_log_interval=25
```

ViT-L：

```bash
cd /cpfs29jb/data/zhaozc/project/drive_future
DRIVOR_EXPERIMENT=training_drivoR_action_dit_dinov3_vitl_navtrain_multigt \
DRIVOR_RUN_ID=action_dit_dinov3_vitl_seed2_run1 \
DRIVOR_SCENE_QUERIES=0 DRIVOR_PRECISION=bf16-mixed \
bash DrivoR/scripts/training/run_drivor_action_dit_dinov3_vitl_training.sh \
  auto_resume=false train_ckpt_path=null runtime.require_resume=false \
  runtime.console_log_interval=25
```

`DRIVOR_SCENE_QUERIES=16` 开启每视角 16 个查询；如需四视图，再追加 `'agent.config.cam_b0=[3]' 'agent.config.cam_l0=[3]' 'agent.config.cam_r0=[3]'`，并使用独立 run ID / 实验名。只打开 query 开关时仍是单前视。

两份 DINOv3 配置均设置 `image_size: [1152, 672]`（W,H），数据侧直接 resize 到 672×1152，骨干不做 padding，得到 3024 个 patch；二维 RoPE 按实际网格生成。权重来源、冻结范围、query 模式完整命令及验证见 [DINOv3 说明](../../DRIVOR_ACTION_DIT_DINOV3.md)。

DINOv3 S/L 的“训练成功后主节点自动测试”完整 DLC 命令见 [DINOv3 说明第 6 节](../../DRIVOR_ACTION_DIT_DINOV3.md#6-训练成功后自动测试-navtest)，同一模板只需切换 `vits` / `vitl`。测试使用对应的 DINOv3 agent、单前视、每卡 batch 1 和 CPU 进程池。

## 新增：Action-DiT 的 human / 非 human GT 权重

加权入口使用 `L_human + 0.5 * mean(L_有效非human_GT)`。human 主项读取原始 NAVSIM GT，导出集合中标为 `human_log` 的行从辅助项排除；仅 human 的场景辅助 loss 为 0。匹配保持独立最近邻，v MSE、时间权重、原始 human GT 加噪、单次 64 候选前向和 Scorer 均沿用现有实现。

单前视版（DLC 8 机 × 8 卡，总 batch 64）：

```bash
cd /cpfs29jb/data/zhaozc/project/drive_future
DRIVOR_EXPERIMENT=training_drivoR_action_dit_navtrain_multigt_weighted \
DRIVOR_RUN_ID=action_dit_multigt_weighted_seed2_run1 \
DRIVOR_PRECISION=bf16-mixed \
bash DrivoR/scripts/training/run_drivor_action_dit_multigt_weighted_training.sh \
  auto_resume=false train_ckpt_path=null runtime.require_resume=false \
  runtime.console_log_interval=25
```

非 human 组的平均 loss 权重可在末尾覆盖，例如 `agent.config.multi_gt.other_weight=0.2`。有 3 条非 human GT 时，默认每条的系数为 `0.5/3`，human 系数为 `1.0`。

多视图新入口为 `run_drivor_action_dit_multiview_multigt_weighted_training.sh`。两种配置、完整命令与缓存说明见 [Action-DiT 加权多 GT](../../DRIVOR_ACTION_DIT_MULTI_GT_WEIGHTED.md)。原等权和匈牙利入口保持原行为，使用新的 weighted 入口才启用该配方。

## 新增：匈牙利一对一匹配，按有效 GT 数平均

新配置使用 `matching=hungarian`、`weighting=equal`。4 条有效 GT 匹配 4 条不同的 proposal，loss 为这 4 对误差的平均；不足 4 条时按实际条数计算。不单独增加 human loss，不使用 human/辅助的 1.0/0.5 配比。Scorer 仍监督全部 64 条，原版保持物理 L1，Action-DiT 保持归一化 v MSE 和原始 human GT 加噪。

原版 DrivoR 的 DLC 启动命令（8 机 × 8 卡，总 batch 64）：

```bash
cd /cpfs29jb/data/zhaozc/project/drive_future
DRIVOR_EXPERIMENT=training_drivoR_Nav1_navtrain_multigt_hungarian \
DRIVOR_RUN_ID=drivor_multigt_hungarian_seed2_run1 \
DRIVOR_PRECISION=bf16-mixed \
bash DrivoR/scripts/training/run_drivor_multigt_hungarian_training.sh \
  auto_resume=false train_ckpt_path=null runtime.require_resume=false \
  runtime.console_log_interval=25
```

需要训练成功后自动测试时，将上述命令最后一行替换为：

```bash
  runtime.console_log_interval=25 && \
if [ "${NODE_RANK:-${RANK:-0}}" = "0" ]; then
  DRIVOR_EXPERIMENT=eval_navtest_drivor_multigt_hungarian \
  DRIVOR_RUN_ID=drivor_multigt_hungarian_seed2_run1 \
  DRIVOR_EVAL_WORKER=single_machine_thread_pool \
  DRIVOR_SCORE_WORKERS=32 NPROC_PER_NODE=8 \
  bash DrivoR/scripts/evaluation/run_drivor_pdm_score_evaluation.sh \
    DrivoR/exp/ke/training_drivoR_Nav1_navtrain_multigt_hungarian/drivor_multigt_hungarian_seed2_run1/lightning_logs/version_0/checkpoints/last.ckpt
fi
```

训练后的测试只在 DLC 主节点运行，使用该节点 8 卡和 CPU 进程池。checkpoint 路径对应新建 run 的 `version_0`；更换实验目录或 version 时同步调整。

两个 Action-DiT 匈牙利入口及其完整 DLC 命令见[匈牙利多 GT 说明](../../DRIVOR_MULTI_GT_HUNGARIAN.md)。它们保持单次 64 候选 X-prediction 训练、原始 human GT 加噪、相同初始化与 10 步 Euler 推理。旧的最近邻和加权入口保持原行为，实验目录彼此独立。

## 新增：使用有效多 GT 监督

三个独立的新入口读取 `drive_future/multi_traj_gt_data/navtrain_multigt_top4_dense_0p1s_f64_8gpu`。用 `gt_valid_mask & ~gt_padded` 得到真实的 1–4 条 GT，关闭额外长 GT 监督；每条 GT 独立从全部 64 条输出中取最近者，多个 GT 可以选同一 proposal。Scorer 仍评分全部 64 条，标签、特征和梯度截断沿用各自版本。

原版 DrivoR 的 [drivoR_multigt.yaml](../../navsim/planning/script/config/common/agent/drivoR_multigt.yaml) 默认对齐 Drive-JEPA 的 loss 配比：`weighting=drive_jepa`、`human_weight=1.0`、`other_weight=0.5`，即 `L_human + 0.5 * mean(L_其他有效GT)`。按 `gt_labels=human_log` 排除辅助集合中的 human 行，只有 human 时辅助 loss 为 0。Action-DiT 两个配置继续对有效 GT 等权平均。

Action-DiT 按确认的设置使用 `input_gt_assignment=original`：64 个输入全部由原始 human GT 和独立噪声构造，每场景共享 t，一次 DiT 前向后才使用多 GT 匹配；不按多 GT 分配加噪输入。

原版 DrivoR：

```bash
cd /cpfs29jb/data/zhaozc/project/drive_future
DRIVOR_EXPERIMENT=training_drivoR_Nav1_navtrain_multigt_jepa_weighted \
DRIVOR_RUN_ID=drivor_multigt_jepa_seed2_run1 \
DRIVOR_PRECISION=bf16-mixed \
bash DrivoR/scripts/training/run_drivor_multigt_training.sh \
  auto_resume=false train_ckpt_path=null runtime.require_resume=false \
  runtime.console_log_interval=25
```

原版脚本默认输出到新的 `training_drivoR_Nav1_navtrain_multigt_jepa_weighted` 实验目录。需要更低辅助权重时，在命令末尾追加 `agent.config.multi_gt.other_weight=0.25`。复现旧等权方案可用 `agent.config.multi_gt.weighting=equal`；模型参数结构兼容，使用独立 run ID 区分实验。

### 原版多 GT：训练完成后自动测试 navtest

以下是可直接粘贴到 DLC 的完整命令，资源仍为 8 机 × 8 卡。训练成功后，只有主节点使用本机 8 卡评测最终的 `last.ckpt`，其他节点完成任务。PDM 评分使用主节点的 32 个 CPU 进程，`DRIVOR_EVAL_WORKER=single_machine_thread_pool` 配合 `use_process_pool=true`，不启动 Ray。

```bash
cd /cpfs29jb/data/zhaozc/project/drive_future

DRIVOR_EXPERIMENT=training_drivoR_Nav1_navtrain_multigt_jepa_weighted \
DRIVOR_RUN_ID=drivor_multigt_jepa_seed2_run1 \
DRIVOR_PRECISION=bf16-mixed \
bash DrivoR/scripts/training/run_drivor_multigt_training.sh \
  auto_resume=false train_ckpt_path=null runtime.require_resume=false \
  runtime.console_log_interval=25 && \
if [ "${NODE_RANK:-${RANK:-0}}" = "0" ]; then
  DRIVOR_EXPERIMENT=eval_navtest_drivor_multigt_jepa_weighted \
  DRIVOR_RUN_ID=drivor_multigt_jepa_seed2_run1 \
  DRIVOR_EVAL_WORKER=single_machine_thread_pool \
  DRIVOR_SCORE_WORKERS=32 NPROC_PER_NODE=8 \
  bash DrivoR/scripts/evaluation/run_drivor_pdm_score_evaluation.sh \
    DrivoR/exp/ke/training_drivoR_Nav1_navtrain_multigt_jepa_weighted/drivor_multigt_jepa_seed2_run1/lightning_logs/version_0/checkpoints/last.ckpt
fi
```

这里的 checkpoint 路径对应上面使用新 run ID 从头启动的 `version_0`；更换实验名、run ID 或恢复到其他 Lightning version 时，同步改成实际 checkpoint 路径。评测使用普通 `agent=drivoR`，加权多 GT 不改变推理网络，也不需要 navtest 多 GT 文件。

测试读取 `data/navsim/navsim_logs/test` 和 `data/metric_cache_navtest`，结果写入 `DrivoR/exp/ke/eval_navtest_drivor_multigt_jepa_weighted/drivor_multigt_jepa_seed2_run1/` 的逐场景 CSV 和 `pdms_summary.txt`。`&&` 保证训练失败时不继续测试；主节点条件避免八台机器重复评测同一份数据。

Action-DiT 单前视：

```bash
cd /cpfs29jb/data/zhaozc/project/drive_future
DRIVOR_EXPERIMENT=training_drivoR_action_dit_navtrain_multigt \
DRIVOR_RUN_ID=action_dit_multigt_seed2_run1 \
bash DrivoR/scripts/training/run_drivor_action_dit_multigt_training.sh \
  auto_resume=false train_ckpt_path=null runtime.require_resume=false \
  runtime.console_log_interval=25
```

Action-DiT 多视图、每视角 16 queries：

```bash
cd /cpfs29jb/data/zhaozc/project/drive_future
DRIVOR_EXPERIMENT=training_drivoR_action_dit_multiview_navtrain_multigt \
DRIVOR_RUN_ID=action_dit_multiview_multigt_seed2_run1 \
bash DrivoR/scripts/training/run_drivor_action_dit_multiview_multigt_training.sh \
  auto_resume=false train_ckpt_path=null runtime.require_resume=false \
  runtime.console_log_interval=25
```

旧的三个入口保持原监督方式。新入口仍使用 8×8 卡、global batch 64 和已有的性能设置；实验目录单独命名。可用 `DRIVOR_MULTI_GT_ROOT` 覆盖 GT 路径。缺少某场景的导出 GT 时，默认退回原始单 GT 并记录 `multi_gt_fallback_fraction`；需要严格覆盖时追加 `agent.config.multi_gt.missing_policy=error`。

配置、X→V loss 公式、缓存和 checkpoint 兼容性见[多 GT 详细说明](../../DRIVOR_MULTI_GT.md)。新脚本在下一次启动时生效。

## 1. 接续现有两个 multinode 任务

旧任务完成 checkpoint 保存并停止后，分别用下面命令重新提交 DLC。它们会从对应 `multinode` 目录自动找最新 `last.ckpt`；没有 checkpoint 会报错，不会静默重新训练。

原版 DrivoR：

```bash
cd /cpfs29jb/data/zhaozc/project/drive_future
DRIVOR_EXPERIMENT=training_drivoR_Nav1_navtrain_traj_long_25epochs \
DRIVOR_RUN_ID=multinode DRIVOR_PRECISION=bf16-mixed \
DRIVOR_PERF_ENABLED=true DRIVOR_PERF_SAMPLE_EVERY=10 \
bash DrivoR/scripts/training/run_drivor_training.sh \
  auto_resume=true runtime.require_resume=true runtime.console_log_interval=25
```

Action-DiT：

```bash
cd /cpfs29jb/data/zhaozc/project/drive_future
DRIVOR_EXPERIMENT=training_drivoR_action_dit_navtrain \
DRIVOR_RUN_ID=multinode DRIVOR_PRECISION=bf16-mixed \
DRIVOR_PERF_ENABLED=true DRIVOR_PERF_SAMPLE_EVERY=10 \
bash DrivoR/scripts/training/run_drivor_action_dit_training.sh \
  auto_resume=true runtime.require_resume=true runtime.console_log_interval=25
```

对应输出目录分别为：

```text
DrivoR/exp/ke/training_drivoR_Nav1_navtrain_traj_long_25epochs/multinode/
DrivoR/exp/ke/training_drivoR_action_dit_navtrain/multinode/
```

续训使用当前源码；Action-DiT 会采用单次 X-prediction 训练和新的 rank 随机流。模型/optimizer/scheduler 的恢复状态来自 checkpoint，随机流的精确位置不在当前 checkpoint 恢复保证内。

## 2. 从头开始两个新实验

使用新 run ID，并显式关闭恢复。原版 DrivoR：

```bash
cd /cpfs29jb/data/zhaozc/project/drive_future
DRIVOR_EXPERIMENT=training_drivoR_Nav1_navtrain_traj_long_25epochs \
DRIVOR_RUN_ID=drivor_nav1_bf16_seed2_fast_run1 DRIVOR_PRECISION=bf16-mixed \
DRIVOR_PERF_ENABLED=true DRIVOR_PERF_SAMPLE_EVERY=10 \
bash DrivoR/scripts/training/run_drivor_training.sh \
  auto_resume=false train_ckpt_path=null runtime.require_resume=false \
  runtime.console_log_interval=25
```

Action-DiT：

```bash
cd /cpfs29jb/data/zhaozc/project/drive_future
DRIVOR_EXPERIMENT=training_drivoR_action_dit_navtrain \
DRIVOR_RUN_ID=action_dit_xpred_seed2_run1 DRIVOR_PRECISION=bf16-mixed \
DRIVOR_PERF_ENABLED=true DRIVOR_PERF_SAMPLE_EVERY=10 \
bash DrivoR/scripts/training/run_drivor_action_dit_training.sh \
  auto_resume=false train_ckpt_path=null runtime.require_resume=false \
  runtime.console_log_interval=25
```

Action-DiT 的新实验仍从指定的 Action-DiT 256 预训练文件初始化主干；`auto_resume=false` 只关闭训练 checkpoint 的自动恢复，不关闭预训练加载。重复新实验时更换 run ID。

## 3. 默认配置与覆盖方式

### 新增：多视图、每视角 16 个 Scene queries

这是独立的新配置和实验目录；原单前视脚本继续使用原生 patches。新变体仅更换视觉特征，其他 Action-DiT、Scorer、训练与推理设置沿用。

```bash
cd /cpfs29jb/data/zhaozc/project/drive_future
DRIVOR_EXPERIMENT=training_drivoR_action_dit_multiview_navtrain \
DRIVOR_RUN_ID=action_dit_multiview_q16_seed2_run1 \
DRIVOR_PERF_ENABLED=true DRIVOR_PERF_SAMPLE_EVERY=10 \
bash DrivoR/scripts/training/run_drivor_action_dit_multiview_training.sh \
  auto_resume=false train_ckpt_path=null runtime.require_resume=false \
  runtime.console_log_interval=25
```

新配置为 `drivoR_action_dit_multiview.yaml`，下游视觉特征是 `4×16=64` 个 256 维 tokens，不是四路完整 patches。详见[多视图变体说明](../../DRIVOR_ACTION_DIT_MULTIVIEW.md)。完整续训使用该新变体自己的 checkpoint。

性能参数已内置在脚本中，不需要额外传 `gpu=true`、Ray 参数或训练采样步数。`num_inference_steps=10` 只控制 Action-DiT 验证/推理；单次训练前向由当前 agent 实现。

每台机器由内部 torchrun 启动 8 个进程，Lightning 使用 DDP，`devices=8`、`num_nodes=8`，`agent.num_gpus=64`。DLC 外层 `WORLD_SIZE=8` 是节点数，内部训练进程的 `WORLD_SIZE=64` 是 GPU 进程总数。

命令末尾的 Hydra 参数优先于脚本默认值。已有同名环境变量会覆盖默认性能配置；如需显式固定资源配置，可在调用前设置：

```bash
export DRIVOR_NUM_WORKERS=4
export DRIVOR_PREFETCH_FACTOR=2
export DRIVOR_CPU_THREADS=1
export NAVSIM_ORACLE_WORKERS=2
export NAVSIM_ORACLE_FAST_PATH=1
export DRIVOR_FUSED_ADAMW=true
export DRIVOR_RANK_SEEDED_TRAINING=true
```

## 已移植的加速设置（2026-10-14）

各入口默认启用以下设置，直接使用对应的 `bash` 命令即可生效：

| 参数 | 默认值 | 用途 |
|---|---|---|
| `DRIVOR_PRECISION` | `bf16-mixed` | BF16 混合精度；Oracle 边界及连续标签保持 FP32 |
| `DRIVOR_NUM_WORKERS` | `4` | 每卡 DataLoader workers，跨 epoch 常驻 |
| `DRIVOR_PREFETCH_FACTOR` | `2` | 每个数据 worker 的预取数量 |
| `DRIVOR_CPU_THREADS` | `1` | OMP/MKL/OpenBLAS/NumExpr 线程数，减少多卡 CPU 争用 |
| `NAVSIM_ORACLE_WORKERS` | `2` | 每卡常驻 CPU 进程，把 64 条候选拆成两份各 32 条 |
| `NAVSIM_ORACLE_FAST_PATH` | `1` | 批量构造轨迹状态数组，避免逐候选创建 EgoState 对象 |
| `NAVSIM_ORACLE_CACHE_ENTRIES` | `8` | 每进程缓存解压字节和 PDM 参考进度；`0` 关闭 |
| `DRIVOR_FUSED_ADAMW` | `true` | 两个模型的 CUDA AdamW 使用 fused 更新 |
| `DRIVOR_PERF_ENABLED` | `false` | 可选每 rank 分段计时，写入 `<run>/performance/` |
| `DRIVOR_AUTO_RESUME` | `true` | 控制 `auto_resume`，命令末尾参数可覆盖 |
| `DRIVOR_REQUIRE_RESUME` | `false` | 要求存在恢复 checkpoint，命令末尾参数可覆盖 |
| `DRIVOR_CONSOLE_LOG_INTERVAL` | `25` | rank 0 控制台日志间隔 |
| `DRIVOR_RANK_SEEDED_TRAINING` | `true` | DDP 就绪后按 global rank 区分模型随机流 |

Dataset 复用已读取的 Scene 图像并只建立一次 token 索引。全局训练指标改为 epoch 汇总，`train_step/*` 是 rank 0 本地曲线；`runtime.console_log_interval=25` 控制 rank 0 的 loss/LR/ETA 打印。可选分段计时不增加统计 collective 或逐步 CUDA synchronize。

Oracle 的场景调度仍使用 `worker=sequential`，候选分块在其内部使用进程池。CPU 进程首次启动有额外开销，之后复用。`NAVSIM_ORACLE_WORKERS=1` 可回退到整批评分，`NAVSIM_ORACLE_FAST_PATH=0` 可回退到原逐候选转换。地图 bbox 裁剪仍默认关闭。没有将评分迁移为 CUDA，也没有缓存旧模型生成的候选标签。

新实验示例：

```bash
cd /cpfs29jb/data/zhaozc/project/drive_future
export DRIVOR_RUN_ID=drivor_fast_seed2_run1
bash DrivoR/scripts/training/run_drivor_training.sh auto_resume=false runtime.console_log_interval=25
```

Action-DiT 使用 `run_drivor_action_dit_training.sh`，资源与精度环境变量同样有效。模型的图像视角、64 候选、Action-DiT 初始化、归一化和验证/推理的 10 步 Euler 保持各自现有配置；原入口保持双 GT loss，多 GT 入口采用本文开头说明的监督方式。

自动恢复现在只查当前 run 的输出目录，优先 `last.ckpt`；显式 `train_ckpt_path` 优先。续训可用 `auto_resume=true runtime.require_resume=true`，缺少 checkpoint 时直接报错；`auto_resume=false` 关闭自动发现。

诊断时在命令前设置 `DRIVOR_PERF_ENABLED=true DRIVOR_PERF_SAMPLE_EVERY=10`，再用 `environment/performance/summarize_rank_performance.py <run>/performance` 汇总。阶段包含关系和异步 GPU 时间的解释见脚本说明。

本次移植已通过 90 项针对性检查，以及原版和 Action-DiT 各自的真实双 A100 smoke：每模型两步训练、每卡一个验证 batch、NCCL DDP、BF16、fused AdamW、两评分进程均正常完成。测试开启了反向异常检测，并包含冷启动，不用于推断正式吞吐或 NAVSIM 成绩。详见[验证记录](../../../artifacts/drivor_speed_port_20261014/validation.json)。

新增设置在下次启动时生效；已经运行的 Python 进程不会自动切换实现。此次加速移植保留了本目录的样本索引顺序；模型随机流现按下述 rank 规则初始化。标准 metric cache 的参考进度补算同时对齐参考版本：先计算 ego-area flags，再计算碰撞、道路合规与进度；原缓存文件不改写。

### 模型随机流按 global rank 区分

两个入口默认 `runtime.rank_seeded_training=true`。模型构造及数据采样仍使用公共 `seed`；DDP 就绪后的 `on_train_start` 执行 FastWAM 的 `process_seed = seed + global_rank`，设置 Python、NumPy、Torch CPU/CUDA 随机流。这里不调用 `pl.seed_everything`，因此不会改写 `PL_GLOBAL_SEED`，DistributedSampler 和独立 DataLoader Generator 仍使用公共 seed。

默认 seed=2 时，rank 0/1/8/63 的模型 seed 分别是 2/3/10/65；训练日志会打印每个 rank 的实际值。只在训练启动时设置，后续 batch 和 epoch 继续推进随机流。`runtime.rank_seeded_training=false` 可显式关闭偏移。

Action-DiT 每个场景仍采样 64 份不同噪声，共享一个 t；原双 GT 配置中普通/长 GT 的对应候选共用噪声和 t，多 GT 配置则只用原始 human GT 构造一组 64 个输入。跨 rank 使用不同随机流，不要求所有随机数值绝不偶然重复。单次 X-prediction 训练方式不变。

本次通过 64 项检查及轻量双 A100/NCCL smoke：模型初始参数相同，Sampler/Loader seed 均为 2；两卡模型 seed 为 2/3，实际 CUDA 噪声和 t 不同，连续两批重新采样，梯度同步后的参数仍相同。Smoke 使用真实 flow 输入构造器、微型预测器和合成数据，不是完整 NAVSIM 训练。见[验证记录](../../../artifacts/drivor_rank_seed_20261014/validation.json)。

此改动只初始化模型随机流，没有新增每个 rank 的 RNG 检查点保存。重新启动或断点恢复会从该 rank 的 seed 初始化，不承诺恢复之前随机流的精确位置。

脚本自动激活 `/cpfs29jb/data/zhaozc/miniconda3/envs/drive_future`，优先导入当前 `DrivoR/` 及其 `nuplan-devkit/` 源码，并切换到 `DrivoR/` 根目录解析 `weights/...`。

所有节点需能访问相同的代码、Conda 环境、权重、NAVSIM 数据与评分缓存；`NAVSIM_EXP_ROOT` 应使用共享目录。输出目录为 `$NAVSIM_EXP_ROOT/ke/$DRIVOR_EXPERIMENT/$DRIVOR_RUN_ID`，不会按各机器的启动时间分别生成目录。

| 分布式环境变量 | 默认值 / 含义 |
|---|---|
| `NNODES` | 优先显式设置，否则读取 DLC 的 `WORLD_SIZE`；无平台变量时默认 `8` |
| `CUDA_VISIBLE_DEVICES` | `0,1,2,3,4,5,6,7`，当前节点的可见 GPU |
| `NPROC_PER_NODE` | 优先显式设置，否则读取 `GPU_NUM` 或可见 GPU 列表长度，默认 `8` |
| `NODE_RANK` / `RANK` | 优先 `NODE_RANK`，否则自动读取 DLC 的 `RANK`，8 机时为 `0..7` |
| `MASTER_ADDR` | 自动读取 DLC 分配的主节点地址 |
| `MASTER_PORT` | 保留 DLC 注入的端口；无平台值时为 `29500` |
| `DRIVOR_RUN_ID` | 可选；未设置时使用 `PAI_JOB_ID`、`SLURM_JOB_ID` 或固定名称 `multinode` |

如需在启动命令中明确机器数和每机卡数，只需设置这两个值，并与 DLC 资源配置一致：

```bash
NNODES=8 NPROC_PER_NODE=8 bash DrivoR/scripts/training/run_drivor_training.sh
```

脚本会同步 torchrun、Trainer 和 `agent.num_gpus`。额外 Hydra 参数仍可覆盖其他训练设置；手动调整 batch、累积步数或总卡数会改变有效总 batch。

配置检查可在任意一台机器上直接执行，不需要设置主节点或节点编号，也不会等待其他节点：

```bash
bash DrivoR/scripts/training/run_drivor_training.sh trainer.params.max_epochs=1 --cfg job --package trainer --resolve
bash DrivoR/scripts/training/run_drivor_action_dit_training.sh --cfg job --package agent --resolve
```

`--cfg` 只输出配置，不启动训练。查看网络参数时可用 `--package agent`，避免输出完整配置中的大量场景 token 列表。

## 数据与评分缓存

| 环境变量 | 默认值 |
|---|---|
| `OPENSCENE_DATA_ROOT` | `drive_future/data/navsim` |
| `NUPLAN_MAPS_ROOT` | `$OPENSCENE_DATA_ROOT/maps` |
| `NAVSIM_METRIC_CACHE_PATH` | `drive_future/data/metric_cache_navtrain` |
| `NAVSIM_EXP_ROOT` | `DrivoR/exp` |
| `DRIVOR_EXPERIMENT` | `training_drivoR_Nav1_navtrain_traj_long_25epochs` |

训练读取 `$OPENSCENE_DATA_ROOT/navsim_logs/trainval` 和 `$OPENSCENE_DATA_ROOT/sensor_blobs/trainval`。DINOv2 权重使用 `DrivoR/weights/vit_small_patch14_reg4_dinov2.lvd142m/model.safetensors`。

两个 Agent 的评分缓存均读取 `NAVSIM_METRIC_CACHE_PATH`，默认使用已有的 `data/metric_cache_navtrain/metadata` 及其索引的缓存文件。`NAVSIM_EXP_ROOT` 用于训练输出；评分缓存与下面的图像下载缓存用途不同。

该目录是标准 NAVSIM metric cache，包含 PDM 参考轨迹。读取时若没有 DrivoR 专用训练缓存的 `pdm_progress` 字段，评分模块会按原版 `train_cache_processor.py` 的算法补算参考进度；已有该字段的缓存直接使用原值。原始缓存文件保持原样。

在线打分与验证默认使用 NAVSIM 的 [`worker=sequential`](../../navsim/planning/script/config/common/worker/sequential.yaml)。两个 Agent 的 `score_worker` 都引用这个配置，每个 DDP 进程在本地计算自己场景的 64 条候选分数。GPU 训练仍由 torchrun + Lightning DDP 并行执行，默认评分流程不启动 Ray 服务。

最小双卡检查使用 4 个真实训练场景、2 个真实验证场景，只跑 2 个训练 step，并启用反向异常检测：

```bash
cd /cpfs29jb/data/zhaozc/project/drive_future
/usr/bin/python3 environment/run_drivor_two_gpu_smoke.py --agent action_dit
```

日志写入 `artifacts/drivor_two_gpu_smoke/`；检查时每卡 batch 为 1、总 batch 为 2，正式 64 卡启动的总 batch 仍为 64。

2026-10-14，Action-DiT 在本地两张 A100 上完成上述 smoke：NCCL DDP、真实图像与缓存、每场景 64 条候选、Sequential 在线评分、2 步反向及每卡 1 个验证 batch 均通过，进程正常退出并释放显存。记录见 [双卡验证结果](../../../environment/drivor_sequential_two_gpu_validation.json)。

## 本地图像与 OSS 图像

日志中的相对或本地绝对 `data_path` 按本地文件 → 已下载的图像缓存 → OSS 的顺序读取；显式的无签名 `oss://bucket/key` 图像 URI 直接读取缓存或 OSS。相机选择、图像 resize 和归一化仍由 Agent 的 YAML 控制。

OSS 的连接与重试方式参考 FastWAM 的 `waymo_dataset.py`：使用 `oss2`，按 PID、配置路径和 bucket 缓存连接，瞬时读取错误最多尝试 5 次。认证失败或对象不存在会直接报错；读取失败不会替换为空白图像。错误消息不包含 SDK 返回的原始认证信息。

| 环境变量 | 含义 / 默认值 |
|---|---|
| `NAVSIM_OSS_CONFIG_PATH` | `/mnt/human/shixi/human/oss_config.json`；凭据只从这个文件读取 |
| `NAVSIM_OSS_SENSOR_BLOBS_PATH` | 对应本地 `sensor_blobs/trainval` 的 OSS 前缀，已配置在启动脚本中 |
| `NAVSIM_OSS_IMAGE_CACHE_DIR` | `$NAVSIM_EXP_ROOT/oss_image_cache`；显式设为空字符串可关闭缓存 |
| `NAVSIM_OSS_USE_PROXY` | 默认 `0`，OSS 直接连接；设为 `1` 才读取环境代理设置 |

已确认的前缀是：

```text
oss://ofasys-multimodal-wlcb-5-toshanghai/anonymous_driving/autonomousvision_navsim/1/download_trainval/openscene-v1.1/sensor_blobs/trainval
```

例如，本地缺失的 `<log>/CAM_L0/<image>.jpg` 会映射为 `<上述前缀>/<log>/CAM_L0/<image>.jpg`。缓存目录直接按 `<cache_dir>/<bucket>/<object_key>` 组织；图像成功解码后才原子写入，供多个 DataLoader worker 和后续 epoch 复用。不做额外的哈希或 SHA 校验。

凭据文件使用按 bucket 名索引的 JSON，条目包含 `access_key_id`、`access_key_secret`、`endpoint`。不要把凭据内容复制到脚本、配置示例或日志。

当前 Conda 环境已经安装 OSS 依赖；在其他环境中可单独安装：

```bash
python -m pip install -r DrivoR/requirements-oss.txt
```

仅使用本地图像时，图像读取模块不会导入 `oss2` 或读取 OSS 凭据。将 `NAVSIM_OSS_SENSOR_BLOBS_PATH` 显式设为空字符串，可关闭本地缺失文件的 OSS 回退。

## Action-DiT 变体

使用独立的 Action-DiT 轨迹网络时，在 DLC 任务的启动命令中填写：

```bash
cd /cpfs29jb/data/zhaozc/project/drive_future
bash DrivoR/scripts/training/run_drivor_action_dit_training.sh
```

该脚本复用上述多机启动、环境、数据和 OSS 设置，默认每卡 batch 1、累积 1、`bf16-mixed`。Action-DiT 配置只启用当前前视 `cam_f0=[3]`，其余相机均为空。Action-DiT 和 Scorer 的 K/V 都是同一张前视图的 3,936 个原生 DINOv2 patch 特征。模型使用 64 份独立噪声、普通/长 GT 各自的 best-of-64 v loss，以及 10 步 X→V Euler 推理。具体权重加载、归一化和训练参数见 [Action-DiT 说明](../../DRIVOR_ACTION_DIT.md)。

训练通过 `forward_train(features, targets)` 将两组插值拼为 128 条，只做一次 DiT 前向。普通 GT 分支的 64 条 X-prediction 直接反归一化、detach 后送 Scorer；长 GT 分支继续参与原有 v loss。`compute_loss` 复用本次前向的 flow loss，不再执行采样或额外 DiT 前向。验证/推理仍为 10 步 Euler；`num_inference_steps` 只控制这两个阶段。

## 已完成的读取验证

相机读取测试覆盖本地优先、OSS URI、重试、错误信息、缓存复用与多进程连接：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH="DrivoR:DrivoR/nuplan-devkit" \
  /cpfs29jb/data/zhaozc/miniconda3/envs/drive_future/bin/python -m pytest -q DrivoR/tests/test_oss_image.py
```

16 项测试通过。针对原版 DrivoR 的四相机配置，真实 NAVSIM 场景也已验证：前视从本地读取，左、右、后视从 OSS 读取，形成 `[4,3,672,1148]` 图像特征；第二次读取完全复用缓存。两个 DataLoader worker 的独立连接、相同图像输出和缓存写入也已验证。Action-DiT 变体只读取当前前视图，其验证见独立模型说明。此处没有启动正式训练或计算 NAVSIM 评测指标。

## 多机启动验证

两个脚本已通过 Bash 语法、Hydra 配置展开和实际启动参数检查。`tests/test_training_launchers.py` 的 10 项测试验证 DLC 环境变量自动接入、节点编号、共享任务目录、配置查询及数量覆盖；`tests/test_action_dit.py` 的 25 项测试验证模型与优化器，其中同时验证两种 Agent 从旧四卡配置改为 64 卡后，基础 LR 仍为 `2e-4`，按原配置计算的调度总步数仍为 33225。合计 **35 项通过**。

本机还模拟 DLC 注入的节点环境，使用脚本生成的 torchrun 参数启动 2 个节点、每节点 2 个 CPU 进程，完成 Lightning DDP 的两步训练，验证了全局/本地 rank、数据分片、梯度同步和参数一致性。该检查没有启动真实 DLC 8 机 64 卡任务，也不覆盖跨机器的 NCCL 网络连通性；详见 [启动验证记录](../../../environment/drivor_multinode_validation.json)。
