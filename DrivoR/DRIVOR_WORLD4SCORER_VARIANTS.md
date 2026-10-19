# 原版 DrivoR：Bank、V-JEPA 与多 GT 未来监督

更新：2026-10-17。以下四组实验均保留 **原版 DrivoR 的轨迹生成器**。第 2 组包含两个骨干版本，因此共有五个训练入口。已有 DrivoR 和 Action-DiT 入口继续保留。

这里的“原版”指网络结构。训练使用此前已经适配DLC、OSS、BF16、CPU评分池和数值稳定性修复的本地DrivoR基线，不等同于官方仓库逐文件不改的完整训练复现。

第一组相对当前 `run_drivor_training.sh` 的核心新增项是 `0.5 * L_Bank`：DINOv2-S、四视角、每视角16个Scene queries、64条候选、原轨迹decoder/回归头和原Scorer均保留。`future.enabled=false`，不实例化独立未来预测器，也不启用multi-GT或V-JEPA；`DrivoRWorldAgent`只是承载可选附加loss的通用包装。

与官方README训练命令相比，当前入口还保留此前确认的差异：官方调用 `run_training_full.py`，其训练使用 `train_logs + val_logs`；本地按用户要求调用 `run_training.py`，训练只用 `train_logs`，验证使用 `val_logs`。此外，官方4卡×batch16改为DLC64卡×batch1（总batch64），默认精度改为BF16，使用fused AdamW、按rank错开的模型随机流和CPU评分池。模型的轨迹/评分decoder结构与官方源码一致，但这些训练设置和数据划分会影响与官方成绩的直接比较。

## 1. 五个入口对应四组实验

| 组别 / 命令中的版本名 | 当前视觉输入 | Bank | 独立未来预测器 | 轨迹监督 | 未来特征目标 |
|---|---|---|---|---|---|
| 1 / `bank` | 原 DINOv2-S，四视角，每视角16 queries | 有 | 无，使用原 Scorer trunk | 原普通＋长 GT | 无 |
| 2 / `vjepa2_vitl` | V-JEPA 2-L，单前视原生 patches | 无 | 无 | 原普通＋长 GT | 无 |
| 2 / `vjepa2_1_vitl` | V-JEPA 2.1-L，单前视原生 patches | 无 | 无 | 原普通＋长 GT | 无 |
| 3 / `vjepa2_world` | V-JEPA 2-L，单前视原生 patches | 有 | 有 | 原普通＋长 GT | 真实 human 未来8帧的冻结 V-JEPA 2.1 特征 |
| 4 / `vjepa2_world_multigt` | V-JEPA 2-L，单前视原生 patches | 有 | 有 | 有效多 GT，独立最近邻匹配、等权平均 | 每条有效 GT 对应的 Cosmos 未来8帧特征 |

对应配置均在 `navsim/planning/script/config/common/agent/`：

- [drivoR_bank.yaml](navsim/planning/script/config/common/agent/drivoR_bank.yaml)
- [drivoR_vjepa2_vitl.yaml](navsim/planning/script/config/common/agent/drivoR_vjepa2_vitl.yaml)
- [drivoR_vjepa2_1_vitl.yaml](navsim/planning/script/config/common/agent/drivoR_vjepa2_1_vitl.yaml)
- [drivoR_vjepa2_world.yaml](navsim/planning/script/config/common/agent/drivoR_vjepa2_world.yaml)
- [drivoR_vjepa2_world_multigt.yaml](navsim/planning/script/config/common/agent/drivoR_vjepa2_world_multigt.yaml)

启动文件统一为 `scripts/training/run_drivor_<版本名>_training.sh`。第 2～4 组已按确认设置使用当前单前视，不加 Scene queries；第 2 组仅更换视觉骨干。

## 2. 视觉骨干与原版轨迹生成器

第 2～4 组图像在数据侧直接 resize 到 **672×1152**，patch16 得到 **3024 个 tokens**。冻结 ViT 原参数，训练每层 Q/V 的 rank-32 LoRA 和 `Linear(1024,256)`。下游场景特征为 `[B,3024,256]`，同时供原轨迹生成器和评分分支使用。

V-JEPA 2 通过同图复制两次适配原生时间核2；V-JEPA 2.1 使用时间核1的专用图像分支和 `img_mod_embed`。位置编码分别保留各自官方 RoPE，包括2.1的坐标缩放。详情见 [V-JEPA 2](DRIVOR_ACTION_DIT_VJEPA2.md) 和 [V-JEPA 2.1](DRIVOR_ACTION_DIT_VJEPA2_1.md) 的骨干说明。

原版轨迹网络继续使用 64 个可学习 trajectory queries、当前 ego 的 `Linear(11,256)`、四层 trajectory decoder 及原有轨迹回归头，输出 `[B,64,8,3]` 的物理坐标轨迹。第 1～3 组启动命令沿用普通 GT 和长 GT 分别 best-of-64 的 L1 监督；第 4 组关闭长 GT，使用多 GT 替代。

## 3. L_Bank 的来源和计算

参考 [World4Scorer](../World4Scorer/README.md) 的 `clover_aux_loss`。其候选 bank 来自 [CLOVER](https://github.com/WilliamXuanYu/CLOVER)，World4Scorer 本身不分发这份数据。

CLOVER 官方 README 的 [Google Drive 数据目录](https://drive.google.com/drive/folders/1oNTv5Pe-naw_i81rqaKk8KIs0VcUqGZ-?usp=drive_link) 中提供 `dataset_decoupled_v2_clean.pkl`，文件 ID 为 `1JUKc9hQ2YH7Ck7ghjZm65OCavWVg6Po_`。该文件包含每场景的 `trajectories_relative`、七项真实评分和候选类型，包含 offroad 等低质量轨迹。这里直接复用这些预计算标签。

官方文件已完整下载并转换。正式Bank覆盖102978个场景，包含5613477条有效候选、1946414条碰撞或越界候选，平均每场景54.511条有效候选。源文件中310个空候选场景按CLOVER加载规则跳过，不补造标签。

[build_world4scorer_bank.py](tools/build_world4scorer_bank.py) 按参考转换器的规则保留每个有效场景的前64条有限候选，产出：

| 文件 | 形状 / 含义 |
|---|---|
| `tokens.npy` | `[N]` 场景 token |
| `bank_candidates.npy` | `[N,64,8,3]` 自车后轴局部坐标，米/弧度，0.5秒间隔 |
| `bank_subscores.npy` | `[N,64,7]`，列序 NC、DAC、EP、TTC、Comfort、DDC、PDMS |
| `bank_mask.npy` | `[N,64]` 有效候选 mask |

每步每场景抽取16条：先从 `NC<0.5 或 DAC<0.5` 的候选中抽最多8条，再从其余有效候选随机补足。后半部分也可能选到不安全轨迹，所以最终不安全数量可能超过8。候选不足16时允许重复抽取。采样使用已有的按 rank 错开的 Torch 随机流。

Bank 坐标先 detach，再经过与生成候选相同的轨迹投影、评分 trunk 和六个评分头。生成候选64条与 Bank候选16条分别前向，不合成一个80-token自注意力集合。

```text
L_Bank = mean(BCE_NC, BCE_DAC, BCE_TTC, BCE_Comfort, BCE_DDC)
         + 0.1 * mean(abs(sigmoid(logit_EP) - target_EP))
总 loss 中的系数 = 0.5
```

NC/DDC 标签0.5按参考映射为0，缺失分项通过 mask 排除。生成候选仍使用原 DrivoR 的六项 BCE 之和。Bank 不作为额外轨迹 GT，也不参与推理时的64条轨迹选择。

第 1 组没有新增预测器或未来对齐头；Bank loss 直接训练原有 Scorer。实现位于 [model.py](navsim/agents/drivoR_world/model.py) 和 [targets.py](navsim/agents/drivoR_world/targets.py)。

## 4. 独立未来预测器与8时刻对齐

第 3、4 组按最新确认使用**独立未来分支**。Scorer 保持原 DrivoR 结构，评分全部64条生成候选及16条 Bank；未来分支只处理与 GT 匹配的候选。

```text
每条 GT → 按原版轨迹 L1 cost 从64条候选中独立选最近者
选中轨迹 [有效匹配数,8,3]
  → 逐点 Linear(3,1024)-GELU-Linear(1024,256)
  → 加8个时刻的可学习位置 embedding
  → 4层预测器，每层：8个轨迹token的Self-Attention
                    → 当前场景patches的Cross-Attention → FFN
  → LayerNorm → Linear(256,1024)
  → [有效匹配数,8,1024] 未来特征
```

预测器随机初始化，hidden256、8 heads、FFN1024，与评分网络的参数独立。其输入是**匹配到的预测轨迹**；GT只用于匹配和确定监督视频。第3组用普通human GT选一条候选；第4组每条有效GT各选一条，允许多个GT选择同一proposal。Self-Attention仅发生在一条轨迹的8个点之间。

未来对齐loss更新选中轨迹及独立预测器。按照最新确认，**只对传入未来预测器的当前场景K/V做detach**；选中轨迹正常反传，保留“未来loss → 轨迹生成器 → 视觉”的间接梯度。不重算轨迹生成器。

Teacher 是完整冻结、无 LoRA 的 V-JEPA 2.1 ViT-L，使用其官方 `ema_encoder`、单图分支和最后一层归一化输出。每张未来图像 resize 到384×384，ImageNet归一化，所有原生 patches在FP32平均池化成1024维向量，再以FP16缓存；逐帧独立编码。训练读取预计算特征，训练步骤中不再运行 teacher。

```text
L_future(b,k,t) = 1 - cosine(predicted_feature, frozen_teacher_feature)
L_future = mean_scene(mean_valid_GT(mean_valid_time(L_future(b,k,t))))
```

每个有效 GT 的8个时刻等权，场景内有效 GT 等权。未来对齐权重1.0；没有 human/非human 的不同权重。

第3、4组训练时：视觉编码一次、原版轨迹生成一次，独立预测器只预测匹配项的8时刻特征。原 Scorer 分别处理64条生成候选和16条Bank。验证/推理跳过独立未来分支，仅保留原规划与评分流程。

| 参数部分 | 轨迹 loss | 生成评分 loss | L_Bank | L_future |
|---|---|---|---|---|
| 轨迹生成器专有参数 | 更新 | 坐标处截断 | 坐标处截断 | 匹配轨迹路径更新 |
| 当前视觉 LoRA / 投影 | 更新 | 更新 | 更新 | 直接K/V路径截断；经轨迹生成器的间接梯度保留 |
| Ego编码 | 更新 | 更新 | 更新 | 经轨迹生成器更新 |
| 评分轨迹投影 / 原评分trunk | 无梯度 | 更新 | 更新 | 无梯度 |
| 六个评分头 | 无梯度 | 更新 | 更新 | 无梯度 |
| 独立未来预测器 / 视觉读出头 | 无梯度 | 无梯度 | 无梯度 | 更新 |
| 未来 teacher | 冻结 | 冻结 | 冻结 | 离线目标，冻结 |

## 5. Multi-GT 与视频的对应关系

轨迹源：`multi_traj_gt_data/navtrain_multigt_top4_dense_0p1s_f64_8gpu`。

视频源：`/cpfs29jb/data/zhaozc/project/Cosmos3forMultiGT/runtime/rollouts/navtrain_full_allcosmos_cuda_v1_gpu64`。

视频41帧、10 FPS，零基索引0为当前帧：

| 轨迹时间/秒 | 0.5 | 1.0 | 1.5 | 2.0 | 2.5 | 3.0 | 3.5 | 4.0 |
|---|---|---|---|---|---|---|---|---|
| 视频零基索引 | 5 | 10 | 15 | 20 | 25 | 30 | 35 | 40 |
| 从1计数的第几帧 | 6 | 11 | 16 | 21 | 26 | 31 | 36 | 41 |

有效 GT 使用 `gt_valid_mask & ~gt_padded`。原始槽位 k 对应 `trajectory_k/future.mp4`，抽帧保留槽位和标签；GT压缩成有效行后，不用压缩后的行号反查视频。每条未来查询与其图像特征在同一缓存中配对保存。

抽帧按 `pose_times_s.npy` 查找上述时刻，实际使用 FFmpeg 的帧索引选择。每场景保存一个 ZIP，包含各有效GT的8张JPEG和 `metadata.json`，避免生成数百万个独立小文件。

Teacher缓存是 `future_*/<token前两位>/<token>.npz`，包含：

```text
features       [K,8,1024] float16
trajectories   [K,8,3]    对应GT的局部坐标
valid          [K,8]      有效GT/时刻mask
times          [8]       0.5,...,4.0
```

第4组轨迹监督采用已有的独立最近邻匹配：每个GT从全部64条候选中取最小L1，允许重复选择同一proposal，再按实际有效GT数平均。未来分支也选择各GT对应的最近proposal，对齐该GT的视频特征。不使用匈牙利匹配或human/其他GT的差异权重。

```text
j(b,k) = argmin_j mean_time(sum_xyz(abs(proposal[b,j] - GT[b,k])))
L_traj(b) = sum_valid_GT(distance[b,k,j(b,k)]) / K(b)
L_future(b) = mean_valid_GT(mean_time(1 - cosine(pred_future[b,j(b,k),t], target[b,k,t])))
```

重复选中同一proposal时，各GT的监督仍分别计算，再平均；不按唯一proposal数重新归一化。Bank的0.5系数是整个辅助评分项的系数，与GT之间的权重无关。

NAVSIM日志索引含103288个场景，多GT/视频目录含103075个场景。缺少导出多GT的场景沿用原始GT回退；缺少未来特征的行屏蔽未来loss，并记录有效目标比例。不会补造未来图像或把其他场景视频错配过来。

## 6. 数据准备与目录

默认缓存根目录为 `DrivoR/exp/world4scorer_data/`：

| 目录 / 文件 | 用途 |
|---|---|
| `bank_clover/` | 第1、3、4组使用的CLOVER候选及七项标签 |
| `real_index.jsonl` | 真实human轨迹和未来8帧的NAVSIM索引 |
| `frames_cosmos/<前两位>/<token>.zip` | 对应有效多GT的8时刻JPEG及槽位元数据 |
| `future_real_vjepa21/<前两位>/<token>.npz` | 第3组真实未来图像的冻结特征 |
| `future_cosmos_vjepa21/<前两位>/<token>.npz` | 第4组Cosmos图像的冻结特征 |

转换官方CLOVER源文件时使用：

```bash
cd /cpfs29jb/data/zhaozc/project/drive_future
/cpfs29jb/data/zhaozc/miniconda3/envs/drive_future/bin/python \
  DrivoR/tools/build_world4scorer_bank.py \
  DrivoR/exp/world4scorer_data/source/dataset_decoupled_v2_clean.pkl \
  DrivoR/exp/world4scorer_data/bank_clover
```

`--preview-scenes`仅用于少量真实场景的smoke，正式转换应省略。转换保留官方候选坐标与标签，不计算新的零标签，也不使用本地poor4替换来源。

独立准备第3组或第4组未来特征时，在DLC配置8节点×8GPU，分别使用：

```bash
cd /cpfs29jb/data/zhaozc/project/drive_future
bash DrivoR/scripts/training/run_drivor_world_data_preparation.sh real
```

```bash
cd /cpfs29jb/data/zhaozc/project/drive_future
bash DrivoR/scripts/training/run_drivor_world_data_preparation.sh cosmos
```

`real`从NAVSIM日志读取当前帧之后0.5…4秒的真实前视图；`cosmos`先按正确视频索引抽帧，再编码。各GPU处理互不重叠的token；缓存文件原子写入，重跑跳过已完成场景。图像在跨场景batch内编码后，按各场景、GT槽位和时刻拆回，不混用标签。

可用 `DRIVOR_PREP_BATCH` 调整每卡teacher图像batch，`DRIVOR_PREP_WORKERS`调整每卡CPU workers。当前teacher为BF16前向、FP32池化、FP16缓存。修改teacher权重或图像预处理后应使用新输出目录，或明确重建缓存。

本地已完成并验证全部103075个Cosmos场景的抽帧，共214190条有效GT、1713520张未来图像。真实未来teacher缓存已有35618个场景；第3、4组的一键DLC入口会继续补齐所需特征后再训练。数据状态见 `exp/world4scorer_data/preparation_status.json`，无需等待本地双卡完成全量teacher编码。

## 7. DLC 一条命令完成预处理、训练和测试

五个版本分别提交任务，均配置 **8节点×8GPU**。训练默认每卡batch1、累积1、总batch64、25epochs、BF16、seed2、LR2e-4。训练成功后，主节点使用本机8卡测试最新保存的 `last.ckpt`；PDM评分采用CPU进程池。

第1组：

```bash
cd /cpfs29jb/data/zhaozc/project/drive_future
DRIVOR_RUN_ID=drivor_bank_seed2_run1 \
bash DrivoR/scripts/training/run_drivor_world_train_eval.sh bank
```

第2组，V-JEPA 2：

```bash
cd /cpfs29jb/data/zhaozc/project/drive_future
DRIVOR_RUN_ID=drivor_vjepa2_vitl_seed2_run1 \
bash DrivoR/scripts/training/run_drivor_world_train_eval.sh vjepa2_vitl
```

第2组，V-JEPA 2.1：

```bash
cd /cpfs29jb/data/zhaozc/project/drive_future
DRIVOR_RUN_ID=drivor_vjepa2_1_vitl_seed2_run1 \
bash DrivoR/scripts/training/run_drivor_world_train_eval.sh vjepa2_1_vitl
```

第3组：

```bash
cd /cpfs29jb/data/zhaozc/project/drive_future
DRIVOR_RUN_ID=drivor_vjepa2_world_seed2_run1 \
bash DrivoR/scripts/training/run_drivor_world_train_eval.sh vjepa2_world
```

第4组：

```bash
cd /cpfs29jb/data/zhaozc/project/drive_future
DRIVOR_RUN_ID=drivor_vjepa2_world_multigt_seed2_run1 \
bash DrivoR/scripts/training/run_drivor_world_train_eval.sh vjepa2_world_multigt
```

第3、4组会先补齐各自的冻结未来特征缓存，再训练；已有缓存会跳过。如已单独完成数据准备，可设置 `DRIVOR_PREPARE_FUTURE=false`。Bank只需准备一次，由第1、3、4组共享。

训练结果目录为 `DrivoR/exp/ke/training_drivoR_<版本名>_navtrain/<RUN_ID>/`，测试结果在 `DrivoR/exp/ke/eval_navtest_drivoR_<版本名>/<RUN_ID>/pdms_summary.txt`。重复新实验请更换run ID；需要断点续训可在命令末尾追加原训练脚本支持的恢复参数。

只训练时直接运行对应的 `run_drivor_<版本名>_training.sh`。单独测试完整checkpoint：

```bash
cd /cpfs29jb/data/zhaozc/project/drive_future
NPROC_PER_NODE=8 bash DrivoR/scripts/evaluation/run_drivor_world_evaluation.sh \
  vjepa2_world_multigt /path/to/last.ckpt
```

完整checkpoint评测保持相同的预测器参数结构，跳过训练用的Bank和未来特征目标读取；多GT目标也在测试入口关闭。

## 8. 验证

当前实现通过 **224项回归检查**，包括原版DrivoR输出兼容、Bank采样和BCE/L1公式、未来分支只使用匹配的预测轨迹、重复GT匹配、有效GT/时刻平均、直接K/V梯度截断与间接视觉梯度保留、视觉和生成器各前向一次、视频帧号、GT槽位与批量teacher编码对应关系，以及DLC准备/训练/主节点评测的执行顺序。

另通过2项正式Bank转换补充检查，覆盖源文件的空候选场景和中断转换的未完成产物；转换结果还与World4Scorer官方工具在64个真实场景上逐数组一致。

五个入口均完成真实双A100/BF16 smoke，每卡两步训练和一个验证batch，正常退出且loss有限。Bank检查使用CLOVER官方文件前512个场景的真实候选/标签，包含29294条有效候选、10389条不安全候选；不是人为伪造负例。未来teacher使用已下载的完整V-JEPA2.1预训练权重。

这些检查验证代码和数据链路，不代表收敛、正式吞吐或全量NAVSIM成绩。验证记录见 [validation.json](../artifacts/drivor_world4scorer_20261016/validation.json)。

全量抽帧核对覆盖103075份归档，结构错误为0；另抽查32个场景、132张首末未来图像，与源视频解码结果一致（JPEG重编码平均绝对误差最大1.758/255）。对应轨迹的最大坐标差为1.722e-6，属于浮点精度范围。记录见 [full_frames_validation.json](../artifacts/drivor_world4scorer_20261016/full_frames_validation.json)。
