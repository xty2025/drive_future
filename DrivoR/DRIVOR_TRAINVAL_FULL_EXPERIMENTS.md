# T07 / T10 / T05 / A23 / A21：trainval 全量训练

带未来 register 的原 A27/A28 也已有独立 full 配置和启动脚本，见 [A27/A28 全量训练说明](DRIVOR_A27_A28_TRAINVAL_FULL.md)。两版保留 human＋D8 未来监督，并通过主目录软链接读取 hist3 补充轨迹。

这五版通过 `navsim/planning/script/run_training_full.py` 启动。沿用官方 `navtrain` 场景筛选，把原来的 train、val 日志集合并并去重后用于训练：**85,109＋18,179＝103,288 个场景**。数据仍读取 `data/navsim`，评分缓存仍读取 `data/metric_cache_navtrain`；已核对该缓存覆盖全部 103,288 个目标 token。

原 val 集也进入训练，因此每轮保留的 FP32 `val_pdms.csv` 是**训练子集监测**，不再是独立验证集成绩。训练后的独立评测仍使用 navtest，FP32、12,146 场景。

## 五组配置

| 原实验 | 轨迹网络 | 轨迹 loss | Full 配置 |
|---|---|---|---|
| T07 | 普通 Transformer | human 物理 L1＋0.5×diverse8 L1 coverage | [full_t07.yaml](navsim/planning/script/config/training/full_t07.yaml) |
| T10 | 普通 Transformer | 仅 human 物理 L1 | [full_t10.yaml](navsim/planning/script/config/training/full_t10.yaml) |
| T05 | 普通 Transformer | human 物理 L1＋长 GT 物理 L1 | [full_t05.yaml](navsim/planning/script/config/training/full_t05.yaml) |
| A23 | Action-DiT | human X-prediction 物理 L1＋0.5×diverse8 L1 coverage | [full_a23.yaml](navsim/planning/script/config/training/full_a23.yaml) |
| A21 | Action-DiT | human X-prediction 转 vMSE＋0.5×diverse8 L1 coverage | [full_a21.yaml](navsim/planning/script/config/training/full_a21.yaml) |

五组均保留当前单前视 DINOv3 ViT-L/16、672×1152、3024 个原生 patch K/V、Q/V LoRA rank 32、64 个 proposal 和原 DrivoR Scorer；不预测未来 register。普通 Transformer 仍为原四层网络；Action-DiT 仍使用原 Action256 预训练初始化、一次训练前向和原多步 Euler 推理。Scorer 继续评分全部 64 个候选，轨迹输入保持 detach。

T07/A23/A21 使用原 diverse8 主目录和已配置的 hist3 补充目录：

```text
/cpfs29jb/data/zhaozc/project/Cosmos3forMultiGT/runtime/multigt_raw64/outputs/navtrain_diverse8_dense_0p1s_f64
/cpfs29jb/data/zhaozc/project/Cosmos3forMultiGT/runtime/multigt_raw64/outputs/navtrain_diverse8_hist3_dense_0p1s_f64
```

生产摘要中主目录完成 103,083 个场景，原跳过的 205 个 history=3 场景在补充目录全部完成，合并覆盖 103,288 个场景。仍按有效 GT mask 使用实际条数，不保证每场景必须填满 8 条。核对记录见 [data_scope_audit.json](../artifacts/trainval_full_variants_20261024/data_scope_audit.json)。

## PAI DLC：仅训练

为每个实验分别创建 **8 节点×8 GPU** 的 PyTorch 分布式任务，填写其中一条命令。平台分发命令并注入节点信息，无需逐节点手工启动。默认每卡 batch 1、梯度累积 1、总 batch 64、seed 2、25 epochs、BF16 训练、峰值学习率 2e-4，每轮保存 checkpoint。

```bash
cd /cpfs29jb/data/zhaozc/project/drive_future

# T07：human + diverse8，普通 Transformer
bash DrivoR/scripts/training/run_drivor_query_human_diverse8_l1_full_training.sh

# T10：仅 human，普通 Transformer
bash DrivoR/scripts/training/run_drivor_query_single_gt_full_training.sh

# T05：human + 长 GT，普通 Transformer
bash DrivoR/scripts/training/run_drivor_query_single_gt_long_full_training.sh

# A23：human X-L1 + diverse8 L1，Action-DiT
bash DrivoR/scripts/training/run_drivor_action_dit_dinov3_vitl_human_xl1_diverse8_l1_full_training.sh

# A21：human vMSE + diverse8 L1，Action-DiT
bash DrivoR/scripts/training/run_drivor_action_dit_dinov3_vitl_human_diverse8_l1_full_training.sh
```

以上五条是五个独立任务的命令，不是在同一个任务中顺序执行五次训练。`DRIVOR_RUN_ID` 默认采用 DLC Job ID；手工运行时也可自行设置。

## 训练结束后自动 FP32 navtest

需要连续训练和测试时，改用对应入口：

```bash
cd /cpfs29jb/data/zhaozc/project/drive_future

# T07
bash DrivoR/scripts/training/run_drivor_query_human_diverse8_l1_full_train_eval.sh

# T10
bash DrivoR/scripts/training/run_drivor_query_single_gt_full_train_eval.sh

# T05
bash DrivoR/scripts/training/run_drivor_query_single_gt_long_full_train_eval.sh

# A23
bash DrivoR/scripts/training/run_drivor_action_dit_dinov3_vitl_human_xl1_diverse8_l1_full_train_eval.sh

# A21
bash DrivoR/scripts/training/run_drivor_action_dit_dinov3_vitl_human_diverse8_l1_full_train_eval.sh
```

这些入口复用已有的训练后测试流程：所有节点训练，node 0 在训练成功后使用该次训练的 `last.ckpt` 做单节点多 GPU、FP32 navtest。测试采用 CPU 进程池评分，不启动 Ray。

## 输出与恢复训练

训练和测试使用新的 `_trainval_full` 实验目录：

| 原实验 | 训练目录前缀（位于 `DrivoR/exp/ke`） |
|---|---|
| T07 | `training_drivoR_query_dinov3_vitl_human_diverse8_l1_trainval_full` |
| T10 | `training_drivoR_query_dinov3_vitl_single_gt_trainval_full` |
| T05 | `training_drivoR_query_dinov3_vitl_single_gt_long_trainval_full` |
| A23 | `training_drivoR_action_dit_dinov3_vitl_human_xl1_diverse8_l1_trainval_full` |
| A21 | `training_drivoR_action_dit_dinov3_vitl_human_diverse8_l1_trainval_full` |

每个目录下按 `<RunID>/` 保存；checkpoint 位于 `lightning_logs/version_*/checkpoints/`。测试目录把上表的 `training_drivoR_` 换成 `eval_navtest_`，同样带 `_trainval_full` 后缀。

默认 `auto_resume=false`，按各模型原有预训练初始化开始新实验。恢复同一 full 训练时指定相同 Run ID，例如：

```bash
export DRIVOR_RUN_ID=<之前的Full训练RunID>
bash DrivoR/scripts/training/run_drivor_query_human_diverse8_l1_full_training.sh \
  auto_resume=true runtime.require_resume=true
```

也可显式指定 `train_ckpt_path=/完整路径/checkpoint.ckpt`。自动发现 checkpoint 只搜索当前 Run ID，沿用当前普通训练入口的规则。

## 入口与训练周期

- [run_training_full.py](navsim/planning/script/run_training_full.py) 合并 `train_logs + val_logs`，然后复用维护中的 `run_training.py` 训练流程；原始数据模式与 cache-only 模式都采用同一合并集合。
- 保留当前的 rank 随机流、DataLoader 参数、FP32 轮末监测和每轮 checkpoint 保存逻辑。
- full 配置按 **103,288** 条初始化 scheduler 参数，并在数据集构建完成后用 `len(train_data)` 更新。默认 64 卡总 batch 64 时，每轮 **1,614** 个 optimizer steps，25 轮 **40,350** 步；不会沿用原 85,000 条配置造成学习率周期过短。
- `trainval_full_scope.json` 保存合并日志集合；`trainval_full_dataset.json` 保存实际训练／监测样本数和 scheduler 使用的数据量。
- 普通入口仍默认 `run_training.py`；两种入口共享 runtime 配置，旧普通训练配置的解析值保持一致。

2026-10-24 用户改为复用现有 DLC：最初提交的 5 个新任务均已取消，T07 接在 `dlc16mhamjyg8qtw` 后，T05 接在 `dlc1aatxjf950vw7` 后，A23 接在 `dlc1aktix72nfi39` 后，A21 接在 `dlc1aut4azaufm8s` 后。T10 后来按用户要求取消续接，改为独立 DLC `dlcatlogbw4zbpn3`（8 机 × 8 卡，训练后 FP32 navtest），任务记录见 [T10 full 独立提交](/cpfs29jb/data/zhaozc/project/drive_future/artifacts/dlc_submissions/t10_trainval_full_20261024_115236/README.md)。新的 Run ID、命令和输出路径见 [DLC 续接记录](../artifacts/dlc_workspace_26585/chain_trainval_full_20261024_012508/README.md)。本地 checkpoint 扫描评测继续执行。

## 本地验证

- 21 项检查通过：train/val 合并与去重、cache-only 路径的日志范围、按实际数据量设置 scheduler、五份配置保留原模型和 loss、十个启动入口、原普通训练入口兼容性，以及一次 CPU 短训练和 FP32 监测。
- 实际 `full_t07` shell 入口的 Hydra 配置查询通过，确认 `training_data_scope=trainval`、`scheduler_dataset_size_from_data=true`、`validation_is_training_subset=true`。
- 普通 `default_training` 配置重构前后的解析值逐项一致；共用配置拆分用于满足 Hydra 的 searchpath 只能位于主配置中的约束。
- 数据范围与 diverse8 覆盖来自真实场景筛选、metric cache 索引和数据生产摘要。CPU 短训练使用测试用小数据，仅验证流程，不代表 NAVSIM 性能。

记录见 [validation.json](../artifacts/trainval_full_variants_20261024/validation.json)。
