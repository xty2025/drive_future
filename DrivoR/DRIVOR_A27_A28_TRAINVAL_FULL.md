# A27 / A28：trainval 全量训练

这两版基于结果表中的原 A27/A28，轨迹与未来 register 都保留 **human＋有效 D8** 监督。训练入口改为 `run_training_full.py`，将原来的 85,109 个 train 场景和 18,179 个 val 场景合并为 **103,288** 个训练场景。

| 配置 | 轨迹 loss | Register loss | Register 进入 Scorer |
|---|---|---|---|
| [full_a27.yaml](navsim/planning/script/config/training/full_a27.yaml) | `L1(H) + 0.5 L1(D8)` | `Rv(H,D8)` | 否 |
| [full_a28.yaml](navsim/planning/script/config/training/full_a28.yaml) | `L1(H) + 0.5 L1(D8)` | `Rv(H,D8)` | 是，预测轨迹/register detach 后融合 |

`Rv(H,D8)` 按有效 GT 平均，权重为 1；各 GT 使用其轨迹匹配到的 proposal 的 register，对齐对应的八帧未来。两版均另外计算原 Scorer loss。仍为当前前视 DINOv3 ViT-L、672×1152、64 个 proposal、每 proposal 32 个未来 register、单次训练前向、十步 Euler 推理和原 EMA teacher。初始化沿用原预训练方式。

此前 `*_register_human_only*` 的两个配置用于另一组消融；本页 full 配置使用原 A27/A28 的 human＋D8 未来监督。

## 轨迹和视频路径

训练统一从主目录按 token 读取 GT：

```text
/cpfs29jb/data/zhaozc/project/Cosmos3forMultiGT/runtime/multigt_raw64/outputs/navtrain_diverse8_dense_0p1s_f64
```

其中 205 个 token 目录已软链接到补充目录：

```text
/cpfs29jb/data/zhaozc/project/Cosmos3forMultiGT/runtime/multigt_raw64/outputs/navtrain_diverse8_hist3_dense_0p1s_f64
```

完整 GT 覆盖为主批次 103,083＋补充 205＝103,288。新增 205 个场景中，161 个属于 train、44 个属于 val。`complete.json`、轨迹、mask 和轨迹元数据都会跟随软链接读取；不是只按主批次旧汇总文件的 103,083 个 token 训练。

D8 对应视频统一来自：

```text
/cpfs29jb/data/zhaozc/project/Cosmos3forMultiGT/runtime/rollouts/navtrain_diverse8_full_allcosmos_cuda_v1_gpu256
```

需要 D8 视频的场景有 103,227 个，视频 token 目录覆盖一致。另有 61 个场景没有有效 D8，这些场景保留 human 轨迹和 human 未来监督，不要求生成不存在的 D8 视频。

补充 205 个场景的 1,621 段有效视频已逐项检查完成标记、slot、候选 ID、标签、文件存在性和时间数组。主批次依据生产完成汇总及 token 目录覆盖核对，没有逐帧解码整个视频库。核对记录见 [data_coverage.json](../artifacts/a27_a28_trainval_full_20261024/data_coverage.json)。

## 抽帧和缓存

视频为 41 帧、10 Hz，第 0 帧是当前帧。仍取零基索引 **5、10、15、20、25、30、35、40**，对应轨迹的 +0.5～+4.0s、2 Hz、8 个点。D8 使用相应视频，human 使用 NAVSIM 真实未来前视帧。

继续复用 `DrivoR/exp/world4scorer_data/frames_diverse8/<token前两位>/<token>.zip`。缓存缺失时按需抽取；无需为 full 实验重建所有旧缓存。读取器增加了软链接缓存兼容：从补充 GT 目录生成的缓存，可通过主目录的相同场景/元数据软链接复用；仍检查实际轨迹、候选、slot 和时间是否一致，不增加哈希校验。

## DLC 启动命令

分别用于两个 8 节点×8 GPU 的 PyTorch DLC 任务。默认 batch=1/卡、梯度累积=1、总 batch=64、seed=2、25 epochs、BF16 训练。训练结束后自动在主节点执行 **FP32 navtest**：

```bash
cd /cpfs29jb/data/zhaozc/project/drive_future

# A27 full：human+D8 未来 register 对齐，register 不进 Scorer
bash DrivoR/scripts/training/run_drivor_action_dit_dinov3_vitl_human_xl1_diverse8_l1_register_full_train_eval.sh

# A28 full：human+D8 未来 register 对齐，register 进 Scorer
bash DrivoR/scripts/training/run_drivor_action_dit_dinov3_vitl_human_xl1_diverse8_l1_register_scorer_full_train_eval.sh
```

只训练时分别使用：

```bash
bash DrivoR/scripts/training/run_drivor_action_dit_dinov3_vitl_human_xl1_diverse8_l1_register_full_training.sh
bash DrivoR/scripts/training/run_drivor_action_dit_dinov3_vitl_human_xl1_diverse8_l1_register_scorer_full_training.sh
```

数据和视频默认即上述路径；需要切换时用 `DRIVOR_DIVERSE8_GT_ROOT`、`DRIVOR_DIVERSE8_ROLLOUTS_ROOT`，缓存目录用 `DRIVOR_DIVERSE8_FRAMES_ROOT`。

每 epoch 保存 checkpoint，并以 FP32 监测原 val 子集。**原 val 已进入训练，因此此时的 val PDMS 是训练子集监测；独立成绩仍看 navtest。**默认每轮 1,614 个 optimizer steps、25 轮共 40,350 步，scheduler 按实际训练样本数更新。

## 输出

`DRIVOR_RUN_ID` 默认采用 DLC Job ID。输出分别是：

```text
DrivoR/exp/ke/training_drivoR_action_dit_dinov3_vitl_human_xl1_diverse8_l1_register_trainval_full/<RunID>
DrivoR/exp/ke/eval_navtest_action_dit_dinov3_vitl_human_xl1_diverse8_l1_register_trainval_full/<RunID>

DrivoR/exp/ke/training_drivoR_action_dit_dinov3_vitl_human_xl1_diverse8_l1_register_scorer_trainval_full/<RunID>
DrivoR/exp/ke/eval_navtest_action_dit_dinov3_vitl_human_xl1_diverse8_l1_register_scorer_trainval_full/<RunID>
```

训练目录中 `trainval_full_scope.json` 记录合并日志集合，`trainval_full_dataset.json` 记录实际训练/监测样本数。默认开始新训练，保留当前 Run ID 的显式恢复能力。

2026-10-24 21:27 已分别提交两个新 DLC 任务，每个 8 机 × 8 卡；提交后服务端状态均为 `Queuing`：

| 实验 | DLC Job ID | 本次 Run ID |
|---|---|---|
| A27 full | `dlc1mewgzvmn8ccp` | `a27_trainval_full_seed2_20261024_212720` |
| A28 full | `dlc1mow2dnqzcpri` | `a28_trainval_full_seed2_20261024_212720` |

服务端命令、资源和挂载已核对；详细启动命令及本次训练/测试输出路径见 [DLC 提交记录](../artifacts/dlc_submissions/a27_a28_trainval_full_20261024_212720/README.md)。

## 验证

- 32 项隔离组件检查通过：读取器使用实际生产方法；覆盖目录软链接、文件软链接、主目录/补充目录产生的缓存、GT 和时间/槽位一致性、两份新配置，以及原五份 full 配置和普通训练入口的兼容性。
- 真实数据 smoke 覆盖主目录 train 场景 `0002267a294e52a7` 和补充目录 val 场景 `1b84a56a7dd75345`；各读取 8 条有效 D8、64 张对应未来帧。前者复用已有 ZIP，后者成功按需抽帧并写入共享缓存。
- DSW 共享文件系统的文件句柄限制阻断了完整 pytest 栈的 pandas/CUDA 导入，因此改用隔离 CPU 组件环境执行已有测试体和真实读取代码；未运行完整模型训练，未改生产 Conda 环境，也不据此报告 NAVSIM 分数。

完整记录：[validation.json](../artifacts/a27_a28_trainval_full_20261024/validation.json)、[component_validation.json](../artifacts/a27_a28_trainval_full_20261024/component_validation.json)、[real_frame_smoke.json](../artifacts/a27_a28_trainval_full_20261024/real_frame_smoke.json)。
