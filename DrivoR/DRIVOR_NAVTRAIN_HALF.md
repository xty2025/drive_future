# NAVSIM 半量 train 配置

新增 `navtrain_half`：训练场景从 **85,109 减为 42,555**，验证仍为 **18,179**。使用 `run_training.py`，没有把 val 合并进训练。

A23、A27、A28 的半量基线和 A28 的 12 个独立改动，共 15 组训练＋FP32 测试入口，见 [半量实验配置与命令](DRIVOR_NAVTRAIN_HALF_ABLATIONS.md)。这些实验统一写入 `exp/navtrain_half_ablations/`。

只替换 scene-filter 的 `tokens`；原 `log_names`、train/val 日志名单、4 帧历史、10 帧可用未来、`frame_interval=1` 均保留。每条轨迹内部仍是 4 秒、2 Hz、8 点；减少的是训练场景起点。

| 检查 | 直接按原 train tokens `[::2]` | 最终：每 log 内按时间隔项 |
|---|---:|---:|
| train 场景数 | 42,555 | 42,555 |
| 保留训练日志数 | 974/978 | 978/978 |
| 日志分布 TV 距离 | 0.036383 | 0.002840 |
| human 轨迹/运动统计最大 KS 距离 | 0.006057 | 0.001497 |
| 轨迹类别最大比例差 | 0.1975 个百分点 | 0.1014 个百分点 |

最终方案每个日志内按时间隔一个取一个；奇数日志分配 floor/ceil 名额，保证总数恰好 42,555，单场景日志保留。选样只用日志与时间，不按未来轨迹筛选；写回时仍保留原 YAML 的 token 相对顺序。

完整 85,109 个 train human GT 已逐场景统计，所有 1,192 个原始 train/val 日志的过滤结果也已核对。路径长度、航向、终点、当前速度、城市、车辆、指令和日志比例接近；图中使用各自样本数归一化。完整统计、图和限制说明见 [分布检查报告](exp/navtrain_half/REPORT.md)。这是数据分布检查，没有运行模型或 NAVSIM oracle。

原始轨迹曲线另见 [轨迹可视化](exp/navtrain_half/TRAJECTORY_VISUALIZATIONS.md)：包含全部轨迹对照、各 1,200 条等量叠加，以及每类各 200 条的静止/直行/左右转对照。均为真实 human GT 的 8 点连线，出图抽样不改变训练 tokens。

## 启动

```bash
cd /cpfs29jb/data/zhaozc/project/drive_future

# 原版 DrivoR，半量 train，完整 val
bash DrivoR/scripts/training/run_drivor_navtrain_half_training.sh

# A27 架构，半量 train：human + D8 轨迹、register 不进 Scorer
DRIVOR_HALF_LAUNCHER=run_drivor_action_dit_dinov3_vitl_human_xl1_diverse8_l1_register_training.sh \
  bash DrivoR/scripts/training/run_drivor_navtrain_half_training.sh

# A28 架构，半量 train：human + D8 轨迹、register 进 Scorer
DRIVOR_HALF_LAUNCHER=run_drivor_action_dit_dinov3_vitl_human_xl1_diverse8_l1_register_scorer_training.sh \
  bash DrivoR/scripts/training/run_drivor_navtrain_half_training.sh
```

`DRIVOR_HALF_LAUNCHER` 可以换成其他已有的普通 `*_training.sh`，包括两个 `human_only_training.sh`。`*_full_*` 和 `*_train_eval.sh` 会在启动前被拒绝。这些是训练命令，不提交 DLC，也不自动启动 navtest；训练期继续完整 val 的每 epoch FP32 评测和原检查点保存策略。

默认仍是 8 机×8 卡、总 batch64、25 epoch。当前 half 配置每 epoch 约 665 个优化器步骤（训练 DataLoader 的 `drop_last` 配置会影响最后一批），原版约 1,330；总更新预算随数据量降低，不改变每场景的 GT 或 64 个 proposal。

入口强制使用 [half_training.yaml](navsim/planning/script/config/training/half_training.yaml) 和 `run_training.py`，并显式关闭 `use_cache_without_dataset`：原裸入口的 cache-only 分支会忽略 scene-filter tokens，不能只替换 YAML 名称后继续用它。最终 CLI 参数保留 `--config-name=half_training`，因此 `human_only` 内层脚本重设配置环境变量也不会切回 `default_training`；Hydra 采用最后一个同名选项。`--cfg job` 查询也已通过实际 Hydra 参数解析检查。

`runtime.scheduler_dataset_size_from_data=true` 会在数据集建立后让 scheduler 使用实际 42,555 条训练数据，避免仍按 85k 场景走学习率周期。新的实验输出目录带 `_navtrain_half`，原全量脚本和配置未修改。

## 配置与复现

- [数据组](navsim/planning/script/config/common/train_test_split/navtrain_half.yaml)
- [完整 half scene-filter](navsim/planning/script/config/common/train_test_split/scene_filter/navtrain_half.yaml)：60,734 个 tokens = 42,555 train + 18,179 val；`run_training.py` 分别按日志名单相交。
- [抽样与分布分析工具](tools/build_navtrain_half.py)
- [启动脚本](scripts/training/run_drivor_navtrain_half_training.sh)
- [回归检查](tests/test_navtrain_half.py)

```bash
# 重新读原始日志并生成配置、统计和图；只需 numpy/pyyaml/matplotlib
python DrivoR/tools/build_navtrain_half.py

# 已有特征统计时重画图，不重复读原始日志
python DrivoR/tools/build_navtrain_half.py --reuse-features

# 只画真实轨迹曲线，不重写采样 YAML 或训练 token 文件
python DrivoR/tools/plot_navtrain_half_trajectories.py
```

本地 12 项检查通过，覆盖真实数据数量/分布、原 loader 的过滤逻辑、原 SE(2) 轨迹转换、Hydra 组合/实际参数解析、五个模型入口的转发（含两种 human-only），以及 full/train_eval 入口的提前拒绝；未启动训练或 DLC。
