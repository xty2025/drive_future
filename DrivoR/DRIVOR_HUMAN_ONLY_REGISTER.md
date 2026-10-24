# A27 / A28：只用 human 未来监督 register

这两版保留 A27/A28 的 human＋diverse8 轨迹监督，仅把 register 的监督范围改为 human。`human_only` 修饰的是未来 register 监督，轨迹仍然使用 `H + D8≤8`。

| 新版本 | 对应基线 | 轨迹 loss | Register loss | Scorer |
|---|---|---|---|---|
| `human_xl1_diverse8_l1_register_human_only` | A27 | `L1(H) + 0.5 L1(D8)` | `Rv(H)` | 原 Scorer，不输入 register |
| `human_xl1_diverse8_l1_register_scorer_human_only` | A28 | `L1(H) + 0.5 L1(D8)` | `Rv(H)` | 轨迹＋预测 register 融合 Scorer |

总 loss 是 `L1(H) + 0.5 L1(D8) + 1.0 Rv(H) + 1.0 Lscore`。

## 匹配与梯度

1. 一次 Action-DiT 前向仍输出 64 条 `[8,3]` 轨迹，以及同位置的 64 组 `[8,4,1024]` register。
2. Human 从 64 条轨迹中选物理坐标 L1 最近者 `m_H`。D8 每条有效 GT 也独立选择最近轨迹，轨迹 loss 仍在有效 D8 内平均后乘 0.5；不同 GT 可以选中同一 proposal。
3. **仅 `register[b,m_H]` 与该场景 human 的真实未来特征对齐**。D8 选中的 proposal 不因其 D8 身份额外获得 register 监督。
4. `Rv(H)` 沿用 X-prediction 转 vMSE、`sigma_min=0.1`、原 timestep 权重；先平均有效未来帧和特征维度，再平均 batch。监督 GT 数为 1，不再除以 human＋D8 的条数。
5. 融合 Scorer 版仍对全部 64 条候选评分，读取每条候选对应的预测 register。轨迹和 register 进入 Scorer 前均 detach；Scorer 的当前视觉梯度沿用 A28。

所有 64 组 action/register 的加噪锚点继续分别来自原 human 轨迹和原 human 未来特征，每组独立高斯噪声、同场景共享 t。DINOv3 ViT-L、当前前视 672×1152、原生 3024 patch K/V、LoRA、单次训练前向、推理十步 Euler、注意力 mask、EMA decay=0.999 均沿用原版。改动没有增加或删除模型参数，同 Scorer 类型的新旧 checkpoint 结构兼容；新训练默认仍按原预训练初始化。

## 数据和兼容性

- 新配置使用 `future_register.supervision_scope: human_only`。未设置该字段的旧配置仍为 `human_and_diverse`，保持 A25～A28 的原行为。
- 只读取 human 的真实未来前视帧，时刻为 +0.5、+1.0、…、+4.0s；经当前 DINOv3 的 EMA teacher 提取每帧四个 register。未来 teacher 输入从 `[B,1+K,8,3,H,W]` 缩为 `[B,1,8,3,H,W]`。
- D8 轨迹及 valid/padding mask 继续读取原主目录。保留 A27/A28 target builder 原有的 main-root 选择规则，不额外引入 history-3 fallback 轨迹。
- 不读取、不抽取、不编码 D8 对应的 Cosmos 视频，target batch 不再包含 D8 的未来图像张量。human-only 模式的缓存名称独立，防止与旧目标缓存混用。
- 仍生成全部 64 组 register；测试时由联合去噪生成，不读取 human 或 D8 的未来 GT 图像。

## 启动

沿用 A27/A28 的 **train-only** 设置：`run_training.py`，训练 85,109 场景、val 18,179 场景；不是 `run_training_full.py`。默认 8 机×8 卡、每卡 batch 1、累积 1、总 batch 64、seed 2、25 epochs、BF16 训练，每 epoch 保存权重并用 FP32 进行 val 监测。

训练后自动 FP32 navtest，两条命令分别用于两个实验：

```bash
cd /cpfs29jb/data/zhaozc/project/drive_future

# A27 对照：仅 human register 对齐，register 不进 Scorer
bash DrivoR/scripts/training/run_drivor_action_dit_dinov3_vitl_human_xl1_diverse8_l1_register_human_only_train_eval.sh

# A28 对照：仅 human register 对齐，register 进 Scorer
bash DrivoR/scripts/training/run_drivor_action_dit_dinov3_vitl_human_xl1_diverse8_l1_register_scorer_human_only_train_eval.sh
```

只训练时，将上述 `_train_eval.sh` 换成 `_training.sh`。`DRIVOR_RUN_ID` 默认使用 DLC Job ID，手工运行可自行指定。训练和测试目录分别为：

```text
DrivoR/exp/ke/training_drivoR_action_dit_dinov3_vitl_navtrain_human_xl1_diverse8_l1_register_human_only/<RunID>
DrivoR/exp/ke/eval_navtest_action_dit_dinov3_vitl_human_xl1_diverse8_l1_register_human_only/<RunID>

DrivoR/exp/ke/training_drivoR_action_dit_dinov3_vitl_navtrain_human_xl1_diverse8_l1_register_scorer_human_only/<RunID>
DrivoR/exp/ke/eval_navtest_action_dit_dinov3_vitl_human_xl1_diverse8_l1_register_scorer_human_only/<RunID>
```

单独测试时调用已有入口，variant 选择对应新版本，例如：

```bash
NPROC_PER_NODE=8 bash DrivoR/scripts/evaluation/run_drivor_diverse8_register_evaluation.sh \
  action_dit_human_xl1_diverse8_l1_register_scorer_human_only /absolute/path/to/last.ckpt
```

配置：[A27 对照](navsim/planning/script/config/common/agent/drivoR_action_dit_dinov3_vitl_human_xl1_diverse8_l1_register_human_only.yaml)、[A28 对照](navsim/planning/script/config/common/agent/drivoR_action_dit_dinov3_vitl_human_xl1_diverse8_l1_register_scorer_human_only.yaml)。两版已于 2026-10-24 提交为独立 DLC 任务：A27 对照 `dlc1gh1ibvu2j54q`，A28 对照 `dlc1gr13pnwwbug8`；完整命令和输出目录见 [提交记录](/cpfs29jb/data/zhaozc/project/drive_future/artifacts/dlc_submissions/human_only_register_20261024_135904/README.md)。

## 本地验证

33 项检查通过，覆盖 human 匹配位置的 register 梯度、D8 轨迹梯度、缺少 D8 视频/图像张量时训练、旧配置行为、模型输出和 checkpoint 参数结构、Scorer detach、EMA 更新以及无未来 GT 的推理。两份真实 shell 入口的 Hydra 参数查询也通过；查询时使用轻量 scene filter，正式训练命令仍采用 navtrain。以上是 CPU 小模型与数据读取 smoke，不代表新的 NAVSIM 性能。记录见 [validation.json](../artifacts/human_only_register_20261024/validation.json)。
