# Action-DiT：human / 非 human 多 GT 加权监督

更新：2026-10-15。新增单前视和多视图两个独立配置；原来的等权多 GT、匈牙利等权入口保持原行为。

[DINOv3 ViT-S / ViT-L 版本](DRIVOR_ACTION_DIT_DINOV3.md) 已于 2026-10-16 改为有效 GT 等权平均，不再默认继承本文配方；本文的独立 weighted 入口保留供旧实验复现。

## Loss 配方

按原版 DrivoR 的 human/辅助目标分组方式，对 Action-DiT 的 v loss 加权：

```text
L_scene = human_weight * L_human
        + other_weight * mean_有效非human_GT(L_other_GT)

L_trajectory = mean_scene(L_scene)
L_total = L_trajectory + L_scorer
```

默认 `human_weight=1.0`，`other_weight=0.5`。human 项单独保留完整权重，只对非 human 组按其实际条数平均；不再把两组的和除以总 GT 数或权重和。

有 1 条 human 和 3 条非 human GT 时：

```text
L_scene = L_human + 0.5 * (L_other1 + L_other2 + L_other3) / 3
```

因此 `other_weight=0.5` 是整个辅助平均项的系数，每条辅助 GT 在此例中的系数为 `0.5/3`。只有 human GT 时，辅助项为 0；batch 汇总仍包含这类场景，不只平均有辅助 GT 的场景。

## 匹配与 v loss

使用 `matching=nearest`：human 和每条有效非 human GT 都从同一组 64 条预测中独立选最近的 proposal，允许多条 GT 选中同一个候选。只对选中的配对计算直接回归 loss。

各 GT 的匹配距离和 v loss 仍为：

```text
C[k,m] = mean_8×3 ((normalize(GT[k]) - X_prediction[m]) / max(sigma, 0.1))²
L_GT[k] = training_weight(t) * min_m C[k,m]
```

human/非 human 的分组权重在各自的回归 loss 上施加，不改变最近邻的选择。归一化、时间权重和小 t 分母下界保持现有 Action-DiT 实现，没有改成 DrivoR 的物理坐标 L1。

## human 身份与训练输入

human 主项使用原始 NAVSIM `targets["trajectory"]`。导出 GT 按 `gt_labels == "human_log"` 识别 human 行，无论它在第几行，都从辅助集合排除。

非 human 的有效条件为 `gt_valid_mask & ~gt_padded & ~is_human`；补齐行不参与平均。缺失导出 GT 时，沿用 `missing_policy=original`，回退为原始 human GT 一条。

全部 64 个输入仍由原始 human GT 加噪构造，使用 64 份独立噪声和场景内共享的 t。**只做一次 DiT 前向**；两组 loss 读取同一个 `pred_x`，不为辅助 GT 追加网络前向。

全部 64 条 X-prediction 继续经轨迹坐标 detach 后送给原 Scorer，评分 loss、在线标签、视觉特征和 ego 编码不变。验证/推理继续使用 64 份纯噪声与 10 步 Euler。

## 配置与启动

| 模型 | 新配置 | 新入口 |
|---|---|---|
| 单前视原生 patch | [drivoR_action_dit_multigt_weighted.yaml](navsim/planning/script/config/common/agent/drivoR_action_dit_multigt_weighted.yaml) | [run_drivor_action_dit_multigt_weighted_training.sh](scripts/training/run_drivor_action_dit_multigt_weighted_training.sh) |
| 四视角、每视角 16 queries | [drivoR_action_dit_multiview_multigt_weighted.yaml](navsim/planning/script/config/common/agent/drivoR_action_dit_multiview_multigt_weighted.yaml) | [run_drivor_action_dit_multiview_multigt_weighted_training.sh](scripts/training/run_drivor_action_dit_multiview_multigt_weighted_training.sh) |

两个新配置均包含：

```yaml
config:
  multi_gt:
    matching: nearest
    weighting: drive_jepa
    human_weight: 1.0
    other_weight: 0.5
```

PAI-DLC 仍配置 8 节点 × 每节点 8 GPU，总 batch 64。单前视版的新实验命令：

```bash
cd /cpfs29jb/data/zhaozc/project/drive_future
DRIVOR_EXPERIMENT=training_drivoR_action_dit_navtrain_multigt_weighted \
DRIVOR_RUN_ID=action_dit_multigt_weighted_seed2_run1 \
DRIVOR_PRECISION=bf16-mixed \
bash DrivoR/scripts/training/run_drivor_action_dit_multigt_weighted_training.sh \
  auto_resume=false train_ckpt_path=null runtime.require_resume=false \
  runtime.console_log_interval=25
```

多视图版：

```bash
cd /cpfs29jb/data/zhaozc/project/drive_future
DRIVOR_EXPERIMENT=training_drivoR_action_dit_multiview_navtrain_multigt_weighted \
DRIVOR_RUN_ID=action_dit_multiview_multigt_weighted_seed2_run1 \
DRIVOR_PRECISION=bf16-mixed \
bash DrivoR/scripts/training/run_drivor_action_dit_multiview_multigt_weighted_training.sh \
  auto_resume=false train_ckpt_path=null runtime.require_resume=false \
  runtime.console_log_interval=25
```

需要降低非 human 的影响，在命令末尾追加 `agent.config.multi_gt.other_weight=0.2`；human 权重也可用 `agent.config.multi_gt.human_weight=1.0` 覆盖。设 `other_weight=0` 则只保留 human 的轨迹回归项，Scorer 仍监督全部候选。

原脚本 `run_drivor_action_dit_multigt_training.sh` 继续使用等权多 GT；匈牙利脚本继续使用一对一等权。加权配置当前采用独立最近邻匹配，不与匈牙利配置混用。

## 兼容性与日志

模型参数结构未改变，同一视觉变体的完整 checkpoint 兼容。新实验使用独立目录，默认关闭恢复的示例会继续使用原有 Action-DiT 预训练初始化。

加权配置复用 `drivor_multi_gt_human_v1` 目标缓存，包含 human 身份 mask；它不能用没有身份 mask 的旧等权目标缓存替代。默认在线读取无需重新生成多 GT 文件。

日志中的 `multi_gt_human_loss`、`multi_gt_other_loss` 均包含原有时间权重，但尚未乘 human/other 分组系数；`flow_multi_gt_loss` 和 `trajectory_loss` 是分组加权后的结果。`multi_gt_other_count` 记录每场景有效非 human GT 的平均条数。

已通过 **144 项针对性检查**，包括辅助条数 0–3、自定义权重、小 t 分母下界、未匹配输出没有直接回归梯度、一次共享 64 候选前向，以及原等权/匈牙利/DrivoR 路径和启动脚本的回归。单前视、多视图两个配置均完成真实双 A100/BF16/NCCL 两步训练和每卡一个验证 batch；平均有效 GT 数为 2.25、非 human GT 数为 1.25、fallback 为 0。

实现见 [multi_gt.py](navsim/agents/drivoR/multi_gt.py) 和 [Action-DiT model](navsim/agents/drivoR_action_dit/model.py)。验证记录见 [validation.json](../artifacts/drivor_action_dit_multigt_weighted_20261015/validation.json)；短 smoke 仅验证训练链路，不代表全量 NAVSIM 成绩或收敛情况。
