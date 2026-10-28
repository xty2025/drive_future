# DrivoR + CLOVER Stage-1 pseudo-expert coverage

本配置在当前 `run_drivor_training.sh` 的 DrivoR 基线上，额外增加 CLOVER Stage-1 的 pseudo-expert coverage loss。模型仍直接使用 `DrivoRModel`：DINOv2 ViT-S/14、四路当前图像（F0/B0/L0/R0）、672×1148、每视角 16 个 Scene queries、原轨迹生成器、64 个 proposals 和原 Scorer。当前基线已经包含此前的 BF16、分布式、OSS 和 CPU Oracle 加速适配，训练入口是 `run_training.py`。

配置：[`drivoR_pseudo_expert.yaml`](navsim/planning/script/config/common/agent/drivoR_pseudo_expert.yaml)。
启动脚本：[`run_drivor_pseudo_expert_training.sh`](scripts/training/run_drivor_pseudo_expert_training.sh)。

## 1. Pseudo-expert 的来源与选择

数据来自 CLOVER README 发布的 [Stage-1 pseudo-expert trajectory package](https://drive.google.com/drive/folders/1oNTv5Pe-naw_i81rqaKk8KIs0VcUqGZ-?usp=drive_link)，已下载到：

```text
DrivoR/exp/world4scorer_data/source/dataset_decoupled_v2_clean.pkl
```

每个场景提供 `token`、相对坐标候选 `trajectories_relative`、预计算 `scores[].pdm_score` 等数据。这是之前 World4Scorer Bank 使用的同一个源文件。本实验按 CLOVER Stage-1 规则筛出高分专家，直接监督轨迹生成器；原始数据中存在低分候选，不能把整个 PKL 都当成专家。

CLOVER README 明确说明候选生成代码尚未发布。当前复用的是作者发布的数据包、筛选算法和 coverage loss，不推断候选由哪个生成模型产生。

选择严格参考 `CLOVER/navsim/agents/clover/pseudo_expert_utils.py`：

1. 保留 `pdm_score >= 0.8` 的候选，按分数稳定降序排列。
2. 从最高分候选开始做贪心最远点采样，最多选 `top_k=8` 条。距离为 8 个时刻、x/y 两轴的平均绝对差，不使用 heading。
3. 当剩余候选到已选集合的最大最近距离小于 0.05 m 时，停止添加。
4. 若 human GT 到所有已选轨迹的平均 xy 差均不小于 0.5 m，且仍有名额，则补入 human GT；已满 8 条时不替换已有专家。
5. 没有合格候选或该 token 缺失时，仅使用 human GT。其余位置 padding，通过 mask 排除。

最终每场景有 1～8 条有效监督轨迹。human GT 保底与原始 human 主 loss 同时存在；因此仅 human 保底的场景，也会额外得到 `0.5 × human coverage`，这是 CLOVER 的原行为。

## 2. Loss 与梯度

输入为最终生成轨迹 `P [B,64,8,3]`、专家 `E [B,8,8,3]` 和有效 mask `[B,8]`。使用原始 ego 相对坐标（x/y 为米、heading 为弧度），不增加归一化或角度 wrapping。

每个场景内，对每条有效专家独立匹配最近 proposal：

```text
d(i,k) = mean_8时刻( |Px−Ex| + |Py−Ey| + |Pheading−Eheading| )
L_PE(scene) = sum_有效专家 k [min_64个proposal i d(i,k)] / 有效专家数
L_PE = mean_batch L_PE(scene)
```

三个坐标维度取和，八个时间点取平均。专家之间等权；不同专家可以选中同一 proposal。没有匈牙利分配。选中的 proposal 接收对应专家的 coverage 梯度；精确并列时，沿用 CLOVER 的 `amin`，在并列项间均分梯度。

默认训练总 loss：

```text
L_total = L_human + L_long + L_score + 0.5 × L_PE
```

实现先调用原始 `DrivoRLoss`，再将 `0.5 × L_PE` 加到 `trajectory_loss`，并按原 `trajectory_weight` 加入总 loss。普通 GT、长 GT、Scorer 的原 loss 均保留。

coverage 只作用于最后一级 64 个 proposals，与 CLOVER 的 `final_only` 配方一致。原轨迹 loss 的 refinement 累计逻辑保持 `trajectory_loss = prev_weight × previous_loss + current_loss`；当前启动脚本的 `prev_weight=0`，所以最终保留最后一级普通/长 GT loss，不能把它描述成五级 loss 直接相加。

coverage 梯度正常回传到轨迹生成器及其视觉输入；专家目标没有梯度。Scorer 继续对生成的 64 条候选训练，并保留原来的轨迹坐标 detach。该配置没有新增网络模块、模型前向或 Scorer 前向，也没有启用 L_Bank / 未来特征预测分支。

日志增加：

| 日志键 | 含义 |
|---|---|
| `pseudo_expert_loss` | 未乘 0.5 的 coverage loss |
| `pseudo_expert_count` | 包括必要 human 补入后的平均有效专家数 |
| `pseudo_expert_fallback_fraction` | 没有合格离线专家、仅回退到 human 的场景比例 |

`agent.loss.pseudo_expert_weight=0` 恢复原 loss 数值；旧配置和旧启动入口均不受影响。

## 3. 已准备的训练缓存

```text
DrivoR/exp/clover_data/pseudo_experts_k8_thr0p8/
  tokens.npy
  pseudo_experts.npy        # [102978,8,8,3], float32
  pseudo_expert_mask.npy    # [102978,8], bool
  manifest.json
```

已从完整源文件处理 102,978 个非空有效场景、5,627,967 条候选。筛选使用全部原始候选，没有经过 Bank 的前 64 条截断或负样本采样。预选结果中，91,867 个场景有 8 条高分专家，3,881 个场景没有满足条件的高分专家，后者在训练时回退 human。源文件中空场景不写入缓存，按缺失 token 同样回退。

FPS 离线执行一次；训练各进程只 mmap 这个小缓存，按 token 取专家并执行 human 补入规则，无需各自加载约 4 GB 的 PKL。缓存中暂不加入 human，以确保使用当前 NAVSIM 场景提供的真实 GT。

重建方式（当前默认缓存已准备好，可直接训练）：

```bash
cd /cpfs29jb/data/zhaozc/project/drive_future
PYTHONPATH=DrivoR \
/cpfs29jb/data/zhaozc/miniconda3/envs/drive_future/bin/python \
  DrivoR/tools/build_clover_pseudo_experts.py \
  DrivoR/exp/world4scorer_data/source/dataset_decoupled_v2_clean.pkl \
  DrivoR/exp/clover_data/pseudo_experts_k8_thr0p8 \
  --top-k 8 --score-thr 0.8
```

自定义目录使用 `DRIVOR_PSEUDO_EXPERT_PACK`。若更改 top_k 或 score_thr，需要重建对应缓存，并同步 `agent.config.pseudo_expert.top_k` / `score_thr`。

## 4. PAI-DLC 启动命令

任务配置 **8 节点 × 每节点 8 GPU**；DLC 自动分配节点和 rendezvous。保持当前基线的总 batch 64（每卡 1、累积 1）、BF16、25 epochs、seed 2、LR 2e-4，不复制 CLOVER 的其他训练日程。

```bash
cd /cpfs29jb/data/zhaozc/project/drive_future
DRIVOR_EXPERIMENT=training_drivoR_pseudo_expert_navtrain \
DRIVOR_RUN_ID=drivor_pseudo_expert_seed2_run1 \
bash DrivoR/scripts/training/run_drivor_pseudo_expert_training.sh \
  auto_resume=false train_ckpt_path=null runtime.require_resume=false \
  runtime.console_log_interval=25
```

若需要同一个 DLC 任务训完自动测试，用下面的入口代替上面的训练入口。所有节点参加训练，训练成功后仅节点 0 用本机 8 卡评测 navtest，自动查找本次 run 最新 `version_*` 下的 `last.ckpt`：

```bash
cd /cpfs29jb/data/zhaozc/project/drive_future
DRIVOR_TRAIN_EXPERIMENT=training_drivoR_pseudo_expert_navtrain \
DRIVOR_RUN_ID=drivor_pseudo_expert_seed2_run1 \
bash DrivoR/scripts/training/run_drivor_pseudo_expert_train_eval.sh
```

命令后的 Hydra 参数用于训练。该组合入口使用上述标准实验/run 目录；如需自定义 `output_dir`，使用独立训练和测试命令。训练保持 BF16，独立测试和训练后自动测试均默认 FP32（`trainer.params.precision=32-true`）、每卡 batch 1、32 个 CPU 评分进程，不使用 Ray。

图像/日志仍从 `data/navsim` 加载，训练 Oracle 缓存仍为 `data/metric_cache_navtrain`。checkpoint 的模型结构与原 DrivoR 完全相同，独立测试在单节点 DLC 上运行：

```bash
cd /cpfs29jb/data/zhaozc/project/drive_future
DRIVOR_EVAL_WORKER=single_machine_thread_pool DRIVOR_SCORE_WORKERS=32 \
NPROC_PER_NODE=8 DRIVOR_EXPERIMENT=eval_navtest_drivor_pseudo_expert \
DRIVOR_RUN_ID=drivor_pseudo_expert_seed2_run1 \
bash DrivoR/scripts/evaluation/run_drivor_pseudo_expert_evaluation.sh \
  /绝对路径/训练产出的checkpoint.ckpt
```

评测不需要 pseudo-expert 缓存，直接使用原 `drivoR` agent。测试读取 `data/metric_cache_navtest`；上述命令的结果写入 `DrivoR/exp/ke/eval_navtest_drivor_pseudo_expert/drivor_pseudo_expert_seed2_run1/`，包括逐场景 CSV 和 `pdms_summary.txt`。

## 5. 验证

- 64 个真实场景、128 组专家选择结果，与 CLOVER 原函数逐项完全一致。
- 19 项测试通过：覆盖 loss 数值、变长专家均分、重复匹配、梯度、保底规则、原 loss 兼容和 DLC 启动参数。
- 两张 A100 上完成真实图像、完整缓存、BF16/DDP smoke：每卡 2 个训练 step 和 1 个验证 batch，coverage 实际参与反传，loss 有限。仅验证运行链路，不代表完整训练效果。
- Smoke 记录：`../artifacts/drivor_two_gpu_smoke/drivor_pseudo_expert_ppjss6il/`。

实现文件位于 `navsim/agents/drivoR_pseudo_expert/`；数据转换入口为 `tools/build_clover_pseudo_experts.py`。
