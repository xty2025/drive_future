# DrivoR/exp/ke 实验配置与 navtest 结果总表

统计时间：**2026-10-24 11:55（北京时间）**。本次依据已保存的训练配置、被测 checkpoint 和测试 CSV 汇总，未完成实验的进度来自本地训练日志。

共 **51 份已完成的 navtest 结果**：Action-DiT 28 份、DrivoR 原始解码器系列 13 份、普通 Transformer 10 份，主表各行均已有完整测试结果。分阶段 register 的 Stage1 归入对应 Stage2 实验，不作为独立规划测试结果。

51 份 CSV 均为同一组 **12,146 个有效且不重复的 navtest 场景，失败 0**。下表 PDMS 为 CSV `token=average` 行的 `score × 100`，已重新计算逐场景均值核对；点击分数可查看原始 CSV 及 NC、DAC、EP、TTC、Comfort、DDC 分项。表内不使用训练分数或 val PDMS 代替测试性能。

**比较口径：**历史测试有 **13 份 BF16、38 份 FP32**，精度逐行标明。官方 103k 发布权重使用的数据规模也不同。`last`、`best e23`、`e24` 按实际测试权重分别记录，不按测试集分数挑选 checkpoint；`e24` 表示从 0 开始计数的第 25 轮。

## 表内符号与 GT 来源

| 符号 | 每场景数量 | 来源与使用方式 |
|---|---:|---|
| **H** | 1 | NAVSIM 原始 human log 的未来轨迹，8 点、0.5～4.0 秒；从 `data/navsim` 场景读取。 |
| **HL** | 1 个附加监督目标 | 同一条 human log 读取 10 个未来点后，用原 DrivoR CubicSpline 方式重采样成 8 点，`long_trajectory_additional_poses=2`。它是长时间范围监督，不是另一个独立驾驶模式。 |
| **M4** | 1～4 个有效目标 | 旧 top4 导出目录：`multi_traj_gt_data/navtrain_multigt_top4_dense_0p1s_f64_8gpu`。包含筛选的 progress/stop 等候选以及部分场景的 `human_log`；**并不保证每个场景都含 human**。缺失或无有效轨迹时，当前 `missing_policy=original` 回退到 H。 |
| **P4** | 0～4 | `multi_traj_gt_data/navtrain_poor4_multigt_dense_0p1s_f64_16gpu`，导出的 poor/negative 轨迹。只在 A10 中当轨迹回归目标，和 M4 合并后最多 8 条，不足时按实际数量。 |
| **PE8** | 最多 8 个辅助目标 | CLOVER 包 `DrivoR/exp/clover_data/pseudo_experts_k8_thr0p8`；源文件为 `DrivoR/exp/world4scorer_data/source/dataset_decoupled_v2_clean.pkl`，过滤 PDMS≥0.8 的候选，以最高分候选起步做 XY 最远点采样。训练时保留 CLOVER 的 human 补入规则，见下文。 |
| **D8** | 0～8 条生成轨迹 | 新 diverse8 导出：`/cpfs29jb/data/zhaozc/project/Cosmos3forMultiGT/runtime/multigt_raw64/outputs/navtrain_diverse8_dense_0p1s_f64`，从每场景 64 条候选筛选；读取已完成场景的有效非 padding 条目。生成的 GT-like 轨迹不是原始 H，H 在主监督中另算。 |
| **Bank16** | 每次抽 16 条；库容量≤64 | `DrivoR/exp/world4scorer_data/bank_clover`，和 PE8 来自同一个 CLOVER PKL。只给 Scorer 增加监督，不增加轨迹生成器的 GT 数。 |

以上数量均指**有效监督条目上限**，不是一律填满的数量。`gt_valid_mask` 还要排除 `gt_padded`；不足 4/8 条时只平均实际有效条目。M4、P4 和 D8 都读取自车坐标下的 `gt_trajectories.npy [K,8,3]`，不把全局坐标 `selected_ego_poses_global.npy` 直接作为回归目标。

**M4 的 human 区别：**普通等权 M4 实验只监督 M4 的有效行，不额外强制加 H；Action-DiT 仍以原 H 构造加噪输入。JEPA 加权实验 D06 则单独监督 H，并把 M4 中 `human_log` 行从辅助项剔除。因此 D06 最多是 `1 + 4` 个监督条目，不一定等于 4。

**PE8 的 human 补入：**若已选 PE8 没有轨迹在 XY 平均 L1 的 0.5 m 范围覆盖 H，且不足 8 条，会再补入 H；全无候选时也会用 H 回退。补入后 mask 为有效，当前 coverage/vMSE 不会按 human 身份剔除它，所以 `H/M4 + PE8` 可能重复监督 human；“最多 12 条”表示标签条目数，不保证 12 条不同轨迹。

**视觉列：**分辨率均写为高×宽；“4 视角”为当前 F0/L0/R0/B0，“前视”只有当前 F0。Q16 是每视角额外注入的 16 个场景 query；原生 patch 模式不额外注入这些 query。骨干内部原有 CLS/register 与后文预测的“未来 register”是不同对象，保留预训练 register 不表示启用了未来对齐。V-JEPA2 适配器将当前图像重复两次满足 tubelet=2，V-JEPA2.1 使用原生单图像分支，二者都没有额外历史帧输入。

**Register 两列分别表示生成与评分两个步骤：**“同步预测”表示生成 64 条轨迹的同一生成器同时输出 64 组未来 register，每组 32 个；“输入 Scorer”表示这些**预测值**是否经 detach、压缩后，与对应轨迹特征融合用于评分。**同步预测但不输入 Scorer**的实验仍有 register 预测与对齐 loss，评分仍沿用原轨迹、当前视觉和 ego 输入。EMA teacher 的 GT register 只作训练目标，不直接输入 Scorer。World 的独立未来特征 predictor 单独标注；A17 则分别注明 Stage1 仅预测 register、Stage2 联合生成与评分。

<!-- DIVERSE8_FUTURE_SUPPLEMENT -->
**新增 diverse8 联合预测实验：**A25～A28、T08/T09 都保留 64 条 proposal，每条同步预测 32 个未来 register。H 的轨迹项保持对应基线的 vMSE/X-L1，D8 始终使用 `0.5×物理 L1 coverage`；未来项把 **H＋所有有效 D8 等权平均，系数 1**。各 GT 按自身轨迹 loss 选择 proposal，再用同 batch、同位置的 register 对齐该 GT 的未来，允许多个 GT 选中同一 proposal。

**轨迹/视频频率：**D8 训练标签为 **4 秒、2 Hz、8 个点**（+0.5、+1.0、…、+4.0s）。配套视频为 10 Hz、41 帧，第 0 帧是当前帧；取**零基索引 5、10、15、20、25、30、35、40**，得到逐点对应的八帧。`selected_ego_poses_global.npy` 的 0.1s 密集表示用于视频条件/插值，不作为 41 点训练标签。

新增视频根目录：`/cpfs29jb/data/zhaozc/project/Cosmos3forMultiGT/runtime/rollouts/navtrain_diverse8_full_allcosmos_cuda_v1_gpu256`。RGB 缓存为 `DrivoR/exp/world4scorer_data/frames_diverse8`，按需抽帧为每场景一个 ZIP，六组复用；H 的未来来自真实 NAVSIM 图像。EMA teacher decay=0.999，未来图像 384×384。该目录与旧 M4 的 `frames_cosmos` 分开。

六组对应的 DLC 编号、配置和日志链接已写入各自表格行。DLC 容器可被复用，区分实验时以本行 Run ID/训练目录为准。详细架构与命令见 [Diverse8 register 实验说明](../../DRIVOR_DIVERSE8_REGISTER_EXPERIMENTS.md)，部署核验见 [六组启动记录](/cpfs29jb/data/zhaozc/project/drive_future/artifacts/dlc_workspace_26585/launch_diverse8_register_20261022_210127/README.md)；已完成项目补入实际测试结果，尚无完整 navtest CSV 的项目保持空白。
<!-- END_DIVERSE8_FUTURE_SUPPLEMENT -->

## Action-DiT

本组轨迹主干均为 **预训练 Action-DiT：30 blocks、hidden=256、FFN=1024、attention=24×128**，从指定 FastWAM Action-DiT 权重初始化；时间与 ego embedding 相加进入 adaLN。测试均为 **64 个 proposal、10 步 Euler**。训练使用一次 X-prediction 得到干净轨迹并交给 Scorer；有 HL 的旧/新单 GT 配方将普通/长 GT 的 64＋64 个输入合批，Scorer 只评分普通组 64 条。

初始化文件：`/cpfs29jb/data/zhaozc/project/FastWam_re/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_256hdim_1024ffn_3outdim.pt`；A17 Stage2 在此结构基础上继续加载 Stage1 的已训练模型和 EMA。

`V` 为 X-prediction 转 vMSE；`L1` 为物理坐标直接 X-L1；`Rv` 为未来 register 的 X-prediction 转 vMSE。精确定义在表后。所有本地视觉骨干均加载预训练、冻结基座、训练 Q/V LoRA rank 32。

| 实验与配置快照 | 视觉骨干 / 输入 / 下游 KV | 轨迹 GT 数量与来源 | 生成轨迹时同步预测 Register？ | 预测 Register 输入 Scorer？ | 未来对齐目标 / 方式 | 轨迹与未来 loss（另加评分项） | Scorer / 评分 loss | 测试权重 / 精度 | navtest PDMS ↑ |
|---|---|---|---|---|---|---|---|---|---:|
| **A01 DINOv2 前视，human＋长 GT**<br>[训练配置][A01-cfg] | DINOv2-S<br>前视 672×1148<br>原生 patch；KV=3936 | H＋HL：2 项 | **否**（仅生成轨迹） | **否** | 无 | `V(H)+V(HL)` | S0 | [v3/e24][A01-ckpt]<br>BF16 | [**90.67**][A01-csv] |
| **A02 DINOv2 前视，旧 multi-GT**<br>[训练配置][A02-cfg] | DINOv2-S<br>前视 672×1148<br>原生 patch；KV=3936 | M4≤4 | **否**（仅生成轨迹） | **否** | 无 | `V(M4)` | S0 | [v0/e24][A02-ckpt]<br>BF16 | [**90.04**][A02-csv] |
| **A03 DINOv2 多视角＋Q16，human＋长 GT**<br>[训练配置][A03-cfg] | DINOv2-S<br>4 视角 672×1148<br>每视角 Q16；KV=64 | H＋HL：2 项 | **否**（仅生成轨迹） | **否** | 无 | `V(H)+V(HL)` | S0 | [v0/e24][A03-ckpt]<br>BF16 | [**89.39**][A03-csv] |
| **A04 DINOv2 多视角＋Q16，旧 multi-GT**<br>[训练配置][A04-cfg] | DINOv2-S<br>4 视角 672×1148<br>每视角 Q16；KV=64 | M4≤4 | **否**（仅生成轨迹） | **否** | 无 | `V(M4)` | S0 | [v0/e24][A04-ckpt]<br>BF16 | [**83.12**][A04-csv] |
| **A05 DINOv3-S，旧 multi-GT**<br>[训练配置][A05-cfg] | DINOv3-S<br>前视 672×1152<br>原生 patch；KV=3024 | M4≤4 | **否**（仅生成轨迹） | **否** | 无 | `V(M4)` | S0 | [v0/last][A05-ckpt]<br>BF16 | [**87.28**][A05-csv] |
| **A06 DINOv3-L，旧 multi-GT 基线**<br>[训练配置][A06-cfg] | DINOv3-L<br>前视 672×1152<br>原生 patch；KV=3024 | M4≤4 | **否**（仅生成轨迹） | **否** | 无 | `V(M4)` | S0 | [v0/last][A06-ckpt]<br>BF16 | [**92.08**][A06-csv] |
| **A07 V-JEPA2-L，旧 multi-GT**<br>[训练配置][A07-cfg] | V-JEPA2-L<br>前视 672×1152<br>原生 patch；KV=3024 | M4≤4 | **否**（仅生成轨迹） | **否** | 无 | `V(M4)` | S0 | [v0/last][A07-ckpt]<br>BF16 | [**85.22**][A07-csv] |
| **A08 V-JEPA2.1-L，旧 multi-GT**<br>[训练配置][A08-cfg] | V-JEPA2.1-L<br>前视 672×1152<br>原生 patch；KV=3024 | M4≤4 | **否**（仅生成轨迹） | **否** | 无 | `V(M4)` | S0 | [v0/last][A08-ckpt]<br>BF16 | [**91.49**][A08-csv] |
| **A09 DINOv3-L，human＋长 GT**<br>[训练配置][A09-cfg] | DINOv3-L<br>前视 672×1152<br>原生 patch；KV=3024 | H＋HL：2 项 | **否**（仅生成轨迹） | **否** | 无 | `V(H)+V(HL)` | S0 | [v0/last][A09-ckpt]<br>FP32 | [**92.42**][A09-csv] |
| **A10 DINOv3-L，旧 multi-GT＋poor GT**<br>[训练配置][A10-cfg] | DINOv3-L<br>前视 672×1152<br>原生 patch；KV=3024 | M4＋P4≤8；合并等权 | **否**（仅生成轨迹） | **否** | 无 | `V(M4∪P4)` | S0 | [v0/last][A10-ckpt]<br>FP32 | [**91.96**][A10-csv] |
| **A11 DINOv3-L，旧 multi-GT＋Bank**<br>[训练配置][A11-cfg] | DINOv3-L<br>前视 672×1152<br>原生 patch；KV=3024 | M4≤4；Bank 不算 GT | **否**（仅生成轨迹） | **否** | 无 | `V(M4)` | S0＋0.5 LBank | [v0/last][A11-ckpt]<br>FP32 | [**92.35**][A11-csv] |
| **A12 DINOv3-L，旧 multi-GT＋PE L1**<br>[训练配置][A12-cfg] | DINOv3-L<br>前视 672×1152<br>原生 patch；KV=3024 | M4≤4＋PE8≤8；最多12条 | **否**（仅生成轨迹） | **否** | 无 | `V(M4)+0.5 L1(PE8)` | S0 | [v0/last][A12-ckpt]<br>FP32 | [**92.61**][A12-csv] |
| **A13 DINOv3-L，旧 multi-GT＋PE vMSE**<br>[训练配置][A13-cfg] | DINOv3-L<br>前视 672×1152<br>原生 patch；KV=3024 | M4≤4＋PE8≤8；最多12条 | **否**（仅生成轨迹） | **否** | 无 | `V(M4)+0.5 V(PE8)` | S0 | [v0/e24][A13-ckpt]<br>FP32 | [**91.93**][A13-csv] |
| **A14 DINOv3-L，multi-GT＋register**<br>[训练配置][A14-cfg] | DINOv3-L<br>前视 672×1152<br>原生 patch；KV=3024 | M4≤4 | **是**（联合去噪）<br>每条候选同时预测 32 个 R | **否**（R 仅用于辅助对齐） | 有；EMA DINOv3-L<br>Rv(M4) | `V(M4)+Rv(M4)` | S0 | [v0/last][A14-ckpt]<br>FP32 | [**92.81**][A14-csv] |
| **A15 DINOv3-L，multi-GT＋register Scorer**<br>[训练配置][A15-cfg] | DINOv3-L<br>前视 672×1152<br>原生 patch；KV=3024 | M4≤4 | **是**（联合去噪）<br>每条候选同时预测 32 个 R | **是**（预测 R detach→压缩→与轨迹融合） | 有；EMA DINOv3-L<br>Rv(M4) | `V(M4)+Rv(M4)` | SR | [v0/last][A15-ckpt]<br>FP32 | [**93.00**][A15-csv] |
| **A16 DINOv3-L，单 human＋register Scorer**<br>[训练配置][A16-cfg] | DINOv3-L<br>前视 672×1152<br>原生 patch；KV=3024 | H：1；无 HL | **是**（联合去噪）<br>每条候选同时预测 32 个 R | **是**（预测 R detach→压缩→与轨迹融合） | 有；真实未来帧<br>Rv(H) | `V(H)+Rv(H)` | SR | [v0/last][A16-ckpt]<br>FP32 | [**93.20**][A16-csv] |
| **A17 DINOv3-L，register 分阶段**<br>[训练配置][A17-cfg] | DINOv3-L<br>前视 672×1152<br>原生 patch；KV=3024 | M4≤4 | Stage1：仅预测 R，Action 输入 GT<br>Stage2：**是**（轨迹＋R 联合去噪） | Stage1：无 Scorer<br>Stage2：**是**（预测 R detach 后融合） | 有；Stage1 5轮＋Stage2 25轮<br>EMA DINOv3-L | `S1: Rv; S2: V(M4)+Rv(M4)` | S1 无评分；S2 SR | [v0/e24][A17-ckpt]<br>FP32 | [**92.84**][A17-csv] |
| **A18 DINOv3-L，PE vMSE＋register**<br>[训练配置][A18-cfg] | DINOv3-L<br>前视 672×1152<br>原生 patch；KV=3024 | M4≤4＋PE8≤8 | **是**（联合去噪）<br>每条候选同时预测 32 个 R | **否**（R 仅用于辅助对齐） | 有；仅 M4 的配对未来<br>PE8 不对齐未来 | `V(M4)+0.5 V(PE8)+Rv(M4)` | S0 | [v0/last][A18-ckpt]<br>FP32 | [**91.90**][A18-csv] |
| **A19 DINOv3-L，PE vMSE＋register Scorer**<br>[训练配置][A19-cfg] | DINOv3-L<br>前视 672×1152<br>原生 patch；KV=3024 | M4≤4＋PE8≤8 | **是**（联合去噪）<br>每条候选同时预测 32 个 R | **是**（预测 R detach→压缩→与轨迹融合） | 有；仅 M4 的配对未来<br>PE8 不对齐未来 | `V(M4)+0.5 V(PE8)+Rv(M4)` | SR | [v0/last][A19-ckpt]<br>FP32 | [**91.66**][A19-csv] |
| **A20 DINOv3-L，human＋长 GT＋PE L1**<br>[训练配置][A20-cfg] | DINOv3-L<br>前视 672×1152<br>原生 patch；KV=3024 | H＋HL＋PE8≤8；最多10项 | **否**（仅生成轨迹） | **否** | 无 | `V(H)+V(HL)+0.5 L1(PE8)` | S0 | [v0/last][A20-ckpt]<br>FP32 | [**92.91**][A20-csv] |
| **A21 DINOv3-L，human＋diverse8 L1**<br>[训练配置][A21-cfg] | DINOv3-L<br>前视 672×1152<br>原生 patch；KV=3024 | H：1＋D8≤8；最多9条 | **否**（仅生成轨迹） | **否** | 无 | `V(H)+0.5 L1(D8)` | S0 | [v0/last][A21-ckpt]<br>FP32 | [**93.32**][A21-csv] |
| **A22 DINOv3-L，human＋diverse8 vMSE**<br>[训练配置][A22-cfg] | DINOv3-L<br>前视 672×1152<br>原生 patch；KV=3024 | H：1＋D8≤8；最多9条 | **否**（仅生成轨迹） | **否** | 无 | `V(H)+0.5 V(D8)` | S0 | [v0/last][A22-ckpt]<br>FP32 | [**92.49**][A22-csv] |
| **A23 human X-L1＋diverse8 L1**<br>[训练配置][A23-cfg] | DINOv3-L<br>前视 672×1152<br>原生 patch；KV=3024 | H：1＋D8≤8 | **否**（仅生成轨迹） | **否** | 无 | `L1(H)+0.5 L1(D8)` | S0 | [v0/last][A23-ckpt]<br>FP32 | [**93.80**][A23-csv] |
| **A24 human＋长 GT，均为 X-L1**<br>[训练配置][A24-cfg] | DINOv3-L<br>前视 672×1152<br>原生 patch；KV=3024 | H＋HL：2 项 | **否**（仅生成轨迹） | **否** | 无 | `L1(H)+L1(HL)` | S0 | [v0/last][A24-ckpt]<br>FP32 | [**93.29**][A24-csv] |
| **A25 DINOv3-L，human vMSE＋D8＋register**<br>[配置][A25-cfg] · [启动][A25-launch] · [日志][A25-log]<br>DLC `dlc1aatxjf950vw7` | DINOv3-L<br>前视 672×1152<br>原生 patch；KV=3024 | H：1＋D8≤8；最多9条 | **是**（联合去噪）<br>每候选 32 个 R | **否**（R 仅用于辅助对齐） | EMA DINOv3-L<br>H 真实未来＋D8 对应视频<br>Rv：有效未来 GT 均分 | `V(H)+0.5 L1(D8)+Rv(H,D8)` | S0 | [v0/last (e24)][A25-ckpt]<br>FP32 | [**93.58**][A25-csv] |
| **A26 DINOv3-L，human vMSE＋D8＋register Scorer**<br>[配置][A26-cfg] · [启动][A26-launch] · [日志][A26-log]<br>DLC `dlc1aktix72nfi39` | DINOv3-L<br>前视 672×1152<br>原生 patch；KV=3024 | H：1＋D8≤8；最多9条 | **是**（联合去噪）<br>每候选 32 个 R | **是**（预测 R detach→压缩→融合） | EMA DINOv3-L<br>H 真实未来＋D8 对应视频<br>Rv：有效未来 GT 均分 | `V(H)+0.5 L1(D8)+Rv(H,D8)` | SR | [v0/last (e24)][A26-ckpt]<br>FP32 | [**93.51**][A26-csv] |
| **A27 DINOv3-L，human X-L1＋D8＋register**<br>[配置][A27-cfg] · [启动][A27-launch] · [日志][A27-log]<br>DLC `dlc16mhamjyg8qtw` | DINOv3-L<br>前视 672×1152<br>原生 patch；KV=3024 | H：1＋D8≤8；最多9条 | **是**（联合去噪）<br>每候选 32 个 R | **否**（R 仅用于辅助对齐） | EMA DINOv3-L<br>H 真实未来＋D8 对应视频<br>Rv：有效未来 GT 均分 | `L1(H)+0.5 L1(D8)+Rv(H,D8)` | S0 | [v0/last (e24)][A27-ckpt]<br>FP32 | [**93.68**][A27-csv] |
| **A28 DINOv3-L，human X-L1＋D8＋register Scorer**<br>[配置][A28-cfg] · [启动][A28-launch] · [日志][A28-log]<br>DLC `dlc1aut4azaufm8s` | DINOv3-L<br>前视 672×1152<br>原生 patch；KV=3024 | H：1＋D8≤8；最多9条 | **是**（联合去噪）<br>每候选 32 个 R | **是**（预测 R detach→压缩→融合） | EMA DINOv3-L<br>H 真实未来＋D8 对应视频<br>Rv：有效未来 GT 均分 | `L1(H)+0.5 L1(D8)+Rv(H,D8)` | SR | [v0/last (e24)][A28-ckpt]<br>FP32 | [**94.21**][A28-csv] |

A12/A13/A18/A19 的两组 GT **分开平均后按 1 和 0.5 相加**，不是把至多 12 条 GT 全部等权平均；A10 才是合并 M4/P4 后按全部有效 GT 数平均。A21/A22/A23 不读取旧 M4 或 CLOVER PE8。

A17 的 Stage1 输入干净 GT Action，仅训练 register 去噪，GT 按有效行均衡分配给 64 个位置；Stage2 恢复 Action 去噪及 Scorer。Stage1 的 5 轮是额外训练量，表中 PDMS 来自 Stage2 的 e24 权重。

## DrivoR 原始解码器系列

本组保留原 DrivoR 轨迹头：**64 个可学习 Action query＋ego，4 层 SA→CA→FFN，维度 256、FFN 1024、1 个 attention head**；输出物理坐标轨迹，无 diffusion/v loss。表中的 V-JEPA 变体只改变感知骨干，World 变体再增加 Bank/独立未来预测辅助支路，因此仍归在原 DrivoR 解码器系列。`prev_weight=0`、`inter_weight=0`，本地实验的轨迹回归有效监督来自最后一组 proposal。

| 实验与配置快照 | 视觉骨干 / 输入 / 下游 KV | 轨迹 GT 数量与来源 | 生成轨迹时同步预测 Register？ | 预测 Register 输入 Scorer？ | 未来对齐目标 / 方式 | 轨迹与未来 loss（另加评分项） | Scorer / 评分 loss | 测试权重 / 精度 | navtest PDMS ↑ |
|---|---|---|---|---|---|---|---|---|---:|
| **D01 官方 Nav1 发布权重（trainval约103k）**<br>[测试配置][D01-cfg] | DINOv2-S<br>4 视角 672×1148<br>每视角 Q16；KV=64 | README 配方：H＋HL† | **否**（仅生成轨迹） | **否** | 无 | `README: L1(H)+L1(HL)†` | S0 | [官方 e24/step40325][D01-ckpt]<br>FP32 | [**93.69**][D01-csv] |
| **D02 官方 Nav1 发布权重（train-only约85k）**<br>[测试配置][D02-cfg] | DINOv2-S<br>4 视角 672×1148<br>每视角 Q16；KV=64 | README 配方：H＋HL† | **否**（仅生成轨迹） | **否** | 无 | `README: L1(H)+L1(HL)†` | S0 | [官方 e23/step31896][D02-ckpt]<br>FP32 | [**93.13**][D02-csv] |
| **D03 本地原版 human＋长 GT，best e23**<br>[训练配置][D03-cfg] | DINOv2-S<br>4 视角 672×1148<br>每视角 Q16；KV=64 | H＋HL：2 项 | **否**（仅生成轨迹） | **否** | 无 | `L1(H)+L1(HL)` | S0 | [v2/best e23][D03-ckpt]<br>FP32 | [**93.15**][D03-csv] |
| **D04 本地原版 human＋长 GT，e24**<br>[训练配置][D04-cfg] | DINOv2-S<br>4 视角 672×1148<br>每视角 Q16；KV=64 | H＋HL：2 项 | **否**（仅生成轨迹） | **否** | 无 | `L1(H)+L1(HL)` | S0 | [v2/e24][D04-ckpt]<br>FP32 | [**93.16**][D04-csv] |
| **D05 原版＋旧 multi-GT，等权 nearest**<br>[训练配置][D05-cfg] | DINOv2-S<br>4 视角 672×1148<br>每视角 Q16；KV=64 | M4≤4 | **否**（仅生成轨迹） | **否** | 无 | `L1(M4)` | S0 | [v0/e24][D05-ckpt]<br>FP32 | [**92.92**][D05-csv] |
| **D06 原版＋旧 multi-GT，JEPA 加权**<br>[训练配置][D06-cfg] | DINOv2-S<br>4 视角 672×1148<br>每视角 Q16；KV=64 | H：1＋M4中的非human≤4 | **否**（仅生成轨迹） | **否** | 无 | `L1(H)+0.5 L1(M4非human)` | S0 | [v0/e24][D06-ckpt]<br>FP32 | [**92.62**][D06-csv] |
| **D07 原版＋旧 multi-GT，Hungarian**<br>[训练配置][D07-cfg] | DINOv2-S<br>4 视角 672×1148<br>每视角 Q16；KV=64 | M4≤4；一对一匹配 | **否**（仅生成轨迹） | **否** | 无 | `L1_Hungarian(M4)` | S0 | [v0/e24][D07-ckpt]<br>FP32 | [**92.74**][D07-csv] |
| **D08 原版＋World4Scorer Bank**<br>[训练配置][D08-cfg] | DINOv2-S<br>4 视角 672×1148<br>每视角 Q16；KV=64 | H＋HL：2 项；Bank 不算 GT | **否**（仅生成轨迹） | **否** | 无 | `L1(H)+L1(HL)` | S0＋0.5 LBank | [v0/last][D08-ckpt]<br>BF16 | [**92.79**][D08-csv] |
| **D09 原版＋CLOVER pseudo-expert**<br>[训练配置][D09-cfg] | DINOv2-S<br>4 视角 672×1148<br>每视角 Q16；KV=64 | H＋HL＋PE8≤8；最多10项 | **否**（仅生成轨迹） | **否** | 无 | `L1(H)+L1(HL)+0.5 L1(PE8)` | S0 | [v0/last][D09-ckpt]<br>FP32 | [**93.49**][D09-csv] |
| **D10 原 DrivoR 头＋V-JEPA2-L**<br>[训练配置][D10-cfg] | V-JEPA2-L<br>前视 672×1152<br>原生 patch；KV=3024 | H＋HL：2 项 | **否**（仅生成轨迹） | **否** | 无 | `L1(H)+L1(HL)` | S0 | [v0/last][D10-ckpt]<br>BF16 | [**93.37**][D10-csv] |
| **D11 原 DrivoR 头＋V-JEPA2.1-L**<br>[训练配置][D11-cfg] | V-JEPA2.1-L<br>前视 672×1152<br>原生 patch；KV=3024 | H＋HL：2 项 | **否**（仅生成轨迹） | **否** | 无 | `L1(H)+L1(HL)` | S0 | [v0/last][D11-ckpt]<br>BF16 | [**93.16**][D11-csv] |
| **D12 V-JEPA2-L＋World，单 human**<br>[训练配置][D12-cfg] | V-JEPA2-L<br>前视 672×1152<br>原生 patch；KV=3024 | H＋HL：2 项 | **否**（World 独立未来特征支路） | **否** | 无32个register；有独立<br>V-JEPA2.1未来特征对齐 | `L1(H)+L1(HL)+Fcos(H)` | S0＋0.5 LBank | [v0/last][D12-ckpt]<br>BF16 | [**93.42**][D12-csv] |
| **D13 V-JEPA2-L＋World，旧 multi-GT**<br>[训练配置][D13-cfg] | V-JEPA2-L<br>前视 672×1152<br>原生 patch；KV=3024 | M4≤4；无 HL | **否**（World 独立未来特征支路） | **否** | 无32个register；有独立<br>V-JEPA2.1未来特征对齐 | `L1(M4)+Fcos(M4)` | S0＋0.5 LBank | [v0/last][D13-ckpt]<br>BF16 | [**92.75**][D13-csv] |

† D01/D02 是官方发布模型在本地的实测结果，不是 `exp/ke/training_*` 下训练出的权重。H＋HL 来自 [官方 README 的 Nav1 训练配方](../../README.md)，此目录未附这两次发布训练的完整 Hydra 快照，不能用测试配置里关闭 GT 读取的 `long_trajectory_additional_poses=-1` 反推其训练时没有长 GT。103k/85k 的来源标记沿用发布文件及本地 release 测试入口。权重内实际元数据分别为 D01 `epoch=24, step=40325`、D02 `epoch=23, step=31896`；文件名中的 `25epochs` 不代表被测权重一定来自 e24。

## 普通 Transformer / learned-query 系列

本组视觉统一为 **DINOv3-L、当前前视 672×1152、3024 个原生 patch KV、QV LoRA rank 32**。规划器是随机初始化的 **64 个可学习 Action query、4 层普通 Transformer、256/1024 维**，最后 MLP 输出轨迹；有 register 时每个 proposal 增加 32 个可学习 future query。训练和测试都是一次普通前向，无 Action-DiT 预训练、随机扩散噪声、timestep 或 vMSE。

`R1` 表示直接 register L1 对齐，不是 Action-DiT 表中的 `Rv`。单 GT 的 T01/T03 **没有长 GT**，T05/T06 使用 H＋HL；T10 是只用 H、无未来 register 的对照。

每个 epoch 的 val PDMS（排除 T02、T04）单独汇总在 [T 系列逐轮验证表](T_EPOCH_VAL_PDMS_SUMMARY.md)，包含 25 轮完整数值、最佳轮次和收敛曲线。

生成器每层 Self-Attention 使用 `action_to_register` mask：Action 能读全部 Action，不能读未来 register；每个 proposal 的 register 只能直接读自己的 Action 和本组 32 个 register。两路 Cross-Attention 都以当前图像 patch 为 K/V，每个 Query 独立读取视觉，不会绕过上述 mask。未来 loss 仍能更新共享模块及视觉 LoRA；前向可见性不等于梯度隔离。

| 实验与配置快照 | 视觉骨干 / 输入 / 下游 KV | 轨迹 GT 数量与来源 | 生成轨迹时同步预测 Register？ | 预测 Register 输入 Scorer？ | 未来对齐目标 / 方式 | 轨迹与未来 loss（另加评分项） | Scorer / 评分 loss | 测试权重 / 精度 | navtest PDMS ↑ |
|---|---|---|---|---|---|---|---|---|---:|
| **T01 单 human＋register，原 Scorer**<br>[训练配置][T01-cfg] | DINOv3-L<br>前视 672×1152<br>原生 patch；KV=3024 | H：1；无 HL | **是**（同一次 Transformer 前向）<br>每条候选同时预测 32 个 R | **否**（R 仅用于辅助对齐） | 有；真实未来帧<br>EMA DINOv3-L，R1(H) | `L1(H)+R1(H)` | S0 | [v0/last][T01-ckpt]<br>FP32 | [**93.21**][T01-csv] |
| **T02 旧 multi-GT＋register，原 Scorer**<br>[训练配置][T02-cfg] | DINOv3-L<br>前视 672×1152<br>原生 patch；KV=3024 | M4≤4；无 HL | **是**（同一次 Transformer 前向）<br>每条候选同时预测 32 个 R | **否**（R 仅用于辅助对齐） | 有；配对 Cosmos未来<br>EMA DINOv3-L，R1(M4) | `L1(M4)+R1(M4)` | S0 | [v0/last][T02-ckpt]<br>FP32 | [**93.20**][T02-csv] |
| **T03 单 human＋register，融合 Scorer**<br>[训练配置][T03-cfg] | DINOv3-L<br>前视 672×1152<br>原生 patch；KV=3024 | H：1；无 HL | **是**（同一次 Transformer 前向）<br>每条候选同时预测 32 个 R | **是**（预测 R detach→压缩→与轨迹融合） | 有；真实未来帧<br>EMA DINOv3-L，R1(H) | `L1(H)+R1(H)` | SR | [v0/last][T03-ckpt]<br>FP32 | [**92.72**][T03-csv] |
| **T04 旧 multi-GT＋register，融合 Scorer**<br>[训练配置][T04-cfg] | DINOv3-L<br>前视 672×1152<br>原生 patch；KV=3024 | M4≤4；无 HL | **是**（同一次 Transformer 前向）<br>每条候选同时预测 32 个 R | **是**（预测 R detach→压缩→与轨迹融合） | 有；配对 Cosmos未来<br>EMA DINOv3-L，R1(M4) | `L1(M4)+R1(M4)` | SR | [v0/last][T04-ckpt]<br>FP32 | [**92.51**][T04-csv] |
| **T05 human＋长 GT，无 register**<br>[训练配置][T05-cfg] | DINOv3-L<br>前视 672×1152<br>原生 patch；KV=3024 | H＋HL：2 项 | **否**（仅生成轨迹） | **否** | 无；不构建未来分支 | `L1(H)+L1(HL)` | S0 | [v0/last][T05-ckpt]<br>FP32 | [**93.74**][T05-csv] |
| **T06 human＋长 GT＋register**<br>[训练配置][T06-cfg] | DINOv3-L<br>前视 672×1152<br>原生 patch；KV=3024 | H＋HL：2 项 | **是**（同一次 Transformer 前向）<br>每条候选同时预测 32 个 R | **否**（R 仅用于辅助对齐） | 有；只对齐 H 的真实未来<br>HL 不提供 register loss | `L1(H)+L1(HL)+R1(H)` | S0 | [v0/last][T06-ckpt]<br>FP32 | [**93.36**][T06-csv] |
| **T07 human＋diverse8，无 register**<br>[训练配置][T07-cfg] | DINOv3-L<br>前视 672×1152<br>原生 patch；KV=3024 | H：1＋D8≤8；无 HL | **否**（仅生成轨迹） | **否** | 无；不构建未来分支 | `L1(H)+0.5 L1(D8)` | S0 | [v0/last][T07-ckpt]<br>FP32 | [**93.82**][T07-csv] |
| **T08 human＋D8 L1＋register**<br>[配置][T08-cfg] · [启动][T08-launch] · [日志][T08-log]<br>DLC `dlc1stkt1n8i1ctp` | DINOv3-L<br>前视 672×1152<br>原生 patch；KV=3024 | H：1＋D8≤8；最多9条 | **是**（同一次 Transformer 前向）<br>每候选 32 个 R | **否**（R 仅用于辅助对齐） | EMA DINOv3-L<br>H 真实未来＋D8 对应视频<br>R1：有效未来 GT 均分 | `L1(H)+0.5 L1(D8)+R1(H,D8)` | S0 | [v0/last][T08-ckpt]<br>FP32 | [**93.53**][T08-csv] |
| **T09 human＋D8 L1＋register Scorer**<br>[配置][T09-cfg] · [启动][T09-launch] · [日志][T09-log]<br>DLC `dlcudc5qjukb34ak` | DINOv3-L<br>前视 672×1152<br>原生 patch；KV=3024 | H：1＋D8≤8；最多9条 | **是**（同一次 Transformer 前向）<br>每候选 32 个 R | **是**（预测 R detach→压缩→融合） | EMA DINOv3-L<br>H 真实未来＋D8 对应视频<br>R1：有效未来 GT 均分 | `L1(H)+0.5 L1(D8)+R1(H,D8)` | SR | [v0/last][T09-ckpt]<br>FP32 | [**93.55**][T09-csv] |
| **T10 单 human，无 register**<br>[配置][T10-cfg] · [启动][T10-launch] · [验证][T10-validation]<br>DLC `dlc1q4uo0bszf0gl` | DINOv3-L<br>前视 672×1152<br>原生 patch；KV=3024 | H：1；无 HL / M4 / D8 / PE | **否**（仅生成轨迹） | **否** | 无；不构建未来分支 / EMA teacher | `L1(H)` | S0 | [v0/last][T10-ckpt]<br>FP32 | [**93.32**][T10-csv] |

## Loss 和 Scorer 的精确定义

**L1 / coverage：**先在物理自车坐标（x、y 单位米，heading 单位弧度）上对三个分量绝对误差求和，再平均 8 个时间点。对每条有效 GT 独立从 64 个 proposal 中选最小者，再按有效 GT 数平均，最后对 batch 平均：

```text
C1(m,g) = mean_time(sum_xyz_heading(abs(pred_physical[m] - GT_physical[g])))
L1(G)   = mean_batch(mean_valid_GT(min_proposal C1(m,g)))
```

`L1(H)+L1(HL)` 是两个 best-of-64 loss 相加，不再除以 2；同一个 proposal 可以同时被多个 GT 选中。唯一例外 D07 采用 Hungarian 一对一分配，只对分配中的 proposal/GT 对计算 L1，并除以有效 GT 数；没有给未匹配 proposal 额外轨迹 loss。P4、PE8、D8 的 coverage 也只更新各自匹配到的输出。

**Action-DiT 的 V：**归一化沿用 FastWAM：`N(x) = 2*(x + [1.57,19.68,1.67])/[66.74,42.0,3.53] - 1`，模型输出干净轨迹的归一化 X-prediction。以下式子与将预测、GT 都从同一 noisy input 转为 v 后做 MSE 等价：

```text
sigma_safe = max(sigma, 0.1)
CV(m,g) = mean_time,xyz_heading(((X_pred[m] - N(GT[g])) / sigma_safe[m])**2)
m*(g)   = argmin_m CV(m,g)
V(G)    = mean_batch(mean_valid_GT(w(t[m*(g)]) * CV(m*(g),g)))
```

其中 `w(t)` 是现有 `ContinuousFlowSchedule.training_weight`：对 `exp(-2*(sigma-0.5)^2)` 减去训练网格最小值后，用该差值的网格均值归一化。每场景的 64 个 proposal 共用一个采样 t，噪声各自独立；主 M4/PE8/D8 监督重用这同一次 X 输出，而不是每条 GT 都再跑一次 DiT。A23/A24 改为物理 X-L1 后不再做 X→v 转换、不除 sigma、不乘此时间权重，但加噪输入和多步推理仍保留。

**未来 register：**每个 proposal 预测 `8 帧 × 4 register × 1024维`，64 个 proposal 对应 64 组。当前视觉编码器加载 DINOv3-L 预训练，基座冻结、QV LoRA 可训练；未来目标来自该编码器的 EMA teacher，`decay=0.999`，未来图像 384×384。8 个时刻为 +0.5、+1.0、…、+4.0s；Cosmos 视频第 0 帧为当前帧，实际使用索引 5、10、…、40。真实 human 未来来自 NAVSIM 图像；M4 的配对未来帧缓存为 `DrivoR/exp/world4scorer_data/frames_cosmos`，原视频目录为 `/cpfs29jb/data/zhaozc/project/Cosmos3forMultiGT/runtime/rollouts/navtrain_full_allcosmos_cuda_v1_gpu64`。

每条 GT 先按轨迹 loss 选 proposal，**该位置的 register 才对齐这条 GT 的未来**，不另外匹配 register；允许多 GT 选同一位置。无效未来帧/GT 不贡献未来 loss。`Rv` 对 register 的 X 误差也除 `sigma_safe` 后求 MSE、乘 `w(t)`，再按有效帧/GT/batch 平均；`R1` 直接平均 register L1，不使用 sigma 或时间权重。两者系数均为 1。Action 看不到 future register，future register 可以读对应 Action；teacher 不接收反向梯度，测试无需读取未来 GT/teacher 目标。

**World 的 Fcos：**D12/D13 不是上述 32-register 去噪网络。其独立 predictor 将选中轨迹的 8 个点编码为 query，经 4 层 Transformer（8 heads）和当前视觉特征交互，输出 8×1024 未来特征；与缓存的冻结 V-JEPA2.1 特征计算 `1 - cosine_similarity`，按有效时刻/GT/batch 平均，系数为 1。输入 predictor 的直接视觉 K/V detach，选中轨迹保留梯度，轨迹生成器向上游视觉的梯度链路正常保留。H＋HL 配方只用 H 对齐真实未来；M4 配方按每个有效 GT 对齐各自 Cosmos 未来。

**S0：**原 DrivoR Scorer，输入 64 条 `trajectory.detach()`，轨迹展平 24 维后 `Linear(24,1024)→ReLU→Linear(1024,256)`，经过 **4 层 TransformerDecoderScorer、256 维、1 attention head、FFN 1024**，加 ego embedding，再由 **6 个评分头**预测 NC、DAC、EP、TTC、Comfort、DDC。视觉 K/V 与表中规划器的当前视觉特征一致，原四视角 Q16 为 64 个，DINOv2 单前视为 3936 个，DINOv3/V-JEPA 单前视为 3024 个；Scorer 的直接视觉分支保留梯度。

训练中给 64 条 proposal 计算在线 NAVSIM oracle 标签，六个分量使用原 `DrivoRLoss.score_loss` 的 BCE 等权求和（含原有 TTC 有效性掩码及 NC/DDC 的 0.5 类映射），总系数 1；不是按 GT 数扩成 64×K 个评分目标。排序使用与 `pNC*pDAC*(5*pTTC+5*pEP+2*pComfort)` 等价的稳定 log 分数，当前 `ddc=0` 不参与排序，但 DDC 头仍有监督。`double_score/agent_pred/area_pred/bev_map/bev_agent` 均关闭，没有额外检测或地图 loss。

**SR：**仍用同一套四层/六头 Scorer。每个候选的预测 register 先 `detach`，展平 32×1024 后 `Linear→256→ReLU`；与该候选的 256 维轨迹 embedding **concat→MLP(512→1024→256)**，再进入 Scorer。Scorer loss 不通过轨迹或 register 输出反传生成器，当前视觉 K/V 分支保持原梯度行为；它消费的是预测 register，不是未来 GT。

**LBank：**在原 64 条生成 proposal 的 `Lscore` 之外，另从该场景库抽 16 条轨迹，优先给 unsafe 条目分配 50% 配额，不足再取其他有效候选/有放回补足。16 条 bank 轨迹与原 64 条**分两次进入同一个 Scorer、共享权重**，未拼成 80-token 自注意力。Bank 标签来自离线 PKL，loss 为 `mean(5个分类BCE) + 0.1*L1(sigmoid(EP), EP_gt)`，外层系数 0.5；轨迹为 detach 常量。Bank 只在训练使用，测试仍只生成/选择 64 条 proposal。

表中所有 loss 总式为“轨迹与未来列 + 评分列”：S0/SR 均代表 `+1.0 Lscore`；带 Bank 的再 `+0.5 LBank`。PE8/D8 的 0.5 是辅助组总权重，组内按有效条目均分。除官方发布模型和 A17 额外 5 轮 Stage1 外，本地训练快照统一为 seed 2、25 epochs、AdamW base LR 2e-4、8机×8卡、每卡 batch 1、累积 1、总 batch 64、BF16 训练；训练入口使用 train/val 分开的 `run_training.py`，`split=trainval` 是原始数据文件 split 名称，不表示把验证场景也并入训练。

<!-- T_EPOCH_NAVTEST_SUPPLEMENT -->
## T 系列第 16～20 轮 navtest 补测

另外新增 **33/40 份**同一训练不同 checkpoint 的 FP32 navtest，不增加主表实验编号，也不覆盖主表固定权重的分数。每份均核对 12,146 个有效且不重复的场景。轮次按 1 开始编号，16～20 轮对应文件 `epoch=15～19`；第 25 轮引用主表已有结果。

| 实验 | 第16轮 | 第17轮 | 第18轮 | 第19轮 | 第20轮 | 第25轮（主表） |
|---|---:|---:|---:|---:|---:|---:|
| T01 | —（未完成） | —（未完成） | [93.1248][T01-epoch18-csv] | [93.1170][T01-epoch19-csv] | [93.1632][T01-epoch20-csv] | [93.2095][T01-csv] |
| T03 | [93.2179][T03-epoch16-csv] | —（未完成） | [92.2195][T03-epoch18-csv] | [92.6493][T03-epoch19-csv] | [92.9249][T03-epoch20-csv] | [92.7240][T03-csv] |
| T05 | —（未完成） | —（未完成） | —（未完成） | —（未完成） | [93.9337][T05-epoch20-csv] | [93.7361][T05-csv] |
| T06 | [93.3478][T06-epoch16-csv] | [93.3945][T06-epoch17-csv] | [93.6240][T06-epoch18-csv] | [93.4068][T06-epoch19-csv] | [93.5702][T06-epoch20-csv] | [93.3606][T06-csv] |
| T07 | [93.5522][T07-epoch16-csv] | [93.3875][T07-epoch17-csv] | [93.9210][T07-epoch18-csv] | [93.9905][T07-epoch19-csv] | [94.0400][T07-epoch20-csv] | [93.8238][T07-csv] |
| T08 | [93.1521][T08-epoch16-csv] | [93.5498][T08-epoch17-csv] | [93.6556][T08-epoch18-csv] | [93.2439][T08-epoch19-csv] | [93.8400][T08-epoch20-csv] | [93.5333][T08-csv] |
| T09 | [93.3263][T09-epoch16-csv] | [93.5165][T09-epoch17-csv] | [93.7364][T09-epoch18-csv] | [93.8204][T09-epoch19-csv] | [93.7398][T09-epoch20-csv] | [93.5494][T09-csv] |
| T10 | [93.0481][T10-epoch16-csv] | [93.4257][T10-epoch17-csv] | [93.1936][T10-epoch18-csv] | [93.2469][T10-epoch19-csv] | [93.2927][T10-epoch20-csv] | [93.3215][T10-csv] |

尚缺 T01 第16/17轮、T03 第17轮、T05 第16～19轮，共 7 项。原评测失败后，自动重试在占卡资源释放检查阶段终止；这些格子没有有效 navtest CSV，不能填作 0 分。

T07 第20轮为 **94.0400**，主表第25轮为 **93.8238**；T03 第16轮为 **93.2179**，第25轮为 **92.7240**。不同 checkpoint 的变化不可忽略；后续选权重仍应依据 val，不能把这里按 navtest 事后取到的最高值当成新的公平主表基线。

完整任务记录及日志见 [补测结果](/cpfs29jb/data/zhaozc/project/drive_future/artifacts/t_navtest_epochs16_20/20261023_213451/RESULTS.md) 和 [运行状态](/cpfs29jb/data/zhaozc/project/drive_future/artifacts/t_navtest_epochs16_20/20261023_213451/state.json)。

T07/T10/T05/A23/A21 的 `trainval full` 是另起的 103,288 场景训练，当前尚无 navtest 结果，没有替换上表原 train-only 结果。其原 val 已并入训练，轮末指标属于训练子集监测；配置与启动命令见 [full 实验说明](../../DRIVOR_TRAINVAL_FULL_EXPERIMENTS.md)。

<!-- END_T_EPOCH_NAVTEST_SUPPLEMENT -->

## 实验对比结论

本节基于主表 **51 份已完成结果**，最新补齐 A25～A28。**本次最高分为 A28：human X-L1＋0.5×diverse8 L1＋register vMSE 对齐，并把预测 register 融入 Scorer，navtest PDMS 为 94.2064。**它比无 register 的同监督基线 A23 高 **0.4018** 点；但只加对齐、保留原 Scorer 的 A27 为 **93.6800**，比 A23 低 **0.1245** 点。因此，“register 对齐”和“register 进入 Scorer”仍需分别判断，不能沿用此前只看普通 Transformer 得出的整体印象。

普通 Transformer 的结论暂未改变：T07 的 human＋diverse8 无未来分支为 **93.8238**，加 register 对齐的 T08 为 **93.5333**，融合 Scorer 的 T09 为 **93.5494**。无 register、human＋diverse8 物理 L1 条件下，Action-DiT A23 与普通 Transformer T07 仍十分接近。下面的比较沿用主表固定 checkpoint；33 份不同轮次补测单独列在前一节。

以下差值均按 CSV 未四舍五入的均值计算，单位为 **PDMS 点**，不是相对百分比。“较可比”表示已核对所列骨干、监督、Scorer 和测试精度等条件；所有结果仍是单次训练/测试，不能据此宣称统计显著或多 seed 稳定。

### 1. 辅助 GT 的物理 L1 coverage 是目前方向最一致的改动

| 比较 | 保持的主要条件 | PDMS 变化 | 可以得到的结论 |
|---|---|---:|---|
| A22 → A21：D8 的 vMSE 改成 L1 | DINOv3-L、human 主项仍为 vMSE、D8 来源与 0.5 权重、原 Scorer、FP32 | 92.49 → 93.32，**+0.84** | 当前 diverse8 配方中，辅助项用物理 L1 更好。 |
| A13 → A12：PE8 的 vMSE 改成 L1 | DINOv3-L、M4 主项、同一 CLOVER PE8、0.5 权重、原 Scorer、FP32 | 91.93 → 92.61，**+0.68** | 相同方向也出现在另一种辅助 GT 来源上。 |
| A09 → A20：H＋HL 再加 PE8 L1 | DINOv3-L、H/HL vMSE、原 Scorer、FP32 | 92.42 → 92.91，**+0.48** | 保留原 human＋长 GT 主项时，额外 coverage 有收益。 |
| D04 → D09：原 DrivoR 加 PE8 L1 | DINOv2-S、四视角 Q16、H＋HL L1、原 Scorer、FP32 | 93.16 → 93.49，**+0.34** | coverage 的正向结果也出现在原 DrivoR 解码器上。 |
| T10 → T07：普通 Transformer 加 D8 L1 | DINOv3-L、human L1、原 Scorer、无 register、FP32 | 93.32 → 93.82，**+0.50** | human-only 基线补齐后，直接增加 `0.5×D8 coverage` 的收益得到本次结果支持。 |

这支持把 **“主 GT 保留，辅助 GT 用 0.5×物理 L1 coverage”**作为优先配方。晚间结果又补齐了主 human loss 的对照：

| 主轨迹 loss 对照（FP32、DINOv3-L、原 Scorer、无 register） | PDMS 变化 | 判断 |
|---|---:|---|
| A21 → A23：D8 L1 不变，仅 human vMSE 改为 X-L1 | 93.32 → **93.80，+0.48** | 在 human＋D8 配方中，主项改为物理 X-L1 也有收益。 |
| A09 → A24：human＋长 GT 两项都由 vMSE 改为 X-L1 | 92.42 → **93.29，+0.86** | 支持这套主监督整体改用物理 X-L1；不能拆分 human 与长 GT 各自贡献。 |

另外，物理 L1 与归一化 vMSE 的坐标权重、sigma 放大和时间权重不同；相同的外层系数 0.5 不代表相同梯度强度。因此这里支持的是**当前完整 loss 配方**，还不能把收益全部归因于 L1/MSE 的函数形式。

### 2. 旧 M4 替代 human，并没有体现出“GT 越多越好”

| 比较 | PDMS 变化 | 比较边界 |
|---|---:|---|
| T01 → T02：普通 Transformer、原 Scorer，H 换 M4 | 93.2095 → 93.2029，**−0.0066** | 基本持平；同时由真实 human 未来换成 M4 配对未来。 |
| A16 → A15：Action-DiT、register Scorer，H 换 M4 | 93.20 → 93.00，**−0.20** | 本次 M4 没有超过单 H；未来目标来源也随 GT 改变。 |
| D04 → D05：原版 H＋HL 换成 M4 | 93.16 → 92.92，**−0.24** | 同时取消了长 GT，不能单独归因于 GT 数量。 |

**GT 来源和主监督结构比槽位数量更值得关注。**M4 并不保证保留 human；即使包含 H，等权平均也会把它的权重变为 `1/K`。这是代码层面已经确认的变化，可能影响对真实驾驶分布的拟合，但现有测试不能证明它就是下降的唯一原因。

也不能反过来断言“只要给 human 更大权重就会改善”：D06 已显式使用 `L1(H)+0.5×L1(非human M4)`，结果仍低于 D05。D8/PE8 的数据质量、覆盖方式及 loss 都与旧 M4 不同，不能把它们视作只改变 GT 数量的同一实验。

### 3. Hungarian 和 JEPA 加权暂时都没有优于简单 nearest

原 DrivoR、同一旧 M4 来源、FP32 测试的三组结果为：

| 匹配 / 汇总方式 | 实验 | PDMS | 相对等权 nearest |
|---|---|---:|---:|
| 每 GT 独立 nearest，有效 GT 均分 | D05 | **92.92** | — |
| Hungarian 一对一，有效 GT 均分 | D07 | 92.74 | **−0.18** |
| human 单独一项＋0.5×其他 GT 均值 | D06 | 92.62 | **−0.30** |

当前证据支持继续把 **nearest＋有效 GT 平均**作为旧 M4 的默认匹配方式。强制不同 GT 占用不同 proposal 在这次实验中没有带来更高的最终 PDMS；但尚未比较 proposal 覆盖率、oracle-best 或 GT 匹配误差，不能据此判断 Hungarian 的轨迹覆盖能力一定更差。

### 4. Register 对齐与 register 进入 Scorer，要分开判断

| 仅改变 Scorer 是否融合预测 register | 原 Scorer → 融合 Scorer | PDMS 差值 |
|---|---:|---:|
| 普通 Transformer，单 H：T01 → T03 | 93.21 → 92.72 | **−0.49** |
| 普通 Transformer，M4：T02 → T04 | 93.20 → 92.51 | **−0.70** |
| Action-DiT，M4：A14 → A15 | 92.81 → 93.00 | **+0.19** |
| Action-DiT，M4＋PE8 vMSE：A18 → A19 | 91.90 → 91.66 | **−0.24** |
| 普通 Transformer，H＋D8：T08 → T09 | 93.5333 → 93.5494 | **+0.0161，基本持平** |
| Action-DiT，human vMSE＋D8：A25 → A26 | 93.5761 → 93.5054 | **−0.0707** |
| Action-DiT，human X-L1＋D8：A27 → A28 | 93.6800 → 94.2064 | **+0.5263** |

**普通 Transformer 当前宜保留原 Scorer；Action-DiT 的 human X-L1＋D8 配方值得保留 register 融合版继续验证。**A27 → A28 的正向结果没有在 human vMSE 的 A25 → A26 中复现，因此仍没有“register 进 Scorer 总是更好”的证据。

以上对照都已经带 register 对齐，只能比较是否把 register 输入 Scorer。辅助预测本身需要固定原 Scorer，比较有／无未来分支：

| 固定原 Scorer、FP32 的配方 | 无 register → 增加 register 对齐 | PDMS 差值 |
|---|---:|---:|
| 普通 Transformer，仅 H：T10 → T01 | 93.32 → 93.21 | **−0.11** |
| 普通 Transformer，H＋HL：T05 → T06 | 93.74 → 93.36 | **−0.38** |
| 普通 Transformer，H＋D8：T07 → T08 | 93.82 → 93.53 | **−0.29** |
| Action-DiT，human vMSE＋D8：A21 → A25 | 93.3236 → 93.5761 | **+0.2525** |
| Action-DiT，human X-L1＋D8：A23 → A27 | 93.8045 → 93.6800 | **−0.1245** |

目前三个普通 Transformer 配方都未显示 register 辅助对齐提高最终 PDMS；其中 T01 使用较早的模块构造顺序，与 T10 的共享参数初值不保证逐项相同，因此单 H 的 −0.11 更不能视为确定的因果效应。H＋D8 再融合 register 的 T09 为 93.55，相比无未来分支的 T07 仍低 **0.27**。

Action-DiT 的新对照均为 FP32：human vMSE 下，辅助对齐 A25 比 A21 高 **0.2525**；human X-L1 下，仅辅助对齐 A27 比 A23 低 **0.1245**，联合预测再融合 Scorer 的 A28 则比 A23 高 **0.4018**。后一个比较同时增加未来预测和评分融合，不能把全部收益称为“对齐 loss 的收益”。同样，联合训练时 Scorer 的 loss 也会改变共享视觉特征，最终分数提高不直接证明只是排序更准。旧 A06/A14 仍有 BF16/FP32 差异，需单独保留比较边界。

### 5. PE8 vMSE 与未来 register 没有产生叠加收益，分阶段也尚未占优

在同为 DINOv3-L、FP32 的 register 配方中：

- 原 Scorer：A14 加 PE8 vMSE 得到 A18，**92.81 → 91.90，−0.91**。
- 融合 Scorer：A15 加 PE8 vMSE 得到 A19，**93.00 → 91.66，−1.34**。
- 已有 PE8 vMSE 的 A13 再加 register 对齐得到 A18，**91.93 → 91.90，−0.03**，没有明显增益。
- A15 的直接联合训练与 A17 的“5 轮 Stage1＋25 轮 Stage2”相比，**93.00 → 92.84，−0.16**；额外 Stage1 没有在本次最终 PDMS 上体现优势。

因此不建议把这些分支默认全部叠加。PE8 没有对应未来帧、两组 GT 分开加权、多个 loss 竞争共享参数，都是值得检查的解释，但**尚无梯度或匹配统计证明具体原因**。这里比较的是 CLOVER PE8 vMSE。新的 D8 L1 有各轨迹对应的视频，A28 已出现正向结果；不能把 PE8＋register 的负结果直接推广到这一配方。

### 6. Action-DiT 更适合当前 DINOv3-L 原生 patch 配方；骨干升级不是普遍增益

同为前视、M4、原 Scorer、BF16 的 Action-DiT 记录：DINOv3-S **87.28**（A05），DINOv3-L **92.08**（A06），V-JEPA2-L **85.22**（A07），V-JEPA2.1-L **91.49**（A08）。

- DINOv3-S → L 为 **+4.80**，当前主线选择 DINOv3-L 有依据。
- V-JEPA2 → 2.1 为 **+6.27**，但同时包含重复当前帧/原生单图像分支等适配差异，应把它看成整套骨干适配方案的比较。
- DINOv2-S 的 A02 为 **90.04**，高于 DINOv3-S 的 A05；两者 patch 大小、分辨率/网格也不同，不能简单按 DINO 版本新旧排序。
- 原 DrivoR 头下，V-JEPA2-L 的 D10 为 **93.37**，V-JEPA2.1-L 的 D11 为 **93.16**，顺序又不同。V-JEPA2 在 Action-DiT 配方中的低分不能推广为“这个骨干本身不行”。

多视角实验同样需要拆开看：A01 → A03 **−1.28**，A02 → A04 **−6.92**。这两组同时把“单前视的 3936 个原生 patch”改成“四视角合计 64 个 scene query”，规划器和 Scorer 的 K/V 都变了。**可以确认这套多视角＋Q16 组合较差，不能确认增加视角本身有害。**如果继续研究，应分别固定视角比较 patch/query，固定特征方式比较单/多视角。

### 7. 简单 Transformer 已有竞争力，尚不能宣布哪个轨迹主干普遍更好

在 DINOv3-L、前视、M4、register 对齐、FP32 的条件下：

- 原 Scorer：Action-DiT A14 **92.81**，普通 Transformer T02 **93.20**，后者高 **0.39**。
- 融合 Scorer：Action-DiT A15 **93.00**，普通 Transformer T04 **92.51**，后者低 **0.49**。

两边轨迹/register loss 分别是 vMSE 与直接 L1，初始化和推理机制也不同。这说明**当前配方对 loss 与 Scorer 组合很敏感，复杂轨迹主干尚未转化为普遍领先**，还不是只替换 Transformer block 的严格对照。

晚间结果补齐了更接近同条件的比较：下面两行都固定 **DINOv3-L、前视、原 Scorer、物理 L1、无 register、FP32**。

| 监督配方 | Action-DiT | 普通 Transformer | Transformer − DiT |
|---|---:|---:|---:|
| human＋长 GT | A24：93.29 | T05：**93.74** | **+0.45** |
| human＋0.5×diverse8 | A23：93.8045 | T07：93.8238 | **+0.0193，基本持平** |

在这两个配方中，没有看到 Action-DiT 更高的 PDMS。普通 Transformer 是 4 层、一次推理，Action-DiT 是 30 层、10 步采样；前者值得作为主要基线。不过本表没有同设备的延迟、吞吐和显存测量，不能报告具体加速倍数。初始化、加噪输入和采样仍属于各自方法的一部分，这也不是只替换一个 Transformer block 的对照。

### 8. Bank、poor 正 GT、World 分支，目前不足以作为默认增强

| 改动 | 观察到的 PDMS 变化 | 当前判断 |
|---|---:|---|
| Action-DiT M4 加 Bank：A06 → A11 | **+0.28** | 基线 BF16、Bank FP32，需同精度补测。 |
| 原版 H＋HL 加 Bank：D04 → D08 | **−0.37** | 基线 FP32、Bank BF16，同样不能单独归因于 Bank。 |
| V-JEPA2 原 DrivoR 头加 Bank＋未来 predictor：D10 → D12 | **+0.05** | 同为 BF16，但同时加了两个分支；差异很小，无法分离各自贡献。 |
| M4 中再混入 poor GT：A06 → A10 | **−0.12** | 测试精度不同，当前没有可靠增益证据；不能直接认定 poor GT 一定导致退化。 |

Bank 是用低分轨迹的真实结果标签训练 Scorer，poor 正 GT 则要求生成器回归这些轨迹，二者作用不同。当前可先维持较简单的训练配方，把这些增强保留为独立对照，而不是凭候选数量或新增 loss 数量默认采用。

### 9. 较高 PDMS 不等于各分项同时更好

下表各项均为新配置减旧配置，单位为原始 0～1 分项乘 100 后的点数；NC、DAC、EP、TTC 都是越高越好。各行均为 FP32 对照。

| 比较 | ΔPDMS | ΔNC | ΔDAC | ΔEP | ΔTTC |
|---|---:|---:|---:|---:|---:|
| D8 辅助 vMSE → L1，A22 → A21 | **+0.84** | −0.74 | −0.17 | **+3.27** | −0.73 |
| Action-DiT M4 的 register 进 Scorer，A14 → A15 | +0.19 | −0.05 | +0.16 | **+0.80** | −0.56 |
| 普通 Transformer 单 H 的 register 进 Scorer，T01 → T03 | −0.49 | +0.07 | −0.35 | −0.03 | **−0.65** |
| 普通 Transformer M4 的 register 进 Scorer，T02 → T04 | −0.70 | −0.14 | −0.30 | +0.16 | **−1.33** |
| human 主项 vMSE → X-L1，A21 → A23 | **+0.48** | **+0.82** | +0.27 | −0.91 | **+1.39** |
| 同为 human＋D8 L1，Action-DiT → Transformer，A23 → T07 | +0.0193 | −0.62 | −0.02 | **+1.74** | **−1.47** |
| 普通 Transformer：增加长 GT，T10 → T05 | **+0.41** | −0.07 | −0.30 | **+1.92** | −0.53 |
| 普通 Transformer：增加 D8 L1，T10 → T07 | **+0.50** | −0.40 | +0.11 | **+1.81** | −0.63 |
| 普通 Transformer：H＋D8 增加 register 对齐，T07 → T08 | **−0.29** | −0.02 | −0.15 | −0.02 | **−0.43** |
| 普通 Transformer：H＋D8 的 register 进 Scorer，T08 → T09 | +0.0161 | +0.15 | −0.28 | +0.48 | −0.15 |
| Action-DiT vMSE 主项：增加 register 对齐，A21 → A25 | **+0.2525** | +0.24 | +0.11 | +0.09 | +0.28 |
| Action-DiT X-L1 主项：仅增加对齐，A23 → A27 | **−0.1245** | −0.20 | −0.21 | +0.36 | −0.28 |
| Action-DiT X-L1 主项：register 进 Scorer，A27 → A28 | **+0.5263** | +0.18 | +0.07 | **+1.62** | **−0.61** |
| Action-DiT X-L1 主项：联合预测＋融合，A23 → A28 | **+0.4018** | −0.02 | −0.13 | **+1.98** | **−0.89** |

A21 相比辅助 vMSE 的优势伴随更高 EP，以及略低的 NC/DAC/TTC；进一步把主 human 改为 X-L1 得到 A23 后，NC/TTC 改善、EP 略降，PDMS 又提高 0.48。A23 与 T07 的总分接近，但 T07 的 EP 更高、NC/TTC 更低，因此“总分接近”不表示驾驶行为或安全分项相同。这是分项变化的描述；PDMS 在逐场景上做组合，不能用这些分项均值精确分解每项对总分的贡献。

新补齐的 T10 对照也显示：增加长 GT 或 D8 后，总分和 EP 上升，但 TTC 均下降；不能把总分提升描述成所有安全分项同时改善。T07 → T08 的 register 辅助预测同时伴随 TTC 下降约 0.43 点；T08/T09 的总分近乎相同，EP、NC、DAC 和 TTC 则各有变化。

A28 相比 A23 的优势也主要伴随更高 EP，同时 TTC 下降约 **0.89** 点、DAC 下降约 **0.13** 点。它是当前最高的综合 PDMS，不能描述为每个驾驶质量分项都更优。

当前 CSV 只评价最终选中轨迹，**不能区分“64 个 proposal 的质量变差”还是“Scorer 没选好”**。诊断需要同一模型同一次采样的 `oracle-best PDMS`、`selected PDMS` 及两者差距，不能把上述下降直接定性为排序错误。

### 10. 已补齐的 human-only 对照与后续实验

本地原版 D04 **93.16** 与官方 train-only 85k 的 D02 **93.13** 接近，至少从总体 PDMS 看，本地基线没有明显失配。优先用 D04 作为本地改动参照；官方约 103k 的 D01 **93.69** 单独列作参考。

T10 已完成，普通 Transformer 的纯 human 基线是 **93.32**，不是此前带长 GT 的 T05。固定 DINOv3-L、前视、原 Scorer、物理 L1、无 register、FP32 后：

| 轨迹监督 | 实验 | PDMS | 相对 T10 |
|---|---|---:|---:|
| 仅 human | T10 | **93.32** | — |
| human＋长 GT | T05 | **93.74** | **+0.41** |
| human＋0.5×diverse8 L1 | T07 | **93.82** | **+0.50** |

这三组使用同一套共享模块构造顺序、相同 seed 和训练配方。T10/T05 只有长 GT 开关不同；T10/T07 增加 diverse8 coverage。本次结果支持保留辅助轨迹监督，同时保留 T10 作为单 human 的干净对照。

| 后续对照 | 要回答的问题 | 解释边界 |
|---|---|---|
| 重复 A23/A27/A28 的训练 seed | X-L1 配方的辅助对齐 −0.1245、评分融合 +0.5263 是否稳定？ | 当前三组均只有一次训练结果，尤其要区分小差异与可重复收益。 |
| 同一次采样的 oracle-best 与 selected PDMS | A28 的优势来自候选集合还是选择候选的能力？ | 现有 CSV 仅包含最终选中轨迹，无法分解这两部分。 |
| T07/T10/T05/A23/A21 的 trainval full | 从约 85k 扩至 103,288 场景后，相同配方的差异是否保留？ | 新任务尚无 navtest；原 val 已进入训练，不能当独立验证集比较。 |
| A27 vs T08；A28 vs T09 | 两种主干在 D8 联合预测任务中表现如何？ | 本次 Action-DiT 分别高 0.1467/0.6570；未来项仍是 vMSE/L1，不能视作完全同 loss 的主干消融。 |
| A06 的 FP32 基线；D08 的 FP32 Bank 结果 | 排除测试精度差异后，register/Bank/poor 对照还是否成立？ | 需固定权重、推理采样设置和测试精度。 |

T05 的 H＋HL 为 93.74，T07 的 H＋D8 为 93.82，差 **+0.09**；这是“去长 GT、加 D8”的配方更换，不能称为纯粹新增 D8 的收益。现在应使用 T10 → T07 的 **+0.50** 描述仅增加 D8 辅助项的这次对照结果。

单 human 条件下，T10 → T01（增加 register 辅助对齐）为 **−0.11**；T10 → T03（辅助预测＋融合 Scorer）为 **−0.60**。后者同时增加两个组成部分，不能只归因于 Scorer，且历史 T01/T03 的参数构造顺序不同。

若固定后续基线，当前可保留 **Action-DiT 的 A23（93.80，无未来分支）和 A28（94.21，register 联合预测＋融合 Scorer）**、**原 DrivoR 的 D04（93.16，纯基线）/D09（93.49，PE8 增强）**、**普通 Transformer 的 T07（93.82，human＋D8 L1）/T05（93.74，human＋长 GT）**。A28 值得继续验证，A23 仍是隔离未来分支作用的必要参照。它们的 GT、视角和未来监督不同，不能把各系列最高分直接当作公平架构排行榜。小幅差异应结合重复训练/采样验证；下一轮配置选择优先依据独立 val，navtest 用于确认固定方案。

## 证据入口

- 最新结果核对记录：[配置快照、测试 CSV 与对照差值](/cpfs29jb/data/zhaozc/project/drive_future/artifacts/ke_experiment_summary_20261024/evidence.json)。

- 晚间增补核对记录：`artifacts/ke_experiment_summary_20261022/evening_update.json`，包含五份新增 FP32 测试的逐场景均值核对及六组新实验的实际训练配置。

- [T07、T10、T05、A23、A21 的 trainval 全量配置与启动命令](../../DRIVOR_TRAINVAL_FULL_EXPERIMENTS.md)：使用 `run_training_full.py`，训练范围合并为 103,288 个场景。
- [GT 匹配、等权/JEPA 加权/Hungarian loss](../../navsim/agents/drivoR/multi_gt.py)、[原 DrivoR loss](../../navsim/agents/drivoR/layers/losses/drivor_loss.py)。
- [Action-DiT flow / 归一化](../../navsim/agents/drivoR_action_dit/flow.py)、[直接 X-L1](../../navsim/agents/drivoR_action_dit/x_l1.py)、[CLOVER coverage](../../navsim/agents/drivoR_pseudo_expert/loss.py)、[diverse8 读取](../../navsim/agents/drivoR_action_dit/diverse_gt.py)。
- [Action/register 联合模型](../../navsim/agents/drivoR_action_dit_register/model.py)、[普通 Transformer](../../navsim/agents/drivoR_query_register/model.py)、[新增三组 GT 对照](../../navsim/agents/drivoR_query_register/gt_variants.py)、[World/Bank](../../navsim/agents/drivoR_world/model.py)。
- 机器可读核对记录：`artifacts/ke_experiment_summary_20261022/evidence.json`、`audit_rows.json`，保存了训练/测试配置字段、实际 checkpoint、CSV 路径、各分项均值及 12,146 场景一致性检查。

A17 的额外预训练配置：[Stage1（5 epochs）][A17-stage1]。


<!-- 每行对应的实际训练快照、checkpoint 和原始 navtest CSV。 -->

[A01-cfg]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_action_dit_navtrain/multinode/code/hydra/config.yaml
[A01-ckpt]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_action_dit_navtrain/multinode/lightning_logs/version_3/checkpoints/epoch=24-step=33250.ckpt
[A01-csv]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/eval_navtest_drivor_action_dit_navtrain_v3_epoch24/action_dit_v3_navtest_run1/2026.10.15.22.09.50.csv
[A02-cfg]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_action_dit_navtrain_multigt/action_dit_multigt_seed2_run1/code/hydra/config.yaml
[A02-ckpt]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_action_dit_navtrain_multigt/action_dit_multigt_seed2_run1/lightning_logs/version_0/checkpoints/epoch=24-step=33250.ckpt
[A02-csv]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/eval_navtest_drivor_action_dit_multigt_seed2_epoch24/navtest_run1/2026.10.16.11.55.47.csv
[A03-cfg]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_action_dit_multiview_navtrain/action_dit_multiview_q16_seed2_run1/code/hydra/config.yaml
[A03-ckpt]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_action_dit_multiview_navtrain/action_dit_multiview_q16_seed2_run1/lightning_logs/version_0/checkpoints/epoch=24-step=33250.ckpt
[A03-csv]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/eval_navtest_drivor_action_dit_multiview_q16_seed2_epoch24/navtest_run1/2026.10.16.12.25.52.csv
[A04-cfg]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_action_dit_multiview_navtrain_multigt/action_dit_multiview_multigt_seed2_run1/code/hydra/config.yaml
[A04-ckpt]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_action_dit_multiview_navtrain_multigt/action_dit_multiview_multigt_seed2_run1/lightning_logs/version_0/checkpoints/epoch=24-step=33250.ckpt
[A04-csv]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/eval_navtest_drivor_action_dit_multiview_multigt_seed2_epoch24/navtest_run1/2026.10.16.12.10.53.csv
[A05-cfg]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_action_dit_dinov3_vits_navtrain_multigt/action_dit_dinov3_vits_seed2_train_eval_run1/code/hydra/config.yaml
[A05-ckpt]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_action_dit_dinov3_vits_navtrain_multigt/action_dit_dinov3_vits_seed2_train_eval_run1/lightning_logs/version_0/checkpoints/last.ckpt
[A05-csv]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/eval_navtest_action_dit_dinov3_vits/action_dit_dinov3_vits_seed2_train_eval_run1/2026.10.17.06.16.00.csv
[A06-cfg]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_action_dit_dinov3_vitl_navtrain_multigt/action_dit_dinov3_vitl_seed2_train_eval_run1/code/hydra/config.yaml
[A06-ckpt]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_action_dit_dinov3_vitl_navtrain_multigt/action_dit_dinov3_vitl_seed2_train_eval_run1/lightning_logs/version_0/checkpoints/last.ckpt
[A06-csv]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/eval_navtest_action_dit_dinov3_vitl/action_dit_dinov3_vitl_seed2_train_eval_run1/2026.10.17.08.43.31.csv
[A07-cfg]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_action_dit_vjepa2_vitl_navtrain_multigt/action_dit_vjepa2_vitl_seed2_train_eval_run1/code/hydra/config.yaml
[A07-ckpt]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_action_dit_vjepa2_vitl_navtrain_multigt/action_dit_vjepa2_vitl_seed2_train_eval_run1/lightning_logs/version_0/checkpoints/last.ckpt
[A07-csv]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/eval_navtest_action_dit_vjepa2_vitl/action_dit_vjepa2_vitl_seed2_train_eval_run1/2026.10.17.12.06.32.csv
[A08-cfg]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_action_dit_vjepa2_1_vitl_navtrain_multigt/action_dit_vjepa2_1_vitl_seed2_train_eval_run1/code/hydra/config.yaml
[A08-ckpt]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_action_dit_vjepa2_1_vitl_navtrain_multigt/action_dit_vjepa2_1_vitl_seed2_train_eval_run1/lightning_logs/version_0/checkpoints/last.ckpt
[A08-csv]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/eval_navtest_action_dit_vjepa2_1_vitl/action_dit_vjepa2_1_vitl_seed2_train_eval_run1/2026.10.17.11.24.11.csv
[A09-cfg]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_action_dit_dinov3_vitl_navtrain_single_gt_long/action_dit_dinov3_vitl_single_gt_long_seed2_train_eval_run1/code/hydra/config.yaml
[A09-ckpt]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_action_dit_dinov3_vitl_navtrain_single_gt_long/action_dit_dinov3_vitl_single_gt_long_seed2_train_eval_run1/lightning_logs/version_0/checkpoints/last.ckpt
[A09-csv]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/eval_navtest_action_dit_dinov3_vitl_single_gt_long/action_dit_dinov3_vitl_single_gt_long_seed2_train_eval_run1/2026.10.19.06.17.21.csv
[A10-cfg]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_action_dit_dinov3_vitl_navtrain_multigt_poor/action_dit_dinov3_vitl_multigt_poor_seed2_train_eval_run1/code/hydra/config.yaml
[A10-ckpt]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_action_dit_dinov3_vitl_navtrain_multigt_poor/action_dit_dinov3_vitl_multigt_poor_seed2_train_eval_run1/lightning_logs/version_0/checkpoints/last.ckpt
[A10-csv]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/eval_navtest_action_dit_dinov3_vitl_multigt_poor/action_dit_dinov3_vitl_multigt_poor_seed2_train_eval_run1/2026.10.19.06.55.08.csv
[A11-cfg]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_action_dit_dinov3_vitl_navtrain_multigt_bank/action_dit_dinov3_vitl_multigt_bank_seed2_train_eval_run1/code/hydra/config.yaml
[A11-ckpt]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_action_dit_dinov3_vitl_navtrain_multigt_bank/action_dit_dinov3_vitl_multigt_bank_seed2_train_eval_run1/lightning_logs/version_0/checkpoints/last.ckpt
[A11-csv]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/eval_navtest_action_dit_dinov3_vitl_multigt_bank/bank_last_fp32_run1/2026.10.19.10.36.26.csv
[A12-cfg]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_action_dit_dinov3_vitl_navtrain_multigt_pseudo_expert/action_dit_dinov3_vitl_multigt_pseudo_expert_seed2_train_eval_run1/code/hydra/config.yaml
[A12-ckpt]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_action_dit_dinov3_vitl_navtrain_multigt_pseudo_expert/action_dit_dinov3_vitl_multigt_pseudo_expert_seed2_train_eval_run1/lightning_logs/version_0/checkpoints/last.ckpt
[A12-csv]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/eval_navtest_action_dit_dinov3_vitl_multigt_pseudo_expert/pseudo_expert_last_fp32_run1/2026.10.19.10.36.48.csv
[A13-cfg]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_action_dit_dinov3_vitl_navtrain_multigt_pseudo_expert_vmse/action_dit_dinov3_vitl_pseudo_expert_vmse_seed2_train_eval_run1/code/hydra/config.yaml
[A13-ckpt]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_action_dit_dinov3_vitl_navtrain_multigt_pseudo_expert_vmse/action_dit_dinov3_vitl_pseudo_expert_vmse_seed2_train_eval_run1/lightning_logs/version_0/checkpoints/epoch=24-step=33250.ckpt
[A13-csv]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/eval_navtest_action_dit_dinov3_vitl_multigt_pseudo_expert_vmse/action_dit_pseudo_expert_vmse_epoch24_fp32_eval1/2026.10.21.00.50.13.csv
[A14-cfg]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_action_dit_dinov3_vitl_navtrain_register/dinov3_register_seed2_run1/code/hydra/config.yaml
[A14-ckpt]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_action_dit_dinov3_vitl_navtrain_register/dinov3_register_seed2_run1/lightning_logs/version_0/checkpoints/last.ckpt
[A14-csv]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/eval_navtest_action_dit_dinov3_vitl_register/dinov3_register_seed2_run1/2026.10.19.11.45.26.csv
[A15-cfg]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_action_dit_dinov3_vitl_navtrain_register_scorer/dinov3_register_scorer_seed2_run1/code/hydra/config.yaml
[A15-ckpt]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_action_dit_dinov3_vitl_navtrain_register_scorer/dinov3_register_scorer_seed2_run1/lightning_logs/version_0/checkpoints/last.ckpt
[A15-csv]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/eval_navtest_action_dit_dinov3_vitl_register_scorer/dinov3_register_scorer_seed2_run1/2026.10.19.12.01.21.csv
[A16-cfg]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_action_dit_dinov3_vitl_navtrain_register_scorer_single_gt/dinov3_register_scorer_single_gt_seed2_train_eval_run1/code/hydra/config.yaml
[A16-ckpt]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_action_dit_dinov3_vitl_navtrain_register_scorer_single_gt/dinov3_register_scorer_single_gt_seed2_train_eval_run1/lightning_logs/version_0/checkpoints/last.ckpt
[A16-csv]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/eval_navtest_action_dit_dinov3_vitl_register_scorer_single_gt/dinov3_register_scorer_single_gt_seed2_train_eval_run1/2026.10.21.10.55.38.csv
[A17-cfg]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_action_dit_dinov3_vitl_navtrain_register_staged_stage2/dinov3_register_staged_seed2_run1/code/hydra/config.yaml
[A17-ckpt]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_action_dit_dinov3_vitl_navtrain_register_staged_stage2/dinov3_register_staged_seed2_run1/lightning_logs/version_0/checkpoints/epoch=24-step=33250.ckpt
[A17-csv]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/eval_navtest_action_dit_dinov3_vitl_register_staged/dinov3_register_staged_seed2_run1_epoch24_fp32/2026.10.20.15.46.13.csv
[A18-cfg]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_action_dit_dinov3_vitl_navtrain_multigt_pseudo_expert_vmse_register/action_dit_pe_vmse_register_seed2_train_eval_run1/code/hydra/config.yaml
[A18-ckpt]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_action_dit_dinov3_vitl_navtrain_multigt_pseudo_expert_vmse_register/action_dit_pe_vmse_register_seed2_train_eval_run1/lightning_logs/version_0/checkpoints/last.ckpt
[A18-csv]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/eval_navtest_action_dit_dinov3_vitl_pseudo_expert_vmse_register/action_dit_pe_vmse_register_seed2_train_eval_run1/2026.10.21.01.49.07.csv
[A19-cfg]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_action_dit_dinov3_vitl_navtrain_multigt_pseudo_expert_vmse_register_scorer/action_dit_pe_vmse_register_scorer_seed2_train_eval_run1/code/hydra/config.yaml
[A19-ckpt]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_action_dit_dinov3_vitl_navtrain_multigt_pseudo_expert_vmse_register_scorer/action_dit_pe_vmse_register_scorer_seed2_train_eval_run1/lightning_logs/version_0/checkpoints/last.ckpt
[A19-csv]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/eval_navtest_action_dit_dinov3_vitl_pseudo_expert_vmse_register_scorer/action_dit_pe_vmse_register_scorer_seed2_train_eval_run1/2026.10.21.02.29.12.csv
[A20-cfg]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_action_dit_dinov3_vitl_navtrain_single_gt_long_pseudo_expert/action_dit_dinov3_vitl_single_gt_long_pseudo_expert_seed2_dlc8hwmx72tdbjoc_chain1/code/hydra/config.yaml
[A20-ckpt]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_action_dit_dinov3_vitl_navtrain_single_gt_long_pseudo_expert/action_dit_dinov3_vitl_single_gt_long_pseudo_expert_seed2_dlc8hwmx72tdbjoc_chain1/lightning_logs/version_0/checkpoints/last.ckpt
[A20-csv]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/eval_navtest_action_dit_dinov3_vitl_single_gt_long_pseudo_expert/action_dit_dinov3_vitl_single_gt_long_pseudo_expert_seed2_dlc8hwmx72tdbjoc_chain1/2026.10.22.07.21.05.csv
[A21-cfg]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_action_dit_dinov3_vitl_navtrain_human_diverse8_l1/action_dit_human_diverse8_l1_seed2_dlc_20261020_095054/code/hydra/config.yaml
[A21-ckpt]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_action_dit_dinov3_vitl_navtrain_human_diverse8_l1/action_dit_human_diverse8_l1_seed2_dlc_20261020_095054/lightning_logs/version_0/checkpoints/last.ckpt
[A21-csv]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/eval_navtest_action_dit_dinov3_vitl_human_diverse8_l1/action_dit_human_diverse8_l1_seed2_dlc_20261020_095054/2026.10.21.14.53.09.csv
[A22-cfg]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_action_dit_dinov3_vitl_navtrain_human_diverse8_vmse/action_dit_human_diverse8_vmse_seed2_dlc_20261020_095054/code/hydra/config.yaml
[A22-ckpt]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_action_dit_dinov3_vitl_navtrain_human_diverse8_vmse/action_dit_human_diverse8_vmse_seed2_dlc_20261020_095054/lightning_logs/version_0/checkpoints/last.ckpt
[A22-csv]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/eval_navtest_action_dit_dinov3_vitl_human_diverse8_vmse/action_dit_human_diverse8_vmse_seed2_dlc_20261020_095054/2026.10.21.14.36.46.csv
[A23-cfg]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_action_dit_dinov3_vitl_navtrain_human_xl1_diverse8_l1/action_dit_dinov3_vitl_human_xl1_diverse8_l1_seed2_dlc112peyzbocg66_20261021_run1/code/hydra/config.yaml
[A23-run]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_action_dit_dinov3_vitl_navtrain_human_xl1_diverse8_l1/action_dit_dinov3_vitl_human_xl1_diverse8_l1_seed2_dlc112peyzbocg66_20261021_run1
[A24-cfg]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_action_dit_dinov3_vitl_navtrain_single_gt_long_xl1/action_dit_dinov3_vitl_single_gt_long_xl1_seed2_dlc16mhamjyg8qtw_20261021_run1/code/hydra/config.yaml
[A24-run]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_action_dit_dinov3_vitl_navtrain_single_gt_long_xl1/action_dit_dinov3_vitl_single_gt_long_xl1_seed2_dlc16mhamjyg8qtw_20261021_run1
[D01-cfg]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/eval_navtest_drivor_Nav1_25epochs/20261014_232522/code/hydra/config.yaml
[D01-ckpt]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/weights/releases/model_weights/drivor_Nav1_25epochs.pth
[D01-csv]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/eval_navtest_drivor_Nav1_25epochs/20261014_232522/2026.10.14.23.49.03.csv
[D02-cfg]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/eval_navtest_drivor_Nav1_train_only_85k_25epochs/20261014_232522/code/hydra/config.yaml
[D02-ckpt]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/weights/releases/model_weights/drivor_Nav1_train_only_85k_25epochs.ckpt
[D02-csv]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/eval_navtest_drivor_Nav1_train_only_85k_25epochs/20261014_232522/2026.10.14.23.37.48.csv
[D03-cfg]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_Nav1_navtrain_traj_long_25epochs/multinode/code/hydra/config.yaml
[D03-ckpt]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_Nav1_navtrain_traj_long_25epochs/multinode/lightning_logs/version_2/checkpoints/best-epoch=23-step=31920.ckpt
[D03-csv]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/eval_navtest_training_drivoR_Nav1_navtrain_traj_long_25epochs_multinode_version_2_best-epoch_23-step_31920/20261015_120016/2026.10.15.12.14.37.csv
[D04-cfg]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_Nav1_navtrain_traj_long_25epochs/multinode/code/hydra/config.yaml
[D04-ckpt]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_Nav1_navtrain_traj_long_25epochs/multinode/lightning_logs/version_2/checkpoints/epoch=24-step=33250.ckpt
[D04-csv]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/eval_navtest_training_drivoR_Nav1_navtrain_traj_long_25epochs_multinode_version_2_epoch_24-step_33250/20261015_114507/2026.10.15.12.00.03.csv
[D05-cfg]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_Nav1_navtrain_multigt/drivor_multigt_seed2_run1/code/hydra/config.yaml
[D05-ckpt]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_Nav1_navtrain_multigt/drivor_multigt_seed2_run1/lightning_logs/version_0/checkpoints/epoch=24-step=33250.ckpt
[D05-csv]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/eval_navtest_training_drivoR_Nav1_navtrain_multigt_drivor_multigt_seed2_run1_version_0_epoch_24-step_33250/20261015_120227/2026.10.15.12.16.54.csv
[D06-cfg]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_Nav1_navtrain_multigt_jepa_weighted/drivor_multigt_jepa_seed2_run1/code/hydra/config.yaml
[D06-ckpt]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_Nav1_navtrain_multigt_jepa_weighted/drivor_multigt_jepa_seed2_run1/lightning_logs/version_0/checkpoints/epoch=24-step=33250.ckpt
[D06-csv]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/eval_navtest_training_drivoR_Nav1_navtrain_multigt_jepa_weighted_drivor_multigt_jepa_seed2_run1_version_0_epoch_24-step_33250/20261015_230301/2026.10.15.23.10.30.csv
[D07-cfg]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_Nav1_navtrain_multigt_hungarian/drivor_multigt_hungarian_seed2_run1/code/hydra/config.yaml
[D07-ckpt]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_Nav1_navtrain_multigt_hungarian/drivor_multigt_hungarian_seed2_run1/lightning_logs/version_0/checkpoints/epoch=24-step=33250.ckpt
[D07-csv]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/eval_navtest_drivor_multigt_hungarian_seed2_epoch24/navtest_run1/2026.10.16.11.41.06.csv
[D08-cfg]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_bank_navtrain/drivor_bank_seed2_run1/code/hydra/config.yaml
[D08-ckpt]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_bank_navtrain/drivor_bank_seed2_run1/lightning_logs/version_0/checkpoints/last.ckpt
[D08-csv]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/eval_navtest_drivoR_bank/drivor_bank_seed2_run1/2026.10.17.09.58.55.csv
[D09-cfg]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_pseudo_expert_navtrain/drivor_pseudo_expert_seed2_run1/code/hydra/config.yaml
[D09-ckpt]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_pseudo_expert_navtrain/drivor_pseudo_expert_seed2_run1/lightning_logs/version_0/checkpoints/last.ckpt
[D09-csv]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/eval_navtest_drivor_pseudo_expert/drivor_pseudo_expert_seed2_run1/2026.10.18.04.29.44.csv
[D10-cfg]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_vjepa2_vitl_navtrain/drivor_vjepa2_vitl_seed2_run1/code/hydra/config.yaml
[D10-ckpt]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_vjepa2_vitl_navtrain/drivor_vjepa2_vitl_seed2_run1/lightning_logs/version_0/checkpoints/last.ckpt
[D10-csv]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/eval_navtest_drivoR_vjepa2_vitl/drivor_vjepa2_vitl_seed2_run1/2026.10.17.17.43.32.csv
[D11-cfg]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_vjepa2_1_vitl_navtrain/drivor_vjepa2_1_vitl_seed2_run1/code/hydra/config.yaml
[D11-ckpt]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_vjepa2_1_vitl_navtrain/drivor_vjepa2_1_vitl_seed2_run1/lightning_logs/version_0/checkpoints/last.ckpt
[D11-csv]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/eval_navtest_drivoR_vjepa2_1_vitl/drivor_vjepa2_1_vitl_seed2_run1/2026.10.17.17.51.09.csv
[D12-cfg]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_vjepa2_world_navtrain/drivor_vjepa2_world_seed2_run1/code/hydra/config.yaml
[D12-ckpt]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_vjepa2_world_navtrain/drivor_vjepa2_world_seed2_run1/lightning_logs/version_0/checkpoints/last.ckpt
[D12-csv]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/eval_navtest_drivoR_vjepa2_world/drivor_vjepa2_world_seed2_run1/2026.10.17.23.04.30.csv
[D13-cfg]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_vjepa2_world_multigt_navtrain/drivor_vjepa2_world_multigt_seed2_run1/code/hydra/config.yaml
[D13-ckpt]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_vjepa2_world_multigt_navtrain/drivor_vjepa2_world_multigt_seed2_run1/lightning_logs/version_0/checkpoints/last.ckpt
[D13-csv]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/eval_navtest_drivoR_vjepa2_world_multigt/drivor_vjepa2_world_multigt_seed2_run1/2026.10.17.23.02.24.csv
[T01-cfg]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_query_register_dinov3_vitl_navtrain_single_gt/query_register_single_gt_seed2_dlc1stkt1n8i1ctp_20261021_run1/code/hydra/config.yaml
[T01-ckpt]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_query_register_dinov3_vitl_navtrain_single_gt/query_register_single_gt_seed2_dlc1stkt1n8i1ctp_20261021_run1/lightning_logs/version_0/checkpoints/last.ckpt
[T01-csv]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/eval_navtest_query_register_dinov3_vitl_single_gt/query_register_single_gt_seed2_dlc1stkt1n8i1ctp_20261021_run1/2026.10.21.16.22.19.csv
[T02-cfg]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_query_register_dinov3_vitl_navtrain_multigt/query_register_multigt_seed2_dlcudc5qjukb34ak_20261021_run1/code/hydra/config.yaml
[T02-ckpt]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_query_register_dinov3_vitl_navtrain_multigt/query_register_multigt_seed2_dlcudc5qjukb34ak_20261021_run1/lightning_logs/version_0/checkpoints/last.ckpt
[T02-csv]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/eval_navtest_query_register_dinov3_vitl_multigt/query_register_multigt_seed2_dlcudc5qjukb34ak_20261021_run1/2026.10.21.16.58.38.csv
[T03-cfg]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_query_register_dinov3_vitl_navtrain_scorer_single_gt/query_register_scorer_single_gt_seed2_dlc1stkt1n8i1ctp_20261021_run1/code/hydra/config.yaml
[T03-ckpt]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_query_register_dinov3_vitl_navtrain_scorer_single_gt/query_register_scorer_single_gt_seed2_dlc1stkt1n8i1ctp_20261021_run1/lightning_logs/version_0/checkpoints/last.ckpt
[T03-csv]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/eval_navtest_query_register_dinov3_vitl_scorer_single_gt/query_register_scorer_single_gt_seed2_dlc1stkt1n8i1ctp_20261021_run1/2026.10.22.06.14.54.csv
[T04-cfg]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_query_register_dinov3_vitl_navtrain_scorer_multigt/query_register_scorer_multigt_seed2_dlcudc5qjukb34ak_20261021_run1/code/hydra/config.yaml
[T04-ckpt]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_query_register_dinov3_vitl_navtrain_scorer_multigt/query_register_scorer_multigt_seed2_dlcudc5qjukb34ak_20261021_run1/lightning_logs/version_0/checkpoints/last.ckpt
[T04-csv]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/eval_navtest_query_register_dinov3_vitl_scorer_multigt/query_register_scorer_multigt_seed2_dlcudc5qjukb34ak_20261021_run1/2026.10.22.07.21.28.csv
[T05-cfg]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_query_dinov3_vitl_navtrain_single_gt_long/query_dinov3_vitl_single_gt_long_seed2_dlc8hwmx72tdbjoc_chain1/code/hydra/config.yaml
[T05-run]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_query_dinov3_vitl_navtrain_single_gt_long/query_dinov3_vitl_single_gt_long_seed2_dlc8hwmx72tdbjoc_chain1
[T06-cfg]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_query_dinov3_vitl_navtrain_single_gt_long_register/query_dinov3_vitl_single_gt_long_register_seed2_dlc1stkt1n8i1ctp_chain1/code/hydra/config.yaml
[T06-run]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_query_dinov3_vitl_navtrain_single_gt_long_register/query_dinov3_vitl_single_gt_long_register_seed2_dlc1stkt1n8i1ctp_chain1
[T07-cfg]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_query_dinov3_vitl_navtrain_human_diverse8_l1/query_dinov3_vitl_human_diverse8_l1_seed2_dlcudc5qjukb34ak_chain1/code/hydra/config.yaml
[T07-run]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_query_dinov3_vitl_navtrain_human_diverse8_l1/query_dinov3_vitl_human_diverse8_l1_seed2_dlcudc5qjukb34ak_chain1
[A17-stage1]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_action_dit_dinov3_vitl_navtrain_register_staged_stage1/dinov3_register_staged_seed2_run1/code/hydra/config.yaml
[A23-ckpt]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_action_dit_dinov3_vitl_navtrain_human_xl1_diverse8_l1/action_dit_dinov3_vitl_human_xl1_diverse8_l1_seed2_dlc112peyzbocg66_20261021_run1/lightning_logs/version_0/checkpoints/last.ckpt
[A23-csv]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/eval_navtest_action_dit_dinov3_vitl_human_xl1_diverse8_l1/action_dit_dinov3_vitl_human_xl1_diverse8_l1_seed2_dlc112peyzbocg66_20261021_run1/2026.10.22.20.39.43.csv
[A23-eval-cfg]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/eval_navtest_action_dit_dinov3_vitl_human_xl1_diverse8_l1/action_dit_dinov3_vitl_human_xl1_diverse8_l1_seed2_dlc112peyzbocg66_20261021_run1/code/hydra/config.yaml
[A24-ckpt]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_action_dit_dinov3_vitl_navtrain_single_gt_long_xl1/action_dit_dinov3_vitl_single_gt_long_xl1_seed2_dlc16mhamjyg8qtw_20261021_run1/lightning_logs/version_0/checkpoints/last.ckpt
[A24-csv]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/eval_navtest_action_dit_dinov3_vitl_single_gt_long_xl1/action_dit_dinov3_vitl_single_gt_long_xl1_seed2_dlc16mhamjyg8qtw_20261021_run1/2026.10.22.19.42.45.csv
[A24-eval-cfg]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/eval_navtest_action_dit_dinov3_vitl_single_gt_long_xl1/action_dit_dinov3_vitl_single_gt_long_xl1_seed2_dlc16mhamjyg8qtw_20261021_run1/code/hydra/config.yaml
[T05-ckpt]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_query_dinov3_vitl_navtrain_single_gt_long/query_dinov3_vitl_single_gt_long_seed2_dlc8hwmx72tdbjoc_chain1/lightning_logs/version_0/checkpoints/last.ckpt
[T05-csv]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/eval_navtest_query_dinov3_vitl_single_gt_long/query_dinov3_vitl_single_gt_long_seed2_dlc8hwmx72tdbjoc_chain1/2026.10.22.20.04.28.csv
[T05-eval-cfg]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/eval_navtest_query_dinov3_vitl_single_gt_long/query_dinov3_vitl_single_gt_long_seed2_dlc8hwmx72tdbjoc_chain1/code/hydra/config.yaml
[T06-ckpt]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_query_dinov3_vitl_navtrain_single_gt_long_register/query_dinov3_vitl_single_gt_long_register_seed2_dlc1stkt1n8i1ctp_chain1/lightning_logs/version_0/checkpoints/last.ckpt
[T06-csv]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/eval_navtest_query_dinov3_vitl_single_gt_long_register/query_dinov3_vitl_single_gt_long_register_seed2_dlc1stkt1n8i1ctp_chain1/2026.10.22.19.49.43.csv
[T06-eval-cfg]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/eval_navtest_query_dinov3_vitl_single_gt_long_register/query_dinov3_vitl_single_gt_long_register_seed2_dlc1stkt1n8i1ctp_chain1/code/hydra/config.yaml
[T07-ckpt]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_query_dinov3_vitl_navtrain_human_diverse8_l1/query_dinov3_vitl_human_diverse8_l1_seed2_dlcudc5qjukb34ak_chain1/lightning_logs/version_0/checkpoints/last.ckpt
[T07-csv]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/eval_navtest_query_dinov3_vitl_human_diverse8_l1/query_dinov3_vitl_human_diverse8_l1_seed2_dlcudc5qjukb34ak_chain1/2026.10.22.19.58.19.csv
[T07-eval-cfg]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/eval_navtest_query_dinov3_vitl_human_diverse8_l1/query_dinov3_vitl_human_diverse8_l1_seed2_dlcudc5qjukb34ak_chain1/code/hydra/config.yaml
[A25-cfg]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_action_dit_dinov3_vitl_navtrain_human_diverse8_l1_register/action_dit_human_diverse8_l1_register_seed2_20261022_210127/code/hydra/config.yaml
[A25-run]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_action_dit_dinov3_vitl_navtrain_human_diverse8_l1_register/action_dit_human_diverse8_l1_register_seed2_20261022_210127
[A25-log]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_action_dit_dinov3_vitl_navtrain_human_diverse8_l1_register/action_dit_human_diverse8_l1_register_seed2_20261022_210127/run_training.log
[A25-launch]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/scripts/training/run_drivor_action_dit_dinov3_vitl_human_diverse8_l1_register_train_eval.sh
[A25-ckpt]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_action_dit_dinov3_vitl_navtrain_human_diverse8_l1_register/action_dit_human_diverse8_l1_register_seed2_20261022_210127/lightning_logs/version_0/checkpoints/last.ckpt
[A25-csv]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/eval_navtest_action_dit_dinov3_vitl_human_diverse8_l1_register/action_dit_human_diverse8_l1_register_seed2_20261022_210127/2026.10.24.10.18.48.csv
[A25-eval-cfg]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/eval_navtest_action_dit_dinov3_vitl_human_diverse8_l1_register/action_dit_human_diverse8_l1_register_seed2_20261022_210127/code/hydra/config.yaml
[A26-cfg]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_action_dit_dinov3_vitl_navtrain_human_diverse8_l1_register_scorer/action_dit_human_diverse8_l1_register_scorer_seed2_20261022_210127/code/hydra/config.yaml
[A26-run]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_action_dit_dinov3_vitl_navtrain_human_diverse8_l1_register_scorer/action_dit_human_diverse8_l1_register_scorer_seed2_20261022_210127
[A26-log]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_action_dit_dinov3_vitl_navtrain_human_diverse8_l1_register_scorer/action_dit_human_diverse8_l1_register_scorer_seed2_20261022_210127/run_training.log
[A26-launch]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/scripts/training/run_drivor_action_dit_dinov3_vitl_human_diverse8_l1_register_scorer_train_eval.sh
[A26-ckpt]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_action_dit_dinov3_vitl_navtrain_human_diverse8_l1_register_scorer/action_dit_human_diverse8_l1_register_scorer_seed2_20261022_210127/lightning_logs/version_0/checkpoints/last.ckpt
[A26-csv]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/eval_navtest_action_dit_dinov3_vitl_human_diverse8_l1_register_scorer/action_dit_human_diverse8_l1_register_scorer_seed2_20261022_210127/2026.10.24.10.31.26.csv
[A26-eval-cfg]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/eval_navtest_action_dit_dinov3_vitl_human_diverse8_l1_register_scorer/action_dit_human_diverse8_l1_register_scorer_seed2_20261022_210127/code/hydra/config.yaml
[A27-cfg]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_action_dit_dinov3_vitl_navtrain_human_xl1_diverse8_l1_register/action_dit_human_xl1_diverse8_l1_register_seed2_dlc16mhamjyg8qtw_20261022_210127/code/hydra/config.yaml
[A27-run]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_action_dit_dinov3_vitl_navtrain_human_xl1_diverse8_l1_register/action_dit_human_xl1_diverse8_l1_register_seed2_dlc16mhamjyg8qtw_20261022_210127
[A27-log]: /cpfs29jb/data/zhaozc/project/drive_future/artifacts/dlc_workspace_26585/launch_diverse8_register_20261022_210127/reuse_state/dlc16mhamjyg8qtw/stage0.node0.log
[A27-launch]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/scripts/training/run_drivor_action_dit_dinov3_vitl_human_xl1_diverse8_l1_register_train_eval.sh
[A27-ckpt]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_action_dit_dinov3_vitl_navtrain_human_xl1_diverse8_l1_register/action_dit_human_xl1_diverse8_l1_register_seed2_dlc16mhamjyg8qtw_20261022_210127/lightning_logs/version_0/checkpoints/last.ckpt
[A27-csv]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/eval_navtest_action_dit_dinov3_vitl_human_xl1_diverse8_l1_register/action_dit_human_xl1_diverse8_l1_register_seed2_dlc16mhamjyg8qtw_20261022_210127/2026.10.24.09.55.15.csv
[A27-eval-cfg]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/eval_navtest_action_dit_dinov3_vitl_human_xl1_diverse8_l1_register/action_dit_human_xl1_diverse8_l1_register_seed2_dlc16mhamjyg8qtw_20261022_210127/code/hydra/config.yaml
[A28-cfg]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_action_dit_dinov3_vitl_navtrain_human_xl1_diverse8_l1_register_scorer/action_dit_human_xl1_diverse8_l1_register_scorer_seed2_20261022_210127/code/hydra/config.yaml
[A28-run]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_action_dit_dinov3_vitl_navtrain_human_xl1_diverse8_l1_register_scorer/action_dit_human_xl1_diverse8_l1_register_scorer_seed2_20261022_210127
[A28-log]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_action_dit_dinov3_vitl_navtrain_human_xl1_diverse8_l1_register_scorer/action_dit_human_xl1_diverse8_l1_register_scorer_seed2_20261022_210127/run_training.log
[A28-launch]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/scripts/training/run_drivor_action_dit_dinov3_vitl_human_xl1_diverse8_l1_register_scorer_train_eval.sh
[A28-ckpt]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_action_dit_dinov3_vitl_navtrain_human_xl1_diverse8_l1_register_scorer/action_dit_human_xl1_diverse8_l1_register_scorer_seed2_20261022_210127/lightning_logs/version_0/checkpoints/last.ckpt
[A28-csv]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/eval_navtest_action_dit_dinov3_vitl_human_xl1_diverse8_l1_register_scorer/action_dit_human_xl1_diverse8_l1_register_scorer_seed2_20261022_210127/2026.10.24.10.23.56.csv
[A28-eval-cfg]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/eval_navtest_action_dit_dinov3_vitl_human_xl1_diverse8_l1_register_scorer/action_dit_human_xl1_diverse8_l1_register_scorer_seed2_20261022_210127/code/hydra/config.yaml
[T08-cfg]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_query_dinov3_vitl_navtrain_human_diverse8_l1_register/query_human_diverse8_l1_register_seed2_dlc1stkt1n8i1ctp_20261022_210127/code/hydra/config.yaml
[T08-run]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_query_dinov3_vitl_navtrain_human_diverse8_l1_register/query_human_diverse8_l1_register_seed2_dlc1stkt1n8i1ctp_20261022_210127
[T08-log]: /cpfs29jb/data/zhaozc/project/drive_future/artifacts/dlc_workspace_26585/launch_diverse8_register_20261022_210127/reuse_state/dlc1stkt1n8i1ctp/stage0.node0.log
[T08-launch]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/scripts/training/run_drivor_query_human_diverse8_l1_register_train_eval.sh
[T08-ckpt]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_query_dinov3_vitl_navtrain_human_diverse8_l1_register/query_human_diverse8_l1_register_seed2_dlc1stkt1n8i1ctp_20261022_210127/lightning_logs/version_0/checkpoints/last.ckpt
[T08-csv]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/eval_navtest_query_dinov3_vitl_human_diverse8_l1_register/query_human_diverse8_l1_register_seed2_dlc1stkt1n8i1ctp_20261022_210127/2026.10.23.14.22.12.csv
[T08-eval-cfg]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/eval_navtest_query_dinov3_vitl_human_diverse8_l1_register/query_human_diverse8_l1_register_seed2_dlc1stkt1n8i1ctp_20261022_210127/code/hydra/config.yaml
[T09-cfg]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_query_dinov3_vitl_navtrain_human_diverse8_l1_register_scorer/query_human_diverse8_l1_register_scorer_seed2_dlcudc5qjukb34ak_20261022_210127/code/hydra/config.yaml
[T09-run]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_query_dinov3_vitl_navtrain_human_diverse8_l1_register_scorer/query_human_diverse8_l1_register_scorer_seed2_dlcudc5qjukb34ak_20261022_210127
[T09-log]: /cpfs29jb/data/zhaozc/project/drive_future/artifacts/dlc_workspace_26585/launch_diverse8_register_20261022_210127/reuse_state/dlcudc5qjukb34ak/stage0.node0.log
[T09-launch]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/scripts/training/run_drivor_query_human_diverse8_l1_register_scorer_train_eval.sh
[T09-ckpt]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_query_dinov3_vitl_navtrain_human_diverse8_l1_register_scorer/query_human_diverse8_l1_register_scorer_seed2_dlcudc5qjukb34ak_20261022_210127/lightning_logs/version_0/checkpoints/last.ckpt
[T09-csv]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/eval_navtest_query_dinov3_vitl_human_diverse8_l1_register_scorer/query_human_diverse8_l1_register_scorer_seed2_dlcudc5qjukb34ak_20261022_210127/2026.10.23.14.11.28.csv
[T09-eval-cfg]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/eval_navtest_query_dinov3_vitl_human_diverse8_l1_register_scorer/query_human_diverse8_l1_register_scorer_seed2_dlcudc5qjukb34ak_20261022_210127/code/hydra/config.yaml
[T10-cfg]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_query_dinov3_vitl_navtrain_single_gt/query_dinov3_vitl_single_gt_seed2_20261022_221845/code/hydra/config.yaml
[T10-launch]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/scripts/training/run_drivor_query_single_gt_train_eval.sh
[T10-validation]: /cpfs29jb/data/zhaozc/project/drive_future/artifacts/query_single_gt_20261022/validation.json
[T10-deployment]: /cpfs29jb/data/zhaozc/project/drive_future/artifacts/dlc_submissions/query_single_gt_20261022_221845/operation.json
[T10-ckpt]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/training_drivoR_query_dinov3_vitl_navtrain_single_gt/query_dinov3_vitl_single_gt_seed2_20261022_221845/lightning_logs/version_0/checkpoints/last.ckpt
[T10-csv]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/eval_navtest_query_dinov3_vitl_single_gt/query_dinov3_vitl_single_gt_seed2_20261022_221845/2026.10.23.17.20.24.csv
[T10-eval-cfg]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/eval_navtest_query_dinov3_vitl_single_gt/query_dinov3_vitl_single_gt_seed2_20261022_221845/code/hydra/config.yaml
[T01-epoch18-csv]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/eval_navtest_query_register_dinov3_vitl_single_gt/epochs16_20_local_20261023_213451_e18_ckpt17/2026.10.23.22.19.22.csv
[T01-epoch19-csv]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/eval_navtest_query_register_dinov3_vitl_single_gt/epochs16_20_local_20261023_213451_e19_ckpt18/2026.10.23.22.27.29.csv
[T01-epoch20-csv]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/eval_navtest_query_register_dinov3_vitl_single_gt/epochs16_20_local_20261023_213451_e20_ckpt19/2026.10.23.22.35.28.csv
[T03-epoch16-csv]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/eval_navtest_query_register_dinov3_vitl_scorer_single_gt/epochs16_20_local_20261023_213451_e16_ckpt15/2026.10.23.22.43.26.csv
[T03-epoch18-csv]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/eval_navtest_query_register_dinov3_vitl_scorer_single_gt/epochs16_20_local_20261023_213451_e18_ckpt17/2026.10.23.22.52.32.csv
[T03-epoch19-csv]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/eval_navtest_query_register_dinov3_vitl_scorer_single_gt/epochs16_20_local_20261023_213451_e19_ckpt18/2026.10.23.23.03.14.csv
[T03-epoch20-csv]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/eval_navtest_query_register_dinov3_vitl_scorer_single_gt/epochs16_20_local_20261023_213451_e20_ckpt19/2026.10.23.23.11.34.csv
[T05-epoch20-csv]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/eval_navtest_query_dinov3_vitl_single_gt_long/epochs16_20_local_20261023_213451_e20_ckpt19/2026.10.23.23.20.15.csv
[T06-epoch16-csv]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/eval_navtest_query_dinov3_vitl_single_gt_long_register/epochs16_20_local_20261023_213451_e16_ckpt15/2026.10.23.23.28.10.csv
[T06-epoch17-csv]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/eval_navtest_query_dinov3_vitl_single_gt_long_register/epochs16_20_local_20261023_213451_e17_ckpt16/2026.10.23.23.36.24.csv
[T06-epoch18-csv]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/eval_navtest_query_dinov3_vitl_single_gt_long_register/epochs16_20_local_20261023_213451_e18_ckpt17/2026.10.23.23.44.27.csv
[T06-epoch19-csv]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/eval_navtest_query_dinov3_vitl_single_gt_long_register/epochs16_20_local_20261023_213451_e19_ckpt18/2026.10.23.23.52.51.csv
[T06-epoch20-csv]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/eval_navtest_query_dinov3_vitl_single_gt_long_register/epochs16_20_local_20261023_213451_e20_ckpt19/2026.10.24.00.01.34.csv
[T07-epoch16-csv]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/eval_navtest_query_dinov3_vitl_human_diverse8_l1/epochs16_20_local_20261023_213451_e16_ckpt15/2026.10.24.00.09.15.csv
[T07-epoch17-csv]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/eval_navtest_query_dinov3_vitl_human_diverse8_l1/epochs16_20_local_20261023_213451_e17_ckpt16/2026.10.24.00.17.01.csv
[T07-epoch18-csv]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/eval_navtest_query_dinov3_vitl_human_diverse8_l1/epochs16_20_local_20261023_213451_e18_ckpt17/2026.10.24.00.24.42.csv
[T07-epoch19-csv]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/eval_navtest_query_dinov3_vitl_human_diverse8_l1/epochs16_20_local_20261023_213451_e19_ckpt18/2026.10.24.00.32.27.csv
[T07-epoch20-csv]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/eval_navtest_query_dinov3_vitl_human_diverse8_l1/epochs16_20_local_20261023_213451_e20_ckpt19/2026.10.24.00.40.05.csv
[T08-epoch16-csv]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/eval_navtest_query_dinov3_vitl_human_diverse8_l1_register/epochs16_20_local_20261023_213451_e16_ckpt15/2026.10.24.00.48.00.csv
[T08-epoch17-csv]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/eval_navtest_query_dinov3_vitl_human_diverse8_l1_register/epochs16_20_local_20261023_213451_e17_ckpt16/2026.10.24.00.55.52.csv
[T08-epoch18-csv]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/eval_navtest_query_dinov3_vitl_human_diverse8_l1_register/epochs16_20_local_20261023_213451_e18_ckpt17/2026.10.24.01.03.43.csv
[T08-epoch19-csv]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/eval_navtest_query_dinov3_vitl_human_diverse8_l1_register/epochs16_20_local_20261023_213451_e19_ckpt18/2026.10.24.01.11.33.csv
[T08-epoch20-csv]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/eval_navtest_query_dinov3_vitl_human_diverse8_l1_register/epochs16_20_local_20261023_213451_e20_ckpt19/2026.10.24.01.19.28.csv
[T09-epoch16-csv]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/eval_navtest_query_dinov3_vitl_human_diverse8_l1_register_scorer/epochs16_20_local_20261023_213451_e16_ckpt15/2026.10.24.01.27.24.csv
[T09-epoch17-csv]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/eval_navtest_query_dinov3_vitl_human_diverse8_l1_register_scorer/epochs16_20_local_20261023_213451_e17_ckpt16/2026.10.24.01.35.21.csv
[T09-epoch18-csv]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/eval_navtest_query_dinov3_vitl_human_diverse8_l1_register_scorer/epochs16_20_local_20261023_213451_e18_ckpt17/2026.10.24.01.43.18.csv
[T09-epoch19-csv]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/eval_navtest_query_dinov3_vitl_human_diverse8_l1_register_scorer/epochs16_20_local_20261023_213451_e19_ckpt18/2026.10.24.01.51.12.csv
[T09-epoch20-csv]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/eval_navtest_query_dinov3_vitl_human_diverse8_l1_register_scorer/epochs16_20_local_20261023_213451_e20_ckpt19/2026.10.24.01.59.02.csv
[T10-epoch16-csv]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/eval_navtest_query_dinov3_vitl_single_gt/epochs16_20_local_20261023_213451_e16_ckpt15/2026.10.24.02.06.35.csv
[T10-epoch17-csv]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/eval_navtest_query_dinov3_vitl_single_gt/epochs16_20_local_20261023_213451_e17_ckpt16/2026.10.24.02.14.07.csv
[T10-epoch18-csv]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/eval_navtest_query_dinov3_vitl_single_gt/epochs16_20_local_20261023_213451_e18_ckpt17/2026.10.24.02.21.38.csv
[T10-epoch19-csv]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/eval_navtest_query_dinov3_vitl_single_gt/epochs16_20_local_20261023_213451_e19_ckpt18/2026.10.24.02.29.09.csv
[T10-epoch20-csv]: /cpfs29jb/data/zhaozc/project/drive_future/DrivoR/exp/ke/eval_navtest_query_dinov3_vitl_single_gt/epochs16_20_local_20261023_213451_e20_ckpt19/2026.10.24.02.36.40.csv
