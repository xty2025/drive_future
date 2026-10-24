# Future register 对齐：代码与小样本诊断

本次检查针对普通 Transformer 的 T08（human＋diverse8 L1＋未来 register、原 Scorer），使用其完成 25 轮后的 `last.ckpt`。结论是：**GT、proposal 与未来视频的配对没有发现错位；更直接的线索是未来分支对 Action 条件利用很弱，预测的 register 缺少候选差异。现有证据不足以把原因归结为 L1 这个损失函数本身。**

审计使用两个真实 NAVSIM 场景，一个来自 train、一个来自 val；保持 64 proposals、当前前视 672×1152、未来帧 384×384。模型处于 eval 模式以排除 Dropout/GridMask 的扰动，前向用 BF16，loss/统计用 FP32。没有 optimizer step、EMA 更新、在线 oracle 或新训练，也未修改生产模型与启动配置。两例诊断不能替代全数据统计或新实验。

复现脚本：[diagnose.py](../artifacts/register_alignment_diagnosis_20261023/diagnose.py)。完整结果：[results.json](../artifacts/register_alignment_diagnosis_20261023/results.json)。

## 1. 当前配对逻辑正确，方向是“每条 GT 选 proposal”

对每个有效 GT `g`，在同一场景的全部 64 条输出中选择：

```text
C[b,g,m] = mean_time(sum_xyz_heading(abs(P[b,m] - GT[b,g])))
i[b,g]   = argmin_m C[b,g,m]

R_selected[b,g,t,r,d] = R_pred[b,i[b,g],t,r,d]
R_target[b,g,t,r,d]   = EMA_DINO(future_RGB[b,g,t])[r,d]
```

当前普通 Transformer 的 future loss 是：先平均该 GT 的 register/channel L1，再平均有效未来时刻，再平均该场景的有效 GT，最后平均 batch。匹配索引不参与求导，但选中的 register 预测保留梯度。不是按 Scorer 最高分选择，也不是按 register 特征重新做匹配。

- Human 和 diverse8 分别按自己的轨迹误差选 winner，再按 `[human, diverse8 slot 0, ..., slot 7]` 拼接匹配索引与未来目标。
- 允许多个 GT 选择同一 proposal；这符合现有 nearest 方案。此时同一组预测 register 会承担多份目标的 loss。两例实际检查都是 9 个有效 GT 选中 9 个不同 proposal，未观察到匹配冲突，不能把这两例的弱效果归因于重复匹配。
- Diverse8 loader 保留原始 slot，并核对轨迹、候选编号、label 和视频缓存 slot；有效 mask 会排除 padding。
- 未来监督是 `+0.5,+1.0,...,+4.0s`，对应 Cosmos 视频零基索引 `5,10,...,40`。Human 的轨迹与未来图像都取同一 Scene 的后续 8 帧。

实现见 [diverse_gt.py](navsim/agents/drivoR_query_register/diverse_gt.py)、[配对辅助函数](navsim/agents/drivoR_action_dit_register/diverse_losses.py)、[matched_future_loss](navsim/agents/drivoR_query_register/losses.py)、[未来帧读取](navsim/agents/drivoR_action_dit_register/diverse_targets.py)。

## 2. 直接证据：预测 register 几乎不依赖 Action 通道

两例均保留原图像、模型参数和匹配索引；只在内存中的模型上临时屏蔽 `Register ← Action` 连接，计算后恢复原 mask。

| 诊断 | train 场景 | val 场景 |
|---|---:|---:|
| 正常 future L1 | 0.073533 | 0.094388 |
| 屏蔽 Register 读取 Action 后的 future L1 | 0.073540 | 0.094358 |
| 用 64 组预测 register 的均值替代各自输出 | 0.074633 | 0.094561 |
| 打乱预测 register 与 proposal 的对应关系 | 0.076219 | 0.094820 |
| 用当前帧的 EMA register 直接复制到所有未来时刻 | 0.083216 | 0.075639 |
| 不同 proposal 预测 register 的平均余弦相似度 | 0.998935 | 0.999205 |
| 不同 GT 未来 register 的平均余弦相似度 | 0.976002 | 0.981121 |

屏蔽 Action 后 loss 变化约 `+0.010% / -0.032%`；换成各候选的平均 register 后，loss 只增加约 `1.50% / 0.18%`。在这两例里，保留单独的候选 register 带来的重建收益很小。

各层匹配 register 对自身 Action 的平均 attention 权重分别为：

- train：`0.31%、0.73%、0.076%、2.46%`；
- val：`0.18%、0.43%、0.036%、2.18%`。

这支持一个具体解释：未来分支主要依赖当前图像、时间/slot embedding 和独立的 future query，较少读取当前 Action 表征。因为每个 proposal 还有自己的 future queries，不能说它完全没有候选身份条件；但它对实际 Action 内容的依赖明显偏弱。仅把 Self-Attention 的方向设正确，并不能保证模型会利用这条连接。

当前 DINO register 是图像级全局 storage tokens，四个槽位不对应固定道路区域或动态物体。它们在不同轨迹未来中的差异，比局部时空 patch 更容易被全局外观淹没。两例中目标跨 GT 的 L1 差异约 `0.057～0.064`，而预测跨 proposal 的差异只有 `0.0106～0.0118`。这是预测差异收缩的线索；本次没有对比完整数据和原始预训练 teacher，不能据此认定 teacher 已经发生表示坍塌。

## 3. Future loss 的梯度没有直接到达轨迹坐标头

当前网络从 shared blocks 的两类输出分别经过轨迹 MLP 和 register head。未来分支没有把已预测的 `[8,3]` 轨迹重新编码为输入；轨迹仅参与离散匹配。因此：

- `future loss → trajectory_head`：无梯度；
- `future loss → predicted trajectory coordinates`：无梯度；
- `future loss → earlier Action states / action_queries / shared blocks / visual LoRA`：有梯度。

这来自计算图结构，不是 Scorer 的 detach 误伤了未来分支。Scorer 仍只对自身的轨迹/register 输入做 detach。

| 参数组 | Future 梯度范数 / 轨迹梯度范数：train | 同比例：val |
|---|---:|---:|
| Action queries | 0.0085% | 0.0050% |
| 共享 query decoder | 2.93% | 2.57% |
| 视觉 LoRA | 2.88% | 1.40% |
| 轨迹输出头 | 0 | 0 |

两例中 Future 对视觉 LoRA 的梯度与轨迹梯度的 cosine 为 `+0.54 / -0.57`，共享 decoder 为 `+0.023 / -0.067`。存在方向不一致的样本，但既没有显示一致的梯度冲突，也没有显示 Future 梯度压倒轨迹梯度。因此不能仅凭 `loss_weight=1` 认定未来 loss 太大，更不宜直接把它统一减小。

这是训练完成时的局部梯度诊断；早期训练的梯度比例尚未测量。

## 4. 多 GT 的 future 权重与轨迹权重并不一致

当前轨迹项为：

```text
L_traj = L_human + 0.5 * mean_valid_D8(L_D8)
```

未来项则对 human 和所有有效 D8 逐 GT 均分。8 条 D8 都有效时，生成视频贡献未来目标的 `8/9`，真实 human 未来只占 `1/9`。

这不是索引 bug，也不违反当前 future 均分实现，但意味着辅助任务主要在拟合 Cosmos 未来的 DINO 表征。它与 human 主导的轨迹监督具有不同的数据来源权重。是否有生成外观偏差，需要单独比较 human future 与 D8 future 的验证误差；不能从总 future loss 下降直接推出真实场景的规划改善。

## 5. 与 V-JEPA2 masked prediction 的实质区别

官方 V-JEPA2 ViT-L 的 `loss_exp=1.0`，因此也是普通 L1：

```text
h = stop_gradient(target_encoder(full_clip))
h = layer_norm(h, normalized_shape=(D,))
h_target = gather(h, masks_pred)
loss = mean(abs(predicted_target_tokens - h_target))
```

| 项目 | 当前普通 Transformer | 官方 V-JEPA2 预训练 |
|---|---|---|
| 预测目标 | 每个未来图像的 4 个 DINOv3 storage/register tokens | 有明确时空位置的 masked video patch tokens |
| 位置对应 | 轨迹匹配索引＋未来时刻＋register 槽位 | `masks_pred` 中的时空 patch 索引 |
| Teacher | 当前 DINOv3 有效 Q/V LoRA 权重的 EMA | 视频 encoder 的 EMA |
| 目标归一化 | DINO 自身的带仿射 LayerNorm 输出 | encoder 输出后再做无仿射 `F.layer_norm` |
| 默认回归函数 | L1 | L1，`loss_exp=1.0` |
| Predictor 的 mask query | 每 proposal 独立的 32 个可学习 queries | mask embedding＋对应时空位置 |

**当前目标并非完全未归一化。**DINOv3 的 `x_storage_tokens` 已经过 backbone 的 norm。本次测得目标 token 的通道标准差约 `0.50`；外加 `F.layer_norm` 后才会成为近似单位标准差。这是可做的对照，但不能把现有实现描述成漏掉了所有归一化。

V-JEPA2 的 `mask` 在这里主要表示“取哪些目标位置”，与 Action/Register 的可见性 attention mask 是两种不同用途。不能直接照搬它的无方向限制 predictor attention，改变当前 Action 不读取 future register 的约束。

官方 V-JEPA2 预训练对 target 做上述归一化，默认不会在最终 predictor 输出上再做同样的归一化。V-JEPA2.1 的 ViT-L 配置同样默认 L1，`normalize_predictor=false`。V-JEPA2-AC/DROID 是另一条动作条件训练路径，显式编码 actions/states，并按 `normalize_reps` 对预测和目标处理，需要与普通 masked pretraining 区分。

参考：[V-JEPA2 loss](../vjepa2/app/vjepa/train.py)、[ViT-L 配置](../vjepa2/configs/train/vitl16/pretrain-256px-16f.yaml)、[mask gather](../vjepa2/src/masks/utils.py)、[predictor](../vjepa2/src/models/predictor.py)、[动作条件 predictor](../vjepa2/src/models/ac_predictor.py)、[V-JEPA2.1 loss](../vjepa2/app/vjepa_2_1/train.py)。

## 6. 建议的对照顺序

1. **先保留配对与 L1，只增加 V-JEPA2 式 target LayerNorm。**对每个 `[1024]` target token 做无仿射 `F.layer_norm`，然后沿用当前匹配索引与有效帧/GT 平均。保持其他模块不变，单独检验目标尺度的影响。这会改变训练目标，需要重新训练或受控微调，不能用旧 checkpoint 的原始 loss 大小直接判定新目标更好。
2. **优先加强显式轨迹条件。**把预测的 8 个 `(x,y,heading)` 点编码后输入未来预测分支，使 Future loss 能回传到轨迹坐标头。仍可保持 `Action 不读 Register`，无需放开当前 mask。这比仅增加共享 Transformer 层数更直接地针对已观察到的条件利用不足。
3. **单独比较 global register 与有空间位置的未来特征。**例如保留每帧 4 个输出槽位，把 target 改成 DINO patch 的 2×2 空间池化特征；或做 V-JEPA2/2.1 video patch target 对照。这借鉴了 masked prediction 的空间对应关系，但改变了预测对象，应使用独立实验名。
4. **记录并控制 Future 对主任务的实际梯度影响。**增加 Action 条件后重新测梯度比例；同时可把 future 的 human/D8 组权重与轨迹侧做对照。当前两个后期样本不支持“一律把 future 系数调小”的结论，也不支持直接调大来解决弱条件利用。

以上是建议的独立对照，没有直接改动已训练模型、现有 loss 默认值、matching、Scorer 或线上任务。

## 7. 验证集与测试集应分开看

| 对照 | 最后一轮 val PDMS | navtest PDMS |
|---|---:|---:|
| T10：human，无 register | 91.58 | 93.32 |
| T01：human＋register，原 Scorer | 92.03 | 93.21 |
| T07：human＋D8，无 register | 93.19 | 93.82 |
| T08：human＋D8＋register，原 Scorer | 93.26 | 93.53 |
| T09：human＋D8＋register，融合 Scorer | 92.96 | 93.55 |

T07/T08 在 val 上近乎持平、略有上升，而 navtest 下降约 0.29；不能解释成所有数据上都出现了训练退化。Val 的 oracle-best PDMS 为 `99.5498 / 99.5570`，也很接近。当前更准确的结论是：**未来辅助目标没有稳定转化成规划泛化收益，且两例直接诊断显示该分支对 Action 内容利用不足。**具体哪项修改能改善 navtest，需要保持其他条件一致的新实验验证。

验证日志摘录：[validation_metrics.json](../artifacts/register_alignment_diagnosis_20261023/validation_metrics.json)。完整测试对照见 [EXPERIMENT_RESULTS_SUMMARY.md](exp/ke/EXPERIMENT_RESULTS_SUMMARY.md)。

## 8. 实算：是否额外加 target LayerNorm，特征差别多大？

使用同一个 T08 EMA teacher（33250 次更新）、上述两个场景的 human＋diverse8 未来图像，共 **18 条 GT、144 张未来图像、576 个 1024 维 register tokens**。先只编码一次，再对同一份 FP32 特征计算 `F.layer_norm(x, (1024,), eps=1e-5)`。两种情况都保留 DINO 自身的内部 norm；这里比较的是是否再加一层无仿射 target LayerNorm。

**数值尺度变化明显，方向和相对关系变化很小。**

| 特征统计，两个场景合并 | 不额外加 target LN | 额外加 target LN |
|---|---:|---:|
| 每 token 通道均值的平均 | 0.01381 | 约 0 |
| 每 token 通道标准差的平均 | 0.50255 | 0.99998 |
| 特征元素绝对值的平均 | 0.28622 | 0.57991 |
| 每 token L2 范数的平均 | 16.0902 | 31.9994 |

前后同一 token 的平均余弦相似度为 **0.999491**，平均夹角约 **1.54°**；元素平均绝对变化为 **0.29608**。较大的绝对变化主要来自缩放。用一个全局仿射关系近似可得：

```text
LN(x) ≈ 1.9783 × x − 0.02732，拟合 R² = 0.99421
```

实际是逐 token 减均值、除标准差，不是所有 token 乘完全相同的常数；此次缩放系数为 **1.783～2.218**，平均 **2.0004**。原四个 register 槽位的标准差不同，因此额外 LN 也会改变槽位间的相对尺度。

| 相同场景，不同 GT 的未来特征关系 | train 场景：不加 → 加 LN | val 场景：不加 → 加 LN |
|---|---:|---:|
| 不同 GT 的平均 L1 距离 | 0.06445 → 0.13198 | 0.05711 → 0.11708 |
| 不同 GT 的平均余弦相似度 | 0.976002 → 0.975997 | 0.981121 → 0.981116 |
| GT 间 L1 距离 / 特征平均绝对值 | 0.22582 → 0.22805 | 0.19898 → 0.20149 |
| 相邻未来时刻的 L1 距离 | 0.03810 → 0.07813 | 0.04214 → 0.08622 |
| 归一化前后 GT 间 L1 距离的 Pearson 相关系数 | 0.999820 | 0.999941 |
| 最近邻 GT 身份保持不变 | 9/9 | 9/9 |

GT 之间的绝对距离约放大 **2.05 倍**，但相对于特征自身幅度的差异几乎不变，余弦相似度与最近邻关系也基本保持。这里的“最近邻 GT”只用于诊断特征几何；生产代码的 proposal 匹配依然按轨迹 L1 进行，不会被 target LN 改写。

因此，额外 target LN 可以统一目标尺度并调整回归任务的标度，但这次计算没有显示它会明显增强不同候选未来的相对区分度。不能把它当成“让 register 开始利用 Action”的充分条件。前文的 LN 实验仍适合作为小改动对照，应降低对它单独解决弱轨迹条件问题的预期。

如果只改变 target，而继续使用按旧尺度训练的预测头，loss 数值会受目标尺度变化影响；不能据此直接判断规划性能会变好或变坏。本次只比较特征，没有重新训练或评测 PDMS。

复现与数据：[比较脚本](../artifacts/register_alignment_diagnosis_20261023/compare_target_layernorm.py)、[完整统计](../artifacts/register_alignment_diagnosis_20261023/target_layernorm_comparison.json)、[同一批原始及归一化特征](../artifacts/register_alignment_diagnosis_20261023/target_layernorm_features.npz)。标准差使用总体定义（`correction=0`），与 LayerNorm 的方差定义一致。

## 9. V-JEPA2-AC：预测器如何利用动作轨迹？

AC 的重点是显式建模“在状态 `s_t` 执行动作 `a_t`，视觉状态将如何变化”。它的 predictor 输入有数值含义的 actions/states，输出下一帧视觉特征；训练数据中的动作由机器人记录给出。它本身不是生成动作的规划头。

### 9.1 动作是相邻位姿的变化

在 [DROID 数据读取](../vjepa2/app/vjepa_droid/droid.py) 中：

```text
s_t = [末端执行器 xyz、Euler 角、gripper 开合度]，7 维
a_t = [xyz_{t+1} - xyz_t、相对旋转的 Euler 角、gripper_{t+1} - gripper_t]，7 维
```

旋转差通过 `R_{t+1} R_t^T` 计算；示例配置默认保持机器人基座坐标系。8 帧对应 8 个状态和 7 个动作。这里没有把 action 当作每个候选的可学习编号，也没有让它只是一个与坐标输出并行的隐向量。

### 9.2 每个时刻的动作、状态和视觉直接做 Self-Attention

以仓库 [DROID 配置](../vjepa2/configs/train/vitg16/droid-256px-8f.yaml) 为例：

```text
视觉特征 z_t [256,1408] → Linear(1408,1024)
动作 a_t [7]           → Linear(7,1024) → 1 个 action token
状态 s_t [7]           → Linear(7,1024) → 1 个 state token

第 t 个时间块 = [action_t, state_t, 256 个视觉 patch tokens]
完整序列     = [块0 | 块1 | ... | 块T-1]
              → 24 层 ACBlock，1024 维，16 heads，FFN 4096
              → 去掉 action/state 位置
              → predictor LayerNorm → Linear(1024,1408)
              → 预测的下一时刻视觉 patch 特征
```

每层 ACBlock 是 `LN → Self-Attention → residual → LN → FFN → residual`。这里没有你当前 decoder 中独立的 Cross-Attention 子层，也不初始化额外的未来 register/mask queries；未来特征由输入视觉 token 的输出位置产生。

代码见 [AC predictor](../vjepa2/src/models/ac_predictor.py) 的 `forward`，以及 [ACBlock / ACRoPEAttention](../vjepa2/src/models/utils/modules.py)。

### 9.3 Mask 是按时间分块的因果关系

- 第 `t` 个时间块可以读取 `0...t` 的动作、状态和视觉 tokens；不能读取 `t+1` 及后续输入块。
- 同一个时间块内部可相互读取，视觉可以读取 `a_t/s_t`，动作/状态位置也可以读取同一时刻视觉。
- 动作/状态 tokens 用时间 RoPE；视觉 tokens 用时间和空间位置 RoPE。
- AC 的布尔 mask 传给 SDPA，`True` 表示允许；当前普通 Transformer 用 `nn.MultiheadAttention`，布尔 `True` 表示禁止。移植时不能原样复用布尔值语义。

它没有你现在的 `Action 不读未来 Register` 这一特定分支结构。其同时间块视觉是当前／已预测的状态特征，不是被直接喂入的未来 GT。

### 9.4 下一帧监督、滚动预测与归一化

[AC 训练代码](../vjepa2/app/vjepa_droid/train.py) 先用冻结的 target encoder 逐帧提取 `h_0...h_7`。单帧复制两次以满足 backbone 的 tubelet=2；此路径没有普通预训练那种随机 masked-patch 任务。

```text
Teacher forcing:
  predictor(h_0...h_6, a_0...a_6, s_0...s_6)
    → h_hat_1...h_hat_7
  与真实 h_1...h_7 计算 L1

Autoregressive，示例 auto_steps=2:
  (h_0, a_0, s_0) → h_hat_1
  (h_0, h_hat_1, a_0, a_1, s_0, s_1) → h_hat_2
  对两步 rollout 计算 L1

L = L_teacher_forcing + L_rollout
```

**这里默认 `normalize_reps=true`，会对 encoder 特征与 predictor 最终输出两侧都做额外 `F.layer_norm`。**普通 V-JEPA2 masked 预训练的默认路径只在 target 侧额外做该归一化，两者应明确区分。

AC 后训练实际使用冻结的视觉特征、训练 predictor，没有对 target encoder 做 EMA 更新。训练脚本虽创建 context encoder 和对应 optimizer group，但该 context encoder 未参与上述 forward，因而没有梯度；不能把这条路径描述成和当前 DINO LoRA＋EMA 联训完全相同。

### 9.5 推理时以候选动作驱动 world model

示例 [MPC/CEM](../vjepa2/notebooks/utils/mpc_utils.py) 在动作序列空间采样候选，把每个候选送入 predictor 做逐步 latent rollout，用预测最终视觉特征到目标图像特征的 L1 距离选择较好的动作序列，再更新采样分布。状态由当前 pose 和动作解析更新，见 [world model wrapper](../vjepa2/notebooks/utils/world_model_wrapper.py)。这不是 NAVSIM PDMS Scorer。

### 9.6 对当前网络最值得借鉴的部分

| 当前普通 Transformer | AC 提供的参考 |
|---|---|
| 每个候选一个可学习 Action query，最终经 MLP 输出 8 个轨迹点 | 数值动作和状态逐时刻经过 Linear，直接进入未来预测器 |
| register 读取 Action 隐向量，没有直接输入已预测坐标 | 视觉预测显式以动作／位姿序列为条件 |
| Future loss 不能直接更新最终轨迹坐标头 | 若输入换成可微的预测轨迹，Future loss 可沿此输入回传到轨迹头 |
| 预测四个全局 register／未来帧 | 预测带空间位置的全部视觉 patch 特征 |
| DINO LoRA＋EMA 联训、仅 target 沿用内部 norm | AC 使用冻结视觉特征，示例额外归一化预测和目标 |

可借鉴的 NAVSIM 链路是：先生成 64 条 `[8,3]` proposal；按现有规则为每个 GT 选 proposal；用**选中的预测轨迹**构造逐时刻状态／增量，送入未来 predictor；再对齐该 GT 的未来特征。预测轨迹作为条件时应保留梯度，用 PyTorch 计算位姿差分与 heading wrapping。DROID 的 NumPy/SciPy 差分函数适合生成数据标签，不能直接搬来处理需要回传梯度的预测轨迹。

如果未来 predictor 输入的是 GT 轨迹，而不是预测轨迹，就不会获得这条到生成器的直接梯度路径。若 Scorer 消费 register，推理时还需为所有待评分 proposal 生成相应未来特征；仅训练匹配者的计算量与推理全部候选的计算量应分别设计。

**从现有小样本证据看，显式数值动作条件比单独增加 target LN 更值得优先检验。**但 AC 的结构允许读取动作，并不从数学上保证一定使用动作；仍应保留 Action 打乱／屏蔽诊断。可以先只借鉴动作条件接口，保持当前 64 proposals、GT 匹配与 Scorer，不必同时搬入 24 层大 predictor 和多步 rollout。以上为代码对照与后续设计建议，未修改现有训练网络。
