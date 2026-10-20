# NVS 维护约束（nvs_v6_wan_e2e 唯一谱系）

本仓库现在只有一条实现谱系：`src/nvs_v6_wan_e2e/`，配置 `config/config_v6_wan_e2e.yaml`，
契约测试 `tests/contract_tests_v6_wan_e2e.py`。`src/nvs_v6/`、`src/nvs_v6_wan/`、
`config/config_v6_final.yaml`、`tests/contract_tests_v6.py` 及其脚本已在 2026-10-14
收敛时删除，需要历史版本请从 commit `52a9d64` 取回。本文件此前描述那两个包的段落
（66 scene token / 12 宽 ego / 224×224 视觉 / scorer K/V detach / 单步
`endpoint = source + head(...)`）对现在的代码**全部不成立**，已随包一起删除。

- 权威契约文档：`NAVSIM_MultiGT_FutureRegister_ActionDiT_Codex_Spec_v6_FINAL.md`。
  它是历史 spec，其中面向 `nvs_v6` 的部分（66 token、12 宽 ego、`xy_scale_m`、
  detach 的 scorer 输入）已被 e2e 契约取代；冲突时以 `config/config_v6_wan_e2e.yaml`
  和 `src/nvs_v6_wan_e2e/config.py` 的严格 loader 为准。
- 面向用户的说明在 `README.md`；本文件是给改代码的 agent 看的约束层，不重复架构描述。
- `config.py` 是严格 loader：它会**拒绝**任何偏离契约的配置写法（包括上面那些旧写法）。
  改契约必须同时改 yaml、loader 断言和负向测试，三处一起。

## 硬性规则（不要绕过）

- **不要声称任何 NAVSIM 指标来自合成冒烟。** oracle 数字必须有 `DEFAULT_CACHE` 下的真实
  metric cache；冒烟只能证明 loss 有限、参数在更新、shape 正确。
- **保留真实 ego 顺序 `pose3, velocity2, acceleration2, command4`。** e2e 里可观测 ego
  特征是 **11 维**（取 `ego_history[:, -1]` 的 7 维拼 4 维 command），**没有** 12 宽 pad；
  `ego_history` 张量本身是 7 宽。旧的 12 宽 checkpoint 接口只属于已删除的 `nvs_v6`。
- **轨迹永远是米制 `[x, y, heading]`，任何地方都不做 rescale。** `/20`、`xy_scale`、
  `encode_trajectory`、`decode_trajectory`、`gt_encoded` 都已删除，不要加回来。
- `native_drivor.py`（原生 DINOv2 Scene + 原生 Ego cache）**只用于显式对比**，禁止出现在
  默认训练路径、真数据冒烟或任何 launcher 里。
- 永远用 `python -m pytest`，不要用 venv 的 `pytest` shim（shebang 在本镜像上是坏的）。
  `PYTHONPATH` 必须写绝对路径：该 venv 的 editable `navsim` `.pth` 指向不存在的
  `/mnt/zczhao_workspace/...`（源头拼写错误），相对路径会静默解析失败。
- `torch.load` 一律 `weights_only=True`（仓库有 PostToolUse 安全钩子会标记 `False`）。
  只有在 payload 确实含非张量对象、且已确认来源可信时才例外，并要在代码里写明原因。
- 本镜像没有 `/usr/bin/time`，计时用 bash 的 `$SECONDS`。机器只有 **50 GB RAM**，
  任何持有 per-scene 大张量的循环都要算一遍驻留量（Stage 1 约 195 MB/scene，
  见 `dlc_navtrain_wan_e2e.sh` 的 preflight）。
- 提交前用 `git status` 复核暂存内容；**永远不要** `git add -A` / `git add .`。
- `dino_tokens.py` 和 `lora.py` 是死代码，但删除需要用户明确授权，不要顺手清掉。
  `manifest.py` 和 `rollout_data.py` 已在本次收敛中删除（同样无人 import，且
  `rollout_data` 只是 `multiview_data.load_multiview_sample` 的透传壳）。

## 契约要点（改动时最容易破坏的几处）

- ActionDiT 的自注意力序列是 **`[Scene(81) | Action A(8) | Register R(32)]` 共 121 token**。
  Scene 由 `text_embedding(S(81))` **只 embed 一次**，然后**用两遍**：既拼进自注意力序列，
  又原样当 cross-attention 的 K/V——这样 Wan 预训练的 `cross_attn.q/k/v/o`、`norm_q/norm_k`
  看到的输入分布没变，226 key `strict=True` 的加载才仍然有意义。**cross-attention 的
  K/V 只有 Scene**，register 从不进去，所以它不可能是 Action 读到未来的第二条路。
  Register 段是 `8 未来帧 × REGISTER_QUERIES_PER_FRAME=4` = **32 token**（旧记录里的
  `R(8)` / 97 token 是 8-token 时代，已作废）。
  RoPE 按分段约定铺：**每段从 0 重启**。Scene 取 `rope_for(81)`、Action 取 `rope_for(8)`
  （必须用 Wan 自己预训练的位置，因为 `head`、`time_queries` 和搬过来的 8 层 block 都是在
  那个位置上训出来的），Register 取 `register_rope(32)` = `rope_for(8).repeat_interleave(4)`
  ——**每帧的 4 个 query 共享该帧的相位**，而不是 `0..31` 顺数。顺数会把一帧的四个 query
  放到四个不同时刻、并把 register 段拉得比 action 段长，破坏"register token `q+4f` 对应
  action token `f`（即 `(f+1)×0.5s`）"这个一一对应。同帧四个 query 靠 `future_queries`
  和 `register_type_embedding` 区分；**RoPE 只携带时间**。
  三段刻意重叠的位置靠三个 type embedding（`scene_type_embedding` / `action_type_embedding`
  / `register_type_embedding`，各自 `randn(1,1,256)×0.02`，在 skip 前缀里保持随机）区分。
  `Head` 读 `x[:, 81:89]` 出 flow velocity，`future_head` 读 `x[:, 89:]` 出预测 register
  （32 行 × Teacher 自己的 1024 宽）。不同 candidate 之间永不互相 attend（每条 candidate
  自己一份 121）。`predict_future=False` 时 register 整段不进序列，变成 `[Scene | Action]`
  = 89 token，单表退化成左上角 2×2，不需要特判；实测此时 velocity 与带 register 时相差
  **8.1e-10**（相对 3.2e-07 ≈ 2.7× float32 eps），是 SDPA 在不同序列长度下的累加噪声，
  **不是**信息流——同一输入重跑逐位相同，register 输入从全零换到 1e3 量级 velocity 也
  逐位相同。
- **自注意力只有一张 segment mask，Stage 1 / Stage 2 / 部署推理完全共用，没有 stage 参数**
  （`NvsActionDiT.SEGMENT_VISIBILITY`，行=query、列=key，1=可读）：

  | query\key | Scene | Action | Register |
  |---|---|---|---|
  | Scene | ✓ | ✗ | ✗ |
  | Action | ✓ | ✓ | **✗** |
  | Register | ✓ | ✓ | ✓ |

  即 `((1,0,0),(1,1,0),(1,1,1))`。**硬不变量：Action 永远读不到 Register（未来）**，
  两个 stage 都一样。Register 读 Action 是必需的——那是 `(Scene, Action) → Future` 这条
  轨迹条件通路本身；关掉它会让 register 在同一场景的 K 条 GT 上逐位相同，把
  `paired_future_l1` 钉在 `scene_mean_target_l1` 地板上、`gain_over_trajectory_blind`
  恒为 0（实测 depth-2 trunk 上关掉该条目 cross-GT `max|dRegister| = 0.0`，打开是 `7.5e-3`）。
  Scene 只读自己，所以 ActionDiT 永不写 scorer 消费的那批 token；`_flow_pass` 返回的
  `scene` 就是 eta 的精修输出，DiT 内部的 scene token 从不外泄（`readout = x[:, 81:]`）。

  **旧版"两张 per-stage 表 + `attention_stage` 必填位置参数 + Stage 2 反转箭头"已整体作废**，
  连同 `STAGE1_/STAGE2_SEGMENT_VISIBILITY`、`attention_stage_of` 一起删除；`config.py` 的
  严格 loader 现在**拒绝**这些键重新出现在 yaml 里。三个入口
  （`forward_register_only` / `forward_flow` / `forward_pairs`）都走同一个 `_flow_pass`，
  一次 forward、一张 mask。

  行为级验证（生产 geometry，`num_heads=4 / depth=8 / R=32 / D=1024`，2026-10-18 实测）：
  register 输入扰动 100× → velocity `max|diff| = 0.000e+00` **逐位相同**；
  `d velocity.sum() / d register输入` **恰为全零**；`future_head` 与 `future_queries`
  从 velocity 拿到的梯度为 **False**；反向对照把表改成全可见后同一扰动使 velocity 变化
  `1.2e-04`（≈|v|max 的 17%），证明这些零**不是空断言**、mask 确实是唯一屏障。
  同一次探测里 `d register / d action = 7.2e-01`、扰动 action 使 register 变化 `3.1e-02`，
  即轨迹条件通路是活的。
  对角块全可见，所以不存在全 `-inf` 的 softmax 行（加 mask 唯一会静默产出 NaN 的方式），
  契约测试用 `mask.any(dim=-1).all()` 钉住。
  **注意 bool 语义方向**：SDPA 的 `attn_mask` 里 `True = 允许`，而
  `nn.MultiheadAttention` 的 `attn_mask` 里 `True = 禁止`，两者相反，别混用。
  **`loss_future` 跨 stage 仍不可直接比数**，但原因换了：不再是 mask 反转，而是
  Stage 1 跑 `timestep=0` + clean GT 轨迹、Stage 2 跑随机 `sigma` + 加噪轨迹，而
  timestep 经 adaLN 调制每个 block——实测 condition 扰动自身量级会使 register 输出变化
  **44%** 的 `|register|`。函数是同一个，工作点不是。
- **Stage 1 的 loss 是 `identity_future_l1` + `lambda_contrast × paired_contrast_future_l1`
  （`--lambda-contrast` 默认 1.0，0 退回纯 identity）。** 加对比项是因为背景在目标里占的
  比重是**实测**出来的，不是猜的：400 个缓存场景上，target 的场景均值占原始 L2 能量的
  **79.28%**（11.83% 全局 per-token DC + 67.46% 场景特有），占 L1 目标的 **59.8%**
  （预测全零 0.7921 → 预测场景均值 0.3184）；直接在渲染像素上量（`scripts/probe_future_background.py`，
  12 场景）是 **90.43% 背景 / 9.57% 轨迹**，且随时域单调：+0.5s 背景 97.35% → +4.0s 83.89%。
  所以纯 identity L1 的大部分梯度在教"输出这个场景的平均未来"，那不需要读轨迹——这正是
  未训练 register 的 `residual_relative_l1` 恰好等于 1.0、以及失败跑 cos 能读 0.65 而 gain 全负
  的原因。对比项监督同场景两条 GT 行的**差**，场景均值被精确消掉；其目标确实由轨迹驱动
  （`corr(||Δtraj||, ||Δreg||)` = +0.58 Pearson / +0.65 Spearman），且 K 行已在同一次 forward 里，
  不付额外前向。
  **去背景不能靠减常数或固定线性投影**，两条实测否证：按 `||Δtraj||` 分 5 个分位桶，桶内
  `Δreg` 方向平均 cos = **−0.002**（同一条轨迹差在不同场景里方向完全不同 → 映射是
  scene-conditional 的）；`Δreg` 近各向同性（pc1 仅 0.5%、前 64 主成分共 14.1%），不存在可隔离的
  低维轨迹子空间。全局线性探针 `GT轨迹(24维) → 场景内残差` 的 R² = **−0.03**（等于零），
  换成配对差分后 ridge → 前 8 主成分 R² = **+0.113**。
  **逐帧读数**：`mean_frame{1..8}_gain_over_scene_mean` 在 verdict 里逐时域给出，能看出学习
  发生在近端（背景主导）还是远端（轨迹主导）。
- **`trajectory_gates`：register 到 action 的单向帧对齐通路（#58 的结构性修复）。**
  实测（真实 DINOv3 scene 流、生产 geometry）register 行在自注意力里只给 8 个 action key
  **8-10%** 的质量（scene 81 个 key 拿 61-66%、register 自己 24-31%），而 register 的**输入**
  按构造与轨迹无关（`future_queries` + ego），所以轨迹只能经一条每跳 ~9% 的通道抵达未来读出，
  8 层复合成 `trajectory_conditioning_fidelity ≈ 5e-4` 的 583× 衰减。
  **过拟合探针证明瓶颈在通路而不在目标**：6 场景 80 步、lr 2e-4，开/关对比项的两次跑在所有
  共享读数上逐位相同，`residual_relative_l1` 都钉在 1.00000、fidelity 5e-5、`learned=False`
  ——连"精确监督跨 GT 差"的对比项都推不动它。
  修复是每个 block 一个 `TrajectoryGate`：register 查询、action 作 K/V 的 cross-attention，
  带**帧对齐掩码**（行 `q + Q*f` 只允许 action token `f`，把 RoPE 相位"暗示"的 1:1 对应做成
  结构）。没有掩码时 gate 的 softmax 在 8 个 action 行上近似均匀，每条 register 行读到的是
  **平均轨迹**，行间仍逐位相同（实测 resid_rel 依旧 1.0）——掩码是它生效的必要条件。
  `o` 用 `normal_(std=1e-2)` 而非零初始化：register 分支本就随机初始化，零初始化保护的
  "预训练状态"不存在，而零初始化实测把整个 warmup 浪费在等 `o` 长到 ~4e-3。
  实测效果：`d register/d action` 从无 gate 的 9.7e-2 升到 **3.14（32×）**；未训练时 register
  跨 GT 行差异从 ~1e-4 升到 **0.4318**。
  **不变量保持**：gate 只写 register 切片，scene/action token 原样穿过，所以
  `Action 永不读 Register` 在结构上仍成立（扰动 register 输入 100× → velocity 逐位不变、
  `d velocity/d register输入` 恰为全零，契约测试钉住）。gate 在 `BACKBONE_SKIP_PREFIXES`
  里（Wan payload 没有它），`predict_future=False` 时冻结以免 DDP reducer 报未用参数。
- **`_split` 必须 transpose，不能 view**：`q/k/v` 出来是 `[B, L, H*D]`，即
  (token, head, dim) 顺序，所以拆头要 `view(B, L, H, D).transpose(1, 2)`。
  写成 `view(B, H, L, D)` 会把连续缓冲区**重新解释**，交给 SDPA 的每个"query 位置"
  在任何 mask 生效**之前**就已经混进了别的 token——mask 形同虚设、`rope_apply` 刚写好的
  相位贴到错位置上，而所有 shape 检查照样通过。`num_heads == 1` 时两种写法结果相同，
  所以这个 bug 只在多头下出现（本包 `num_heads = 4`）。
  回归测试 `test_attention_keeps_the_token_axis_intact_across_the_head_split` 用
  单位 mask 断言"query `i` 的输出逐位独立于其它行的输入"，
  `test_the_per_stage_masks_block_the_leak_in_both_directions_of_autograd` 再对
  两个阶段各做一遍前向扰动 + 反向梯度，共 4 条箭头。**改注意力实现前先跑这两个。**
- **`future_fuse` 与 EMA 教师已整体删除，不要加回来。** 它曾经就是一条显式的
  `Register → Action` 残差：θ 的一份冻结 EMA 副本先在 stage2 表下跑一遍产出
  `teacher_register`，再 cross-attend 进 Action token。那正是"Action 不能看见未来"这条
  硬不变量要关掉的箭头，用一个旁路把它重新打开，mask 就形同虚设。删除后未来对动作的影响
  **只剩一条路**：`loss_future` 的梯度塑造 Action 流与 register 段**共享的那 8 个 Wan
  block**，前向上未来永不抵达 action token。
  连带删除的还有 `_ema_theta` / `enable_ema()` / `update_ema()` / `restore_ema()` 与
  checkpoint 里的 `ema_theta` 顶层条目。三处守卫钉住它们不会悄悄回来：
  `config.py` 的严格 loader **拒绝** yaml 里出现 `future_fuse` / `future_fuse_stages` /
  `future_fuse_out_proj_init` / `ema_theta` / `attention_stage_of` /
  `STAGE{1,2}_SEGMENT_VISIBILITY` 这些退役键；契约测试用
  `RETIRED_BYPASS_PREFIXES = ("future_fuse", "ema_theta", "_ema_theta")` 断言模型上
  没有这些参数、也没有 `enable_ema` / `update_ema` 方法；
  `scripts/probe_stage2_eval_path.py` 对 checkpoint payload 做同样的断言。
  **scorer 也不消费 `predicted_register`**：`scorer.py` 里是 `del predicted_register`，
  审计字段 `predicted_register_input: False`，且 `config.py` 用
  `predicted_register_is_input is False` 强制这一点（"strict DrivoR score path cannot
  consume predicted future registers"）。所以未来既到不了 action token，也到不了 score。
  旧记录里 `future_fuse` 的 `kdim/vdim`、两个 LayerNorm、`out_proj` 零初始化、
  `658,944 参数 / 10 tensor`、`EMA_DECAY=0.996` 等全部作废。
- **Stage 2 的 `loss_future` 与 Stage 1 不可直接比数**，原因见上面 mask 那条：wiring 相同、
  工作点不同（σ 与 timestep 经 adaLN 调制整个 trunk）。看 Stage 2 的 `loss_future` 曲线时
  不要把它当成 Stage 1 的延续，回升也不代表 Stage 1 白训了。
- `scene_tokens = 5 帧 × 16 + 1 ego = 81`，scorer 的默认值同为 81。ego 是 Scene 的
  **最后一个** slot，scorer 用 `scene_context[:, -1:]` 原样取出（不过 LayerNorm）相加。
- Scorer 的 detach 语义**逐字照抄 DrivoR**：只有 proposals 被 detach，scene K/V **不**
  detach，所以 `loss_score` 会回传塑造 η。对应配置
  `input_detach: [trajectory, predicted_register]`、`scene_kv_detached: false`、
  `updates_scene_lora: true`。**不要"修回去"。**
  注意 `input_detach` 里列了 `predicted_register` 只是沿用 DrivoR 的字段形状，
  **scorer 并不真的吃它**（见上面 `del predicted_register` 那条）；
  `predicted_register_is_input` 必须是 `false`。
- 分辨率：Student 收 DrivoR 官方的 `672×1148`（W,H；`1148 = 14×82`，是 DrivoR 为它的
  **patch14 DINOv2** 选的），而本包的 DINOv3 是 **patch16**，`1148 % 16 = 12`，
  所以宽度必须闭合到 16 的倍数。**做法是 bicubic 重采样到 `672×1152`**
  （`right_resample_px = (-1148) % 16 = 4`，`F.interpolate(mode="bicubic",
  align_corners=False)`）→ patch16 grid `42×72`、`image_patches = 3024`。
  **旧版的"4 列零填充（归一化之后施加）"已作废**：零列会伪造 patch 内容——一个横跨
  真实像素与补零的 patch 编码的是"图像边缘有一块黑"，而那不是场景里存在的东西。
  严格 loader 因此**拒绝**零填充配置（`right_zero_pad_px != 0` 或
  `student_resampling` 不是 `bicubic_to_patch_multiple` 都会 `ValueError`，
  负向测试 `test_strict_config_rejects_zero_padding_instead_of_resampling` 钉住）。
  `_prepare()` 仍然**拒绝任何自己做的重采样**：输入必须正好是 `672×1148`，
  连已经填好/插好的 `1152` 也拒（那意味着 loader 自己插值过，重采样必须在模型内部
  这一处发生）。**不要用 timm 的 `dynamic_img_pad=True` 绕过 assert**——它会跳过检查但
  `patch_embed.grid_size` 仍停在 71，forward 吐 72 列直接 `IndexError`。
  构造 backbone 时必须声明 `img_size=(672,1152)`。
- 位置编码逐字照抄 DrivoR：DINOv3 的 `pos_embed is None`，所以 `_pos_embed` 直接返回原张量
  ——**纯 RoPE，无绝对位置编码，零插值**。`rope_prefix_tokens = 5 + 16 = 21`。
  `_widen_rope_prefix()` **只在 `__init__` 调一次**：在 forward 里临时改
  `EvaAttention.num_prefix_tokens` 再还原会让 `torch.utils.checkpoint` 反向重算时 RoPE
  静默错位（注意 timm 的 `EvaAttention` 持有自己独立的同名属性，改 Eva 模块上那个没用）。
- Teacher 在未来帧的**原生 `464×832`** 上编码，零 resize 零插值，收到别的尺寸直接
  `ValueError`。Teacher grid `29×52 = 1508 patch` 比 Student 的 `42×72` **粗**，这是刻意的且合法：
  DINOv3 的 RoPE 按轴归一化（`normalize_coords='separate'`），token 的旋转只取决于它在
  画面里的**分数位置**，与被切成多少块无关。空间上**不再做任何切分**：读出是
  `FutureQueryExtractor`——query 就是每帧自己的 CLS token，logits 是 8 头 cosine、
  `QUERY_TEMPERATURE = 0.005`，value 不归一化。所以旧版象限边界那套整除算术
  （`29 → 14/15`、`52 → 26/26`）连同它那一整类 bug 一起消失了：没有分区就没有边界。
  `query_future_register` **不做任何时间池化**（早期版本把每两帧拼成一个 8D 向量，等于把
  +0.5s 和 +1.0s 折进同一个线性子空间）；契约测试用一个**手写参考实现**逐 (帧, 头) 显式切片
  对照，钉死 `view(B,L,H,Dh).transpose(1,2)` 这个分头约定，并验证扰动第 3 帧只改动第 3 帧的
  register 行。
- **loader↔Teacher↔缓存这个接缝上出过三个"单元各自都对、拼起来就错"的 bug，都是只有真跑真实
  数据才露出来的**，三个都曾在契约测试全绿的状态下存在：
  1. `query_future_register` 收到的是 loader 的 **7 维** `[1,K,8,V,3,H,W]`
     （`torch.stack(future_videos).unsqueeze(0).unsqueeze(3)`），守卫却写 `ndim != 6`，
     而报错信息里印的正是那个 7 维布局。三个真实调用方——Stage 0、Stage 1、可分性诊断，
     全部经过 `load_multiview_sample`——于是**在第一个 scene 就抛异常**：这个函数在真实数据上
     从未成功执行过一次。84 个测试全绿，因为被测的是 extractor 和 loader 各自，连接两者的那次
     调用没人测。现在的守卫是 `ndim != 7 or shape[2] != FUTURE_FRAMES`，seam 测试用一个记录
     形状的假 Teacher 钉死"K 循环每次只剥第 1 维、交给 `_check_shape` 的是 `[1,8,V,3,H,W]`"
     ——把守卫改回 `!= 6` 它就红。
  2. `load_multiview_sample` 必须在 Teacher 构造**之前**就分配 `future_register` 占位张量，
     所以宽度只能来自默认值。参数原名 `target_dim=256`，那是 `d_model`；query 提取之后
     register 宽度是 Teacher 的 `feature_dim=1024`。结果 `with_future_register` 拒收真实
     target，报 `must be (1,4,8,256), got (1,4,8,1024)`——看起来像 Teacher 算错了，其实是
     占位张量分错了宽度，而且这一刻整段 future 视频**已经解码并编码完了**。参数现在叫
     `register_target_dim`，默认取 `config.EXPECTED_REGISTER_TARGET_DIM`（就是
     `load_v6_config` 校验 yaml `future_target.register_target_dim` 的那个常量，三处宽度因此
     不能各自漂移），宽度不符时报错会点名这个参数、帧数不符时则不会（那是另一种错，不该用
     Teacher 宽度去解释）。契约测试在 `tmp_path` 里搭一个结构真实的假 scene 真跑一遍 loader，
     同时钉死默认宽度与显式覆盖：只查默认值的话，分配处硬编码 1024 也能过，覆盖那条会抓住它。
  3. `build_scene_payload` 从 `metadata["history_paths"]` 取那五条已解析的 JPEG 路径，而
     loader 是在**构造 observation 时**解析它们的，只写进 `observation.camera_metadata`
     ——`metadata` 里根本没有这个键（其余标量都在，所以这一行看着完全合理）。于是 Stage 0
     在**每一个** scene 上抛 `KeyError: 'history_paths'`。`build_scene_payload` 当时零测试
     覆盖，和前两个 bug 同一个原因。现在从 `observation.camera_metadata` 取，缺失时报错点名
     `load_multiview_sample`（否则读起来像缓存条目损坏，而不是"这个 observation 不是 loader
     造的"）；契约测试走完整回路 loader → payload → `atomic_torch_save` →
     `read_scene_payload(weights_only=True)` → `targets_from_payload` → `load_history_uint8`，
     并断言 `"history_paths" not in sample.metadata`，把那条结构性事实本身钉住，免得 fixture
     哪天两处都带上而让 revert 变绿。两次 revert 实验都验过：改回 `metadata[...]` 得到生产里
     那个一模一样的 `KeyError`，删掉守卫得到 `TypeError`。
- **静默全灭比崩溃更贵：`except Exception` 的容错粒度要和退出码分开设计。** Stage 0 逐 scene
  捕获异常是**必需**的——103075 个 scene 里坏一个不该毁掉整个构建——但它把上面第 3 个 bug 变成
  了一次"成功"运行：8/8 全失败，脚本照样打印 `"event": "done"`、在**不存在的** `targets/` 旁边
  写下 manifest、exit 0。区分点不是"有没有失败"，而是"**有没有产出**"：系统性故障必然是全灭，
  个别数据问题只会打掉几个。所以 `shard_produced_nothing(result, cached_scene_count)` 判两件事
  ——本 shard 有 scene 却既没新建也没复用一个 payload，或者目标目录里一个 target 都没有
  （第二条只有 rank 0 能判，因为只有它扫目标目录；`world` 大于 scene 数时空 shard 不算失败）。
  判定放在 **barrier 之后**：在 `build_shard` 里 raise 会把其它 rank 卡在集合通信里，把一个
  清楚的失败变成挂死。`failed_scene_count` 现在进 manifest，让部分失败活得比日志行久。
  这个策略函数自己有测试（含 rank 0 建了但没落盘、以及 resume 全部 reuse 两种边界），因为
  **一条从未执行过的失败分支不算被覆盖**。
- Teacher 编码必须**流式归约**：`[8,29,52,1024]` float32 是 50 MB，Student 尺寸下 103 MB，
  而 `query_future_register` 的产出 `[gt,8,1024]` 只有 128 KB。产出 register 后立即丢弃
  grid；目标构建完就 `replace(sample, future_rgb=None)`。`TimmDinoV3Teacher` 另有
  `frame_chunk=2`，因为 attention 对 patch 数是平方复杂度。
- Wan backbone：30 block → 8 层，`layer_map = [0,3,7,11,15,18,22,26]`，226 个 key
  `strict=True`。meta 必须逐项匹配
  `{hidden_dim:256, ffn_dim:1024, num_layers:8, num_heads:4, attn_head_dim:64,
  text_dim:256, freq_dim:256, eps:1e-6}`。Wan 的约定 `num_heads × attn_head_dim == hidden_dim`
  把 **heads 钉死在 4**（`d_model=256` 时）；构造 `NvsActionDiT` 时 `text_dim` 与 `ego_dim`
  都必须显式等于 `hidden_dim`。skip 前缀 `action_encoder.` / `head.` / `mode_projection.` /
  `ego_to_action_projection.` / `time_queries` / `future_queries` / `future_head.` /
  `scene_type_embedding` / `action_type_embedding` / `register_type_embedding`
  保持随机初始化——Wan 的 payload 里没有未来分支也没有分段 type embedding，load 必须原样放过它们。
- **单次 forward 出轨迹，训练和推理都没有 rollout**（`action_sampling_loops: 0`）。
  训练每次 forward 随机采一个 σ；推理把 σ 钉在 1（`INFERENCE_SIGMA`），此时
  `add_noise` 原样返回 anchor、`recover` 减掉整条 velocity，所以
  `JointStage2Module.inference()` 与训练路径是**同一个代数表达式**。任何引入多步采样的
  改动都会让评测偏离训练路径。
- 优化器参数组必须**完整、唯一、互斥**，`build_optimizers` 会断言。`parameter_groups`
  只返回 `eta` / `theta` 两组（`scene_blocks.` 归 eta，其余 `theta.` 归 theta），
  跳过 `requires_grad=False` 的参数；出现第三个前缀直接 `ValueError`。
  `future_fuse*`（三个模块，按前缀匹配）**归 theta 组**：它们存在的意义就是让未来条件
  抵达 action token，而 Stage 1 那个 `theta.head.*` 冻结**不该**顺带冻掉整个 θ。
  但它们自己在 **Stage 1 是被冻结的**（见下），而 `parameter_groups` 会跳过
  `requires_grad=False` 的参数，所以 Stage 1 的 theta 组里**没有** `future_fuse`——
  这不是漏配，是 mask 的必然结果。两组在两个 stage 都非空。

## 模型结构与数据契约速查

- 三个参数组：`η`(eta) = DINOv3 LoRA + scene queries + neck + hist_encoding + scene_blocks；
  `θ`(theta) = Wan2.2 ActionDiT（含 `future_queries` 与 `future_head`，未来分支就在 θ 里面）；
  `ψ`(psi) = `ScratchDrivoRScorer`（官方 `Linear(24,1024)-ReLU-Linear(1024,256)` + 四层
  `TransformerDecoderScorer` + 六头 `Scorer`，全部随机初始化，**永不**加载 DrivoR
  checkpoint）。独立的 `φ` 分支已删除。
- Stage 1 训 `η + θ 的主干`（scene_blocks、Wan blocks、action_encoder、time_queries、
  mode_projection、ego_to_action_projection、future_queries、future_head、三个 type
  embedding），**冻结 `theta.head.*`、`future_fuse*`、EMA 副本和 ψ**：
  Stage 1 的 loss 根本不读 velocity head，而它那套近零、保 anchor 的初始化必须原样
  活到 Stage 2；`future_fuse` 则**本身就是 Stage 1 mask 要关掉的那条 `Register → Action`
  箭头**，而且 `_flow_pass` 在 stage1 下压根不调它——冻住是为了让参数诚实地无梯度，
  而不是"可训练但没人用"（`find_unused_parameters=False` 时 DDP reducer 会拒收后者）。
  `predict_future=False` 时同样冻住，因为没有 register 段就没有教师。
  Stage 1 **没有任何 action 目标**，只训未来表征。输入是**不加噪**的 clean 轨迹
  （同一条 flow 的 σ=0 端），mode code 必须是**全零**而不是 per-row 码——
  per-row 常量码会让网络光看行号就答出"我是第几条 GT"，根本不去读 action token。
  Stage 2 三组全开，输入是加噪 action，σ 随机。两个 stage 都在
  `optimizer.step()` 之后调 `model.update_ema()`（**每个 optimizer step 一次，
  不是每个 micro-batch 一次**）。
- 每个 scene 最多 4 条 GT（`MAX_GT = 4`）。action loss 用 Hungarian 在 `[K, 64]` 上的
  **一份** assignment；future loss **不共享**它，而是 `dense_future_l1`——每条 candidate
  配自己最近的那条**有效** GT，64 行 register 全都有梯度。mask 和 quality 必须用
  **同一个 index** gather，否则 candidate 会继承别的 GT 的有效性。
- Future Register 目标是 `[B, K, 8, D]`，`D = register_target_dim = 1024`，即 **Teacher 自己的
  feature 宽度，不是 `d_model=256`**。那个 `8` 是**逐帧的未来 register**，一帧一个，
  和 8 个未来时间戳、8 个轨迹 pose 都 1:1 对齐；**不做任何时间池化**。
  8 个未来帧来自 41 帧 @10fps 的 `future.mp4`，索引 `range(5,41,5)` → +0.5…+4.0s，
  正好对上 DrivoR `num_poses = 8`。每帧 = 一次 **CLS-query cosine attention 读出**
  （τ=0.005，8 头，value 不归一化）→ 1024 维，再过 **train-fitted 对角标准化**
  （逐维 mean/std，相对最大 std 有 `TARGET_STD_FLOOR = 1e-2` 下限，被 clamp 的维数写进 audit）。
- **为什么不是象限 mean-pooling**：两因素 ANOVA 在真实 Teacher 特征上测得，GT 判别方差的
  **81.0%** 在 GT 特有的**空间 pattern** 里，只有 19.0% 在每 GT 的平均电平上（逐 scene
  75.1 / 87.1 / 80.1 / 82.3%）。mean-pooling 把那 81% 直接销毁；ordered-2×2 相对单个全局
  均值只多买到 ~3% 的相对可分性，却付 4 倍宽度。
- **为什么 τ=0.005、为什么 query 必须无参数**：可分性和轨迹相关必须一起读，只看前者会被骗。
  随机 query 对照的可分性 0.4957 甚至**高于** CLS 的 0.4934，但轨迹相关只有 0.4121
  （CLS 0.6647）——高比值可能是"保住了轨迹信号"，也可能是"把场景扔进噪声里"。
  τ 表（熵 / 可分性 / 轨迹相关）：0.10 → 熵 0.936，可分性**低于**被替换的池化，直接淘汰；
  0.02 → 0.436 / 0.3593 / 0.7184；0.01 → 0.320 / 0.4498 / 0.7032；
  **0.005 → 0.098 / 0.5091 / 0.6890（选定）**；0.003 → 0.055 / 0.5357 / 0.6816。
  降 τ 单调地拿可分性换轨迹相关，0.005 是选定的折中点而**不是**拐点——0.003 可分性更高，
  代价是轨迹相关继续掉。
  原始**点积** softmax 熵 0.9987、可分性 0.1414，比它要替换的 mean-pooling 还差——
  DINOv3 特征强各向异性，共享均值方向主导每个 query/patch 内积。
  `FutureQueryExtractor` **零参数**：query 就是每帧自己的 CLS token。可学习的 extractor 会在
  Student 追目标的同时把目标挪走；随机初始化的可学习 query 实测注意力熵 **0.999997**，
  就是穿了 attention 外衣的 mean pool。`MAX_QUERY_ATTENTION_ENTROPY = 0.5` 是它的门禁，
  依据是实测间隙（选择性读出 ≤ 0.436，退化读出 ≥ 0.936），`--allow-degenerate-query` 放行。
- **为什么是对角标准化而不是白化 PCA**：PCA 的秩是 `min(target_dim, fit_rows)`，秩之外的
  维度**逐位为零**。4-scene 拟合（秩 128/256）下"直接吐零"的 L1 是 **0.335**，而真训 20 步
  是 0.353——塌缩解比训练结果更好。对角 mean/std 没有子空间，因此没有秩，
  `compressor_full_rank` 门禁和 `--allow-rank-deficient-target` 开关随之作废。
  `fingerprint()` = `sha256(feature_dim|temperature|heads|mean|scale)`，所以温度和头数不同的
  两次提取不可混用；每个 payload 带 `target_fingerprint`，trainer 逐 scene 校验
  （宽度检查抓不到这种"两半在不同目标空间里建出来"的缓存，形状是一样的）。
  `load_standardizer` 见到没有 `feature_dim` 的 payload 直接报错要求重建。
  **刻意不做 per-token LayerNorm**：对角标准化已经把各维尺度拉平，再 LN 会连 token 的
  整体幅值一起丢掉，而幅值正是目标要携带的量（这也是 value 侧不归一化的同一个理由）。
  注意目标侧的 stop-grad 是**结构性**的
  （离线缓存 + 冻结 Teacher，永远不可微），EMA 只出现在**条件侧**
  （`future_fuse` 的教师是 θ 的 EMA 副本），所以 `future_relative_l1` 的分母是
  一个固定的实数量级，不会因为教师漂移而失真。
- Score 监督是在线 NAVSIM oracle 标签 `[B, 64, 7]`（NC, DAC, EP, TTC, C, DDC, Final），
  scorer 目标是 `[B, 64, 6]`。**永远不是** `64×K` 的 candidate-by-GT 矩阵。Multi-GT 不改动
  DrivoR 的 scorer 网络或 loss。
- 数据根目录 `/mnt/data/multi_future_view/navtrain_full_allcosmos_cuda_v1_gpu64/...`
  （103075 scene）；`/mnt/data/multi_view/...` 是指向同一目录的**符号链接**。
  历史帧文件名是**相机 token 不是帧 token**，必须经
  `frames[i]["cams"]["CAM_F0"]["data_path"]` 解析；日志是 2 Hz。
  `trajectory_k` 的**目录序号不等于候选轨迹序号**（实测 193/200 不等），配对必须走
  metadata 里的显式字段。
- metric_cache 路径与 DrivoR `train_metric_cache` 完全兼容：
  `data_runtime/train_metric_cache/<log>/unknown/<token>/metric_cache.pkl`。
  `/mnt/zhaozc_workspace/official_DrivoR_clean/data_runtime/train_metric_cache` 是指向
  FastWAM 离线 cache 的符号链接，**被 DrivoR 在线评测拒绝**。

## 方案A — Joint 阶段对齐 DrivoR 感知容量（2026-10-13）

DrivoR 能安全使用 `final_score_weight=1.0`（score loss 直接回传感知）是因为它只训练 LoRA，
base backbone 全程冻结，感知容量小。我们之前的 Joint 额外训练了 `scene_blocks`
（8 层 1007-token self-attn，6.3M 参数），感知容量远大于 DrivoR。当 source_loss 权重拉到
1.0 时，未 detach 的 score 梯度会把感知冲垮（历史上 sel_pdms 0.978→0.85、ade 0.25→0.56
的回归即源于此）。

方案A：Joint 阶段冻结 `scene_blocks`，把可训练感知容量退回 DrivoR 的 LoRA 量级，于是
source_loss 可以安全地按 DrivoR 原值 1.0 训练；rank loss 按 DrivoR 原值 0.1 独立加权。

改动（3 处，都在 e2e 路径）：

1. `scripts/run_navtrain_wan_e2e_ddp.py` → `build_optimizers()`：在未来分支冻结之后、
   scorer 解冻之前冻结 `scene_blocks.`：

   ```python
   for name, parameter in model.named_parameters():
       if name.startswith("scene_blocks."):
           parameter.requires_grad_(False)
   ```

   `parameter_groups` 会跳过 `requires_grad=False` 的参数，因此「generator 参数集合 ==
   eta+theta 可训练集合」的断言仍然成立。

2. 同文件训练日志的 `trainable` 字典新增 `"scene_blocks_frozen"`（实测 6318080），
   setup 日志中应 >0，用于确认冻结生效。

3. `scripts/dlc_navtrain_wan_e2e.sh` 的 Stage 2 段：`--lambda-score 1.0`（对齐 DrivoR
   `final_score_weight`）、`--lambda-rank 0.1`（对齐 `preference_rank_weight`）、
   `--lr 2e-4 --scorer-lr 5e-4`。rank loss 在损失块中已「去嵌套」：
   `loss = λ_action·loss_action + λ_velocity·loss_velocity + λ_score·source_loss
   + λ_rank·rank_loss + λ_future·loss_future`，rank 有效系数即 0.1
   （不再被 lambda_score 二次缩放成 0.02）。

回退方法：删掉 `build_optimizers()` 里那个冻结循环即恢复 `scene_blocks` 可训
（`scene_blocks_frozen` 日志字段可保留）；把 launcher 的 `--lambda-score` 改回 `0.2`
即恢复温和 score 权重。两者独立，完全回到方案A之前需同时执行这两步。

## 方案B — Scene 进自注意力 + EMA 教师反向条件化 + per-stage mask（2026-10-15）

> **本节及下面「训练入口与流水线」「评测」「验证」「已知限制」里的架构描述已被第三轮指令
> 部分推翻，读到冲突处以「契约要点」为准。** 具体作废清单：
> ① **两张 per-stage mask 表 → 一张共用表**，`attention_stage` 参数已删除；
> ② **`future_fuse` + EMA 教师 → 已整体删除**（它就是 `Register → Action` 的显式旁路）；
> ③ **`Register(8)` / 97 token → `Register(32)` / 121 token**（8 帧 × 每帧 4 个 query）；
> ④ **4 列零填充 → bicubic 重采样到 `672×1152`**；
> ⑤ 所有 tensor 数 / 参数量 / 测试项数（`768+108=876`、`84 项`、`93 passed` 之外的）
>    与 `dit_sequence_tokens=97` 这类实测记录都属于旧 geometry，**不要拿来对账**。
> 下面保留它们是当作决策史料（为什么 token 顺序是 scene/action/register、为什么 L1 是 loss
> 而 cosine 只当读数、为什么 Stage 1 只训未来表征），不是当前实现的规格。

这一节经历了三轮用户指令，**后一轮推翻前一轮关于 mask 的部分**，其余要求仍然有效：

- 第一轮：「**不要设置mask啊**，你这里也可以用EMA的机制进行计算，L1loss是一定的。……
  我的核心目标是让模型学会怎么预测未来，**并且把未来条件辅助生成动作**。……
  Action DiT 里的输入 token 顺序是 **scene, action, register** 这个顺序。stage1 就只是
  训个未来表征。」
- 第二轮：「停，你把 mask 那部分改回去，参照 *Future Register Attention Mask
  Design Review*……这部分得严格按照给你的提示做啊，**不能信息泄漏**。」
- 第三轮（**现行**）：「**stage1和stage2都用相同的mask机制就行了。一定不要让action看见
  future。**」——于是两张表合成一张 `((1,0,0),(1,1,0),(1,1,1))`，Action 在**任何** stage
  都读不到 Register；而为了让"未来条件辅助生成动作"在不打开 `Register → Action` 的前提下
  仍然成立，`future_fuse` 与 EMA 教师被删除，未来只通过**共享 trunk 的梯度**影响动作。

第一、二轮当时据此做了五件事（其中第 2、5 项已被第三轮撤销）：

1. **序列从 `[Action(8) | Register(8)]` = 16 token 改成 `[Scene(81) | Action(8) |
   Register(8)]` = 97 token**（FastWAM 布局），RoPE 分段从 0 重启，加三个 type embedding。
   Scene 仍然**同时**是 cross-attention 的 K/V——这不是冗余：Wan 预训练的
   `cross_attn.*` / `norm_q` / `norm_k` 是在"Scene 当上下文"这个分布上训出来的，
   把它从 cross-attention 里撤掉会让 226-key 的 `strict=True` 加载失去意义。
2. **加了 EMA 教师 + `future_fuse`**，让未来条件真的反向辅助动作生成。
   注意 FastWAM 的 EMA 是套在**目标编码器**上的，而我们的目标是离线冻结缓存，
   没有可 EMA 的东西——所以 EMA 落在 θ 上，用在**条件侧**，这才是方差平滑真正起作用
   的地方（早期一个很烂的 register 不该去动摇保 anchor 的 `Head`）。
3. **L1 保持为 loss**，cosine 只当读数报出来（详见「评测」一节）。
4. **撤回上一轮我提的"删掉 8 个未来 token、改成 action token 上挂第二个头"的方案。**
   「把未来条件辅助生成动作」要求未来必须能抵达 action 流，删掉 token 就做不到；
   现在的实现是 register 既做 DiT 的输出（被 L1 监督），又经 EMA 教师 + `future_fuse`
   做 action 的输入。**这两个角色不冲突，别只保留一个。**
5. **按 design review 装上两张 mask，并据此把 `future_fuse` 从"写 Scene"改成"写 Action"、
   且只在 Stage 2 跑。** 第二轮的指令还暴露出第一轮实现里一处真实违规：当时
   `future_fuse` 在 Stage 1 也跑，等于把 Stage 1 存在的唯一理由——关掉
   `Register → Action`——自己打开了。

**与 FastWAM 的方向差别（这是理解两张表的钥匙）**：FastWAM 的目标是"未来预测增强表征"，
所以它禁止 `Future → Action`；**本项目的目标恰恰是 `Future → Action`**，因为 Future
Register 是一个 planning condition（future-conditioned planning）。禁止的是
**`R^GT` 直接进 Action**（推理时没有真实未来），以及 **Stage 1 里的 `Register → Action`**
（那时 register 是被监督的目标，不是条件）。Stage 2 必须用 `R^pred`，两遍式：
先 `(Scene, Mode) → R_pred`，再 `(Scene, Action, R_pred) → Action`。

同时被这次改动**作废**的旧说法：「Scene 是结构性只读的，ActionDiT 拿不到写它的路径」
——这句在**第一轮**（无 mask）下确实作废过，但**第二轮把两张表的 Scene 行都设成只读自己**，
所以它现在**又成立了**：ActionDiT 不写 Scene。交给 scorer 的那份
`PairOutputs.scene_context` 因此逐位等于 η 精修后的输出，融合只发生在 Action 段上。

## 训练入口与流水线

`scripts/dlc_navtrain_wan_e2e.sh` 是**唯一**的流水线 launcher，按序跑：

- **Stage 0** `build_future_target_cache.py`（torchrun）→ `standardizer.pt` + per-scene
  targets。`standardizer.pt` 已存在时自动跳过，`NVS_FORCE_STAGE0=1` 强制重跑。
  一次冻结 Teacher 过全量数据，之后训练不再解码 mp4。
  **Stage 0 没有 `--target-dim`**：register 宽度就是 Teacher 的 `feature_dim`，由 checkpoint
  决定，不是命令行参数（传了会 argparse 失败）。可调的是 `--query-temperature` /
  `--query-heads`，两者都进目标指纹。
- **Stage 1** 有两个 trainer，共用**同一份数字**：
  `run_stage1_cache_ddp.py`（生产，torchrun ×16，读 Stage 0 缓存）与
  `run_stage1_multiview.py`（冒烟/诊断，单进程，在线用 Teacher 提目标、在自己的
  `--scenes` 上拟合 standardizer）。度量、基线、判定规则全在
  `src/nvs_v6_wan_e2e/stage1_eval.py`，两个脚本都 import 它，谁都不许留私有副本
  （`test_both_stage1_trainers_share_one_copy_of_the_verdict_maths` 用 `is` 断言身份）。
  两者都输出 `stage1_checkpoint.pt`，payload 形状对 Stage 2 兼容（DDP 版是超集，
  多带 `opt_step`/`optimizer`/`ema_theta`/`config`）；汇总 JSON 分别是
  `stage1_cache_ddp_result.json` 与 `stage1_multiview_result.json`。
  生产版**读缓存**，所以不再每次重启重付 163 GPU-hours 的 Teacher 提取；
  冒烟版仍然自己拟合，因此它的指纹与缓存的不同——同一总体统计量、不同 scene 数，
  差的是估计精度，不是目标空间。它的 `--target-dim` 是 **`d_model`**，不决定 register 宽度。
- **Stage 1 DDP 的两个承重点**（都不是样板）：
  ① 训练前向必须走 `Stage1Module.forward`（它就是 `forward_register_only`），
  因为直接调内层 module 的方法会绕过 `DDP.forward`，而 reducer 的
  `prepare_for_backward` 正是在那里跑 `_find_tensors(output)`——绕过的后果是梯度
  永不 all-reduce，16 个 rank 各训各的模型，而 16 条 loss 曲线看起来全都健康。
  ② `setup_distributed` 用 `enabled = has_env and world > 0`（**不是 `> 1`**），
  所以 `--nproc_per_node=1` 也会真的包 DDP、真的建 reducer——这是在单卡上验证
  「每个 trainable Parameter 都拿到梯度」这条契约的唯一办法，裸 `python` 会跳过包装
  而空洞地通过。
- **Stage 1 的 train/eval 划分是 builder 的**，从 `_ready_scenes.json` sidecar 加
  manifest 自己的 `eval_scenes`/`seed` 经 `split_scene_tokens` 重新推出，
  **不能**用 CLI flag 重算：standardizer 是在 builder 的 train scene 上拟合的，
  本 run 称为 "eval" 的 scene 可能参与过它自己 target 被标准化所用的 mean/std。
  这条主张由 `test_the_ddp_trainer_reproduces_the_builders_train_eval_split`
  对着持久化的 `fit_scene_ids` 断言，不只是写在 docstring 里。
  过滤用的是 standardizer 自己的指纹，不是 manifest 里那份副本
  （manifest 在 shard 结束后才写，中间被杀就会留下一份不再描述旁边 payload 的指纹）。
- **Stage 2** `run_navtrain_wan_e2e_ddp.py --stage1-checkpoint "$STAGE1_CKPT"`。
  `standardizer.pt` 存在则 `--future-target cache --lambda-future 0.1`，否则
  `--future-target none` 并打印显著警告（此时 `lambda_future` 强制为 0、
  `future_queries`/`future_head` 冻结）。
  register 宽度由**三个来源求一致**：`--register-target-dim`（默认 0 = 不指定）、
  缓存 standardizer 的 `feature_dim`、Stage-1 payload 里的 `register_target_dim`；
  出现分歧直接 `ValueError` 而不是挑一个。宽度错了本来只会在很久以后以
  `strict=True` 的一句 "size mismatch" 出现，看不出是目标空间的问题。
  解析结果会写回 `args.register_target_dim`，因为 checkpoint 存的是 `vars(args)`，
  评测脚本正是从这里读回宽度的。

  **未来 token 进不进 DiT 序列，和它训不训是两个独立开关。**
  `predict_future = train_future or bool(stage1_checkpoint)`：恢复过 Stage 1 就必须
  保留 97 token 序列（Stage 1 就是按这个长度训出来的，砍掉 8 个 register 等于拿模型
  没见过的序列去评测），此时未来分支和 `future_fuse` 靠**冻结**而不是**移除**来避开
  DDP 的 unused-parameter 检查——`requires_grad=False` 的参数 reducer 根本不管。
  只有「既无 cache 又无 Stage 1 checkpoint」时才退化成 `[Scene | Action]` = 89 token，
  这时融合根本不跑，`future_fuse*` 同样必须冻结。
  checkpoint 里显式存了 `predict_future`，`load_joint_checkpoint` 读它，
  两个评测脚本因此会自动复现训练时的序列长度。
  **`load_joint_checkpoint` 还必须 `restore_ema(ckpt.get("ema_theta"))`**，否则评测
  路径没有教师、融合被整个跳过，评测复现的就不是训练时那个 forward。

跳过某阶段用 `NVS_SKIP_STAGE0/1/2=1`；追加参数用 `NVS_STAGE{0,1,2}_EXTRA_ARGS`。
preflight 会：跳过 Stage 1 但 `STAGE1_CKPT` 不存在时**中止**；Wan payload 缺失时
**非致命**警告（两阶段都会退化成随机 θ 初始化）；按 195 MB/scene 核算 Stage 1 的
RAM 需求，超过物理内存一半就 `exit 2`；
**multiview root 下至少要有一个 ready scene**，否则 `exit 2`。

最后这条是实跑撞出来的，而且**原来的默认值本身就是坏的**：
`NVS_MULTIVIEW_ROOT` 默认指向
`.../rollouts/single_gpu_test_gtfallback_v2`，那里 4 个 scene 目录**全都没有
`completion.json`**，而 `multiview_data.scene_is_ready` 恰恰以它为准入条件，
于是 Stage 0 与 Stage 1 在 source 完整个环境之后才死于
`FileNotFoundError: no ready Multi-GT scenes`。默认值已改成 README 数据表里那份
103075-scene 的全量语料 `/mnt/data/multi_future_view/navtrain_full_allcosmos_cuda_v1_gpu64/...`
（实测前 200 个 scene 目录 200 个都 ready）。
「目录存在」和「目录里有可用 scene」是两件事，`[ ! -e "$p" ]` 那轮循环查不出后者。
检查用 `find -mindepth 2 -maxdepth 2 -name completion.json -print -quit`：
命中即停，全量语料上 **0.09 s**；写成 shell glob（`for scene in "$ROOT"/*/`）
会在 `break` 生效前把 103075 个条目全展开，实测 **38.85 s**。
它只查 `completion.json`，是 `scene_is_ready` 的**廉价近似而不是替代**——
抓得住「root 指错」和「rollout 还没跑完」，抓不住「有 completion.json 但 mp4 缺失」。

Stage 2 的模型构造 / checkpoint 恢复 / 评测推理**只有一处实现**：
`run_navtrain_wan_e2e_ddp.py` 里的 `build_generator_and_scorer()`、
`load_joint_checkpoint()` 和 `JointStage2Module.inference()`。两个评测脚本
（`eval_navtrain_wan_e2e_pdms_bev.py`、`eval_navtest_official_pdms_e2e.py`）都 import 它们，
**不要在评测脚本里重新拼一遍模型构造**——它们曾经各自写死 `heads=8`、
`dino_lora_rank=16`、以及不存在的 `dino_lora_alpha` / `use_all_native_tokens` 参数，
加上一个过时的 3 参 `net(...)` 调用，于是**静默坏掉**（构造期就 TypeError，但没人跑过）。
`load_joint_checkpoint` 的 `--heads` / `--target-dim` / `--depth` 默认 0 表示「读 checkpoint
里的 config」，只有显式传非零值才覆盖。`register_target_dim` 同样默认 0，但它的回退是
**`target_dim`**——那是拆分前的布局（future head 输出的就是 `d_model`），只对老 checkpoint
生效；拆分后存的 checkpoint 自带这个键。评测脚本**没有**对应的 `--register-target-dim`
flag，这是刻意的：给了就等于允许把一个 checkpoint 按错误宽度重建 future head。

`stages.py` **不含** Stage 2：它需要梯度累积、LR schedule 和中间 flow 张量
（`velocity`、`predicted_register`、`logits`）用于日志，塞不进一个自包含的
`zero_grad/backward/step` helper；再放一份只会和生产路径漂移。

## 评测

- 两个 ADE 口径必须**同时**报：**matched-GT ADE**（只覆盖 Hungarian 赢家）和
  **all-64 nearest-GT ADE**（覆盖所有被监督的 candidate）。只报前者正是「个别收到 GT
  监督的很高、总体质量偏烂」被掩盖的原因。两者共用 `nearest_gt_ade()` 这一个实现，
  所以 batch 汇总和 per-scene 记录不可能各说一套。
- `eval_navtest_official_pdms_e2e.py --shard-count N` 分片后必须用
  `merge_navtest_shards.py` 合并；它会顺带证明分片正确（各分片 scene 集合两两不相交、
  并集等于 `scenes_indexed_total`，且带 `--reference` 时每条 per-scene 记录与不分片跑
  **逐字相同**）。合并后由同一份代码重算 summary，避免 sharded 与 baseline 用不同归约。
- `official_navtest_oracle.py` 走官方 navsim 评测路径：官方 `MetricCache` +
  官方 `PDMScorer` + `pdm_score` 聚合。
- **Stage 1 未来表征的评价口径**（`stage1_eval.py` 的 `future_fit_metrics`，
  两个 Stage-1 trainer 共用）：
  **训练 loss 永远是 L1**（Stage 1 `identity_future_l1` / Stage 2 `dense_future_l1`），
  这是用户钉死的（「L1loss是一定的」）——FastWAM 拿 `1 − cosine` 当 loss，我们**不抄**，
  只把 cosine 当**读数**。报出来的四个直接读数：
  `paired_future_l1`（训练目标本身，对角标准化后 1024 维空间的加权 L1）、
  `future_cosine_similarity`（逐有效帧的余弦，正是 FastWAM 训的那个量的补）、
  `future_relative_l1 = paired_l1 / weighted|target|`（**无量纲**：0.0 = 完美，
  1.0 = 「和直接预测全零一样错」，>1.0 = 比预测零更差）、`target_abs_mean`（分母本身，
  用来判断 relative 是不是被一个接近零的分母放大了）。
  两个模型无关的对照是 `scene_mean_target_l1`（整个 scene 的 GT 平均目标，
  「不看轨迹、只背场景先验」能拿到多少）和 `trajectory_blind_l1`（把 action token
  换成同 scene 里另一条 GT 的 action，测「未来到底有没有真的由轨迹决定」）。
  **cosine 与 relative_l1 只报不设门槛**：Stage 1 从未在真实 Teacher 缓存上跑过，
  没有任何有校准意义的阈值，写死一个只会变成自我安慰。
  `CONDITIONING_SUMMARY_KEYS` 是 before/after 汇总的**唯一**真相来源，
  测试钉住它 `<=` `future_fit_metrics` 实际发出的键集（少一个键汇总就静默丢一项度量）。
  LR 调度同理只有一份：`schedule_factor` 在 `stages.py`，Stage 1 与 Stage 2 都 import，
  所以命令行上的 `--lr` 在两个阶段意味着同一条曲线。

## 验证

```bash
export PYTHONPATH=/mnt/zhaozc_workspace/project/NVS/src:/mnt/zhaozc_workspace/project/NVS/vendor/DrivoR:/mnt/zhaozc_workspace/project/NVS/vendor/DrivoR/nuplan-devkit
bash scripts/run_contract_tests.sh   # 93 passed（本机：vendor/SimWAM、Wan payload、DINOv3 权重与 CUDA 齐备）
```

环境相关的用例一律 **skip 而不是 fail**（SimWAM 1 个、Wan payload 3 个、
真实 DINOv3 checkpoint 2 个、需要 CUDA 的 checkpointed backward 1 个，
外加 `requires_drivor` 标记的那批），所以缺其中任何一项的新克隆上 passed 数会更低。
不要把 `pytest.importorskip` / `pytest.skip` 换回硬依赖。

`vendor/` 被 gitignore，只有 `vendor/DrivoR`（1398 个文件，含 `nuplan-devkit`）是
force-add 进仓库的；`SimWAM`、`FastWAM_...`、`Drive-JEPA`、`DrivoR_official` 只在本地。
SimWAM **不是运行时依赖**（包内从不 import 它），只被
`test_wan_scheduler_matches_simwam_source` 用来把我们的 flow 移植和上游 scheduler 源码
逐行对齐。

> **下面这段真数据 Stage 1 记录属于已被替换的独立-φ 架构，不是当前架构的证据。**
> 当前架构（未来 query 在 DiT 序列里、8 个逐帧 register 目标、`{eta, theta}` 两组参数）
> **还没有跑过真数据**。列在这里只为保留"曾经在真数据上端到端跑通过"这一事实本身。
>
> **而且它跑的时候 `_split` 还是错的**（`view(B,H,L,D)` 而非
> `view(B,L,H,D).transpose(1,2)`），注意力在任何 mask 生效前就已经把 token 混起来了。
> 所以下面每一个数字——包括 `loss_future` 1.1769 → 0.1602 这条"看起来在学"的曲线——
> 都**不能**当作当前实现的证据，也不能当作"修好 `_split` 后应该复现"的目标值。
> 同理，`diagnose_target_separability.py` 那条 `gain_over_shuffled` 只有 0.4% 的记录
> 也早于两次方向反转和这次修复，**需要重跑**。

真数据 Stage 1（4 个非退化 scene、60 步、candidates=64、depth=8，910s）已验证：
`loss_future` 1.1769 → 0.1602 单调下降且全部有限；`paired_better_than_shuffled: true`
（0.16291 < 0.16593）；`mean_register_std_across_candidates` 0.975 → 0.411（>0，确实在预测
彼此不同的 register）；`parameter_groups_updated == {eta: true, theta: false, phi: true}`；
`scorer_audit.logits_finite: true`、`scene_context_shape == [1,81,256]`；Wan backbone
`loaded: true` / `loaded_keys: 226`；4 个 scene 的 `history_policy` 全为
`real_CAM_F0_sensor_blob`，`matched_pairs` 全为 4。结果在
`artifacts/stage1_multiview/stage1_multiview_result.json`。
（`parameter_groups` 现在只返回 `{eta, theta}`，Stage 1 两组都会动；那三个字段名里的
`phi` 已经不存在了。）

方案A 单卡冒烟（`--batch-size 1 --limit-scenes 4 --max-steps 2 --epochs 1`）通过：
`scene_blocks_frozen=6318080`、`scene_kv_detached=false`、两步 loss 全部有限、
无 `build_optimizers` 断言失败、无 CPU/CUDA 设备不匹配、exit 0。**冒烟的 sel_pdms/ade
仅为合成小样本，不得据此宣称 NAVSIM 指标。**

Stage1→Stage2 链路复测（`[scene|action|register]` + EMA 教师 + `future_fuse` 之后，
真实 DINOv3-large + 真实 226-key Wan payload，生产 geometry `target_dim(d_model)=256 /
register_target_dim=1024 / heads=4 / depth=8`）：generator **768** tensor、scorer **108**
tensor，合计 **876**；Stage 1 payload 用 `strict=True` 装进新建的 Stage 2 模型后
**key 集合完全一致、权重逐位相同**，`load_joint_checkpoint` 解析出
`register_target_dim=1024 / target_dim=256`；`predict_future=False` 的同一份 payload 也能
`strict=True` 装上（register 分支两种情况下都分配了参数，只是不进序列），DiT 序列长度相应从
**97 token** 退回 **89 token**，两种情况的 tensor 总数都是 876。
Stage 1 可训练参数 **eta 9,812,480 / theta 9,433,344**（θ 总参 9,434,627，差额正是 Stage 1
冻结的 `head.*` 1,283）。θ 自身是 **242 tensor = 226 个 Wan 加载 + 16 个 NVS-only**
（`time_queries`、`future_queries`、三个 type embedding、`action_encoder.*`、`mode_projection.*`、
`ego_to_action_projection.*`、`head.*`、`future_head.1.*`，合计 **336,899 参数**——其中
`future_head` 一项就占 **263,168**，因为它输出的是 Teacher 的 1024 宽而不是 256），
所以 Stage 1 的产物绝大部分仍落在被两个 stream 共享的 8 个 Wan block 里，而不是这些头上。
`future_fuse*` 是 **model 级**的 10 个 tensor / **658,944 参数**（`q/k/v_proj_weight`、
`in_proj_bias`、`out_proj.*`、两个 LayerNorm）；Stage 1 冻结它，所以它**不在** Stage 1 的
theta 组里（`parameter_groups` 跳过 `requires_grad=False`）。
`ema_theta` 是 payload 里**独立的 242-tensor 条目**，实测 Stage 1 payload 的 `"model"`
条目里 EMA tensor 数为 **0**（即 EMA 没有污染 `strict=True` 的 key 集合）。
> 旧记录里的「755 + 108 = 863 key」「16→8 token 退化」「未来分支只有 3 个 key /
> 67,840 参数」「853 vs 853 key」是 **Scene 仅作 cross-attention 上下文、无 EMA** 时代的数字；
> 「764 + 108 = 872」「NVS-only 139,523」「theta 9,500,931」则是 **register 宽度还等于
> d_model** 时代的数字，同样过时。其中 9,500,931 反推 = 9,235,968 + 263,680（旧
> `future_fuse`）+ 1,283，说明那次测量**还没**在 Stage 1 冻结 velocity head 与 fusion，
> 别拿它跟现在的口径直接相减。
> 这些 tensor 数与参数量都**不受** mask / `_split` 改动影响（两者都不改形状），
> 但**会**随 register 宽度变：动 `register_target_dim` 就必须重测这一段。

评测路径复测：`scripts/probe_stage2_eval_path.py`，**27 项全过**，跑约 2.3 分钟（`_split`+svd 修好后复测 138s，exit 0），
真实 DINOv3-large Student + 真实 226-key Wan payload + 生产 geometry，覆盖契约套件够不着的
`run_navtrain_wan_e2e_ddp.py`（合成 observation、scratch init，**不含任何 NAVSIM 数据**）：
`ImmutableModeBuffer` 的 64 模态 anchor 展布仍然是 **38.4000 m/rad**（与旧记录逐位一致，
说明模态格没被动过）；错误 geometry 被 Wan meta 检查拦住（`target_dim=128` 撞
`time_embedding` 输入宽度、`heads=8` 撞 `meta.num_heads`，都抛 `ValueError`）；
用 forward pre-hook 实测生产序列是 **`self=97 cross=81`**，cross-attention 的
`ctx_mask` / block 的 `context_mask` **仍为 None**（整个 Scene 都是上下文，不屏蔽），
但**自注意力现在必须带 mask**：三个入口各自被逐个 forward 记录，
`forward_pairs`（4 次）、`forward_flow`（2 次）都拿到 **stage2** 表，
`forward_register_only`（2 次）拿到 **stage1** 表，且两张表逐位等于
`NvsActionDiT.STAGE{1,2}_SEGMENT_VISIBILITY` 展开的结果、彼此不相等；
`net.inference()` 与 `forward_flow(data=anchor, σ=1)`
**逐位相同**、跨调用**确定性**；σ=1 时 noisy action 就是 anchor（**无 GT 泄漏**，
`max|diff|=2.86e-06`）且 `v_target` **恰为 0**，而换一个真 GT 时 `max|v_target|=2.071`
——即 `v_target` 与 σ 无关，σ=1 不会把监督抹掉；scratch head 下 `max|traj-anchor|`
**0.0117 m/rad**。
**这个数字以前是不可复现的，现在才是**：probe 原来**没有给建模播种**，而 PyTorch 默认
generator 是非确定性播种的（实测 `torch.initial_seed()` = 8630428001369399702 这类值），
所以 scratch head 每个进程都不一样——连开三次新进程实测 `future_head` 权重和分别是
**+3.48 / −7.01 / −7.28**，`max|traj-anchor|` 随之在 0.0109 / 0.0159 / 0.0163 之间跳。
旧记录里把这些跳变解释成「加了 `k/v_proj_weight` 和第二个 LayerNorm 导致 RNG 流整体位移」
**是错的因果**：那些改动确实会位移 RNG 流，但真正的成因是**根本没播种**，所以同一个 commit
连跑两次也会给出不同的数。现在 `main()` 在 `probe_geometry()` 之后、建模之前
`torch.manual_seed(0)`（必须放在建模紧前面，因为 `probe_geometry` 自己会构造两个模型、
先吃掉一段流），连跑两次都是 **0.0117**、27/27 全过，这项检查从此是回归信号而不是噪声。
`_split` 修好之前的 **0.036** 仍然是有效的旁证——拆头错误会把 token 混起来、等效放大
注意力输出，那次跳变量级远超播种噪声——但**不要**再把 0.0109↔0.0159↔0.0163 之间
的差别当成任何行为变化来解读。
量级仍然是 38.4 m 展布的 0.03%，符合 `Head.HEAD_INIT_STD = 1e-3 / 33.37` 的近零锚点保持
初始化，**不是**训练后的偏移量；
`forward_register_only` 返回 `(1, 64, 8, 1024)`，即每条 candidate 每个未来帧一个
**Teacher 宽度**的 register token——这项现在拿生产常量 `EXPECTED_REGISTER_TARGET_DIM` 对照，
而不是拿模型自己碰巧持有的宽度，所以 probe 退回拆分前 geometry 时会**失败**而不是打印
`(1,64,8,256)` 算过；另有一项钉死融合的 `in_proj_weight is None` 且
`k_proj_weight = (256, 1024)` / `out_proj = (256, 256)`，即 K/V 读 register 宽度、
输出写 action 宽度；
`save_checkpoint` 把 `predict_future` 从模型自身记进 payload、`config` 是
`weights_only=True` 能吃的普通 dict（9 个 key，多出来的那个就是 `register_target_dim`）、
**`ema_theta` 是独立条目**（实测 `model=768 / ema_theta=242 / scorer=108`）；`load_joint_checkpoint` 重载后
`inference` **逐位复现**、EMA 教师**逐位复现**并解析出 `dit_sequence_tokens=97`；
把教师故意改成与学生不同后重载仍逐位相等，所以这条**真的在证明教师被 round-trip 了**；
删掉 `predict_future` 键、只在 `config` 里留 `stage1_checkpoint` 的**旧格式** checkpoint
也能回退到 `predict_future=True`；删掉 `ema_theta` 的**pre-EMA payload** 仍然装上教师
（快照当前 θ）并报 `ema_teacher_restored_from_payload=False`。
契约套件（**84 项**）另外覆盖：无 GT 的 scene 不监督任何 candidate、`dense_future_l1`
按最近有效 GT 配对、`strict` config loader 拒绝方向性改动（含两张 mask 表被逐字钉死、
`attention_stage_of` 映射、`future_fuse_stages` 只能是 `[stage2]`）、97-token 序列与**分段 RoPE**
（action 段与 register 段逐位相同且都等于 `rope_for(8)`）、三个 type embedding 打破
重叠相位的平局、**两张 mask 表逐箭头对照 design review**、
**`_split` 的 token 轴完整性**（单位 mask 下 query `i` 的输出逐位独立于其它行的输入）、
**四条箭头的泄漏行为**（两个阶段各做前向扰动 + 反向梯度，被关掉的那条前向逐位不变、
反向梯度恰为全零，且同一次调用里未被关闭的方向确实有信号，避免"全零"是空断言）、
**Scene 行在两个阶段都逐位不受 Action/Register 扰动影响**、
EMA 对 `state_dict`/`parameters`/DDP 不可见但跟随 `.double()`/`.float()`、
`future_fuse` 零初始化时 step 0 与"没开 EMA"逐位相同、scorer 的 scene K/V 保持未融合
且教师拿不到梯度、**Stage 1 冻结 `future_fuse*` 与 velocity head 而 Stage 2 训练它们**、
EMA 教师过 checkpoint round-trip、**Teacher 目标读出**（CLS-query 逐帧、cosine logits、
输出是 patch 的凸组合，用**手写参考实现**逐 (帧, 头) 显式切片对照以钉死
`view(B,L,H,Dh).transpose(1,2)`；单 patch 时读出**逐位**等于该 patch 的 3 倍，证明 value
侧未归一化；归一化熵随 τ 单调、生产 τ 下低于 `MAX_QUERY_ATTENTION_ENTROPY`、τ 放大 50 倍后
高于它，即"退化成 mean pool"这件事**可被检出**；扰动第 3 帧只改第 3 帧的 register 行）、
**对角标准化**（32 行 / 64 维拟合后 `not (out == 0).all(dim=0).any()`——这正是 PCA 会失败的
地方：秩之外的维度逐位为零；非 train split / 维度不符 / 未 fit 就 transform / 指纹不匹配
全部报错；`standardizer.pt` 保存加载往返 + 指纹校验 + 拒绝没有 `feature_dim` 的 pre-query
payload + 拒绝被改过 `scale` 的篡改缓存）。
> 上面两个"泄漏行为"测试做过**反向对照**：把 `_split` 改回错误的 `view(B,H,L,D)`，
> 两项立刻 FAIL（`stage1: Register -> Action` 报 `assert True is False`），改回正确写法后
> 84 项全过。**改注意力实现前先跑这两个测试。**
> 旧记录里"压缩器那四项"的 `svd` 误拆包反向对照已随代码一起删除——它覆盖的
> `Ordered2x2Compressor` 不存在了。它当时的教训仍然成立且被新测试继承：**一个从未被
> 真实数据跑过的 fit 路径，可以带着精确匹配的错值活很久**，所以新读出的每一项断言都对照
> 一个独立写出来的参考实现，而不是对照被测代码自己的属性。

## 已知限制（不要粉饰）

> **口径警告（第 2 条起）**：下面这些真数据 Stage 1 冒烟记录是在**已退役的**
> ordered-2×2 象限池化 + 256 维白化 PCA 目标下量的。它们的 `paired_future_l1` /
> `constant_target_l1` / `scene_mean_target_l1` / `register_abs_mean` 等**绝对数值与现在的
> 1024 维对角标准化目标不可比**，不要拿来做基线，也不要拿现在的数字去和它们相减。
> 保留下来的是**结论**而不是数值：60 步只买到"全局常量"这一档、光记住每 scene 的未来均值
> （0.5839）就比训出来的 register（0.7515）更好、`gain_over_trajectory_blind` ≈ 0——
> 这三条正是"象限 mean-pooling 提取不到信息、必须换成 query 读出"的实测依据，
> 也是这次改动的动机本身。第 1 条（缓存从未建过）已经按新目标重算。

- **`/mnt/zhaozc_workspace` 下没有任何 `standardizer.pt`。** Stage 0
  （`build_future_target_cache.py`，缓存的唯一生产者）从未跑过：每 scene 一次冻结
  Teacher，`[K,8,1024]` float32 = 128 KB（K=4），加约 1 KB 的轨迹/掩码/质量字段；
  103075 scene 语料约 13 GB（旧的 256 维目标是 3.4 GB，宽度翻 4 倍）。
  `run_stage1_multiview.py` 有自己的**内联**目标构建
  （`_build_future_targets`），所以真数据 Stage 1 冒烟**不需要**缓存也能跑——
  但那条路径拟合的 standardizer 不落盘，`run_navtrain_wan_e2e_ddp.py` 的
  `--future-target cache` 仍然无缓存可用。
- **真数据 Stage 1 冒烟（4 scene / 20 步，`_split` 与 svd 都修好之后）跑通了，
  但它证明不了"学到了未来"，而且在这个配置下原理上就证明不了。**
  audit：`compressor_rank: 128` / `compressor_output_dim: 256` /
  `compressor_full_rank: false` / `compressor_clamped_components: 25`，
  `attention_stage: stage1`、`segment_visibility: [[1,0,0],[1,1,0],[1,1,1]]`、
  `future_fuse_frozen_in_stage1: true`、`dit_sequence_tokens: 97`、
  `parameter_groups_updated: {eta: true, theta: true}`、`all_losses_finite: true`。
  `loss_future` 0.6724 → 0.3534。**但 4 个 scene 只有 128 行拟合数据，
  rank=128 < 256，目标向量的后 128 维结构性恒为 0**，于是"全零预测器"的 L1 就等于
  `mean_target_abs_mean = 0.335`——**比 20 步实跑的 0.3534 还好**；
  `mean_constant_target_l1 = 0.327`、`mean_scene_mean_target_l1 = 0.265` 也都更好。
  同时 `register_std_across_candidates` 6.6e-05 → 3.0e-06、
  `register_response_to_trajectory` 7.8e-05 → 3.6e-06，即 register 正在塌成
  **与轨迹无关的常量**；脚本自己的判定是
  `train_beats_trajectory_blind_baseline: false`、
  `train_learned_trajectory_conditional_future: false`。
  **别把这轮的 loss 下降当成学习证据**：它大半来自"往零收缩"这个平凡方向。
  任何有意义的 Stage 1 判定都必须先让 `compressor_full_rank: true`。
- **满秩重跑（12 scene / 60 步）也仍然是负结果，但这次的负结果是可信的、
  而且给出了明确的里程碑。** 命令
  `--scenes 12 --eval-scenes 2 --skip-degenerate --steps 60 --check-scorer`，
  926 s（约 15.4 s/步），lr 2e-4，batch 4，`skipped: []`（这批 14 个 scene 都是
  4 条非退化 GT）。
  audit：`compressor_rank: 256` / `fit_rows: 384`（=12×4×8）/ `full_rank: true` /
  `explained_variance_ratio: 0.9987` / `clamped_components: 27`；
  `attention_stage: stage1`、`segment_visibility: [[1,0,0],[1,1,0],[1,1,1]]`、
  `future_fuse_frozen_in_stage1: true`、`future_fuse_in_theta_group: false`、
  `{eta: 9812480, theta: 9235968}`；`target_spread_across_gt` 从 4-scene 的
  0.4350 升到 **0.9524**——**目标本身现在真能区分不同未来了**，降秩这个借口没了。
  `loss_future` 0.8906 → 0.7882（均值 0.8311 → 0.7632），`loss_decreased: true`、
  `all_losses_finite: true`、`parameter_groups_updated: {eta: true, theta: true}`。
  **判定五项全 false**：`train/eval_beats_trajectory_blind_baseline`、
  `train/eval_learned_trajectory_conditional_future`、
  `generalizes_beyond_fitted_scenes`。
  读法（这是关键，别只看 loss 降了）：train `paired_future_l1` 0.9000 → **0.7515**，
  而同一批数据上 `constant_target_l1 = 0.7492`、`scene_mean_target_l1 = 0.5839`。
  即 **60 步只买到了"全局常量"这一档**（`gain_over_constant_target = -0.0031`，
  还差一点没追平常量预测器），而**光记住每个 scene 的未来均值就能到 0.5839**，
  模型离它还差 0.168。旁证：`register_abs_mean` 0.4703 → **0.0532**（8.8× 收缩，
  白化目标本来就零均值，所以"输出趋零"就是全局常量预测器）、
  `distance_to_mean_target_l1` 0.4856 → **0.1312**、`collapse_toward_mean` 0.175、
  `register_std_across_candidates` 7.19e-05 → **4.45e-06**（16× 更小）、
  `trajectory_conditioning_fidelity` 8.16e-05 → **5.06e-06**、
  `gain_over_shuffled_pair` **2.6e-08**、`gain_over_trajectory_blind` **-1.3e-08**
  （换成别的 GT 的 action token，预测**几乎逐位不变**）。
  `future_relative_l1` train 1.1945 → **0.9938**、eval 1.0124 → **0.9992**：
  它刚好落在**自己那条 1.0 的校准线上**，而 1.0 的定义就是"和全零预测一样错"，
  所以这个指标在这里读出来的正是"只学到了全局常量"——校准线起作用了。
  `future_cosine_similarity` train -0.0013 → +0.0423、eval 0.0451 → 0.0349
  （train 微升 eval 微降，量级上等于没学到方向）。
  **这不是管线 bug**：两组参数都真的更新了，只是量极小——eta 范数
  109.4075 → 109.4845、theta 196.4850 → 196.4949、`future_query_update_l2` 0.1035；
  且末段仍在单调下降（step 54→60 约 7e-05/步），说明还停在"往全局常量收缩"阶段。
  **60 步不足以区分"会学会"和"学不会"**，但足以确立两件事实：L1 最先买到的是
  全局常量，以及在这个阶段 register 的轨迹依赖性是**下降**的。
  **下一次真跑要盯的第一个里程碑就是 `paired_future_l1` 跌破 0.5839**
  （scene 均值这条 model-independent 基线）；跌破之前谈"未来表征"都为时过早。
  eval 侧 `target_abs_mean = 2.7504` vs train `0.7561`（3.6×），因为白化基只在
  train 上拟合；所以 eval 的原始 L1 2.7482 **不可读**，只有 `relative_l1` 0.9992
  是可比的——这正是当初把 `relative_l1` 加进来的理由，实测成立。
  `--check-scorer` 的形状与来源审计全对：`scene_context [1,81,256]`（=
  `scene_tokens: 81`）、`trajectory [1,64,8,3]` 且 `trajectory_units` 标注
  metres/radians DrivoR-native 无重缩放、`score_logits [1,64,6]` 六个 PDMS 头、
  `logits_finite: true`、`selected_candidate: 62` / `selected_utility: 0.544`；
  `feature_audit` 为 `initialization: scratch_random` / `checkpoint_loaded: False` /
  `native_dinov2_scene_cache: False` / `native_ego_cache: False` /
  `predicted_register_input: False`——两条 comparison-only 的禁用路径确实是关的。
  未训练生成器的 `proposals_vs_multigt`：`min_ade_over_gt_m` **6.096**
  （= 64 条提案各自对最优 GT 的 ADE 再平均，是**覆盖度**）、`best_pair_ade_m`
  **1.093**（64×K 里的单点 **oracle**）、`best_pair_fde_m` 1.820、
  `mean_speed_mps` 7.103。**注意这两个键都与 scorer 的选择无关**
  （定义在 `stage1_eval.trajectory_errors`），
  别把 6.096 读成"scorer 选错了"。
- **verdict 的头号布尔值原来叫 `beats_trajectory_blind_baseline`，名字与它实际门的
  量矛盾，已改名为 `beats_scene_mean_baseline`。** 它门的一直是
  `gain_over_scene_mean_target >= margin`（0.02），而 `trajectory_blind_l1` 是**另一个、
  更弱的**控制项，有自己的 `gain_over_trajectory_blind`。旧名字让人把布尔值和
  `-1.3e-08` 那个数配对读，得出"和轨迹盲基本打平"的结论；实际触发它的是
  `-0.2871`。契约测试里 `collapsed` 那一例正是反证：fidelity=0.05 的 register 按
  标度无关读数**就是轨迹盲的**，旧键名却对它报 `True`。门本身没改（scene 均值是
  更紧的基线，`flat` 那一例必须继续判 false），只改了名字；`trajectory_blind` 的门
  仍在 `learned_trajectory_conditional_future` 的 `all(...)` 里面。**上面两次实跑记录
  的是旧键名**，因为跑在改名之前。除脚本、契约测试与文档外没有别的消费者
  （已全仓 grep 确认），所以没有 launcher 需要同步。
- `checkpoints/`、`artifacts/`、`logs/`、`reports/` 都在 `.gitignore` 里，所以 36 MB 的
  Wan payload 和所有跑批记录**不会**进仓库；换机器必须先用
  `preprocess_wan_action_dit.py` 从 `/mnt/LLM_weight/Wan-AI/Wan2.2-TI2V-5B` 重建。
- **（已退役）`Ordered2x2Compressor` 的降秩门禁。** 代码没了，但两条教训是**结构性**的，
  新目标必须继续满足：
  1. **目标空间里任何"结构性恒为常数"的子空间都会奖励塌缩。** 旧 PCA 的秩是
     `min(target_dim, 拟合行数)`，秩之外的投影列是**精确的 0**、`scale` 是 1.0，所以 4 个
     scene（rank 128/256）时**一半目标逐位为零**，"输出全零"直接拿到 `L1 = mean|target|`
     = 0.335，比真训 20 步的 0.3534 还好——L1 稳定下降而模型什么都没学。
     对角 mean/std **没有子空间因此没有秩**，`compressor_full_rank` 门禁和
     `--allow-rank-deficient-target` 开关随之作废；契约测试用
     `not (out == 0).all(dim=0).any()`（32 行 / 64 维拟合）把这一条钉死。
     `TARGET_STD_FLOOR = 1e-2 ×` 最大 std 接管了旧 `scale=1.0` 的角色：近常数的维度被
     地板挡住而不是被放大成噪声，被 clamp 的维数写进 audit。
     **注意"拟合 scene 数"现在的意义变了**：它只决定 mean/std 的**估计精度**，
     不再决定目标有没有秩，所以"满秩只需 8 个 scene"这类门槛已经不存在。
  2. **审计行必须在任何会退出的门禁之前落地。** 旧 Stage 0 是 `save_compressor` 先落盘、
     审计行后打印，等于一个降秩基在任何人看到数字之前就进了持久缓存，之后每个 stage 都
     继承一个**奖励塌缩**的目标。新脚本保持修好之后的顺序：
     `log(standardizer_fit)` → 熵门禁（`> MAX_QUERY_ATTENTION_ENTROPY` 且没有
     `--allow-degenerate-query` 就抛）→ `save_standardizer`。
     拟合只在 rank 0 执行，但 torchrun 在任一 worker 非零退出时会拆掉整个作业组，
     且同位置本来就有一个 `RuntimeError`（没有 scene 可拟合）走同一条路，
     所以门禁没有引入新的挂死风险。
     现在被门禁的量是**query 注意力熵**而不是秩：它抓的是"读出退化成 mean pool"，
     这是新目标唯一的那个静默失效模式（熵 ≈ 1 时所有下游数字仍然有限）。
- **两个 trainer 现在都审计参数组的实际切分**，因为"eta/theta 各多少"这种文档数字
  只有审计能证：Stage 1 记 `trainable_parameters_by_group` 与
  `future_fuse_in_theta_group`（实测 `{eta: 9812480, theta: 9235968}`，
  `future_fuse_in_theta_group: false`，两者相加**恰好**等于 `trainable_parameters:
  19048448`）；Stage 2 记同名字段并**断言方向相反**——
  `predict_future=True` 时 `future_fuse` 必须在优化器里（否则 Stage 2 打开的
  `Register → Action` 箭头根本没有可训参数），`predict_future=False` 时必须不在。
- 抽样 200 个 scene 实测 **19%（38/200）** 的 multiGT scene 是退化的（label 重复或全
  `human_log`，GT 间 pair L1 仅 0.005–0.044，非退化 scene 为 0.133–0.167）。用
  `--skip-degenerate` 过滤；它只读 `trajectory_k/metadata.json`，不会为要丢弃的 scene
  付 mp4 解码代价。同一批的对应关系核验：轨迹与日志 `pose_interpolation` 对齐的只有
  **241/800**（30%）。
- DrivoR 训练时开的 `GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1,
  prob=0.7)`（`dinov2_lora.py:341`，`use_grid_mask = True`）**本包尚未移植**。
- **Scene 进自注意力 + EMA 教师让 DiT 的开销涨了约 12×**：注意力/FFN 从
  `16² + 16×81` 变成 `97²`（约 6×），教师那次 forward 再翻一倍。相对
  672×1148×5 帧的 DINOv3-large 仍然很小（probe 墙钟 114s → 146s，其中新增了 6 项 mask
  检查和一次 `forward_register_only`，噪声量级），
  但**没有在真实 batch 上基准测过**，别把 12× 当成端到端的加速比倒数来用。
- **加了 per-stage mask 之后 SDPA 走不了 flash 后端**：任意 bool mask 会让
  `scaled_dot_product_attention` 回落到 memory-efficient 或 math 实现。这是一笔
  真实但**尚未基准测过**的额外开销，和上面那条叠在一起看。mask 每次 forward 只构造
  一次（`[97, 97]` bool，约 9 KB），相对 8 个 block 的注意力可以忽略。
- Stage 1 **冻结 velocity head（`theta.head.*`）、`future_fuse*`、EMA 副本和 ψ，
  但不冻结整个 θ**：未来的
  `future_queries` 就在 θ 里面，所以 Stage 1 训的是 DiT 主干本身。因此
  `scorer_audit.proposals_vs_multigt` 里的 ADE/FDE 反映的是**近乎随机初始化的轨迹头**
  叠加在米制 mode 锚点上的结果，不是训练后的轨迹精度。60 步真数据
  跑出的 `min_ade_over_gt_m: 6.10` / `best_pair_ade_m: 1.32` 只说明 Stage 1 没有动
  velocity head（符合契约），**且那 60 步属于已被替换的独立-φ 架构**。
  `min_ade_over_gt_m` 的口径是「64 个 proposal 各自对 4 个 GT 取最小 ADE，
  再对 proposal 求均值」，衡量 proposal 集合对 GT 集合的覆盖度。
- `dino_tokens.py`、`lora.py` 是死代码，删除需用户明确授权。
