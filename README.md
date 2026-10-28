# NVS — DrivoR scene perception + Wan2.2 Action DiT (joint, end-to-end)

One lineage only: `src/nvs_v6_wan_e2e`. The earlier `src/nvs_v6` and
`src/nvs_v6_wan` packages, their configs, scripts and tests were removed on
2026-10-14; recover them from commit `52a9d64` if you need the history.

The model generates 64 trajectory candidates per scene in a **single forward**
(no sampling loop), scores all of them with DrivoR's scorer, and picks one.

## Layout

| Path | What it is |
|---|---|
| `src/nvs_v6_wan_e2e/` | the package. See the module table below. |
| `config/config_v6_wan_e2e.yaml` | the strict contract loader's input (`config.py` rejects deviations). |
| `tests/contract_tests_v6_wan_e2e.py` | direction, leakage, matching, gradient, flow, scorer, resolution and shape checks. |
| `scripts/dlc_navtrain_wan_e2e.sh` | the two-stage pipeline launcher (Stage 0 → 1 → 2). |
| `scripts/run_stage1_cache_ddp.py` | Stage 1 (production): streams the Stage-0 cache, sharded over 16 ranks by `torch.distributed.run`. |
| `scripts/run_stage1_multiview.py` | Stage 1 (smoke/diagnostic): a handful of resident scenes, targets extracted inline from the Teacher. Single process. |
| `scripts/run_navtrain_wan_e2e_ddp.py` | Stage 2: full-navtrain joint generator + scorer training (DDP). Also the single source of truth for model construction, checkpoint restore and eval inference. |
| `scripts/build_future_target_cache.py` | Stage 0: one frozen-Teacher pass per scene → `standardizer.pt` + per-scene register targets. |
| `scripts/eval_navtrain_wan_e2e_pdms_bev.py` | navtrain PDMS + ego-BEV visualization. |
| `scripts/eval_navtest_official_pdms_e2e.py` | held-out navtest PDMS via the official navsim scorer. |
| `scripts/official_navtest_oracle.py` | official `MetricCache` + `PDMScorer` + `pdm_score` batch oracle. |
| `scripts/merge_navtest_shards.py` | recombine sharded navtest runs and prove the sharding was disjoint and lossless. |
| `scripts/preprocess_wan_action_dit.py` | build `checkpoints/ActionDiT_Wan22_256hdim_3outdim.pt` from a local Wan2.2-TI2V-5B checkout. |
| `scripts/verify_multiview_correspondence.py`, `scripts/diagnose_target_separability.py`, `scripts/_make_bev_contact_sheet.py` | data-integrity and diagnostic utilities. |
| `vendor/` | third-party source copies. Only `vendor/DrivoR` (1398 files, including `nuplan-devkit`) is force-added to git; `SimWAM`, `FastWAM_...`, `Drive-JEPA` and `DrivoR_official` stay local because `.gitignore` excludes `vendor/`. |

`artifacts/`, `checkpoints/`, `logs/` and `reports/` are gitignored, so no
weights and no run logs are in this repository. The Wan payload must be
regenerated locally with `scripts/preprocess_wan_action_dit.py`; both stages
degrade to random theta initialization (with a logged warning) when it is absent.

SimWAM is not a runtime dependency — the package never imports it. It is only
used by `test_wan_scheduler_matches_simwam_source`, which diffs our flow port
against the upstream scheduler instead of against hard-coded numbers, and which
**skips** (does not fail) on a clone without `vendor/SimWAM`.

### Package modules

`data.py` Observation/PairTargets boundaries · `vision.py` DrivoR-exact DINOv3
Student · `teacher.py` frozen DINOv3 Teacher + CLS-query attention readout + diagonal target standardizer ·
`multiview_data.py` `/mnt/data/multi_view` readers · `real_data.py` NAVSIM
`CAM_F0` history · `target_cache.py` Stage 0 cache I/O · `modes.py` the immutable
64-anchor grid + rollout · `flow.py` rectified-flow schedule · `wan_dit.py` /
`wan_init.py` Wan2.2 Action DiT and its weight transfer · `model.py`
`DirectedPairDiT` (η/θ) · `scorer.py` `ScratchDrivoRScorer` (ψ) ·
`selection.py` DrivoR's utility + argmax · `losses.py` · `oracle.py` online
NAVSIM labels · `stages.py` the Stage 1 step, the optimizer plumbing and the shared
LR schedule · `stage1_eval.py` the Stage-1 future metrics, baselines and verdict rule,
imported by *both* Stage-1 trainers so a number in one log means what it means in the
other ·
`config.py` the strict loader · `native_drivor.py` comparison-only, never on a
default training path.

`dino_tokens.py` and `lora.py` are dead code kept only because removing them
needs explicit authorization.

## Architecture

**Scene (η).** DrivoR's exact recipe on DINOv3. Frames arrive at DrivoR's
official `672×1148` (`drivoR.yaml:67 image_size: [1148, 672]`), produced by PIL
BICUBIC resize of the real 1920×1080 `CAM_F0` — never interpolation on our side.
1148 is not divisible by patch16, so `vision.pad_to_patch_multiple()` closes the
12 px remainder with **4 zero columns after normalization** (they land on the
dataset mean colour, not a black bar); the backbone is built at `img_size=(672,
1152)` → grid `42×72`. `_prepare()` refuses to resample and raises on any other
input size. 16 learnable scene queries are prepended per frame (official
`num_scene_tokens: 16`) and the first 16 backbone outputs are kept. Position
encoding is DrivoR's verbatim: DINOv3 has `pos_embed is None`, so
`_pos_embed` returns the tokens unchanged — **RoPE only, no absolute embedding,
no interpolation**. LoRA is rank 32 on q/v, `w_B` zero-initialized, base frozen.

Scene is **81 tokens** = 5 frames × 16 + exactly 1 ego token. The ego token is
11-wide: `pose3, velocity2, acceleration2` from the last history frame plus the
4-way `route_command`. There is no 12-wide pad in this package.

**Trajectories.** DrivoR-native metric `[x, y, heading]` in metres/radians,
8 steps at 0.5 s, `TRAJECTORY_STATE_DIM = 3`. All `/20`, `xy_scale`,
`encode_trajectory` and `decode_trajectory` are gone; nothing is rescaled.

**Action (θ).** SimWAM's Wan2.2 `ActionDiT` at `hidden_dim = 256`, loaded
`strict=True` (226 keys) from `checkpoints/ActionDiT_Wan22_256hdim_3outdim.pt`.
30 Wan blocks → 8 layers via `layer_map = [0,3,7,11,15,18,22,26]`; the
per-dimension 1D linear interpolation and `alpha = sqrt(d_src/d_tgt)` scaling are
SimWAM's own. Wan's convention `num_heads × attn_head_dim == hidden_dim` pins
**heads = 4** at `d_model = 256`. `action_encoder.`, `head.`, `mode_projection.`,
`ego_to_action_projection.`, `time_queries`, `future_queries` and `future_head.`
stay randomly initialized — Wan's payload has no future branch, so the load must
leave it at its own init. `future_queries` is `randn(8, 256) * 0.02` rather than
V-JEPA's exact zeros so the eight tokens are distinguishable in the residual
stream from step 0, not only through their RoPE phase.

**Rectified flow with an anchor prior.** SimWAM's
`WanContinuousFlowMatchScheduler` semantics — `x_σ = (1-σ)·data + σ·noise`,
`v_target = noise - data`, σ = timestep/1000, continuous timesteps, `shift = 5.0`
time-warping — with one deliberate substitution: the σ = 1 endpoint is **not**
Gaussian noise but the immutable 64-anchor grid, so

```
x_σ      = (1-σ)·norm_odo(gt) + σ·norm_odo(anchor)
v_target = norm_odo(anchor) - norm_odo(gt)
x0       = x_σ - σ · velocity_to_metric_scale(v̂)
```

`norm_odo` is SimWAM's min-max map to `[-1,1]` (`x: 2(x+1.57)/66.74 - 1`,
`y: 2(y+19.68)/42 - 1`, `heading: 2(h+1.67)/3.53 - 1`); the Jacobian
`(33.37, 21.0, 1.765)` converts the velocity loss back to metres. The loss is
weighted by SimWAM's bell `training_weight(t)`, which is exactly 0 at both ends
and peaks at t = 500.

Training samples **one random σ per forward**. Inference pins σ = 1, where the
anchor *is* the noisy action and the recovery subtracts the full velocity, so
`JointStage2Module.inference()` is algebraically the same expression as the
trained path. **`action_sampling_loops = 0`: no rollout at train or eval.** The
64 anchors are velocity `[-4,-2,-1,0,1,2,3,4]` m/s × yaw rate
`[-0.30,-0.18,-0.08,0,0.04,0.08,0.18,0.30]` rad/s (`ImmutableModeBuffer`,
8×8 `torch.cartesian_prod`), rolled out by `StructuredSourceInitializer.clean`
with a small-angle unicycle reading `ego_history[:, -1, 3]` as vx.

**One joint DiT sequence.** The ActionDiT self-attends
`[Scene S(81) | Action A(8) | Register R(8)]` — **97 tokens** — following
FastWAM's layout. The Scene is embedded
*once* by `text_embedding(S(81))` and then used *twice*: prepended into the
self-attention sequence **and** kept as the cross-attention K/V. The second use is
not redundant — Wan's pretrained `cross_attn.q/k/v/o` and `norm_q/norm_k` were
trained on "Scene as context", so dropping it would make the 226-key
`strict=True` backbone load meaningless.

**The self-attention is masked, and the mask is per stage.** Two 3×3 visibility
tables (rows = query, columns = key) live on `NvsActionDiT` and are expanded to a
boolean `[97, 97]` mask that broadcasts over batch and heads:

| Stage | Query \ Key | Scene | Action | Register |
|---|---|---|---|---|
| **1** | Scene | ✓ | ✗ | ✗ |
| | Action | ✓ | ✓ | ✗ |
| | Register | ✓ | ✓ | ✓ |
| **2** | Scene | ✓ | ✗ | ✗ |
| | Action | ✓ | ✓ | ✓ |
| | Register | ✓ | ✗ | ✓ |

Stage 1 learns `(Scene, Action) → Register`. The register **is** the prediction
target, so `Register → Action` must stay closed — otherwise the trajectory stream
is trained by the future it is supposed to predict. Stage 2 is
future-conditioned planning, so that arrow **reverses**: `Register → Action` opens
and `Action → Register` closes instead, which keeps `R_pred` an independent
`(Scene, Mode)` estimate rather than a description of the trajectory it is about
to condition. **The two stages cannot share a table.** Under both, Scene reads
only itself — Scene is the scorer's K/V, so the ActionDiT never writes to the
tokens DrivoR's scorer consumes.

Note the direction of the boolean: for `F.scaled_dot_product_attention` a `True`
in `attn_mask` means *allowed*, while for `nn.MultiheadAttention` it means
*forbidden*. Every table has a fully visible diagonal block, so no softmax row is
ever all-masked — the one way adding a mask silently produces NaN.

`attention_stage` is a **required** positional on both `NvsActionDiT.forward` and
`DirectedPairDiT._flow_pass`, with no default on purpose: a default would let one
stage silently run the other's information flow. The entry points map 1:1 —
`forward_register_only → "stage1"`, `forward_flow`/`forward_pairs → "stage2"` —
and both the contract suite and the probe assert the mapping by capturing the
mask on every forward.

RoPE follows FastWAM's **per-segment** convention: every segment restarts at 0
(Scene `0..80`, Action `0..7`, Register `0..7`). The Action segment must keep
Wan's own pretrained positions, since `head`, `time_queries` and all eight
transferred blocks were trained with the action there. Action and Register
deliberately *sharing* a phase is semantically right, not a shortcut: register
token `i` is the future frame at `(i+1) × 0.5 s` and action token `i` is the
waypoint at that same instant. Three type embeddings (`scene_type_embedding`,
`action_type_embedding`, `register_type_embedding`, each `randn(1,1,256) × 0.02`,
kept random at the Wan load) break the tie between the overlapping ranges.

`Head` reads `x[:, 81:89]` for the flow velocity and `future_head` reads
`x[:, 89:]` for the predicted register. Each candidate carries its own 97 tokens;
candidates never attend to each other. `predict_future=False` drops the register
segment entirely, giving `[Scene | Action]` = **89 tokens**, where both tables
truncate to the same top-left 2×2 — so the two-segment case needs no special
handling.

**Splitting the heads must transpose, not reshape.** `q`/`k`/`v` emit
`[B, L, H*D]` in (token, head, dim) order, so `_split` is
`view(B, L, H, D).transpose(1, 2)`. A plain `view(B, H, L, D)` reinterprets the
contiguous buffer and hands SDPA a strided mix of every token: each "query
position" then already contains other tokens' features *before* any mask applies,
which silently defeats `attn_mask` and attaches the RoPE phases to the wrong
positions — while every shape check still passes. With `num_heads == 1` the two
forms coincide, so the bug only appears with multiple heads (this package uses 4).
Two regression tests guard it: an identity-mask test asserting query `i`'s output
is bitwise independent of the other rows' inputs, and a leakage test that runs
both stages through a forward perturbation *and* a backward pass, checking all
four arrows.

The future queries are `randn(8,256) * 0.02` plus the projected clean ego — **no
trajectory projection and no mode code** — so a row's identity can only come from
the action tokens beside it in self-attention, never from a per-row constant.
`future_head` is `LayerNorm(elementwise_affine=False) → Linear(hidden_dim, register_target_dim)`,
i.e. `Linear(256, 1024)` in production: the head emits the **Teacher's** feature width, not
`d_model`. The standardized target itself is deliberately *not* normalized, because the
diagonal standardizer already equalizes per-dimension scale and discarding the token's
magnitude would throw away the one thing the readout is supposed to carry. Keeping the two
widths apart is load-bearing — Wan's payload requires `num_heads * attn_head_dim == hidden_dim`
(4 x 64 == 256), so widening `hidden_dim` to 1024 to match the target would break the weight
transfer.

**The future also conditions the action (`future_fuse`).** Predicting the future
is only half the goal — the future has to *help generate* the action. This is the
second pass of the design review's two-pass Stage 2 flow, implemented **outside**
the shared trunk so the mask tables stay the only thing governing information
flow inside it. A **frozen EMA copy of θ** (`EMA_DECAY = 0.996`, roughly a
250-step lag) runs first **under the Stage 2 mask** and produces
`teacher_register`; running it under that table is what makes the teacher's
register action-independent, i.e. exactly `(Scene, Mode) → R_pred`. `future_fuse`
(`nn.MultiheadAttention` + `future_fuse_norm`) then cross-attends it into the
**Action** tokens the **student** θ consumes. It writes to Action rather than
Scene because the Stage 2 table says Scene accepts no future modification — and
Scene is the scorer's K/V. That teacher/student asymmetry is what breaks the
cycle: the student's own register never contributes to the tensor that conditions
it. `future_fuse.out_proj`'s weight and bias are **zero-initialized**,
so step 0 is bitwise identical to running without the EMA — the same
anchor-preserving philosophy as `Head.HEAD_INIT_STD = 1e-3 / ODO_METRIC_JACOBIAN_MAX`
— while `out_proj.weight` itself still receives gradient immediately, so it is not
a dead start. FastWAM EMAs its *target* encoder; ours is a frozen offline cache
with nothing to EMA, so the EMA lives on θ and on the *conditioning* side, which
is where variance smoothing actually matters (a bad early register must not
destabilize the anchor-preserving head).

**`future_fuse` is skipped *and* frozen in Stage 1.** It is itself a
`Register → Action` path — the one arrow the Stage 1 table exists to close — so
running it there would violate the mask from the outside. Freezing it also removes
a trainable-but-gradless parameter, which is what DDP's reducer rejects under
`find_unused_parameters=False`. The student's register segment is deliberately
*not* seeded from `teacher_register` either: `future_head` could then learn a
trivial copy and `loss_future` would degenerate.

An honest consequence of the Stage 2 table: it closes `Action → Register`, so the
register's conditioning differs between the two stages and Stage 2's `loss_future`
does not measure the same function Stage 1 trained. That is the trade-off the
design review asks for (`R_pred` must stay independent of the trajectory it
conditions), but do not read Stage 2's `loss_future` curve as a continuation of
Stage 1's.

The EMA copy is attached with `object.__setattr__(self, "_ema_theta", ...)`,
outside the module dict, so it is invisible to `state_dict()`, `parameters()`,
`named_modules()` and DDP; `_apply` is overridden so `.to()` / `.double()` /
`.cuda()` propagate to it. **Every rank must call `enable_ema()` itself** — DDP
never broadcasts the copy. In a checkpoint `ema_theta` is its **own top-level
entry**, never merged into `"model"`, so `strict=True` and the 226-key check are
untouched; `restore_ema(None)` snapshots θ and returns `False` (a pre-EMA
payload). **The Scene handed to the scorer is bitwise the refined, unfused Scene**
— `PairOutputs.scene_context` — since the fusion now writes only to Action.

**L1 stays the loss.** `identity_future_l1` (Stage 1) / `dense_future_l1`
(Stage 2) remain the training objective. FastWAM trains `1 − cosine`; we report
cosine as a *reading*, not as a loss — see the Stage 1 metrics below.

**Scorer (ψ).** `ScratchDrivoRScorer` randomly initializes the official DrivoR
`Linear(24,1024)-ReLU-Linear(1024,256)` trajectory projection, the four-layer
`TransformerDecoderScorer` and the six-head `Scorer`. It never loads a DrivoR
checkpoint. All 64 candidates become trajectory queries attending to the Scene
tokens as K/V, then the last Scene token is added as the ego term, yielding
`[B, 64, 6]` logits for `NC, DAC, DDC, TTC, EP, comfort`.

Detach semantics are DrivoR's verbatim — **only proposals are detached**:

```python
embedded_traj = self.pos_embed(proposals.reshape(B, N, -1).detach())
tr_out = self.scorer_attention(embedded_traj, scene_features)   # scene_features NOT detached
tr_out = tr_out + ego_token
```

so `loss_score` does shape η. Selection is `aggregate_score_logits` → `argmax`,
returning `([B] pick, [B,64] utility)`.

## Supervision

Multi-GT, up to 4 GT trajectories per scene (`MAX_GT = 4`). The Stage 2 loss is
five independently weighted terms:

```
loss = λ_action·loss_action + λ_velocity·loss_velocity
     + λ_score·source_loss  + λ_rank·rank_loss + λ_future·loss_future
```

| Term | Weight | Note |
|---|---|---|
| `loss_action` | 1.0 | DrivoR's trajectory objective generalized to Multi-GT. Composed inside `multigt_trajectory_loss` of: Hungarian matched L1 at full weight (one assignment per forward over `[K, 64]`), a DrivoR dense nearest-GT pull at `dense_weight = 0.1` gated by `dense_gate_m`, and an anti-collapse diversity penalty at `diversity_weight = 1.0` |
| `loss_velocity` | 1.0 | SimWAM's `v_loss` on the rectified-flow velocity in `norm_odo` units, bell-weighted by t, so metres and radians contribute comparably |
| `source_loss` | 1.0 | DrivoR `trajectory_weight = final_score_weight = 1.0` |
| `rank_loss` | 0.1 | DrivoR `preference_weight × preference_rank_weight = 1.0 × 0.1`, weighted **independently** of `λ_score` — nesting it there diluted it 5× |
| `loss_future` | 0.1 | `dense_future_l1`: each candidate is paired with its **nearest valid GT** and supervised on that GT's 8 per-frame targets, so all 64 register rows get gradient. Only when a Stage 0 cache exists; with no cache the branch is **frozen**, and it is additionally left out of the DiT sequence (`predict_future=False`) only when there is no Stage 1 checkpoint to restore either — freezing rather than removing is what keeps `future_head` out of the DDP reducer instead of handing it an identically-unused parameter |

Because the dense pull covers every candidate against its own nearest valid GT,
the losers of the Hungarian matching are never left unsupervised — which is what
stops the set from degenerating into one good trajectory plus 63 junk ones.

Online score labels are `[B, 64, 7]` (`NC, DAC, EP, TTC, C, DDC, Final`), never
a `64×K` candidate-by-GT matrix.

Reporting separates the two ADE numbers on purpose: **matched-GT ADE** covers
only the Hungarian winners, **all-64 nearest-GT ADE** covers every supervised
candidate. Quoting the first alone is exactly how a collapsed candidate set hides.

### How the future representation is scored

Stage 1 has no trajectory objective, so "did it learn the future?" needs its own
readings. **The training loss is L1 and stays L1** — FastWAM trains `1 − cosine`,
we do not; cosine is reported as a *reading* only.

`future_fit_metrics` in `src/nvs_v6_wan_e2e/stage1_eval.py` reports, per row
(register `k` is scored against GT `k`'s own query-extracted future). It is a
library rather than a per-script helper because there are two Stage-1 trainers and
the verdict rule is the one part that must not drift between them —
`test_both_stage1_trainers_share_one_copy_of_the_verdict_maths` asserts both import
the same object and that neither kept a private copy:

| Field | What it answers |
|---|---|
| `paired_future_l1` | The objective itself: weighted L1 in the standardized 1024-d target space |
| `future_cosine_similarity` | Per-valid-frame cosine of prediction vs target — the direction match, scale-free |
| `future_relative_l1` | **Dimensionless**: `paired_l1 / weighted|target|`. `0.0` = perfect, `1.0` = "as wrong as predicting all zeros", `>1.0` = worse than predicting zeros |
| `target_abs_mean` | The denominator itself, so you can tell whether a large `relative_l1` came from a near-zero target magnitude |
| `scene_mean_target_l1` | Model-independent baseline: what you get by memorizing the scene's mean future and never looking at the trajectory |
| `trajectory_blind_l1` | Model-side control: swap in another GT's action tokens from the same scene. If this does not get *worse*, the future is not actually determined by the trajectory |

`trajectory_conditioning_fidelity` and `trajectory_response_alignment` summarize
those two controls. The before/after headline is exactly
`CONDITIONING_SUMMARY_KEYS`, and `describe()` must stay a subset of it (a
contract test enforces the `<=`).

**Neither metric gates the verdict, but `future_relative_l1` does carry one hard,
calibrated bar by construction: below `1.0` means the register beats emitting
nothing, above `1.0` means its output is worse than silence.** That bar matters most
on `eval`, whose targets are standardized by mean/std fit on the *train* scenes only, so
their magnitudes are not comparable to train's. (A 12-scene fit measured `target_abs_mean`
**2.75 on eval against 0.76 on train**, making the raw eval L1 of 2.78 unreadable and the
relative one of 1.01 the only comparable number — but those two values were measured under
the retired whitened-PCA target and are not comparable to the current diagonal one. The
*reason* the relative reading is the comparable one is unchanged.) What still
has no *absolute* threshold is "good *enough*" — but it does have a data-derived
anchor: `scene_mean_target_l1`, the loss of memorizing each scene's mean future while
ignoring the trajectory entirely. A future representation that cannot beat it has
learned nothing trajectory-conditional. On the 12-scene full-rank run that anchor sat
at **0.5839** while the trained register reached only **0.7515** (and the best global
constant 0.7492), which is why that run's verdict is `false` on all five booleans
despite `future_relative_l1` dipping to 0.9938. Read these next to
`scene_mean_target_l1`, not in isolation.

## Training

Stage 1 first, then Stage 2 — `scripts/dlc_navtrain_wan_e2e.sh` runs them in order.

**Stage 0** (`build_future_target_cache.py`, once). The generated
`trajectory_k/future.mp4` files are 41 frames at 10 fps; future indices
`range(5,41,5)` give 8 frames at +0.5…+4.0 s, matching DrivoR's `num_poses = 8`.
The frozen Teacher encodes them at their **native `464×832`** (both divisible by
16 → grid `29×52` = 1508 patches) with zero resize and zero interpolation — it
raises on any other size. Each of the **8 register tokens is one future frame**, read
out by `FutureQueryExtractor`: the frame's own **CLS token as the query**, cosine
logits over the patch grid at `QUERY_TEMPERATURE = 0.005` split across
`QUERY_HEADS = 8` heads, **values left un-normalized**. Nothing is pooled across
time: an earlier revision folded each *pair* of frames into one target, which put
+0.5 s and +1.0 s in the same linear subspace and left +2.5…+4.0 s — the part PDMS
actually scores — thinly supervised. The readout is then **diagonally standardized**
(per-dimension mean/std fit on train only, each std floored at
`TARGET_STD_FLOOR = 1e-2` of the largest and the clamped count audited), which is
what makes the target's scale readable without introducing a basis.

Why a query rather than spatial pooling: a two-way ANOVA on real Teacher features
puts **81.0%** of the GT-discriminating variance in the GT-specific *spatial pattern*
and only 19.0% in the per-GT average level (per scene: 75.1 / 87.1 / 80.1 / 82.3%).
Mean-pooling destroys that 81% outright, and ordered 2×2 quadrants bought only ~3%
relative separability over a single global mean at 4× the width. Why τ = 0.005:
separation alone is not a sufficient statistic — a **random-query control** scored
separation 0.4957, *above* the CLS query's 0.4934, but trajectory correlation only
0.4121 against the CLS query's 0.6647, so a high ratio can mean "kept the trajectory
signal" or "threw the scene away into noise". The raw **dot-product** softmax (no
cosine) measured entropy 0.9987 of `log2(P)` and separation 0.1414 — *worse* than the
pooling it was meant to replace, because DINOv3 features are strongly anisotropic and
a shared mean direction dominates every product. Why the query has **zero parameters**:
a learned extractor moves the target while the Student chases it, and a randomly
initialized learnable query measured attention entropy **0.999997** — a mean pool
wearing an attention costume. `MAX_QUERY_ATTENTION_ENTROPY = 0.5` is the gate that
catches that, justified by a measured gap (selective readouts ≤ 0.436, degenerate
ones ≥ 0.936).

The Teacher's grid is deliberately coarser than the Student's `42×72`; that is sound
because DINOv3's RoPE normalizes coordinates **per axis** (`RotaryEmbeddingDinoV3`,
`normalize_coords='separate'`), so a token's rotation depends only on its fractional
image position, not on how many patches the frame was cut into. Cutting quadrants
also used to require integer-division boundary arithmetic on an odd grid
(`29 → 14/15`, `52 → 26/26`); with no spatial partition there are no boundaries and
that whole class of bug is gone. Reduction is streamed — a `[8,29,52,1024]` float32
grid is 50 MB and the machine has 50 GB RAM. The cached register is `[K,8,1024]`,
128 KB at K = 4.

**Stage 1** (`run_stage1_cache_ddp.py` in production, `run_stage1_multiview.py` as the
smoke and diagnostic harness — two trainers, one copy of every number, see
`stage1_eval.py` above). Trains **η + the DiT trunk of θ** —
`scene_blocks`, the Wan blocks, `action_encoder`, `time_queries`, `mode_projection`,
`ego_to_action_projection`, `future_queries`, `future_head` — and freezes
`theta.head.*`, **`future_fuse*`** (three modules, matched by prefix), the EMA copy
and ψ. The
velocity head is frozen because nothing in the Stage 1 loss reads it and its
near-zero anchor-preserving initialization has to survive to Stage 2. The fusion is
frozen because it *is* the `Register → Action` arrow the Stage 1 mask exists to
close, and `_flow_pass` skips it under `stage1` anyway — freezing keeps the
parameter honestly gradless instead of trainable-but-unused, which is the state
DDP's reducer rejects under `find_unused_parameters=False`. It also means
`parameter_groups` drops it from the theta group in Stage 1, since that function
skips `requires_grad=False`. The action input is **noise-free** by construction —
`forward_register_only` runs the clean trajectory at σ = 0, the same end of the
flow Stage 2 noises — and the mode code is **zeros**, not the per-row immutable
code: a code constant per row index would let the network answer "which GT am I"
from the row alone instead of reading the action tokens the future queries attend
to. Loss is the future-register alignment alone, identity-paired row for row.
The number of fit scenes now sets **estimation precision only**. Under the retired
PCA target it also set the target's *rank* (`min(target_dim, fit_rows)`), and below
8 scenes at K = 4 the dimensions past the rank were **bitwise zero** in every row —
so the constant-zero predictor scored L1 0.335 against 0.353 for 20 real training
steps, and L1 fell without anything being learned. A diagonal mean/std has no
subspace and therefore no rank, which removes that entire failure class; `fit_rows`
is still audited, and what is gated now is the **query attention entropy** instead.

The two trainers differ in where their targets come from, and only in that. The
production trainer reads the Stage-0 cache — the 128 KB `[K,8,1024]` block plus the
persisted standardizer — so it neither re-decodes `future.mp4` (~5.7 s of GPU per
scene, 163 GPU-hours over the corpus, repaid on every restart) nor refits the
scaling. The smoke trainer extracts targets inline and fits its own standardizer over
its `--scenes`, which is why its fingerprint differs from the cache's: same
population statistic over 8 scenes instead of `--fit-scenes`, a precision difference
rather than a different target space. It also holds every scene resident as float32
(~195 MB each), which caps it at a few hundred scenes; the production trainer streams
through `Stage1SceneDataset` with the history left `uint8` on the host (11.6 MB) and
decoded per `__getitem__`.

Two things about the DDP wrap are load-bearing rather than boilerplate. The trainable
forward goes through a `Stage1Module` whose `forward` *is*
`forward_register_only`, because calling that method on the inner module would bypass
`DDP.forward` — and `prepare_for_backward`'s `_find_tensors(output)` runs there, so
bypassing it means gradients are never all-reduced: 16 ranks each training their own
model, with 16 healthy-looking loss curves and no error anywhere. And
`setup_distributed` sets `enabled = has_env and world > 0`, not `> 1`, so a
`--nproc_per_node=1` run still wraps and still builds the reducer; that is the only
way to exercise the "every trainable Parameter receives a gradient" contract on a
single card, since a bare `python` run would skip the wrap and pass vacuously.

The train/eval split is the **builder's**, re-derived from `_ready_scenes.json` with
the manifest's own `eval_scenes`/`seed`. Recomputing it from CLI flags would be a
leak, not a typo: the standardizer was fit on the builder's train scenes, so a scene
this run called "eval" could have helped define the mean/std its own targets were
standardized by. `test_the_ddp_trainer_reproduces_the_builders_train_eval_split`
asserts that against the persisted `fit_scene_ids`.

**Stage 2** (`run_navtrain_wan_e2e_ddp.py`). Trains **η + θ + ψ** jointly from the
Stage 1 checkpoint (`--stage1-checkpoint`). Input is the noised action at a random
σ; the scorer is on. Parameter groups must stay complete, unique and disjoint —
`build_optimizers` asserts it.

Whether the future tokens sit in the sequence is decided **independently** of
whether they train:

```python
predict_future = train_future or bool(args.stage1_checkpoint)
```

With a Stage 1 checkpoint but no `--future-target cache`, the eight register
tokens stay in the sequence and their branch — together with `future_fuse` — is
**frozen** rather than removed: the trunk was shaped around 97 tokens, so dropping
8 would evaluate a sequence Stage 1 never saw, and `requires_grad=False` is
exactly what keeps `future_head` out of the DDP reducer.
`predict_future=False` (`[Scene | Action]` = 89 tokens) is for the case with
neither a cache nor a Stage 1 checkpoint; the fusion never runs there, so
`future_fuse*` must be frozen too — a trainable-but-gradless parameter is what the
reducer rejects when `find_unused_parameters` is off. `save_checkpoint` records the
resolved value, and `load_joint_checkpoint` reads it back so evaluation rebuilds
the sequence length the weights were actually trained at.
**`load_joint_checkpoint` also calls `restore_ema(ckpt.get("ema_theta"))`** — without
a teacher the fusion is skipped entirely, and evaluation would silently reproduce a
*different* forward than the one that was trained.

## Commands

```bash
cd /mnt/zhaozc_workspace/project/NVS
source /usr/local/PPU_SDK/envsetup.sh ppu

# Contract suite (the wrapper owns the interpreter; see below)
bash scripts/run_contract_tests.sh

# Evaluation-path probe: the two things the contract suite cannot reach —
# the production 672x1148 forward, and run_navtrain_wan_e2e_ddp.py's own
# inference / save_checkpoint / load_joint_checkpoint.  ~2.5 min, synthetic input.
python scripts/probe_stage2_eval_path.py

# Full two-stage pipeline
bash scripts/dlc_navtrain_wan_e2e.sh
```

Per-stage skips: `NVS_SKIP_STAGE0/1/2=1`. Stage 0 is skipped automatically when
`standardizer.pt` already exists, unless `NVS_FORCE_STAGE0=1`. Extra arguments go
through `NVS_STAGE0_EXTRA_ARGS` / `NVS_STAGE1_EXTRA_ARGS` / `NVS_STAGE2_EXTRA_ARGS`.

Stage 1's knobs are `NVS_STAGE1_STEPS` (20000), `NVS_STAGE1_EPOCHS` (64 — must exceed
`STEPS / steps_per_epoch` or both the loop and the cosine schedule end early),
`NVS_STAGE1_BATCH` (4), `NVS_STAGE1_GRAD_ACCUM`, `NVS_STAGE1_WORKERS` (6) and the
rank-0 verdict sizes `NVS_STAGE1_TRAIN_METRIC_SCENES` / `NVS_STAGE1_EVAL_METRIC_SCENES`
(32 each) / `NVS_STAGE1_MEAN_TARGET_SCENES` (512) / `NVS_STAGE1_EVAL_EVERY` (1000).
`NVS_STAGE1_SCENES` and `NVS_STAGE1_EVAL_SCENES` are **gone**, not renamed: they sized
the resident scene list, and the production trainer streams the cache instead, so it
trains on every cached scene and there is nothing for them to set. `NVS_STAGE1_DIR`
now defaults to `artifacts/stage1_cache_ddp`.

Two interpreter rules that are easy to get wrong:

- Always `python -m pytest`, never the venv's `pytest` shim — its shebang is
  broken on this image.
- `PYTHONPATH` entries must be **absolute**. The venv's editable `navsim` `.pth`
  points at a nonexistent `/mnt/zczhao_workspace/...` (typo in the source venv),
  so relative paths silently fail to resolve.

```bash
export PYTHONPATH=/mnt/zhaozc_workspace/project/NVS/src:/mnt/zhaozc_workspace/project/NVS/vendor/DrivoR:/mnt/zhaozc_workspace/project/NVS/vendor/DrivoR/nuplan-devkit
```

## Data and caches

| Purpose | Path |
|---|---|
| Multi-GT generated futures (103075 navtrain scenes) | `/mnt/data/multi_future_view/navtrain_full_allcosmos_cuda_v1_gpu64/navtrain_full_allcosmos_cuda_v1_gpu64` |
| Same data via symlink | `/mnt/data/multi_view/...` |
| DrivoR-compatible online metric cache | `/mnt/zhaozc_workspace/project/DrivoR_multigt_kvcache_final_20261004/data_runtime/train_metric_cache` |
| Wan2.2 source weights | `/mnt/LLM_weight/Wan-AI/Wan2.2-TI2V-5B` |

Metric-cache layout: `data_runtime/train_metric_cache/<log>/unknown/<token>/metric_cache.pkl`.
`/mnt/zhaozc_workspace/official_DrivoR_clean/data_runtime/train_metric_cache` is a
symlink to the FastWAM offline cache and is **rejected** for DrivoR online
evaluation.

History frame filenames are **camera tokens, not frame tokens** — resolve them
through `frames[i]["cams"]["CAM_F0"]["data_path"]`. Logs are 2 Hz. `trajectory_k`
directory order does **not** equal candidate index (measured: `folder_slot !=
source_candidate_index` for 193/200 scenes), so pairing must go through the
explicit fields in `metadata.json`. About 19% of multi-GT scenes are degenerate
(duplicate or all-`human_log` labels, inter-GT pair L1 of 0.005–0.044 vs
0.133–0.167 for the rest); `--skip-degenerate` filters them by reading only
`trajectory_k/metadata.json`, so no mp4 is decoded for a scene about to be dropped.

That label test is necessary but not sufficient, and Stage 1 applies a second gate
beside it. `degenerate_multi_gt` is `len(set(labels)) < len(labels)`, so a scene
carrying four *distinct* labels whose trajectories sit centimetres apart passes it —
one was measured in the smoke cache at 0.028 m mean pairwise separation over the 4 s
horizon, while its four register targets still differed by 0.332 in standardized
units. Such a scene is worse than useless to the register: its K inputs are identical
to within float noise while its K targets are not, so the only way to reduce the loss
on it is to stop reading the action tokens. `target_cache.MIN_GT_SPREAD_M` (0.1 m) is
the floor, measured on the trajectory rather than the label and applied under the same
`--skip-degenerate` flag.

The floor is set from data, not taste. Across the eight cached scenes
`target_cross_gt_rms` tracks the metric spread at pearson **+0.874**, so the target is
largely trajectory-explainable and a small spread is not by itself a problem; target
spread per metre of trajectory spread was 0.36–2.2 for seven of them and **11.7** for
the 0.028 m scene. 0.1 m isolates that outlier with a 3.5x margin on both sides and
drops 1 of 8, where 0.5 m would also drop a 0.216 m scene whose ratio of 2.2 sits
inside the pack.

`filter_usable_tokens` (which builds the train/eval split) and
`Stage1SceneDataset.load_payload` (which re-checks at load time) must apply the gate
identically. If the dataset is stricter, `__getitem__` walks forward to a token the
split never approved — silently training on an eval scene or scoring a train scene as
held out, with the only symptom being a metric attributed to the wrong scene.
`test_the_split_filter_and_the_dataset_agree_on_which_scenes_are_degenerate` asserts
the agreement as a property over every token under both settings of the flag.

## Current honest status

- The contract suite passes (**88 tests**) against the current architecture, and the
  Wan payload loads strict with `loaded_keys: 226`.

> **Measurement caveat.** The two real-data Stage 1 runs recorded below
> (4-scene/20-step and 12-scene/60-step) measured every absolute number against the
> **retired** whitened-PCA `[K,8,256]` target, so they are **not comparable** to the
> current CLS-query + diagonally-standardized `[K,8,1024]` one: whitening makes the
> target zero-mean unit-variance along a fitted basis, diagonal standardization does
> not; the width differs 4×; the L1 denominator is a different quantity. They are
> kept because the **conclusions survive the change of target** — both are negative
> results, and both show that L1 buys the *global constant* first, that the
> register's trajectory-dependence *decreased* while the loss fell, and that
> `paired_future_l1` has to drop below `scene_mean_target_l1` before "learned a
> future representation" means anything. That milestone is target-independent. So are
> the structural readings (`attention_stage`, `segment_visibility`,
> `dit_sequence_tokens: 97`, `future_fuse_frozen_in_stage1`,
> `parameter_groups_updated`), none of which passes through the target. What must
> **not** be reused as a threshold to reproduce: 0.5839, 0.7515, 2.7504, and every
> `compressor_*` field — those keys no longer exist, the corresponding audit keys are
> `standardizer_fit_rows` / `standardizer_fit_scenes` /
> `standardizer_clamped_components` / `standardizer_fingerprint` and
> `query_attention_entropy_{mean,max,rows}`.
>
> The **2026-10-15** run recorded further below is *not* covered by this caveat: it
> is the first under the current target. It happens to share the older retired-target
> run's 4-scene / 20-step shape, so tell them apart by date, not by shape.

- **Retired: `Ordered2x2Compressor`.** It mis-unpacked `torch.linalg.svd`, which
  returns `(U, S, Vh)` — the singular values are the **second** element, so `singular`
  was bound to a `[rows, rank]` matrix. It crashed on the first real fit with
  `ValueError: only one element tensors can be converted to Python scalars`, and the
  projection basis was still correct (`v` bound to `Vh` either way), so nothing about
  the shapes looked wrong. It survived because the cache had never been built, i.e.
  the fit path had never run. The class is gone, but the two transferable lessons are
  why the incident stays recorded: an assertion that reads a value off the object
  under test is a tautology — what caught this class of bug was checking
  `explained_variance_ratio()` against an independent `svdvals` of the same centred
  rows rather than a loose band, where a silent mis-binding reports **0.1147** against
  a truth of **0.8385**; and a code path that has never executed is not covered by
  "the tests pass". The diagonal standardizer that replaced it has no basis, no
  singular values and no rank, so that specific bug cannot recur — but both lessons
  apply to it unchanged, which is why `clamped_components` is audited rather than
  trusted and why the attention entropy is re-measured after the readout instead of
  being assumed selective.
- **Three seam bugs found only by running the pipeline on real data.** All three lived
  under a green contract suite for the same reason the `svd` one did: every unit was
  individually correct and the call that joins them was never executed. (1)
  `query_future_register` guarded `ndim != 6` while its own docstring, its error
  message and all three of its callers describe the loader's **seven**-dimensional
  `[1,K,8,V,3,H,W]` — so Stage 0, Stage 1 and the separability diagnostic each raised
  on their first scene, and the function had never once succeeded on real data. (2)
  `load_multiview_sample` must size its `future_register` placeholder *before* the
  Teacher exists, and the parameter doing it was `target_dim = 256` — `d_model`, which
  stopped being the register width when the pooled-then-PCA target was retired.
  `with_future_register` then rejected the real `[1,K,8,1024]` target with a message
  that read like a Teacher fault, at a point where the future video had already been
  decoded *and* encoded. The parameter is now `register_target_dim`, defaulted to the
  same `EXPECTED_REGISTER_TARGET_DIM` that `load_v6_config` checks the yaml against,
  and a width mismatch names it while a frame-count mismatch does not. (3)
  `build_scene_payload` read the five resolved history JPEGs from
  `metadata["history_paths"]`, but the loader resolves them while constructing the
  *observation* and records them in `observation.camera_metadata` — the key is simply
  absent from `metadata`, where every other scalar in that payload does live, which is
  what made the line look correct. Stage 0 therefore raised
  `KeyError: 'history_paths'` on **all eight** smoke scenes. It now reads the
  observation, and a missing key names `load_multiview_sample`, because the failure
  otherwise reads as a corrupt cache entry rather than as an observation that did not
  come from the loader. All three sides of the seam now have a test: a shape-recording
  stub Teacher pins the per-GT peel down to `[1,8,V,3,H,W]`; a fixture scene under
  `tmp_path` drives the real loader, asserting the default width *and* an explicit
  override — a hardcoded 1024 at the allocation site would satisfy the first and fail
  the second; and the same fixture drives the whole cache round trip
  `build_scene_payload → atomic_torch_save → read_scene_payload(weights_only=True) →
  targets_from_payload → load_history_uint8`, asserting
  `"history_paths" not in sample.metadata` so the structural fact is pinned too and a
  fixture that one day carried both copies could not turn a revert green. Reverting
  each fix reddens its own test, which was verified rather than assumed: the payload
  revert reproduces the production `KeyError` verbatim, and dropping the new guard
  reproduces a bare `TypeError: 'NoneType' object is not iterable`.
- **A silent total wipeout costs more than a crash, so `except`-per-item tolerance and
  the exit code have to be designed separately.** Stage 0's per-scene `except` is
  *required* — one unreadable scene out of 103075 must not condemn the build, and a
  resumed run legitimately builds zero **new** payloads while reusing thousands — but
  it is also what turned bug (3) into a run that reported success: 8 scenes attempted,
  8 failed, and the script still printed `"event": "done"`, wrote a `_manifest.json`
  beside a `targets/` directory that did not exist, and exited 0. The distinction is
  not "were there failures" but "**was anything produced**": a systematic fault is
  total by definition, while a data fault costs a few scenes. `shard_produced_nothing`
  now fails the run when a shard has scenes but produced neither a new nor a reused
  payload, or when the destination holds no target at all — the second clause is
  rank 0's alone, since only it scans the destination, and an empty shard slice
  (`world` larger than the scene count) is not a failure. The decision is taken
  **after** the barrier: raising inside `build_shard` would leave the other ranks
  blocked in the collective and turn a legible failure into a hang. `torchrun` fails
  the job if any rank returns non-zero, so per-rank is enough. `failed_scene_count`
  now rides in the manifest so a *partial* failure outlives its log line, and the
  policy function has its own test — including "rank 0 built three but nothing landed"
  and "a resume reused everything" — because a failure branch that has never executed
  is not covered by anything.
- The previously recorded 4-scene / 60-step real Stage 1 run (`loss_future`
  1.1769 → 0.1602, `parameter_groups_updated == {eta: true, theta: false, phi:
  true}`) belongs to the **superseded** separate-φ architecture. It is not
  evidence about the current one, which **has since been re-run on real data twice**
  (4-scene/20-step and 12-scene/60-step, both documented below, both negative).
  `parameter_groups` now returns `{eta, theta}` and Stage 1 moves both.
  **It also ran with the broken head split** (`view(B,H,L,D)` instead of
  `view(B,L,H,D).transpose(1,2)`), so attention mixed tokens before any mask could
  apply. Neither that curve nor the `diagnose_target_separability.py` reading of
  `gain_over_shuffled ≈ 0.4%` — which predates both direction reversals *and* the
  fix — should be treated as a baseline to reproduce. That script still needs
  re-running; note its `gain_over_shuffled` is a **target-side** quantity and must
  not be confused with the **model-side** `gain_over_shuffled_pair` in
  `register_verdict`, which the full-rank run measured at 2.6e-08.
- **Stage 0 has now run, on 8 real scenes, and its cache reads back — but no
  production-scale cache exists.** (2026-10-15, `/tmp/nvs_smoke_cache`: `--limit 8
  --fit-scenes 4 --eval-scenes 2 --skip-degenerate`, one GPU.) This is the run that
  exposed seam bug (3) above: its first attempt wrote `standardizer.pt`, a manifest and
  **zero** targets, and exited 0. After the fix it built **8/8** payloads, exit 0,
  `failed_scene_count: 0`. Each is **137,624 bytes**, matching the predicted ~128 KB of
  `[K=4, 8, D=1024]` float32 register data, 1.1 MB for all eight. The read-back is
  verified end to end, not assumed: `read_scene_payload(weights_only=True)` returns 22
  keys; `history_paths` holds five **real sensor-blob** JPEGs
  (`.../52_01598_02013/CAM_F0/00a12db907405002.jpg`) under
  `history_policy: real_CAM_F0_sensor_blob` with `history_valid` all true — so the
  loader found the camera files and did not take its `initial_front.jpg` fallback;
  `future_register` is `(4, 8, 1024)` float32 at `absmean 0.9233 / std 1.1449`, i.e.
  the diagonal standardizer really did land the target near unit scale;
  `targets_from_payload` re-inflates to `(1, 4, 8, 1024)`; and `Stage1SceneDataset`
  plus `collate_stage1_batch` produce `(3,5,1,3,672,1148)` history and
  `(3,4,8,1024)` register with `Observation.validate()` passing. The `uint8` history
  item measures **11.57 MB**, which is the number the module docstring claims against
  the eager float path's 46.3 MB — now measured rather than computed. The reused
  `standardizer.pt` fingerprint `53d2f54b28513cd0…` matches the first run's
  `standardizer_fit` event, so the write→load round trip is stable across processes.
  **Cost, measured on this run:** payload building ran at `scenes_per_s 0.176` =
  **5.7 s/scene** single-GPU, so the full 103075-scene corpus is ~**163 GPU-hours**,
  i.e. ~10 h on 16 GPUs. The readiness scan is separate and much cheaper once warm:
  `iter_ready_scenes` costs **26 ms/scene** (1000 scenes in 25.5 s) → ~**44 min** for
  the whole corpus on rank 0, once, behind the `_ready_scenes.json` sidecar. Two
  cautions from measuring it: the *first* touch of a scene directory is far more
  expensive than the warm 26 ms (the first 8 ready scenes cost 48.8 s cold and 1.2 s
  warm), so a `--limit 8` smoke run is dominated by cold-start latency and is **not**
  a scaled-down estimate of the full scan; and the sidecar is keyed on the literal
  `{multiview_root, cache_root, limit, skip_degenerate}`, so the symlink spelling of the
  corpus root mentioned in `multiview_data.py` invalidates an index built under the real
  one. The launcher uses the real spelling throughout, so it will not flip-flop there.
  `run_stage1_multiview.py` builds its
  targets **inline** instead, so a real-data Stage 1 smoke needs no cache — but that
  standardizer is never written to disk, so Stage 2's `--future-target cache` still has
  nothing to read outside a smoke directory. Until a production Stage 0 runs, the
  launcher prints a warning and passes
  `--future-target none`; because it also passes `--stage1-checkpoint`, the eight
  register tokens **stay** in the DiT sequence and the branch is frozen instead of
  removed — `predict_future=False` needs neither a cache nor a Stage 1 checkpoint.
  Stage 2 collects `register_target_dim` from up to three sources — `--register-target-dim`,
  the cache standardizer's `feature_dim`, and the Stage 1 checkpoint's recorded width —
  and **raises** if they disagree rather than picking one, so a cache and a checkpoint
  built against different targets cannot be silently combined. Stage 1, which fits its
  own standardizer inline, writes the width into its checkpoint config for that reason.
- **The first real-data Stage 1 smoke after seam fixes (1) and (2) ran clean and proved
  nothing.** This run predates fix (3), which Stage 0 exposed afterwards.
  4 scenes / 20 steps: `attention_stage: stage1`,
  `segment_visibility: [[1,0,0],[1,1,0],[1,1,1]]`, `dit_sequence_tokens: 97`,
  `future_fuse_frozen_in_stage1: true`, `parameter_groups_updated: {eta: true,
  theta: true}`, `all_losses_finite: true`, `loss_future` 0.6724 → 0.3534. So the
  plumbing works end to end on real Teacher output for the first time.
  **But the run cannot show learning, and at `--scenes 4` it structurally cannot.**
  `compressor_rank: 128` against `compressor_output_dim: 256`
  (`compressor_full_rank: false`), and the projection is *exactly* zero past the
  fitted rank — so **half of every target row is bitwise zero**. Emitting zeros
  therefore scores `L1 = mean|target| = 0.335`, which is **better than the 0.3534
  twenty real steps reached**; the best constant scores 0.327 and the per-scene mean
  0.265. Meanwhile `register_std_across_candidates` fell 6.6e-05 → 3.0e-06 and
  `register_response_to_trajectory` 7.8e-05 → 3.6e-06, i.e. the register is
  collapsing to a trajectory-independent constant. The script's own verdict agrees:
  `train_beats_trajectory_blind_baseline: false`,
  `train_learned_trajectory_conditional_future: false`. Do not read that falling loss
  as progress — most of it is the trivial shrink-toward-zero direction.
  `run_stage1_multiview.py` **used to refuse to start** on a rank-deficient compressor
  unless `--allow-rank-deficient-target` was passed, printing the audit line first so
  the numbers were on the record. Both the gate and the flag are **gone**, because the
  diagonal standardizer has no rank to be deficient in — which is the point of the
  replacement, not a loss of a safeguard.
- **The full-rank rerun is also a negative result — but a trustworthy one, and it
  fixes a concrete milestone.** `--scenes 12 --eval-scenes 2 --skip-degenerate
  --steps 60 --check-scorer`, 926 s (~15.4 s/step), lr 2e-4, batch 4, `skipped: []`
  (all 14 scenes in this draw had 4 non-degenerate GTs). The compressor was
  `compressor_rank: 256` / `fit_rows: 384` (= 12×4×8) / `full_rank: true` /
  `explained_variance_ratio: 0.9987` / `clamped_components: 27`, and
  `target_spread_across_gt` rose from 0.4350 to **0.9524** — the targets genuinely
  separated different futures at that point, so rank deficiency was no longer an
  excuse for that run. (Under the diagonal standardizer there is no rank to be
  deficient in at any scene count.)
  The mask audit is unchanged and correct: `attention_stage: stage1`,
  `segment_visibility: [[1,0,0],[1,1,0],[1,1,1]]`,
  `future_fuse_frozen_in_stage1: true`, `future_fuse_in_theta_group: false`,
  `{eta: 9812480, theta: 9235968}` — the θ figure is from the pre-split width and has
  since been re-measured at **9,433,344**; see the handoff bullet below, and do not
  subtract one from the other. `loss_future` 0.8906 → 0.7882 (mean 0.8311 →
  0.7632), `loss_decreased: true`, `all_losses_finite: true`,
  `parameter_groups_updated: {eta: true, theta: true}` — and **all five verdict
  booleans are `false`**.
  Read it this way, not as "the loss fell": train `paired_future_l1` went
  0.9000 → **0.7515**, while on the same data `constant_target_l1 = 0.7492` and
  `scene_mean_target_l1 = 0.5839`. So **60 steps bought exactly the global-constant
  tier** (`gain_over_constant_target = -0.0031`, still a hair worse than a constant),
  whereas *memorizing each scene's mean future alone* would already reach 0.5839 —
  the model is 0.168 short of a baseline that uses no trajectory at all.
  Corroborating: `register_abs_mean` 0.4703 → **0.0532** (an 8.8× shrink; the
  standardized target is zero-mean by construction — true of the diagonal one too, so
  this reading survives the rewrite — which makes "output ≈ 0" *the* global constant
  predictor), `distance_to_mean_target_l1` 0.4856 → **0.1312**,
  `collapse_toward_mean` 0.175, `register_std_across_candidates` 7.19e-05 →
  **4.45e-06** (16× smaller), `trajectory_conditioning_fidelity` 8.16e-05 →
  **5.06e-06**, `gain_over_shuffled_pair` **2.6e-08**, `gain_over_trajectory_blind`
  **-1.3e-08** — swapping in another GT's action tokens leaves the prediction
  **essentially bitwise unchanged**.
  `future_relative_l1` moved train 1.1945 → **0.9938** and eval 1.0124 → **0.9992**,
  i.e. it landed *on* its own calibrated 1.0 bar. Since 1.0 means "as wrong as
  predicting zeros", the metric is reporting precisely "only the global constant was
  learned" — the bar is doing its job. `future_cosine_similarity` went train
  -0.0013 → +0.0423 but eval 0.0451 → 0.0349 (up on train, down on eval; at that
  magnitude, no direction was learned).
  **This is not a plumbing bug.** Both groups really updated, just barely: eta norm
  109.4075 → 109.4845, theta 196.4850 → 196.4949, `future_query_update_l2` 0.1035;
  and the tail is still descending monotonically (~7e-05/step over steps 54→60),
  i.e. training is still inside the "shrink to the global constant" phase. **60 steps
  cannot separate "will learn" from "cannot learn"** — but it does establish that L1
  buys the global constant first, and that the register's trajectory-dependence
  *decreased* while it did. **The first milestone for any real run is
  `paired_future_l1` dropping below that run's own `scene_mean_target_l1`** — the
  model-independent scene-mean baseline, which the audit prints rather than hardcodes
  precisely because its value moves with the target definition. On this run it was
  0.5839; before the trained register beats it, talking about a "future
  representation" is premature.
  On eval, `target_abs_mean = 2.7504` vs train `0.7561` (3.6×) because the
  standardization statistics are fit on train only — still true of the diagonal
  standardizer — so eval's raw L1 of 2.7482 is **unreadable** and only
  `relative_l1` (0.9992) is comparable — which is exactly why `relative_l1` was added,
  now confirmed empirically.
  The `--check-scorer` audit is clean on shapes and provenance:
  `scene_context [1,81,256]` (matching `scene_tokens: 81`), `trajectory [1,64,8,3]`
  with `trajectory_units` recorded as metres/radians DrivoR-native with no rescaling,
  `score_logits [1,64,6]` over the six PDMS heads, `logits_finite: true`,
  `selected_candidate: 62` / `selected_utility: 0.544`; `feature_audit` reads
  `initialization: scratch_random`, `checkpoint_loaded: False`,
  `native_dinov2_scene_cache: False`, `native_ego_cache: False`,
  `predicted_register_input: False` — both comparison-only paths are confirmed off.
  The untrained generator's `proposals_vs_multigt`: `min_ade_over_gt_m` **6.096**
  (each of the 64 proposals' best-GT ADE, then averaged — a **coverage** number),
  `best_pair_ade_m` **1.093** (the single best of all 64×K pairs — an **oracle**
  number), `best_pair_fde_m` 1.820, `mean_speed_mps` 7.103. **Neither key involves
  the scorer's selection** (both come from `stage1_eval.trajectory_errors`),
  so do not read 6.096 as "the scorer picked badly".
- **The verdict's headline boolean was misnamed and has been renamed**
  `beats_trajectory_blind_baseline` → `beats_scene_mean_baseline`. It always gated on
  `gain_over_scene_mean_target >= margin` (0.02), while `trajectory_blind_l1` is a
  *different, weaker* control with its own `gain_over_trajectory_blind`. The old name
  invited pairing the boolean with the full-rank run's `-1.3e-08` and concluding "a
  tie with trajectory-blind", when the value that actually triggered it was
  `-0.2871`. The contract test's `collapsed` case is the proof: a register with
  `fidelity = 0.05` **is** trajectory-blind by the scale-free reading, yet the old key
  reported `True` for it. The gate itself is unchanged — the scene mean is the tighter
  baseline and the `flat` case must still fail — and `trajectory_blind` is still gated
  inside `learned_trajectory_conditional_future`'s `all(...)`. **Both recorded runs
  above print the old key**, having run before the rename. A repo-wide grep confirmed
  the script, the contract test and the docs are the only consumers, so no launcher
  needed updating.
- **First real-data Stage 1 under the query-extracted target (2026-10-15): the
  plumbing works, and the negative result reproduces.** `--scenes 4 --eval-scenes 2
  --steps 20 --skip-degenerate --check-scorer`, 317 s (~15 s/step at batch 4 with
  `--grad-checkpointing`, one PPU-ZW810E), lr 2e-4, `skipped: []`, all six scenes
  `real_CAM_F0_sensor_blob` with 4 GTs each. The width agrees at every hop:
  `teacher_feature_dim` = `register_target_dim` = **1024** in both the target audit
  and the model audit, `register_target_shape [1,4,8,1024]`, and the scorer is handed
  `predicted_register_shape [1,64,8,1024]`. `loaded_keys 226`,
  `trainable_parameters 19,245,824` (`eta 9,812,480 / theta 9,433,344`),
  `future_fuse_frozen_in_stage1: true`, `parameter_groups_updated {eta: true,
  theta: true}`, `all_losses_finite: true`, `loss_future` 0.9312 → 0.6182.
  **The readout is selective, and the audit's *max* is the number that matters:**
  `query_attention_entropy_mean 0.0953` against `query_attention_entropy_max 0.3869`
  — 4× the mean, both under `MAX_QUERY_ATTENTION_ENTROPY = 0.5`. No head has
  flattened into the mean pool this readout replaced, but a mean-only audit would
  have hidden the worst one, which is why both are recorded.
  Target quality on this six-scene draw: `within_scene_gt_distance 0.5553` against
  `across_scene_gt_distance 1.2598` → `separation_ratio 0.4407`, and
  `trajectory_distance_pearson 0.8102` / `spearman 0.8539` over 24 GT pairs. These
  are **not** comparable to the τ table in `teacher.py`, which
  `diagnose_target_separability.py` measured on a different sample — that script has
  still not been re-run under this target. `standardizer_fit_rows 128` (= 4×4×8) for
  1024 dimensions with `clamped_components 1` is a small-fit artifact rather than a
  defect: production passes `--fit-scenes 512`, i.e. 16384 rows.
  **The verdict is the same negative result as before, and the target can no longer
  be blamed for it.** Train `paired_future_l1` 0.9312 → **0.6089** while
  `scene_mean_target_l1` on the same data is **0.3441**, so
  `train_gain_over_scene_mean_target = -0.769` and
  `train_beats_scene_mean_baseline: false`. It *did* clear the weaker tiers —
  `gain_over_constant_target +0.234`, `gain_over_scene_blind +0.388` — which the
  12-scene run under the retired target had not. On the two held-out scenes
  `paired_future_l1` 1.0341 → 1.0027 against `scene_mean_target_l1` 0.3548
  (`eval_gain_over_scene_mean_target = -1.826`), `eval_future_relative_l1 = 1.0925`
  — still *above* its calibrated 1.0 bar, i.e. worse than emitting zeros — and
  `generalizes_beyond_fitted_scenes: false`.
  The trajectory controls are the telling part: `train_gain_over_shuffled_pair
  = -3.3e-07` and `train_gain_over_trajectory_blind = +4.9e-08`, so swapping in
  another GT's action tokens leaves the prediction bitwise-identical to seven digits,
  exactly as under the retired target. Meanwhile train `future_cosine_similarity`
  went 0.0084 → **0.5795** and `future_relative_l1` 1.1434 → 0.7421: twenty steps
  bought the *scene-level* direction and none of the trajectory-conditional one.
  The collapse readings move the same way they did before —
  `register_std_across_candidates` 6.5e-05 → 1.8e-05,
  `trajectory_conditioning_fidelity` 1.2e-04 → 3.1e-05,
  `register_response_to_trajectory` 7.7e-05 → 2.1e-05,
  `collapse_toward_mean 0.799` — i.e. trajectory-dependence *decreased* while the
  loss fell. **Removing the rank-deficiency excuse did not by itself produce
  trajectory conditioning.** The milestone is unchanged and is this run's own
  `scene_mean_target_l1`: 0.3441 on train, 0.3548 on eval.
  One reading genuinely improved: `mean_target_abs_mean` is **0.8146** on train
  against **0.9179** on eval (1.13×), where the retired whitened target measured
  0.7561 against 2.7504 (3.6×). Eval's raw L1 is therefore readable on this run
  rather than only its `relative_l1` — but `relative_l1` stays the number to compare,
  since the standardization statistics are still fit on train only.
  `--check-scorer` is clean: `scene_context [1,81,256]`, `trajectory [1,64,8,3]`,
  `score_logits [1,64,6]`, `logits_finite: true`, `selected_candidate 63` /
  `selected_utility 0.3918`, `feature_audit` with `initialization: scratch_random`,
  `checkpoint_loaded: False`, `predicted_register_input: False`; the untrained
  generator's coverage/oracle pair is `min_ade_over_gt_m 6.093` /
  `best_pair_ade_m 1.097` (`fde 1.805`, `mean_speed_mps 7.103`).
  At ~15 s/step single-process, the launcher's `STAGE1_STEPS=20000` is ~83 GPU-hours.
- Stage 1 → Stage 2 handoff re-measured after the register target became a
  **query extraction at the Teacher's own width** (`d_model = 256`,
  `register_target_dim = 1024`, `heads = 4`, `depth = 8`), with the real
  DINOv3-large Student and the real 226-key Wan payload: the generator state dict is
  **768 tensors** and the scorer's **108**, totalling **876**; the key sets match
  exactly and a `strict=True` reload is **bitwise identical**, and
  `load_joint_checkpoint` resolves `register_target_dim=1024 / target_dim=256` from it.
  The same payload also loads into a `predict_future=False` build — the sequence drops
  from **97 to 89 tokens** but the tensor count stays 876, since the register branch
  is allocated either way.
  Stage 1's trainable parameters are **eta 9,812,480 / theta 9,433,344**, summing to
  `trainable_parameters: 19,245,824`; θ's total is **9,434,627** and the 1,283
  difference is exactly the `head.*` parameters Stage 1 freezes. The theta figure
  **excludes** `future_fuse*` (**10 tensors / 658,944 parameters**, model-level, not
  θ-level) because Stage 1 freezes them and `parameter_groups` skips frozen
  parameters — the audit reports `future_fuse_in_theta_group: false` for Stage 1 and
  the Stage 2 trainer asserts the opposite. θ itself is
  **242 tensors = 226 Wan-loaded + 16 NVS-only** (`time_queries`, `future_queries`,
  the three type embeddings, `action_encoder.*`, `mode_projection.*`,
  `ego_to_action_projection.*`, `head.*`, `future_head.1.*`, **336,899 parameters** —
  of which `future_head` alone is **263,168**, since it now emits the Teacher's 1024
  rather than `d_model`), so Stage 1's product still lives mostly in the eight Wan
  blocks both streams share, not in those heads. `ema_theta` is a **separate
  242-tensor payload entry** and the number of EMA tensors inside the Stage 1
  `"model"` entry measures **0** — the EMA never pollutes the `strict=True` key set.
  > The previously recorded **764 / 872 tensors**, **NVS-only 139,523** and
  > **theta 9,235,968** are pre-split figures, from when the register was `d_model`
  > wide. Do not subtract across the two conventions: the tensor counts moved because
  > the fusion grew a separate K/V projection and a second LayerNorm (+4), and the
  > parameter counts moved because the future head's output quadrupled.
- `scripts/probe_stage2_eval_path.py` passes **27/27** on the same production
  geometry (~2.3 min): the 64-mode anchor grid still spans exactly **38.4000 m/rad**
  (bitwise identical to the pre-split record, i.e. the mode grid was not touched),
  forward pre-hooks measure the production sequence as **`self=97 cross=81`** with
  the **cross-attention still unmasked** (`ctx_mask` / `context_mask` are `None`)
  but the **self-attention masked per entry point** — every forward through
  `forward_pairs` (4) and `forward_flow` (2) carries the Stage 2 table and every
  forward through `forward_register_only` (2) carries the Stage 1 table, each
  bitwise equal to the expanded `STAGE{1,2}_SEGMENT_VISIBILITY` and the two
  differing from each other — `inference()` is bitwise equal to
  `forward_flow(data=anchor, σ=1)` and deterministic across calls, σ=1 leaks no GT
  (`max|diff| = 2.86e-06`, and `v_target` is exactly 0 there while a real GT still
  yields `2.071`), and a saved checkpoint reloads to bitwise-identical inference
  **and a bitwise-identical EMA teacher** (`model=768 / ema_theta=242 / scorer=108`,
  `dit_sequence_tokens=97`, `config` holding **9 keys** — the extra one being
  `register_target_dim`). The probe deliberately perturbs the teacher away from
  the student before saving, so that reload assertion really proves the teacher
  round-trips rather than passing trivially. An older payload with no
  `predict_future` key still falls back to main()'s rule, and a pre-EMA payload
  still installs a teacher while reporting
  `ema_teacher_restored_from_payload=False`.
  Two checks are **new** and exist because the probe used to be self-confirming:
  `forward_register_only` returning `(1, 64, 8, 1024)` is now compared against the
  production constant `EXPECTED_REGISTER_TARGET_DIM` rather than against
  `model.theta.target_dim`, and it asserts `target_dim != d_model`, so a probe that
  silently built the pre-split geometry **fails** here instead of printing
  `(1, 64, 8, 256)` as a pass; and one check pins the fusion's asymmetry —
  `in_proj_weight is None`, `k_proj_weight = (256, 1024)`, `out_proj = (256, 256)`,
  i.e. K/V read the register width and the output writes the action width.
  Its observation is synthetic and its head is at scratch init, so nothing it prints
  is a quality number — including `max|traj-anchor| = 0.0117 m/rad`.
  **That reading used to be irreproducible and now is not.** The probe never seeded the
  model build, and PyTorch's default generator is seeded non-deterministically, so the
  scratch head differed per process: three fresh unseeded builds measured `future_head`
  weight sums of **+3.48 / −7.01 / −7.28**, and `max|traj-anchor|` wandered across
  0.0109 / 0.0159 / 0.0163 accordingly. The earlier explanation for those steps — that
  adding `k/v_proj_weight` and the second LayerNorm shifted the construction-time RNG
  stream — was the **wrong causal story**: those changes do shift the stream, but the
  real cause was that nothing seeded it, so the same commit run twice also disagreed.
  `main()` now calls `torch.manual_seed(0)` after `probe_geometry()` and immediately
  before the build (it has to be immediately before, since `probe_geometry` constructs
  two models and would consume the stream otherwise); two consecutive runs both give
  **0.0117** at 27/27, so the check is a regression signal rather than noise.
  The pre-`_split` **0.036** remains a valid tell — the broken head split mixed tokens
  and effectively amplified the attention output, a jump far larger than seeding noise —
  but the 0.0109 ↔ 0.0159 ↔ 0.0163 differences must not be read as behaviour changes.
  The magnitude is still 0.03% of the 38.4 m/rad spread, consistent with
  `Head.HEAD_INIT_STD = 1e-3 / 33.37`'s near-zero anchor-preserving init — an
  anchor-preservation reading, not an accuracy.
- Stage 1's reported ADE/FDE reflect a **frozen** velocity head on top of metric
  anchors, not trained accuracy — Stage 1 freezes `theta.head.*` by contract, so
  its near-zero anchor-preserving initialization reaches Stage 2 untouched.
- Moving the Scene into self-attention and adding the EMA forward raises the DiT's
  attention/FFN work by roughly **12×** (`97²` versus `16² + 16×81`, doubled by the
  teacher). That is still small next to DINOv3-large at 672×1148×5 frames — the
  probe's wall clock moved 114 s → 146 s, which also absorbs six new mask checks
  and one extra `forward_register_only` — but **it has not been
  benchmarked on a real batch**, so do not read 12× as an end-to-end slowdown.
  Separately, an arbitrary boolean mask means `scaled_dot_product_attention`
  **cannot take the flash backend** and falls back to the memory-efficient or math
  kernel; that is a real additional cost which has likewise not been benchmarked.
- DrivoR's `GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1,
  prob=0.7)` augmentation is **not** ported.
- No NAVSIM score in this repository is a performance claim. Oracle numbers
  require the real metric cache; synthetic smoke output never produces one.
