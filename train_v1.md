# train_v1 — design before implementation

Status: for discussion, 2026-11-23. Built from the two-part framework below, not from plan_v2/plan_v3 or the REVIEW's proposed fix. It replaces the v0 critic → GAE → batch-normalisation → progress-gate path in `training/runs/online_curriculum_01`. It reuses v0's rollout, capture and NFT-loss plumbing. Decisions are logged in §10.

## 0. Decisions at a glance

| Item | Choice |
| --- | --- |
| Action preference | $p^a_k = \tfrac12 + \tfrac12\,\mathrm{clip}(G_k/G_{\max}(s_k),-1,1)$ |
| Return | $G_k = r_k + \gamma G_{k+1}$, Monte Carlo, $G_T=0$, $\gamma=0.95$ per chunk |
| Reward | $r_k = \Delta P_k + S_k$, where $P$ comes from the target object's *held* and *placed* states |
| $G_{\max}$ | $2-P_k$: the largest return reachable from the chunk's own start state. The clip never fires. |
| Removed from v0 | value table, GAE, per-group normalisation, progress gate, timeout bootstrap, administrative truncation |
| Task subset | Single-object pick-and-place family (31 tasks, §3). A screening epoch under the base checkpoint picks the training tasks, and its data becomes training round 0. |
| Video branch | Its own preference $p^v_k$ from a VLM-judged dream reward, immediate only ($\gamma_v=0$). V1 and V2 are both implemented and selectable. The simulator gates the judge. |
| Wrong-grasp penalty | Not in the reward. It is logged as a diagnostic of the alignment hypothesis (§5). |

## 1. Framework

### 1.1 Chunks and episodes

At boundary $b_k$ the policy jointly samples $x_0^k$, which holds future video latents and 32 action rows. Rows execute until the next boundary (32 steps, 2.13 s at 15 Hz) or until termination. An episode ends only on official success or official timeout, at the task's `episode_length_s`. There is no administrative truncation. If the wall-clock budget expires, unfinished episodes are dropped rather than bootstrapped, because their Monte Carlo return is undefined.

### 1.2 Progress $P$

All tasks in the subset have one target object $o$ and one receptacle $c$:

$$
P_k=\begin{cases}1 & o \text{ placed in/on } c\\ \tfrac12 & o \text{ held}\\ 0 & \text{otherwise}\end{cases}
$$

- **Held** means gripper contact (RoboLab `object_grabbed`) *and* the object raised at least 3 cm above its reset height. Both must hold for the last 8 simulator steps before $b_k$. RoboLab's own ladder uses contact alone, which also fires on nudges. The lift condition removes those.
- **Placed** is the task's official success predicate, with the gripper detached. In this family, placed ⇔ success, so $P=1$ only at the terminal boundary.
- $P$ describes the current state; it is not a monotone ladder. A drop sends it back to 0. Holding a wrong object gives $P=0$.

### 1.3 Reward

$r_k = (P_{k+1}-P_k) + S_k$, where $S_k=1$ if and only if official success occurs inside chunk $k$.

| Event in chunk | $r_k$ |
| --- | ---: |
| Grasp | +0.5 |
| Drop | −0.5 |
| Place from held (success) | +1.5 |
| Grasp and place in one chunk | +2 |
| No change, or wrong object held | 0 |

### 1.4 Return

$G_k=r_k+\gamma G_{k+1}$, with $G_T=0$ at both success and timeout. With $\gamma=0.95$ per chunk, a success 10 chunks away keeps a weight of 0.60, and one at 28 chunks (a 60 s horizon) keeps 0.24. With $\gamma=1$, every regression-free success would get $G/G_{\max}=1$ on every chunk, fast or slow. Discounting is the only thing here that ranks faster successes higher.

$\Delta P$ carries no $\gamma$, so it is not potential-based shaping. This matters. With Monte Carlo returns, the potential-based form $\gamma P_{k+1}-P_k$ telescopes to $\gamma^{T-k}P_T-P_k$, and intermediate progress would earn nothing beyond timing. The undiscounted $\Delta P$ keeps credit on the chunk that made the progress. The side effect is that a grasp–drop cycle nets $\gamma^i\cdot0.5(1-\gamma^d)>0$. The bound in §2 caps the total from cycling at $1-P_k$, below the $2-P_k$ that completing the task earns. Grasp→drop cycles are logged per episode.

### 1.5 Preference and loss

$p^a_k = \tfrac12+\tfrac12\,\mathrm{clip}(G_k/G_{\max}(s_k),-1,1)$ is routed to the action tokens. The NFT loss is the same as in v0: $\beta=0.1$, reference MSE 0.01, rank-16 LoRA on both heads plus the MLP down projections in all 28 GEN blocks, learning rate 1e-4, one captured $\sigma$ per chunk, and the executed-row mask. Only these change:

- A chunk enters the actor batch only if at least one modality is non-neutral ($p\ne\tfrac12$). If every chunk is neutral, the update is skipped.
- Within the batch, the action loss is masked where $p^a=\tfrac12$. The video loss is *kept* at $p^v=\tfrac12$ (neutral or abstained). At ½ the NFT term reduces to the trust-region pull toward $v_{\mathrm{old}}$ plus the reference MSE. This keeps the video stream anchored while action gradients move the shared LoRA (see §6, pending Q5).
- No cohorts, gate or normalisation pool. $p$ depends only on the chunk's own episode.
- The old adapter is set to the trained adapter after each update, as in v0.

### 1.6 Why $G$ without a baseline is valid, and what it costs

At the start of an update ($v_\theta=v_{\mathrm{old}}$), one sample's NFT gradient is proportional to $(2p-1)(v_{\mathrm{old}}-v)$, where $v$ is that sample's flow-matching target. Within a state, $\mathbb E[(2p-1)(v_{\mathrm{old}}-v)\mid x_t]=\mathrm{Cov}(2p-1,\,v\mid x_t)$, provided $v_{\mathrm{old}}$ is the conditional mean velocity of the policy's own samples. A per-state offset in $2p-1$ therefore cancels in expectation: replacing $A$ with $G$ changes the variance, not the direction. Three consequences follow.

- A failed episode with no progress has $G=0$ and is exactly neutral. This removes the REVIEW's timeout artefact by construction.
- Most of the signal is positive; only drops produce negatives. The update behaves like return-weighted self-imitation and has higher variance than an advantage-based update.
- The cancellation is only approximate, because the sampler uses CFG with 4 steps. On tasks where nearly every chunk has $p>\tfrac12$, the residual acts as self-distillation of the sampled distribution (sharpening). Saturated tasks are therefore kept out of training and used only as retention checks (§3).

## 2. $G_{\max}$

The requirement is a number per chunk that comes only from that chunk's own state and guarantees $|G/G_{\max}|\le1$ without clipping.

**Bound.** For $P\in[0,1]$ and any $\gamma\le1$, summation by parts gives $\sum_{j\ge k}\gamma^{j-k}\Delta P_j\in[-P_k,\,1-P_k]$. Success adds at most 1, once. Hence $-P_k\le G_k\le 2-P_k$, and

$$
G_{\max}(s_k)=2-P_k \qquad (2 \text{ before the grasp},\ 1.5 \text{ while holding}).
$$

- $p=1$ is the best return reachable from this state (immediate completion). $p=\tfrac12$ means no net progress. The worst case, losing everything, gives $p=\tfrac12-\tfrac12P_k/(2-P_k)$, which is ⅓ while holding.
- The clip never fires, so the map is linear and the covariance argument in §1.6 holds exactly.
- $G_{\max}$ depends only on $s_k$. A positive factor that depends only on the state rescales the step size at that state; it does not change the improvement direction. Chunks from unrelated states never affect each other's scale.
- **Rejected alternatives:**
  - Batch standard deviation couples unrelated states; this is the v0 failure.
  - A per-state empirical maximum needs several samples per state; we have one.
  - A global constant of 2 is valid, but it caps a holding chunk at $p=0.875$ even for an immediate success.
  - Separate negative and positive scales ($P_k$ and $2-P_k$) would make the map non-linear and turn rare drops into the strongest gradients.

| Situation (γ = 0.95) | $G$ | $G_{\max}$ | $p^a$ |
| --- | ---: | ---: | ---: |
| Chunk 0 of a success with grasp in chunk 2 and place in chunk 6 | 1.554 | 2 | 0.89 |
| Chunk 3 of the same episode (holding) | 1.286 | 1.5 | 0.93 |
| Grasp chunk, then a drop 3 chunks later, then timeout | 0.071 | 2 | 0.52 |
| The drop chunk itself | −0.5 | 1.5 | 0.33 |
| No progress | 0 | any | 0.50 |

The general form for later multi-object tasks is $G_{\max}=(1-P_k)+1$.

## 3. Task subset

**Why narrow.** Similar tasks are needed so that (i) $P$ comes from each task's own definition with no per-task engineering, (ii) the VLM asks the same two predicates everywhere, (iii) $G_{\max}$ and $\gamma$ mean the same thing on every task, and (iv) horizons are comparable.

**Inspection of all 120 tasks** (success term plus `subtasks`):

| Group | Count (approx.) | Why not in v1 |
| --- | ---: | --- |
| Multi-object `logical="all"` | ~40 | $P$ needs per-object averaging; horizons of 90–300 s |
| `any` / `choose K` / counting | ~12 | Ambiguous which object counts toward $P$ |
| Ordered, stacking, reorientation, spatial relations, pick-only, out-of-container, bowl stacking | ~37 | Different predicates, or no place rung |
| **Single object into or onto a receptacle** | **31** | **Chosen family** |

**The 31 candidates.** Split is the existing scene-disjoint `training/task_split.json`.

| Into a container (train split) | s | Onto a surface (train split) | s |
| --- | ---: | --- | ---: |
| BananaInBowl | 50 | BananaOnPlate | 40 |
| RubiksCubeTask | 40 | FruitsOnionToPlate | 60 |
| YogurtInBowl | 40 | MouseOnKeyboard | 60 |
| ThrowAwayApple | 60 | ButterAboveRaisin | 40 |
| MustardInLeftBin / MustardInRightBin | 30 | MustardAboveRaisin | 40 |
| BowlInBin, CannedFoodInBin, CoffeePotInBin | 60 | PutBowlOnShelfTop | 60 |
| ClampInRightBin, FoodPacking1Boxes, FoodPacking1Cans | 60 | | |
| FruitsOnion, WoodSpatulaToBowl, PinkSpoonInPot, CookingPickPastaTool | 60 | | |
| MarkerInMug, SauceBottlesCrate | 40 | | |
| LargerObjectRaisinBoxInBin, SmallerObjectButterInBin | 30 | | |

The held-out tasks are never trained. **Validation:** BigPumpkinInBin and SmallPumpkinInBin. **Test:** SmartphoneInBin, SpoonInMug and ToyInBin. All are 60 s tasks in the same family.

**Known base rates.** These come from the sweep (10 episodes each), and from v0 online runs where noted.

| Task | Outcome | Group |
| --- | --- | --- |
| BananaInBowl | 9/10 | Saturated |
| BananaOnPlate | 10/10; 25/36 online | Saturated |
| MustardInRightBin | 9/10 | Saturated |
| YogurtInBowl | 0/10 (5 grasped); 11/104 online; 69/104 had a wrong grab | Learnable |
| ThrowAwayApple | 0/10 (10 grasped, 0 placed); 3/88 online | Learnable |
| FoodPacking1Cans | 0/16 online, no progress | Dead |

The other 25 tasks have no data, and I will not guess them.

**Screening epoch = training round 0** (settles Q3). There is no separate sweep. The first batch of each task (12 environments, default prompt, official horizon, random reset seeds) runs under the base checkpoint, which stays frozen until every task's batch is done. Its episodes are then reused:

- **Training data:** the selected training tasks' episodes become the first training collection. They are on-policy, because the old adapter is the base model.
- **Calibration data:** the captured dreams feed the calibration in §4.5.

Rules:

- **Screen the 26 train-split candidates only.** The 5 held-out tasks are not touched, because evaluation is on hold (§6.2). The existing sweep data for 5 of the 26 (BananaInBowl, BananaOnPlate, MustardInRightBin, YogurtInBowl, ThrowAwayApple) used the fixed authored layout. With randomization on they are re-screened. Wall time is at most about 5.4 h (§9.5).
- **No updates between tasks** during screening. Otherwise later tasks would be screened by an updated policy, and their base rates would be confounded.
- **Initial-pose randomization is on** (Q7 settled): `DREAM_INIT_POSE_NOISE=0.1`, which `dream_eval/run_pilot.sh` passes as RoboLab's `--init-pose-noise 0.1`. This is RoboLab's documented example value: each object group moves up to ±10 cm in x/y, and yaw defaults to ±60° (180° × min(1, 0.1/0.3)). RoboLab draws each layout collision-free from (seed, task, episode index). Log its per-environment `noise_scale`, which drops below 1 on crowded tables.
- **Seeds.** The batch seed fixes both the 12 layouts and the policy's sampling noise (request seed = crc32(seed:key)).
  - `default` batches use one fixed seed per task, reused every epoch. These are 12 tracked layouts, so epochs can be compared on identical starts.
  - `specific` batches draw a fresh seed every epoch, so training keeps seeing new layouts and does not fit only the tracked 12.
  - The screening epoch uses the tracked `default` seeds.
- **12 episodes is a coarse screen.** A task whose true held rate is 25% falls below 3/12 about 39% of the time. Borderline tasks can be re-screened in a later epoch.

**Selection rule**, applied after the screening epoch:

| Group | Rule | Use |
| --- | --- | --- |
| Train | ≥ 3/12 episodes reach held and ≤ 9/12 succeed | Trained |
| Retention | ≥ 10/12 succeed | Evaluated, not trained (see §1.6) |
| Dead | < 3/12 reach held | Excluded; every chunk would be neutral under $G$ |

**Train split** within the 31-task family:

| Role | Tasks | Count |
| --- | --- | ---: |
| Candidate pool (train split of `training/task_split.json`) | the 26 in the table above | 26 |
| **Trained** | up to 6 chosen from the pool by the selection rule after screening. Which ones is decided by the screening data, not assumed. | ≤ 6 |
| Pool tasks not chosen (dead, saturated, or over the cap) | not trained | ≥ 20 |
| Held out, never trained or screened | validation: BigPumpkinInBin, SmallPumpkinInBin. Test: SmartphoneInBin, SpoonInMug, ToyInBin. | 5 |

Only `default` and `specific` prompts are trained; `vague` is held out on all tasks.

Cap training at **6 tasks**. v0 measured 10.7–13.8 environment steps/s across both GPUs, which gives roughly 450–550 k steps in a 13 h run once update time is subtracted. That is about 100 episodes per task.


## 4. Video branch with its own reward

### 4.1 Same algorithm

$p^v_k=\tfrac12+\tfrac12\,\mathrm{clip}(G^v_k/G^v_{\max},-1,1)$ is applied to chunk $k$'s video loss, and $p^a_k$ to its action loss. Both losses pass through the shared GEN LoRA, so the dream reward's gradient also reaches action predictions. That coupling is the hypothesis under test.

Settled for now: $\gamma_v=0$, so $G^v_k=r^v_k$. A dream at chunk $k$ predicts only the next 32 steps. Later progress is already credited to actions through $G_k$, and later dreams receive their own rewards. $G^v_{\max}=1$, because $|r^v|\le1$ under both forms below.

### 4.2 What the dream reward is

Settled: both forms are implemented and chosen by configuration. V2 runs first.

The matched pair is dream frame $t_k$ and the real frame after $t_k$ executed steps ($t_k\le32$). On each frame the VLM answers *held(o)?* and *placed(o, c)?*. This gives a dream stage $D_k\in\{0,\tfrac12,1\}$, or an abstention when both answers are yes or the yes/no probability mass is low.

- **V1, imagined progress** (literal reading): $r^v_k=D_k-P_k$.
- **V2, realised imagined progress** (recommended):
  - If the dream matches reality ($D_k=P_{k+1}$): $r^v_k=D_k-P_k$.
  - Otherwise: $r^v_k=-|D_k-P_{k+1}|$.

On real frames, the judge's errors are mostly **false advances**: 23/71 false placement calls at stage 1, and 50/137 false "holding" calls before the grasp. V1 pays for exactly those errors. The cheapest way for the video head to gain reward would be ambiguous "almost placed" imagery, and nothing ties the dream to what the actions then do. V2 pays only for progress the simulator confirms, and penalises both hallucinated and missed progress. It combines "dream makes good progress" with the proposal's prediction–reality alignment in a single term. V1 remains a useful later ablation ("does an optimistic dream pull the actions?") once V2 works.

### 4.3 Judge gate and terminal frames

- **Judge inputs** (Q4 settled: keep the 4B judge; frame choice as designed here). Real and dream frames are always prepared identically, using the policy's camera composite.
  - *Held:* a single end frame, i.e. dream frame $t_k$ vs real frame $t_k$. Single frames already score AUROC 0.90 on this predicate.
  - *Placed:* a before/after pair, i.e. (dream frame 0, dream frame $t_k$) vs (real observation at $b_k$, real frame $t_k$). The question asks whether $o$ went from not-in to in-and-released. On real frames, pairs gave AUROC 0.80 (the score rose in 18/21 pairs), whereas single frames had 23/71 false advances.
  - Dream frames are decoded after collection from the captured latents, and only for the frames needed (0 and $t_k$). Rollout never decodes video.
- The same questions are asked on the matched real frame. If the VLM's reading of the real frame differs from $P_{k+1}$, the chunk abstains ($p^v=\tfrac12$, so the video loss keeps only its regulariser). If the judge cannot read the real scene at that state, its reading of the dream of that state is not trustworthy either, and the simulator tells us when that happens.
- **Required fix:** success ends mid-chunk, and v0 capture skips boundaries that have no live real frame. As a result, the placement rung has never been scored (`vlm_critic_eval` had zero placement positives at full boundaries). Capture the real frame at the termination step and compare it with dream frame $t_k$.

### 4.4 Can the VLM be reliable on a subset? Evidence so far

The judge is Qwen3-VL-4B-FP8, scored on real frames of single-object tasks.

| Predicate | Pooled | Per task | At threshold 0.5 |
| --- | --- | --- | --- |
| Held, once already held | AUROC 0.90 | Banana 0.98, Apple 0.92, Mustard 0.84, Yogurt 0.63 (n = 6) | 10/21 false positives |
| Held, before the first grasp | AUROC 0.80 | Yogurt 0.93, BananaOnPlate 0.90, Banana 0.76, Apple 0.78, **Mustard 0.49** | 50/137 false positives |
| Placed, single frame at stage 1 | — (no positives) | BananaInBowl 8/14, BananaOnPlate 8/13 false positives | 23/71 false positives |
| Placed, before/after pair | AUROC 0.80; score rose in 18/21 pairs | Bagels 0.56 | — |
| Within-stage approach (ψ) | Adjacent-boundary direction correct 60% (grasp), 18% (placement) | — | — |

For dream frames there is **no ground truth yet**. Real and dream judgments disagreed on 19/64 pairs. The judge saw the wrong object in 7/14 real frames but only 1/14 dreams. Neither number separates judge error from dream error.

**Answer:** on current evidence, the held rung is plausibly reliable on some objects (banana, apple, yogurt), but it depends on the object (mustard fails). Single-frame placement is not reliable on any measured task. Reliability on dream frames is unmeasured. So "reliable on a subset" has to be *measured on the chosen subset, on dream frames*, before the video branch is turned on.

### 4.5 Calibration (step 0c)

From the sweep dreams on candidate training tasks, label about 300 matched dream/real end-frame pairs. Stratify by task × true stage × whether a transition happened. Label each image as held, placed, or unreadable (**Q4**).

- **Gate, per task and per predicate, measured on dream frames:** AUROC ≥ 0.85, and a false-advance rate ≤ 15% at the chosen threshold.
- A predicate that fails is dropped for that task. If placement fails, for example, $D\in\{0,\tfrac12\}$ and placement chunks abstain.
- A task where both predicates fail gets no video reward ($p^v=\tfrac12$, regulariser only). This also feeds back into which tasks are chosen for training.
- Calibration compares the single-frame and pair prompts for placement on dream frames, and keeps the better one per task.

## 5. Wrong-grasp penalty

**Choice:** keep it out of the reward in every arm and log it.

- **Outside the research question.** The proposal asks (1) whether predicted states can provide intermediate reward and (2) whether better prediction–reality alignment improves actions. A wrong-object penalty is neither. It is privileged simulator information (which object the gripper touched) added as shaping. It is a performance trick that strengthens the baseline.
- **Mostly covered already.** Under §1.2 a wrong grasp earns $P=0$ and costs time through $\gamma$. Only the explicit negative is missing.
- **Where it does fit is as a diagnostic for (2).** A wrong grasp is the clearest dream–reality divergence: the dream may show the target held while the robot holds a distractor. V2 penalises exactly this case ($D=\tfrac12$ vs $P=0$) through the proposal's own mechanism. Log the wrong-grasp rate per arm, and the dream/real disagreement at wrong-grasp events.
- If you want it for performance (it is the dominant Yogurt failure), add it equally to all arms, never to only one.

## 6. Experiment arms and evaluation

### 6.1 Arms

All arms use matched tasks, seeds and environment steps. The arms differ only in the video preference; the action loss always uses $p^a$ from $G$.

| Arm | Video preference | Question it answers |
| --- | --- | --- |
| **T** (task only) | $p^v=\tfrac12$ on every chunk: regulariser only (anchored, no preference) | Baseline: the only reward in the system is the task return on actions |
| **T+V2** | $p^v$ from V2 | Does realised dream progress (alignment) improve actions? |
| **T+V1** | $p^v$ from V1 | Does an optimistic dream pull the actions? Run after T+V2. |
| **T+J** (optional) | $p^v=p^a$ | Is a dedicated dream reward better than reinforcing the dream along with its action? |

**Q5 — what the video loss gets in the task-only arm:**

| Option | Pros | Cons |
| --- | --- | --- |
| **(a) $p^v=p^a$** (joint) | Treats (video, actions) as one policy sample, which is the natural NFT use. The dream stays the plan that came with the reinforced action. Same as v0. | It is itself a dream reward: the return, including *future* chunks, applied to a 32-step prediction. T vs T+V2 would then differ in two things (task signal removed from video *and* dream reward added), so attribution is unclear. It rewards dreams of good-return chunks whether or not they were accurate. |
| **(b) Video loss masked** | Purest action-only RL | Nothing anchors the video. Action gradients move the shared GEN LoRA, so the dream drifts with no trust region or reference term. Because sampling is joint, that drift feeds back into actions. Masking saves no forward compute either, since the pass is joint. |
| **(c) $p^v=\tfrac12$, regulariser kept** (recommended) | The video stays near $v_{\mathrm{old}}$/$v_{\mathrm{ref}}$ with no preference. T and T+V2 differ in exactly one thing, the dream reward. It matches the abstain/neutral handling inside T+V2. | Arguably a weaker baseline than (a): the video cannot improve in T. If T+V2 wins, a reviewer may ask whether (a) would also have won. T+J answers that. |

**Settled: (c)** for the primary baseline. Order the arms T, then T+V2, then T+V1, and add T+J only if T+V2 beats T. One seed per arm first; a second seed only for a pair that differs.

### 6.2 Evaluation: on hold

On hold until you have reviewed the next training result. Settled so far: hold out `vague` (never trained) and use no separate base evaluation for training tasks. Until then, the only performance numbers come from training rollouts: epoch-to-epoch success and $P$ on the tracked `default` seeds, with the screening epoch as epoch 0. Dedicated evaluation rollouts (held-out `vague`, held-out tasks, retention) and their base references are designed once you decide.

### 6.3 Logging

- Histograms of $p^a$ and $p^v$, the neutral share, and the clip rate (which must be 0).
- Grasp→drop cycles and the wrong-grasp rate.
- Judge abstention rate, and dream/real stage agreement.

## 7. Code changes in `./training`

- `native_progress.py`: per boundary, record the target's held flag (contact plus lift above reset height, debounced over 8 steps) and the official success flag. Also capture the real frame at the termination step.
- `reward.py`: replace the critic, GAE and EMA with $P$, $r$, the Monte Carlo $G$, $G_{\max}=2-P_k$, and $p^a$. Add V1 and V2 $r^v$ (selected by configuration) and $p^v$ with the judge gate.
- `online.py` / `curriculum.py`: add the screening epoch (frozen base, one batch per task, no updates, then selection). Remove the progress gate, the value table and administrative truncation (drop unfinished episodes instead). Skip the update when every chunk is neutral. Train on `default` and `specific` only.
- `cosmos_nft.py` / `native_update.py`: give each chunk separate $p^a$ and $p^v$. Mask the action loss at $p^a=\tfrac12$; keep the video loss at $p^v=\tfrac12$ as regulariser only. The arm setting chooses the video preference source.
- The VLM judge (Qwen3-VL-4B-FP8, unchanged) runs after collection on the freed GPU: 2 questions × 2 frames per chunk.
- Fixed-seed evaluation runner for the blocks in §6.2.

## 8. Order of work

1. **0a.** Round 0 under the frozen base checkpoint, with dream capture:
   - Screening: 26 tasks, `default`, fixed seeds (about 5 h). Select the tasks.
   - Then 6 `specific` batches with fresh seeds (training data), about 1 h. No evaluation batches (§6.2 is on hold).
2. **0b.** Offline replay of the new reward on the screening data and the v0 traces. Produce $P$/$G$/$p$ histograms, compare held (with lift) against RoboLab's contact-only grab counts, and confirm a clip rate of 0 and the neutral share.
3. **0c.** Dream-frame calibration (§4.5).
4. **0d.** Q8 co-run test (§9.3), run 2026-11-23.
5. **1.** Short paired T vs T+V2 run on 2 training tasks. Check signs, masks and adapter drift.
6. **2.** 13 h runs in the arm order of §6.1. Evaluation waits for your review.

## 9. Using both GPUs, and the update batch

### 9.1 What v0 did and where time went

- **Rollout.** Each GPU hosted one rollout lane: RoboLab with 12 environments plus a resident Cosmos3 policy server.
- **Update.** After both lanes drained, a single-GPU actor ran on `gpus[0]` while GPU 1 sat idle.
- **Timing per round.** Collection took about 2,140 s and the update 500–690 s, for 1,050–1,420 chunks at microbatch 1 and accumulation 4, which is about 190–250 optimizer steps. So about 25% of wall time was update-only, half the hardware was idle during it, and the GPUs were lightly used even during rollout. Benchmarks showed about 12% on the policy GPU and 32% on the simulator; RoboLab stepping dominates.
- **Machine.** 2× H800 with 80 GB each, 96 CPU cores, 251 GB RAM.

### 9.2 Layout: both GPUs symmetric, everything resident, phases exclusive

Each GPU keeps these processes loaded, but only one phase runs at a time:

1. **Rollout lane** (as v0): RoboLab with 12 environments and the Cosmos3 policy server. About 29 GB.
2. **Actor rank:** the DDP world size is 2, one rank per GPU. It loads the base weights once (the reference policy is the base with LoRA disabled), plus the current LoRA and the frozen "old" LoRA, which is the adapter that collected the data. About 19 GB at microbatch 1.
3. **Dream decoder:** VAE decode of only frames 0 and $t_k$ from the captured latents, after the collection closes.
4. **Judge** (all arms, one GPU only; see §9.6): resident vLLM Qwen3-VL-4B-FP8 with a small memory fraction. About 1,000 chunks × 2 frames × 2 predicates is only a few thousand short queries, so it takes minutes.

### 9.3 Schedule

```text
collect C_n with θ_n on both lanes → decode dream frames + judge → DDP update on both GPUs (old = θ_n) → publish θ_{n+1} to both policy servers
```

Originally proposed: overlap the update with the next collection, with a one-collection lag. The Q8 test rejected it.

- **Q8 test, 2026-11-23** (YogurtInBowl, 12 envs, 256 steps, `--init-pose-noise 0.1`, same seed on every lane; logs in `/tmp/q8_out`, scripts in `/tmp/q8_test`):

| Configuration | Per-lane sim step (12 envs) | Lane env-steps/s | Notes |
| --- | ---: | ---: | --- |
| One lane per GPU, no actor (run 2) | ~2.0 s | 6.07 (GPU 0) / 5.86 (GPU 1) | ~29 GB and ~40% GPU utilization per lane |
| One lane per GPU, no actor (run 3, GPU 0) | ~1.8 s | 6.82 | |
| Lane + looping actor on the same GPU (run 3, GPU 1) | **13.4 s** | **~1.26 (−80%)** | 49 GB total. The actor kept its pace: 480 chunk-σ in 293 s = 1.64/s, about −14% vs the 1.9/s benchmark. |
| Two lanes on one GPU, other GPU free (run 4) | ~5.0 s each | 2.4 each, **4.8 combined** | Less than one lane alone |

**Decision (Q8 settled):**

- **No co-running.** The actor saturates the GPU and the policy server's requests queue behind it.
- **No dedicated rollout GPU.** Two lanes on one GPU are slower in total than one lane, so the rollout bottleneck is GPU-side, not just the CPU simulator.
- **Use phase-exclusive scheduling.** Collect with one lane per GPU, then update with DDP on both GPUs (world size 2), then publish the adapter to both policy servers. Policy servers and actor ranks stay resident (about 29 + 19 GB per GPU) but never run at the same time.
- §9.4 keeps 4 σ per chunk. If the update exceeds ~25% of wall time, drop to 2 σ.

### 9.4 Update batch

| Item | Choice | Why |
| --- | --- | --- |
| Collection | 4 batches (2 per GPU) × 12 envs = 48 episodes, covering 4 (task, level) pairs. 12 training pairs → 3 collections per epoch, rotated so consecutive collections differ. | v0 size, about 35 min. One task per RoboLab process. |
| Chunks in the update | Those with $p^a\ne\tfrac12$, or $p^v\ne\tfrac12$ in T+V arms | Neutral chunks carry no preference (§1.5) |
| Noise levels | All 4 captured $\sigma$ per chunk (as DiffusionNFT does with `timestep_fraction=1`) | Data is the expensive side: a chunk costs about 2.7 s of the whole system at about 12 env-steps/s, while one chunk-σ costs about 0.5 s of one GPU. v0 used only one σ per chunk. |
| Optimizer step | 16 chunks × 4 σ = 64 samples. Per rank: 8 chunks, microbatch 1, accumulate 32. All 4 σ of a chunk go in the same step. | With one sample per state and no baseline, a single gradient is mostly noise (§1.6). The useful signal is the covariance of $p$ with $v$ *across* states, which needs many states per step. 64 vs v0's 4. |
| Order | Shuffle over the whole collection, so each step mixes tasks, episodes and signs | Avoids steps dominated by one episode or task |
| Passes | 1 (each chunk-σ used once) | NFT targets are fixed per sample; reusing them fits noise and drifts further from the reference |
| Steps per collection | About 25–40, given 400–600 non-neutral chunks (to be confirmed by replay 0b) | Down from v0's 190–250 small steps |
| Loss normalisation | Per chunk: token mean per modality. Per step: mean over the chunks contributing to that modality. | Action-masked chunks don't dilute the action loss |
| Learning rate | 1e-4 (v0) for the smoke test; adjust using the per-step $\lVert v_\theta-v_{\mathrm{old}}\rVert$ and $\lVert v_\theta-v_{\mathrm{ref}}\rVert$ | Larger batches and fewer steps move the adapter less per collection than v0 did |

**Update time estimate.** 400–1,000 chunks × 4 σ at about 1.9 chunk-σ/s per GPU (rank-16 benchmark, no checkpointing) on 2 GPUs is about 7–18 min standalone. That fits inside a 35 min collection even with a 2× co-running slowdown.

### 9.5 Time per epoch (estimate)

**Basis.** A lane takes about 2.0 s per simulator step with 12 active environments (measured above). A batch runs until its slowest episode ends. Training tasks have failures by construction, so a batch costs about horizon steps × 2.0 s: **15 / 20 / 25 / 30 min** for 30 / 40 / 50 / 60 s tasks. Episodes that end early only shrink the policy batch, so this is an upper bound.

**Epoch** = 6 tasks × {`default`, `specific`} = 12 batches = 3 collections of 4 batches (2 per lane).

- **Rollout:** 3 collections × 2 batches per lane × batch time = **2.0 h** (all 40 s tasks) to **3.0 h** (all 60 s tasks).
- **Update:** per collection, 48 episodes × (19–28 chunks) × non-neutral fraction $f$ × 4 σ ÷ 3.8 chunk-σ/s on two GPUs gives 16$f$–24$f$ min. At $f=0.5$ that is **0.4–0.6 h per epoch**; $f$ is measured in replay 0b.
- **Judge** (all arms; see §9.6): a few minutes per collection.
- **Total: about 2.5–3.7 h per epoch, so 3.5–5 epochs (500–700 episodes) in a 13 h run.** The exact figure depends on which tasks screening picks; their horizons dominate.

**One-off costs:**

- Screening the 26 candidates: Σ horizons = 4×450 + 7×600 + 750 + 14×900 = 19,350 steps × 2.0 s ÷ 2 lanes ≈ **5.4 h** upper bound.
- 6 `specific` round-0 batches: about **0.75–1.5 h**.

### 9.6 Analysis counters (small, always saved)

**Disk.** `/home` has 125 GB free (93% used); `/` has 429 GB. The NFT latents cost about **1.6 MB per chunk** (measured: 307 MB per 193 chunks), i.e. 1.5–2 GB per collection. Policy:

- Latents go to scratch on `/` and are deleted once the update that consumes them finishes; only the latest collection is kept.
- Sensor video stays off.
- RoboLab's per-batch HDF5 (about 9 MB) is kept only until the per-boundary geometry below has been extracted, then deleted.
- Everything below is a few hundred KB per collection, stored as Parquet: one file per collection plus one per update.

| Table | One row per | Fields |
| --- | --- | --- |
| `chunks` | executed chunk | **ids:** arm, epoch, round, policy hash, task, level, seed, env, episode, k, start step, executed rows, end reason. **Simulator** at $b_k$ and $b_{k+1}$: P, contact, lift above reset, held flag, wrong-object-held id, end-effector–target and target–receptacle distances (within-stage progress without a VLM). **Return:** ΔP, S, r, G, $G_{\max}$, $p^a$. **Dream:** VLM P(yes) for held and placed on dream and real (single and pair prompts), D, real-frame stage, gate pass, $r^v$ under **both** V1 and V2, and the $p^v$ actually used. **Action descriptors:** mean abs joint delta, gripper flips. |
| `episodes` | episode | success, final P, steps, layout `noise_scale`, time to first hold, # grasps / drops / wrong grabs / grasp→drop cycles |
| `update_samples` | chunk × σ used in an update | σ, preference term, trust-region term and reference term per modality, from the forward pass already computed |
| `steps` | optimizer step | lr, chunk count, loss components per modality, grad norm (total, heads, GEN blocks), $\lVert v_\theta-v_{\mathrm{old}}\rVert$ and $\lVert v_\theta-v_{\mathrm{ref}}\rVert$ per modality, adapter norm |
| `rounds` | collection | wall time (rollout, judge, update), env steps, peak GPU memory, episodes and successes, neutral share, counts of $p>\tfrac12$ and $p<\tfrac12$, abstain rate, clip count (must be 0) |

**Run the judge in every arm, including T.** Its output trains nothing in T and V1/V2-off arms, but dream–reality agreement must be measured the same way in all arms to analyze the mechanism. It costs minutes per collection.

**Saved frames.** A random 10% of judged (dream, real) end-frame pairs are saved as 160 px JPEGs, about 2 MB per collection. They cost little and are exactly the material the Q4 calibration labelling needs.

**Checkpoints:** adapter + optimizer at every epoch end, plus the last 3 rounds (about 60 MB each).

## 10. Decisions log

| # | Question | Status |
| --- | --- | --- |
| Q1 | Dream reward V1 or V2 | Both kept as options; V2 runs first |
| Q2 | Dream return horizon | $\gamma_v=0$ for now |
| Q3 | Separate sweep or first rollouts | First rollout batch of each task under the frozen base checkpoint, reused as training round 0. No base evaluation for training tasks: fixed seeds per (task, level) and comparison between epochs. |
| Q4 | Judge | Keep Qwen3-VL-4B-FP8. Frame design in §4.3: single frame for held, before/after pair for placed, terminal frame captured. |
| Q5 | Video loss in the task-only arm | (c): $p^v=\tfrac12$, regulariser only |
| Q6 | Prompt levels | Hold out `vague`. All evaluation is on hold until you review the next training result. |
| Q7 | Initial-pose randomization | On: `--init-pose-noise 0.1`. Tracked fixed seeds for `default`, fresh seeds for `specific`. Screen all 31 tasks. |
| Q8 | Co-running the actor with rollouts | Tested 2026-11-23: co-running cuts the rollout by 80%, and two lanes on one GPU are slower than one. Phase-exclusive: collect on both GPUs, then DDP update on both GPUs. |
| — | Train split | Up to 6 tasks chosen by screening from the 26 train-split candidates; 5 held out (§3) |
| — | Analysis counters | §9.6 |
