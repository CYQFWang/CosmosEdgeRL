# train_v2 — dream–reality alignment arms

Status: 2026-11-24. Builds on `train_v1.md` (framework, reward, return, G_max, task family) and the v1 run `training/runs/v1_T_01`. This document records what v1 showed, what we changed, and the three matched arms.

## 1. What v1 showed

| Finding | Evidence |
| --- | --- |
| Task-only NFT (arm T) learned little on its own tasks | Tracked `default` layouts, paired change in success vs base: +0.07 / +0.11 / +0.11 / +0.03 at epochs 1–4. The CIs all touch 0 (n = 72 per epoch). |
| No transfer, no forgetting | 10 untrained tasks, `e02c2` vs base: Δsuccess −0.008 [−0.067, +0.050] (n = 120). |
| The policy barely moved | Distance from base flat at ~0.04 over 13 updates. Paired trajectories diverge, but outcomes do not shift. |
| Raw G mostly encodes state value | Start-state features explain 46% of Var(G). Raw p^a is 43% positive and 1.5% negative; after a state baseline, 28% / 37%. Per-state contrast is rare anyway, because most chunks have ΔP = 0 for any sample. |
| Learning rate is not the lever | Offline probe on 256 chunks. 1e-4 improves held-out preferred chunks (≈ −0.6%) with drift < 1%. 3e-4 reverses by 16 steps, and ≥ 1e-3 worsens held-out fit while drift grows (+4.5%, +12.7%). Run-to-run noise is about ±0.5%. Extra passes at 1e-4 change nothing measurable and do no harm. |
| VLM dream judging does not work in this setup | Real-frame state accuracy is at most 70–75%. "Plausible" was answered yes for 40/40 dreams and "closer" for 18/20 real pairs: the model answers from the task prior. |
| What the dream does and does not carry | It does not show contact events: release was missing in all 5 placement chunks, even when the jointly sampled action opened the gripper. Frames after t ≈ 20 degrade (blur, melting textures, gripper–object fusion). The wrist view cannot resolve target identity or the correct bin. The first half (frames 0–16) is sharp and predicts the executed motion. |
| Sharp-half alignment is a usable, label-free signal | On the 21 Nov pilot (307 chunks): side-view SSIM of dream frame 16 vs the real frame at step 16. Separates a chunk's own dream from other chunks' dreams (AUROC 0.98–0.99). Beats "nothing moves" in 81–87% of chunks. Beats a counterfactual dream from the *same state* (resampled video noise) in 83–92% (n = 24), so the dream carries its own action's consequences. It is lower at wrong grabs (gain −0.006 vs +0.059, n = 9). |

## 2. Decisions

| Item | v2 choice |
| --- | --- |
| VLM judge | Off. No reliable signal; decoding is kept for alignment. |
| Dream signal | Alignment gain on the sharp half (§3), current chunk only (γ = 0) |
| Arms | **T′** (task only, rerun), **A-video** (gain → video tokens), **A-action** (gain added to the action preference) |
| Training pairs | 8 (task, level) pairs, selected per pair (rule: base success 2–9/12, or held ≥ 3/12). BowlInBin `default`; ButterAboveRaisin both levels; FruitsOnion both levels; MustardInLeftBin `default`; SmallerObjectButterInBin both levels. Dropped as saturated at base: BowlInBin, MustardInLeftBin and MustardInRightBin `specific` (11–12/12), MustardInRightBin `default` (9/12). `default` reuses the screening seeds (tracked layouts, base reference on disk); `specific` uses fresh seeds per epoch; `vague` is held out. |
| Start | Base policy, straight into epochs (no screening, no separate round 0) |
| Learning rate | 1e-4, chosen by the offline probe |
| Passes per collection | 1 (no measured benefit from a second pass; A-video/A-action updates would otherwise take as long as the rollout) |
| Everything else | As v1: NFT β = 0.1, reference weight 0.01, rank-16 LoRA on heads + all GEN blocks, 16 chunks × all σ per step, DDP on both GPUs, phase-exclusive scheduling, randomization 0.1 |

## 3. Alignment signal

For chunk k with at least 16 executed steps:

$$
a_k = \mathrm{SSIM}\big(\hat x^{\,dream}_{16},\, x^{real}_{16}\big) - \mathrm{SSIM}\big(x^{real}_{0},\, x^{real}_{16}\big)
$$

- **Inputs:** the static side-camera band of the policy composite, rows 360–528, grayscale, area-downsampled to 80×21.
- **Geometry:** the policy generates the top 528 of the composite's 540 rows. This is a crop, not a resize (measured: MSE 22.6 vs 1,229 for a resize), so real frames are cropped identically.
- **What it rewards:** subtracting the "nothing moves" baseline rewards a correctly predicted *change*. A static plan scores about 0.
- **Distribution on the pilot (side view, n = 307):** mean +0.032, std 0.041, range −0.044 to +0.193, 19% negative.
- **Bound:** B = 0.2, just above the observed maximum, so the clip practically never fires (the G_max principle). Clip counts are logged per collection.
- **Missing values:** chunks with fewer than 16 executed steps have no a_k and stay neutral on the alignment term.
- **Data needed:** the real frame at step 16, saved by the runner (`DREAM_ENDPOINT_ONLY=0`), and dream frame 16, decoded by the actor ranks after collection.
- **Code:** `training/align.py` is identical to the tested metric in `dream_eval/alignment/test_alignment.py`.

## 4. Arms

All three use the same task return G_k, G_max = 2 − P_k, and p^a = ½ + ½ clip(G_k / G_max) unless stated.

| Arm | Action tokens | Video tokens | Question |
| --- | --- | --- | --- |
| **T′** | p^a | ½ (regulariser only) | Baseline under v2 settings |
| **A-video** | p^a | p^v = ½ + ½ clip(a_k / B) | Does a more accurate world model (the dream learns what the actions do) improve actions through the shared weights? |
| **A-action** | p^a = clip(p^a_T + c·clip(a_k/B, −1, 1), 0, 1), with c = 0.1 | ½ | Does rewarding actions for carrying out their own dream (plan-following) improve task success? |

- **A-action adds alignment after the task preference, in probability units.** The task term is exactly T′'s, so a difference is attributable to alignment. (The earlier form, (G + λa/B)/(G_max + λ), was rejected: λ also sat in the denominator and shrank the task signal.)
  - c = 0.1 is the largest shift alignment can cause, close to one grasp's effect (a grasp chunk sits around p^a ≈ 0.625).
  - A typical gain (≈ 0.04, i.e. a/B ≈ 0.2) shifts p^a by about ±0.02.
  - The clip fires only when p^a_T > 0.9 and the gain is positive. It is logged as `p_action_clip`.
- A chunk enters the update if p^a ≠ ½ or p^v ≠ ½. In A-video almost every chunk with ≥ 16 steps does, so its updates take longer than T′'s.
- Alignment is decoded and logged in **all** arms, including T′, so the mechanism (does alignment change?) is measured the same way everywhere.

**Risks and what to monitor:**
- **A-video, blur:** SSIM is less blur-seeking than pixel error, but watch for own-SSIM rising while the gain stays flat.
- **A-action, predictable motion:** the "nothing moves" baseline removes the trivial route; watch success, not alignment alone.

## 5. Run protocol and budget

- **Per arm:** 3 epochs (`max_epochs` = 3; equal environment data per arm, not equal hours), from the base policy, with the same pairs, seeds and schedule. Epoch checkpoints are kept.
- **Each epoch:** 8 batches = 2 collections of 4.
- **ETA:** A-video and A-action ≈ 1.9 h per epoch (≈ 5.7 h per arm); T′ ≈ 1.7 h per epoch.
- **Order:** A-video → A-action → T′, run by `training/runs/v2_queue.sh` (log: `v2_queue.log`). The queue stops at the first arm that doesn't finish cleanly; rerunning it skips finished arms.
- **Run directories:** `training/runs/v2_AV_01`, `v2_AA_01`, `v2_T_01`, each with `start.sh`, `stop.sh`, `check.sh`.
- **Per-collection counters (v1 counters plus):**
  - per chunk: `align_gain`, `align_own`, `align_copy`, `p_action_used`;
  - per collection: `align_coverage`, `align_mean`, `align_clip`.
- **Evaluation:** same as v1, with no new base runs needed. Tracked `default` layouts are paired with the screening base per epoch, and the untrained-task set is paired with screening (`training/v1_eval.py --pair-screening`) for each arm's chosen checkpoint.

## 6. Implementation (on the v1 code)

| File | Change |
| --- | --- |
| `training/v1_run.py` | `start: "train"` with `selected` (skips screening and round 0); `real16`; `alignment`, `alignment_bound`; arms `AV` / `AA` (`alignment_weight` = λ); `judge` optional; `passes` |
| `training/v1_actor.py` | `passes` (reshuffled each pass) |
| `training/align.py` | Alignment metric |
| `training/lr_probe.py` | Offline learning-rate / passes probe |
| `dream_eval/alignment/test_alignment.py` | Offline metric validation (21 Nov pilot) |
| `dream_eval/plausibility/*` | VLM pilots (kept for the record) |

## 7. Status

All design items decided (2026-11-24). A-video started 05:46. A-action uses c = 0.1. T′ follows.
