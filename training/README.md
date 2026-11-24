# Cosmos3 training review

This document is not latest and some designs have been updated. 

## Implemented for review

- `reward.py`: simulator stage delta, success bonus, and wrong-grab penalty, with draft weights 1/2/1. Per-task stage critic uses bootstrapped discounted returns and an EMA. Score complete episodes against frozen tables before updating those tables. GAE uses gamma 0.95 and lambda 0.8; success zeros the bootstrap, timeout retains it.
- The visual critic only consumes discrete current-grasp and next-subgoal predicates. The same question bank is used on dream and real frames. For single-object tasks it maps to stage 0/1/2; stage disagreement is penalized in value units on video only. Missing or partial-chunk comparisons contribute no honesty signal. Keep the simulator stage authoritative for real advantage.
- `actor_batch.py`: prototype verl DataProto packing for precomputed GAE and separate video/action NFT reward probabilities and executed-token masks. It is not a registered engine or a training launcher. Confirm latent packing and sigma values against the actual model, not the smoke shapes. Draft honesty weight is 0.25; calibrate it on held-out real frames before enabling.

## Model wiring

The workspace includes opt-in native rollout capture (`COSMOS_CAPTURE_NFT=1` with `dream_eval/run_pilot.sh`) and a workspace-local loss registered as `cosmos3_edge_nft` when `training.cosmos_nft` is imported. Each `*_nft.pt` file stores joint latents before action postprocessing, the true schedule, and the conditional/unconditional inputs and masks. The native smoke loads the same vLLM-Omni Edge checkpoint, accumulates one update on rank-4 adapters on the two output projections, and loads the resulting checkpoint in a fresh OpenPI serving worker for another RoboLab rollout.

Following verl-omni's direct-preference integration guide, the target runtime uses `trainer_type=direct_preference`, `sample_source=online`, `paired_preference=false`, and no rollout logprobs. Remaining integration work is config validation, FSDP engine registration and dispatch, importing the registered loss on Ray workers, running the closed-loop RoboLab collector before batch preparation, and syncing the old adapter into the vLLM-Omni worker. The workspace code is not a drop-in trainer launch config.

1. Preserve policy version and executed action count alongside the native rollout trace when extending collection across iterations. Pre-NFT pilot artifacts only contain decoded RGB and postprocessed commands and cannot be used as NFT samples. Export debounced per-object state from RoboLab, including terminal steps, and pair observations to the request ID.
2. Implement a registered Cosmos3 video/action pipeline and loss, using the existing verl-omni LTX2 OmniNFT pipeline as the engine structure and the Cosmos Framework `Cosmos3FlowPolicy` NFT velocity path as the reference for local conditioning and old/current/reference forwards. Verify checkpoint and configuration compatibility. The stock OmniNFT batch builder in `verl_omni/trainer/diffusion/diffusion_algos.py` hard-codes video/audio and centers reward across prompt groups, which zeros singleton rewards. Use a dedicated registered path consuming precomputed temporal GAE. Do not encode actions as audio or pass GAE through group centering.
3. Train GEN LoRA with UND frozen; keep the actual policy mask, sampler CFG, and four deployed sigmas. Compute masked modality MSE separately and normalize by valid elements per modality. Keep the base reference frozen and use a versioned EMA adapter for sampling and old velocity. Prove latent and action parity at a fixed seed, then run a forward/backward and save/reload test. Verify masked elements and video-only honesty routing independently before a live update.

## Joint dream and action training

Cosmos3 Edge has a text-only understanding tower (UND) and a shared 28-layer generation stack (GEN). UND supplies text keys and values. GEN processes video and action tokens together, then splits them into separate velocity predictions through the video projection (`proj_out`) and the domain-aware action projection (`action_proj_out`).

This creates two routes from dream training to actions. With only output adapters, the dream loss cannot directly update action weights, but a changed video velocity changes the latents that both streams see at the next sampling step. With trainable shared GEN adapters, the dream loss can also update weights used to predict action velocities. Freezing the action head does not freeze the action policy.

Freeze UND, the VAE, the VLM judge, and the base reference. For adapter runs, also freeze the base GEN weights, input projections, timestep embeddings, and norms initially. Keep the old policy snapshot fixed within each rollout batch.

| Option | Trainable parameters | What it tests |
| --- | --- | --- |
| 1. Action head only | Adapter on `action_proj_out` | Simulator-reward baseline without dream supervision. Dreams can still change through joint sampling. |
| 2. Both output heads | Adapters on `proj_out` and `action_proj_out` | Cheapest joint option; separates direct gradients while preserving sampler coupling. The rank-4 smoke used [17,408 trainable parameters](head_adapters.py). |
| 3. Both heads + late GEN LoRA | Output adapters plus LoRA in the last four GEN blocks | Tests whether changing shared representations helps; profile native backward on one H800 first. |
| 4. Both heads + all GEN LoRA | Output adapters plus LoRA across all 28 GEN blocks | More capacity if late-block LoRA plateaus. Fewer trainable weights do not eliminate activation memory. |
| 5. Full GEN fine-tuning | GEN blocks and modality projections, with UND and VAE still frozen | Highest memory and optimizer cost, and the greatest risk of changing pretrained behavior. Defer on two GPUs. |

**First comparison:** Start from the same checkpoint and train both heads at matched simulator steps, with the same simulator-derived advantage on both streams. Add the discrete, stage-weighted dream-reality mismatch penalty to video only in the treatment arm. Include a video-head-only mismatch control: train only the video adapter on mismatch and leave the action head and shared stack frozen. Action changes in that control can come through joint sampling without a direct action-weight update.

Measure simulator success, grasp retention, and wrong-object grabs alongside dream agreement. Better agreement alone can mean the model predicts its failures more accurately, not that the policy improves.

## Run order and two-GPU budget

1. Dry run the simulator critic on the full logged cohort. Test per-object debounce, regressions, final completion, wrong-object counts, and timeout bootstrap against representative videos. Export per-task reward signs, advantages and sample counts. Finish the multi-object parser before including those tasks; the discrete visual judge currently covers single-object two-stage tasks only.
2. Run a one-task, four-environment collection and one GPU update with the Banana control. Pin RoboLab to GPU 0 and Cosmos3 to GPU 1. For training, stop or offload the rollout process before loading the actor on GPU 1; start with one GPU, microbatch 1 and gradient accumulation. Time collection, judge, update and weight sync and record peak memory. Only consider two-GPU actor sharding in a separate update phase after releasing the simulator if single-GPU memory is insufficient.
3. Do a short T0 seed on learnable tasks with different outcomes. Freeze the critic per rollout batch, score complete episodes, normalize advantage per task on that collection, update the actor, then update the per-task EMA tables and synchronize the next policy snapshot. Run baseline evaluation with fixed held-out seeds and no exploration protocol changes. Monitor success, stage reach, regressions, wrong grabs, loss by modality, and clipped reward probability fraction.
4. Compare T0 against R1 at matched simulator steps and two seeds. Add the harder tasks only when collection shows useful within-task contrast. For R1 run the judge on paired real/dream frames after collection; initially use one GPU in a separate phase and omit unclear pairs from honesty. Check real-frame calibration by task and grasp/placement stage. Keep Qwen out of the environment reward. Reserve T with a discrete approach critic and R3 with same-state candidate ranking for follow-up ablations; neither is implemented.
5. Budget the proposed 13-hour runs using measured throughput; report both environment steps and wall time. Checkpoint adapter, old adapter, optimizer, critic tables, policy version, and RNG state. Evaluate fixed reset seeds on train tasks, saturated retention tasks, and unseen tasks/instruction levels. Separate learning runs from evaluation and report paired seed deltas.

## Timed native smoke

From the workspace root, run with an unused absolute `OUT` directory and set `SIM_GPU` and `POLICY_GPU` to distinct H800 UUIDs:

```bash
bash training/run_native_smoke.sh
```

It collects two Banana episodes capped at 128 steps, scores paired frame-32 images with Qwen on the freed simulator GPU, runs one optimizer step with four native microbatches on the policy GPU, and collects a 32-step rollout after loading the checkpoint. It disables sensor MP4 recording and frame-16/event scoring; the current serving hook still decodes and saves the full dream array. See `timing.json`, `rollout/update_metrics.json`, and `rollout/rollouts/episode_results.jsonl` for wall times and RoboLab's timing breakdown.

This measures a cold, sequential iteration through the registered verl loss, not steady-state Ray/FSDP training. The initial base policy is both old and reference, the stage values are zero, and the smoke selects four higher-magnitude-advantage chunks. The head-only update is not a proxy for full GEN adapter training cost. The next throughput changes to benchmark are keeping Qwen and the policy resident, sending only needed decoded endpoints, and swapping the trained adapter without restarting serving. Establish a fresh policy-version boundary before collecting the next on-policy batch.

`python -m training.smoke` in the verl-omni environment checks the CPU batch contract.

## Benchmark split and option 4

[`task_split.json`](task_split.json) records 96 train, 12 validation, and 12 test tasks from RoboLab's 120-task benchmark. Every task keeps its official vague, default, and specific prompts in the same split (288/36/36 task-prompt pairs). Scenes and normalized exact instruction strings are disjoint across splits. Previous pilot scenes stay in train. Regenerate with `python -m training.task_split`; validate with `python -m training.audit_smoke`. Select one prompt level per rollout using `DREAM_INSTRUCTION`.

The stock vLLM-Omni sampler resets the transformer UND cache per request and reuses it inside that request. `COSMOS_UND_CACHE=8` enables a bounded worker-local cache across episode, environment, and chunk IDs. Conditional and unconditional token sequences have separate entries when their inputs differ. It keys on exact formatted input tokens, text RoPE frequencies, and device, never raw instructions or episode IDs. GEN RoPE remains request-local. Do not use the cache while updating or swapping UND; GEN-only adapters leave it valid. A full two-episode rollout observed 22 hits, two misses, and an exact match on a fresh UND K/V recomputation. The two cached entries occupied about 19.5 MiB. These are cache-reuse measurements, not a controlled rollout speedup estimate.

Use the same base rollout traces and Qwen labels for the initial comparison. Enable `COSMOS_CAPTURE_NFT=1`, score paired frame-32 endpoints with `JUDGE_ABSTAIN=1`, then run:

```bash
CONTROL_GPU=<uuid> MISMATCH_GPU=<uuid> \
  bash training/run_option4_pair.sh <scored-rollout> <fresh-output>
python -m training.audit_pair <fresh-output>
```

Option 4 adds rank-4 LoRA to both heads and the MLP down projection in every one of 28 GEN blocks. Base weights stay frozen. The native worker uses checkpointed GEN blocks and PyTorch SDPA for backward. The measured run used all 12 chunks from 346 simulator steps, one matched batch from two complete Banana episodes. Both arms completed three optimizer steps in 7.6 s each, after 5.8 to 6.0 s of model loading. Every adapter group had nonzero finite gradients. Peak PyTorch allocation was 8.25 GiB per H800; NVML reported 9,355 MiB and 100% peak GPU utilization. Sampled full-process utilization averaged 19 to 20%, including idle startup and shutdown. This is one GPU per arm, not two-GPU model scaling.

The judge asks for visible grasp or released placement only, using identical questions on dream and real frames. It can answer Unknown; abstentions and contradictory held-and-placed answers produce no mismatch reward. On this batch, real-stage agreement was 4/9 readable boundaries. Mismatch changed two video probabilities by -0.0625 and changed no action probabilities; the shared GEN gradients can still change subsequent action predictions. Four chunks earned positive environment rewards, eight earned zero, and no wrong grabs or negative rewards occurred in this collection. Nonzero penalties and terminal handling have separate unit checks. The native smoke used zero stage values, and final boundaries without a next policy frame were unscored by the VLM. Do not interpret this comparison as evidence of policy improvement.

**Scale-up gates.** The executed trainer accepts single-object grasp/place tasks. See the [full-split reward audit](#full-split-reward-audit) for the simulator signals to validate before expanding the sampler. Calibrate the visual judge on held-out real frames before enabling mismatch for additional task families. This run uses the registered verl loss in a native worker. Multi-iteration Ray/FSDP training with an EMA rollout adapter remains unverified.

Both checkpoints reloaded in vLLM-Omni and completed BananaInBowl on evaluation seed 1 with the specific prompt: 127 steps without mismatch and 142 with mismatch. This is one matched evaluation on a seen task, not evidence of a treatment effect. The live cache also passed its exact K/V parity check during both evaluations. The 346 recorded training actions were checked against the simulator HDF5 export and matched exactly.

## Judge evidence and monitoring

The judge measures visible grasp and placement changes in the single-object experiment. The simulator supplies the action reward. Use the same stage-specific predicate, cameras, and executed timestep on both sides for the optional video-only mismatch term. The judge does not see the instruction level, simulator outcome, or whether an image is real or dreamed. Yes means visible evidence that the object is held or has been placed and released. No means visible evidence that the predicate is false. Unknown and contradictory predicates are masked out. An earlier frame can resolve object identity and contact, and the answer always refers to the endpoint.

`dream_eval/pairs.py --evidence ab` compares frame 32 alone with a paired frame 16/32 panel on exactly the same chunks. On 48 historical boundaries, endpoint-only scored 26/46 readable real stages correctly, with 13 false advances; temporal evidence scored 27/44, with 11 false advances. Both detected 10 of 12 actual advances. Temporal evidence reduced judge mismatch counts from eight to five, but also reduced paired readable coverage from 46 to 42. Lower mismatch is not itself a better critic. Keep endpoint-only as the default and use `JUDGE_EVIDENCE=temporal` as an ablation with `JUDGE_ABSTAIN=1`; collect midpoint frames for the temporal test. The current exporter skips chunks without a matched post-action boundary, including terminal chunks. Do not compare a partial real trajectory with dream frame 32.

For the next calibration, preregister stage and task strata, retain a representative stable-boundary sample, and separately oversample near-misses, release, slip, and wrong-object events for error analysis. Label endpoint state from synchronized simulator predicates plus a small human review of ambiguous clips. Test object-centered crops or targeted temporal queries on ambiguous cases; measure false advances, abstention, and residual dream-reality error by stage before using mismatch for training. Report simulator performance on held-out seeds with the same budget, not raw VLM agreement as the RL study outcome.

**UND A/B.** `training/benchmark_und.py` alternates cached and uncached native policy forwards on one H800, with identical captured observations and seeds. Across 16 pairs, median latency fell from 506.7 to 455.6 ms (10.1%, or 1.11x throughput), with exact action and dream-latent parity. The measurement includes policy input construction and denoising, not transport, simulation, or dream decode.

**GPU balance.** The paired update uses both GPUs concurrently, one arm per GPU. In the monitored run, the 7.6 s update windows averaged 76.1% and 76.3% sampled GPU utilization; the complete processes averaged 20.5% and 19.2%. Peak device memory was 9,355 MiB per GPU. The full loop does not keep both GPUs busy: policy calls and RoboLab stepping alternate, and judge and actor startup occur in separate phases. Prioritize persistent workers and larger rollout batches or multiple staggered environments. When using two GPUs for different arms, keep their checkpoints, policy versions, and collections separate.

**Prompt rewards.** With `DREAM_ALL_LEVELS=1`, the rollout runner keeps a single policy server and writes separate `vague/`, `default/`, and `specific/` data under `OUT`. Reward grouping uses the prompt field actually used by the policy. `python -m training.rollout_rewards <vague-run> <default-run> <specific-run> --output <report.json>` reports simulator return, reward per chunk, success, step count, and wrong grabs independently of the visual mismatch. Missing or mixed prompt metadata is reported as unknown. In the matched Banana smoke, all three levels earned simulator return 4.0, reaching success in 149, 163, and 162 steps respectively; one episode per level is a plumbing check.

**TensorBoard.** The option-4 launcher emits live optimizer scalars per arm and writes GPU utilization, memory, and power series after sampling stops. Use `python -m training.monitor <logdir> --rewards <report.json> --judge <evidence-audit.json> --cache <cache-benchmark.json>` to add rollout, critic, and cache results. The verified dashboard is under `training/runs/option4_monitored`:

```bash
~/verl-omni/.venv/bin/tensorboard --logdir training/runs/option4_monitored --host 127.0.0.1
```

## Rank and rollout throughput

Run `training/benchmark_ranks.sh <scored-rollout> <output-dir>` with separate
`CONTROL_GPU` and `MISMATCH_GPU` UUIDs. It runs both objective arms with
identical traces, noise and accumulation for ranks 8, 16 and 32 without activation
checkpointing, and repeats rank 16 with checkpointing.

| Rank | Adapter parameters | Chunks/s per GPU | Peak PyTorch allocation |
| --- | ---: | ---: | ---: |
| 8 | 2,557,952 | 1.90 | 18.70 GiB |
| 16 | 5,115,904 | 1.90 | 18.72 GiB |
| 32 | 10,231,808 | 1.90 | 18.77 GiB |
| 16, checkpointed | 5,115,904 | 1.60 | 8.28 GiB |

These are the averages across both arms over 12 native microbatches and three
optimizer steps each. The sweep tests hardware costs, not task performance by
rank. Start at `LORA_RANK=16 CHECKPOINT_BLOCKS=0`. Enable checkpointing for
memory-constrained runs. The tested checkpointed and non-checkpointed modes
matched initial losses but did not produce identical final adapter weights.

`training/benchmark_rollouts.sh <absolute-output-dir> [tasks...]` compares
layouts on one simulator GPU and a separate policy GPU. A layout `2x2` means
two task worker processes with `NUM_ENVS=2` per worker, not four environments
per task. Parallel workers share the policy and UND cache, but write separate
logs and checkpoints. The benchmark measures environment steps per second over
the union of the active episode intervals and separately includes cold launch
time, scene loading and shutdown. Use `python -m training.utilization_report`
with `ranks` or `rollouts` and the corresponding benchmark directory to
reproduce the tables.

| Task workers x envs per worker | Episodes | Env steps/s (active) | Env steps/s (cold) |
| --- | ---: | ---: | ---: |
| 1x1 | 2 | 1.36 | 0.82 |
| 1x2 | 4 | 2.28 | 1.46 |
| 1x4 | 8 | 3.56 | 2.42 |
| 2x1 | 2 | 1.32 | 0.86 |
| 2x2 | 4 | 2.44 | 1.62 |

Start with `NUM_ENVS=4 DREAM_TASK_WORKERS=1`: four replicated environments
for one task, with tasks rotated sequentially through the worker. This was
the fastest layout on the two-task, 64-step Banana/Yogurt probe. At the same
total environment count, two task workers raised sampled simulator utilization
from about 32% to 48% but reduced throughput and raised peak simulator memory
from about 12.0 to 21.2 GiB. The policy GPU averaged about 12% in the fastest
layout. This is not saturation; benchmark eight same-task environments or
batched policy requests next, and confirm the layout on full episodes before
scaling to the 96-task split.

## Full-split reward audit

The 96 training tasks share one reward rule across vague, default and specific prompts. Read the active RoboLab success termination and its arguments directly from the environment configuration. This preserves the benchmark's object IDs, destinations, count, contact, release, orientation and tolerance semantics without a task-specific judge or prompt interpretation.

| Audited definition family | Train tasks |
| --- | ---: |
| Placement helpers (29 single, 35 multi-object `all`, 4 `any`, 5 `choose`) | 73 |
| Pickup, including alternatives | 7 |
| Stacking or nesting | 7 |
| Reorientation | 4 |
| Custom removal chains | 2 |
| Centering | 1 |
| Sequential placement trackers | 2 |

**Progress.** Use one generic partial-credit rule for a single native success predicate that explicitly applies independent checks over multiple objects with `logical="all"`. Evaluate the same callable on each declared object with its original parameters. The potential `P` is the fraction of those checks currently true. For example, a released mug in the bin raises `P` by 0.5 on WhiteMugsInBin, and removing it lowers `P` by 0.5. Use binary official success for every other task, including exact-count, multi-term alternatives, coupled stack conditions, and any predicate whose decomposition is unverified. Reorientation uses RoboLab's configured `object_upright` check with its existing axis and release arguments.

Use the termination even when the tracker disagrees. MouseOnKeyboard terminates on support on the keyboard. BananasInCrate terminates when exactly two of its five candidates satisfy the configured containment check. FoodPackingByColor uses its native grouped success conditions. Record instruction and benchmark discrepancies for interpretation.

**Wrong-grasp events are diagnostics only.** The full-split reward and critic give them zero weight. The event detector reported 74 raw flags in 100 historical episodes, including two for a valid bagel. It can fire on contact with a slightly closed hand and can use group labels in place of object IDs. Successful pickup still earns official task success where the benchmark declares a pickup condition.

**Rollout and validation.**

1. The collector checks eligible conjunctions against the original termination at reset, executed chunk boundaries, and the final step. It records the count and uses the measured reset state. A disagreement falls back to binary success for the batch and is logged. Binary tasks have `P = 0` until official success and `P = 1` at success.
2. Each 32-action chunk earns `2 * (P_end - P_start) + 2 * success`. The bonus is paid on the chunk that terminates successfully. The critic uses integer condition counts and bootstraps timeouts. The full-split actor uses the simulator critic. The separate dream mismatch experiment remains scoped to its calibrated tasks.
3. The report includes return, success, signed progress, partial-credit coverage and parity fallbacks by prompt level, with the task name attached to each episode. The online round also records collection and update time. `python -m unittest training.test_native_progress` checks boundary alignment, prefilled progress, regression, timeout bootstrap, count semantics and independence from raw wrong-grasp flags.

For a fresh `training.online` run, use `"reward_mode": "native"` and `"tasks": "train"` in `config.json`, alongside the existing GPU, optimizer and retention settings. A list of training task names limits a native smoke run to a subset. Run `python -m training.online <run-dir> --resident` from the verl-omni environment. For standalone collection, set `DREAM_NATIVE_REWARD=1` with `dream_eval/run_pilot.sh`, then use `python -m training.rollout_rewards --reward-mode native <rollout-dir>`. Existing four-task configs and historical dream-mismatch runs use the legacy scorer.

The [curriculum and two-GPU rollout plan](curriculum_rollout_plan.md) covers the proposed starting mixture, the progress gate for online updates, and the actor lifetime.
