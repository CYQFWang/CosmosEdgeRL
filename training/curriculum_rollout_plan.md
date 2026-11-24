# Curriculum and two-GPU rollout plan

## Starting mixture

Use RoboLab's structural labels as the starting inventory. The training split contains 51 simple, 31 moderate, and 14 complex tasks. Select the opening mixture by observed success, repeatable native progress, and cost per episode. Weights below apply to launched task batches, with one task, one prompt level, and independent reset seeds across the environments in a batch.

| Weight | Task | Evidence and initial horizon |
| --- | --- | --- |
| 25% | YogurtInBowlTask | Two of four successes with the default prompt and zero of four with the specific prompt. Use the native horizon of 600 steps. |
| 25% | ThrowAwayAppleTask | Two of four successes in the recovered handoff. Use the native horizon of 900 steps. |
| 10% | BananaOnPlateTask | Short retention task with strong historical base-policy success. Use the native horizon of 600 steps. |
| 20% | NonHammerToolsInRightBinTask | All four episodes ended with positive partial progress. Start with a cap of 1,536 steps. |
| 10% | ToolOrganizationTask | Two of four episodes ended with positive partial progress. Start with a cap of 1,536 steps. |
| 10% | Rotating probes | Rotate WhiteMugsInBinTask, FoodPacking1CansTask, and RecycleCartonsVerticalCrateTask before adding further tasks from the training split. Start with a cap of 1,536 steps or the native horizon, whichever comes first. |

The opening pool contains eight distinct tasks, with five recurring tasks and three rotating probes. Over 20 launched batches, draw 5/5/2/4/2/2 from the rows above. Shuffle their order and rotate prompt levels within each task with a 2:1:1 default/vague/specific ratio across successive draws. This puts most early capacity on short tasks with a chance of reward while rechecking harder tasks under current checkpoints. The shorter training horizon trades coverage of late progress for throughput. One ToolOrganization placement first appeared in the chunk starting at step 2,368. Keep the original horizons for fixed-seed evaluation.

After each block, check the number of episodes with positive endpoint progress, task-specific success, and effective episodes per GPU-hour set seeds at evaluation time. Reallocate probe slots using evidence from repeated seeds before increasing their share. Keep the [native reward and critic](README.md#full-split-reward-audit), including its zero weight for wrong-grasp diagnostics. [Training split](task_split.json) and [online history](runs/online_native_full_01/history.json)

## Multi-epoch review

Use three epochs for the initial comparison. An epoch groups 10 launched task batches for reporting, up to 120 training episodes at 12 environments per batch. Draw fresh training reset seeds and keep the online update gate active throughout. When the tenth batch finishes, drain it and record the latest checkpoint and cumulative optimizer steps, including epochs in which the gate produces no update.

| Epoch | YogurtInBowl | ThrowAwayApple | BananaOnPlate | NonHammerToolsInRightBin | ToolOrganization | Probe |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 3 | 2 | 1 | 2 | 1 | 1 WhiteMugsInBin |
| 2 | 2 | 3 | 1 | 2 | 1 | 1 FoodPacking1Cans |
| 3 | 3 | 2 | 1 | 2 | 1 | 1 RecycleCartonsVerticalCrate |

Each epoch covers six tasks. All eight appear by epoch 3. Evaluate the starting round-15 checkpoint and each epoch checkpoint on the same four held-out reset seeds per task, using the default prompt and original task horizons. That gives 32 matched evaluation episodes per checkpoint. Report paired success, start and end native potential, and time to success per task. Plot those results alongside online reward by prompt, effective episodes per GPU-hour The pilot measures behavior on the eight curriculum tasks across checkpoints. Generalization to unseen tasks requires evaluation on the existing validation split.

## When to update

1. Freeze and hash the checkpoint for collection. Launch tasks independently on the two GPUs. Score completed trajectories on CPU using the critic snapshot from the start of the collection.
2. Start checking after 16 valid completed episodes. Close collection when at least two independently seeded episodes show official success or positive net native progress and at least one task and prompt group has contrasting endpoint outcomes. Use the outcome contrast to avoid updating on uniform successes or uniform failures. Signed progress and regressions still flow into reward for every valid trajectory.
3. Once the gate fires, stop scheduling new batches. Finish active episodes or deliberately truncate at a completed 32-action boundary, export the final native condition state, and bootstrap the critic. Use all valid trajectories collected under that snapshot, including zero-reward episodes. Start with an overall collection budget of 48 episodes or 35 minutes. Count active reservations against the episode budget and request simulator truncation at the deadline. If the budget closes without satisfying the gate, log the batch, skip the actor update, and rotate tasks. A supervisor timeout remains a process failure.
4. Load the rank-16 actor only after collection has drained. Restore both adapter and optimizer state from the latest checkpoint, update, and publish the new checkpoint atomically. Exit the actor process and confirm its GPU allocation is gone before scheduling rollouts from the new checkpoint. Retain the existing three-checkpoint window and record per-task and per-prompt rewards, rollout timing, GPU telemetry, and optimizer metrics.

## GPU allocation

Start with two independent rollout workers and 12 environments per worker. Each H800 runs one RoboLab batch and one persistent vLLM-Omni Cosmos3 Edge policy server pinned to that device. Both servers use the same adapter hash, with their own port, output directory, and text cache. Dispatch the next task to the first free GPU so a short episode batch can be followed by new work while the other GPU continues a longer batch.

Run one transient actor on a single H800 during updates. The current `training.online --resident` path retains the actor on CUDA. The proposed scheduler will keep the policy servers resident and run `training.native_update` as a subprocess per update. The round-15 checkpoint is the starting point after review. During rollout, check GPU process memory and sample utilization and power on both devices. Compare layouts at equal horizons using aggregate environment steps/s, effective episodes per GPU-hour headroom. Report cold startup separately. [Current worker lifecycle](online.py) · [Earlier layout benchmark](README.md#rank-and-rollout-throughput)

### Rollout-only benchmark

The two H800s ran YogurtInBowl (GPU 0) and NonHammerToolsInRightBin (GPU 1) with the same round-15 checkpoint, default prompts, and 64 simulator steps per environment. All 48 episodes returned native condition metadata. A process check during collection showed RoboLab and vLLM-Omni on both devices. The benchmark launched no actor.

| Environments per GPU | GPU 0 active steps/s | GPU 1 active steps/s | Combined steps/s | GPU 0 / GPU 1 mean utilization | GPU 0 / GPU 1 peak memory |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 3.04 | 6.04 | 5.70 | 36.0% / 46.2% | 22.1 / 21.7 GiB |
| 8 | 4.61 | 8.45 | 6.21 | 43.5% / 54.2% | 25.6 / 25.3 GiB |
| 12 | 6.06 | 9.35 | 11.59 | 45.7% / 48.1% | 28.0 / 27.8 GiB |

Per-device active rates and utilization use each worker's episode window. The combined rate uses the elapsed span from the first episode start to the last episode end. The eight-environment batches overlapped for only 6.8 seconds because their scene startup times differed. The twelve-environment batches overlapped for 76.2 seconds. Twelve environments also had the highest per-device rates including startup, at 3.94 and 5.30 steps/s. These are short throughput probes. Reward yield and GPU saturation on full collection cycles remain unverified.

At 12 environments the workers spent roughly 31 seconds each in policy calls within their 64-step episodes. Cosmos3 currently inherits RoboLab's serial per-environment inference loop. Batch the boundary requests and stagger task starts as the next throughput tests. Validate fixed-seed action parity and use full horizons before tuning the rollout size again.
