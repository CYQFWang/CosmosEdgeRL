"""Run a single in-process Cosmos3 head-LoRA update through the verl loss registry."""

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

import torch
from verl_omni.trainer.diffusion.diffusion_algos import get_diffusion_loss_fn
from vllm.config import set_current_vllm_config
from vllm_omni.diffusion.data import OmniDiffusionConfig, TransformerConfig
from vllm_omni.diffusion.forward_context import set_forward_context
from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker

from training.cosmos_nft import joint_velocity, prepare_joint_forward
from training.head_adapters import HeadAdapters
from training.reward import update_values
from training.rollout_rewards import critic_key, episodes, prompt_level, summarize


def examples(root, scores, values, *, simulator_only=False, reward_mode="legacy", additional_rollouts=()):
    labels = {} if simulator_only else {
        row["id"]: row["answers"] for row in map(json.loads, (scores / "predictions.jsonl").read_text().splitlines())
    }
    runs = [run for source in (root, *additional_rollouts)
            for run in episodes(source, labels, values, reward_mode=reward_mode)]
    cases = [(path, chunk, result, prompt_level(log)) for log, rows in runs for path, chunk, result in rows]
    if not cases:
        raise ValueError("No complete rollout logs")
    return sorted(cases, key=lambda item: -abs(item[2]["advantage"])), summarize(runs), runs


def main(argv=None, resident=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("model")
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--accumulate", type=int, default=0)
    parser.add_argument("--values", type=Path)
    parser.add_argument("--all-gen", action="store_true")
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--no-checkpoint-blocks", action="store_true")
    parser.add_argument("--mismatch-weight", type=float, default=0.25)
    parser.add_argument("--reward-mode", choices=("legacy", "native"), default="legacy")
    parser.add_argument("--additional-rollout", action="append", type=Path, default=[])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--port", type=int, default=18232)
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--online-policy-version")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--round", type=int, default=0)
    args = parser.parse_args(argv)
    if args.samples < 0 or args.accumulate < 0 or args.rank < 1:
        parser.error("Sample and accumulation limits must be nonnegative and rank must be positive")
    checkpoint_blocks = args.all_gen and not args.no_checkpoint_blocks
    values = {task: {int(stage): value for stage, value in stages.items()}
              for task, stages in json.loads(args.values.read_text()).items()} if args.values else {}
    online = args.online_policy_version is not None
    if args.resume and not online or online and (args.mismatch_weight or args.samples):
        parser.error("Online T0 requires --samples 0, --mismatch-weight 0, and a policy version")
    if args.reward_mode == "native" and args.mismatch_weight:
        parser.error("Native rewards require --mismatch-weight 0")
    if args.additional_rollout and (not online or args.reward_mode != "native"):
        parser.error("Additional rollout roots require online native rewards")
    sources = [path.resolve() for path in (args.run, *args.additional_rollout)]
    if len(sources) != len(set(sources)):
        parser.error("Rollout roots must be distinct")
    cases, prompt_rewards, runs = examples(
        args.run, args.run / "scores", values,
        simulator_only=online or args.reward_mode == "native", reward_mode=args.reward_mode,
        additional_rollouts=args.additional_rollout,
    )
    if online:
        by_task = defaultdict(list)
        for log, rows in runs:
            by_task[critic_key(log)].append(rows)
        for task, entries in by_task.items():
            task_results = [result for rows in entries for _, _, result in rows]
            advantages = [result["advantage"] for result in task_results]
            center = mean(advantages)
            scale = pstdev(advantages)
            for result in task_results:
                result["normalized_advantage"] = (result["advantage"] - center) / scale if scale > 1e-6 else 0.0
            values[task] = update_values(
                [[chunk for _, chunk, _ in rows] for rows in entries], values.get(task, {}),
            )
        generator = torch.Generator().manual_seed(42 + args.round)
        cases = [cases[index] for index in torch.randperm(len(cases), generator=generator).tolist()]
        rollout_pids = set()
        by_source = defaultdict(set)
        for path, _, _, _ in cases:
            trace = torch.load(path, weights_only=True)
            if trace.get("policy_version") != args.online_policy_version:
                raise RuntimeError(f"Stale policy sample: {path}")
            if "policy_worker_pid" in trace:
                rollout_pids.add(trace["policy_worker_pid"])
                by_source[path.parent].add(trace["policy_worker_pid"])
        if any(len(pids) > 1 for pids in by_source.values()):
            raise RuntimeError("A rollout source crossed policy workers")
    cases = cases[:args.samples] if args.samples else cases
    accumulate = args.accumulate or len(cases)
    output_dir = args.output or args.run
    output_dir.mkdir(parents=True, exist_ok=True)
    skip_reason = None
    if online and not any(result["reward"] for _, _, result, _ in cases):
        skip_reason = "no simulator reward signal"
    elif online and args.reward_mode == "native" and not any(
        result["normalized_advantage"] for _, _, result, _ in cases
    ):
        skip_reason = "no advantage contrast"
    if skip_reason:
        if args.tensorboard and args.reward_mode == "native":
            from training.monitor import SummaryWriter, write_rewards

            with SummaryWriter(str(output_dir / "tensorboard")) as writer:
                write_rewards(writer, prompt_rewards, 0)
        (output_dir / "update_metrics.json").write_text(json.dumps({
            "steps": 0, "policy_version": args.online_policy_version, "prompt_level_rewards": prompt_rewards,
            "stage_values": values, "skip_reason": skip_reason, "reward_mode": args.reward_mode,
            "rollout_worker_pids": sorted(rollout_pids), "actor_pid": os.getpid(),
        }, indent=2) + "\n")
        return
    architecture = json.loads((Path(args.model) / "transformer" / "config.json").read_text())
    config = OmniDiffusionConfig(model=args.model, model_class_name="Cosmos3OmniDiffusersPipeline",
                                 tf_model_config=TransformerConfig.from_dict(architecture),
                                 enforce_eager=True, model_config={"guardrails": False},
                                 diffusion_attention_config={"default": "TORCH_SDPA"} if args.all_gen else {})
    started = time.monotonic()
    signature = (args.model, args.rank, args.all_gen, checkpoint_blocks, args.learning_rate, args.port)
    reused = resident is not None and "worker" in resident
    if reused:
        if resident["signature"] != signature or resident["version"] != args.online_policy_version:
            raise RuntimeError("Resident actor configuration or policy version changed unexpectedly")
        worker, config = resident["worker"], resident["config"]
    else:
        worker = DiffusionWorker(0, 0, config, distributed_init_method=f"tcp://127.0.0.1:{args.port}",
                                 set_proc_title=False)
        if resident is not None:
            resident.update(worker=worker, config=config, signature=signature)
    writer = None
    try:
        if args.tensorboard:
            from training.monitor import SummaryWriter, write_rewards

            writer = SummaryWriter(str(output_dir / "tensorboard"))
            write_rewards(writer, prompt_rewards, 0)
        transformer = worker.model_runner.pipeline.transformer
        noise_generator = torch.Generator(device=worker.device).manual_seed(42 + args.round)
        if reused:
            adapter, optimizer, old_adapter = (resident[name] for name in ("adapter", "optimizer", "old_adapter"))
        else:
            transformer.requires_grad_(False)
            torch.manual_seed(37)
            adapter = HeadAdapters(transformer, rank=args.rank, all_gen=args.all_gen, checkpoint_blocks=checkpoint_blocks)
            optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.learning_rate)
            old_adapter = None
            if args.resume:
                weights = torch.load(args.resume / "heads.pt", weights_only=True, map_location="cpu")
                adapter.load_state_dict(weights)
                optimizer.load_state_dict(torch.load(args.resume / "optimizer.pt", weights_only=True, map_location="cpu"))
        if online and old_adapter is None:
            old_adapter = HeadAdapters(transformer, rank=args.rank, all_gen=args.all_gen)
            old_adapter.requires_grad_(False)
        if old_adapter is not None:
            old_adapter.load_state_dict(adapter.state_dict())
            old_adapter.active = False
        baseline = {name: tensor.detach().clone() for name, tensor in adapter.state_dict().items()} if online else None
        if resident is not None:
            resident.update(adapter=adapter, optimizer=optimizer, old_adapter=old_adapter)
        objective = get_diffusion_loss_fn("cosmos3_edge_nft")
        device = worker.device
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
        load_seconds = time.monotonic() - started
        started = time.monotonic()
        update_start_unix = time.time()
        metrics = []
        audit = []
        steps = 0
        gradient_norms = {name: 0.0 for name in adapter.layers}
        window = []
        with (set_current_vllm_config(worker.vllm_config),
              set_forward_context(vllm_config=worker.vllm_config, omni_diffusion_config=config)):
            for index, (path, chunk, result, level) in enumerate(cases):
                trace = torch.load(path, weights_only=True)
                denoise = index % len(trace["train_sigmas"])
                sigma, batch = prepare_joint_forward(trace, denoise, chunk.executed, device, noise_generator)
                inputs = batch["video"][0], batch["action"][0]
                adapter.active = False
                with torch.no_grad():
                    base = joint_velocity(transformer, trace, *inputs, denoise)
                    old = base
                    if old_adapter is not None:
                        old_adapter.active = True
                        old = joint_velocity(transformer, trace, *inputs, denoise)
                        old_adapter.active = False
                adapter.active = True
                current = joint_velocity(transformer, trace, *inputs, denoise)
                modalities = {}
                probabilities = {}
                for name, prediction, previous, frozen in zip(batch, current, old, base, strict=True):
                    noisy, clean, mask = batch[name]
                    advantage = result.get("normalized_advantage", result["advantage"])
                    if name == "video":
                        advantage += args.mismatch_weight * (result["honesty"] or 0)
                    probabilities[name] = 0.5 + 0.5 * max(-1, min(1, advantage / 2))
                    probability = torch.tensor([probabilities[name]], device=device)
                    modalities[name] = noisy, clean, sigma, prediction, previous, frozen, probability, mask
                output = objective(config=None, model_output={"modalities": modalities}, data={})
                (output.loss / min(accumulate, len(cases) - (index // accumulate) * accumulate)).backward()
                metrics.append(float(output.loss.detach()))
                window.append((level, metrics[-1], probabilities, result))
                if (index + 1) % accumulate == 0 or index + 1 == len(cases):
                    norms = []
                    for name, module in adapter.layers.items():
                        grad = module[1].weight.grad
                        if grad is None:
                            raise RuntimeError(f"Missing gradient for {name}")
                        norms.append(grad.float().norm())
                    for name, norm in zip(adapter.layers, torch.stack(norms).tolist(), strict=True):
                        if not 0 <= norm < float("inf"):
                            raise RuntimeError(f"Nonfinite gradient for {name}")
                        gradient_norms[name] = max(gradient_norms[name], norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    steps += 1
                    if writer:
                        writer.add_scalar("train/loss", sum(item[1] for item in window) / len(window), steps)
                        writer.add_scalar("train/elapsed_s", time.monotonic() - started, steps)
                        writer.add_scalar("train/chunks", index + 1, steps)
                        writer.add_scalar("train/peak_allocated_gib", torch.cuda.max_memory_allocated() / 2**30, steps)
                        grouped = defaultdict(list)
                        for prompt, _, routed, reward_row in window:
                            grouped[prompt, "sim_reward"].append(reward_row["reward"])
                            for name, probability in routed.items():
                                grouped[prompt, f"{name}_reward_prob"].append(probability)
                            if reward_row["honesty"] is not None:
                                grouped[prompt, "honesty"].append(reward_row["honesty"])
                        for (prompt, name), scalars in grouped.items():
                            writer.add_scalar(f"train/prompt/{prompt}/{name}", sum(scalars) / len(scalars), steps)
                    window.clear()
                audit.append({**result, "trace": path.name, "rollout": str(path.parent.parent),
                              "prompt_level": level, "executed": chunk.executed,
                              "start": chunk.start, "end": chunk.end, "wrong_grabs": chunk.wrong_grabs,
                              "real_stage": chunk.real_stage, "dream_stage": chunk.dream_stage,
                              "sigma": float(sigma.item()), "reward_prob": probabilities})
        torch.cuda.synchronize()
        update_seconds = time.monotonic() - started
        update_end_unix = time.time()
        if not all(torch.count_nonzero(module[1].weight) for module in adapter.layers.values()):
            raise RuntimeError("Expected an optimizer update in both heads")
        changed = sum(not torch.equal(tensor, baseline[name]) for name, tensor in adapter.state_dict().items()) if online else None
        if online and not changed:
            raise RuntimeError("Online optimizer did not change any adapter tensor")
        checkpoint = output_dir / "heads.pt"
        torch.save({name: tensor.cpu() for name, tensor in adapter.state_dict().items()}, checkpoint)
        adapter.load_state_dict(torch.load(checkpoint, weights_only=True, map_location="cpu"))
        if online:
            torch.save(optimizer.state_dict(), output_dir / "optimizer.pt")
        if resident is not None:
            resident["version"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        result = {"load_s": load_seconds, "update_s": update_seconds, "losses": metrics,
                  "update_start_unix": update_start_unix, "update_end_unix": update_end_unix,
                  "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                  "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                  "adapter_groups": len(adapter.layers),
                  "lora_rank": args.rank, "checkpoint_blocks": checkpoint_blocks,
                  "steps": steps, "gradient_norms": gradient_norms, "stage_values": values,
                  "trainable_parameters": sum(param.numel() for param in adapter.parameters()),
                  "mismatch_weight": args.mismatch_weight, "reward_mode": args.reward_mode, "chunks": audit,
                  "policy_version": args.online_policy_version, "changed_adapter_tensors": changed,
                  "actor_pid": os.getpid(), "actor_reused": reused,
                  "rollout_worker_pids": sorted(rollout_pids) if online else [],
                  "prompt_level_rewards": prompt_rewards}
        (output_dir / "update_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    finally:
        if writer:
            writer.close()
        if resident is None:
            worker.shutdown()


def worker_loop():
    resident = {}
    try:
        for line in sys.stdin:
            job = json.loads(line)
            with Path(job["log"]).open("w") as log:
                stdout, stderr = os.dup(1), os.dup(2)
                try:
                    sys.stdout.flush()
                    sys.stderr.flush()
                    os.dup2(log.fileno(), 1)
                    os.dup2(log.fileno(), 2)
                    main(job["argv"], resident)
                    if torch.cuda.is_initialized():
                        torch.cuda.empty_cache()
                finally:
                    sys.stdout.flush()
                    sys.stderr.flush()
                    os.dup2(stdout, 1)
                    os.dup2(stderr, 2)
                    os.close(stdout)
                    os.close(stderr)
            Path(job["done"]).touch()
    finally:
        if "worker" in resident:
            resident["worker"].shutdown()


if __name__ == "__main__":
    if sys.argv[1:] == ["--worker-loop"]:
        worker_loop()
    else:
        main()
