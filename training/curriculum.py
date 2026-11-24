"""Collect on two GPUs and load the training actor only for updates."""

import argparse
import fcntl
import hashlib
import json
import os
import random
import shutil
import sys
import time
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from training.online import (
    CACHE, MODEL, WORKSPACE, ResidentWorkers, completed_rounds, measure_gpu,
    read_json, record, run_command, write_json,
)
from training.rollout_rewards import episodes, summarize


def schedule(config):
    occurrences = defaultdict(int)
    schedule = []
    for epoch in range(1, config["epochs"] + 1):
        copies = (3, 2, 1, 2, 1) if epoch % 2 else (2, 3, 1, 2, 1)
        tasks = [task for task, count in zip(config["tasks"], copies, strict=True) for _ in range(count)]
        tasks.append(config["probes"][(epoch - 1) % len(config["probes"])])
        rng = random.Random(config["seed"] + epoch)
        rng.shuffle(tasks)
        if epoch == 1:
            # Open on the two tasks with observed successful episodes.
            for task in config["tasks"][:2]:
                tasks.remove(task)
            tasks[:0] = config["tasks"][:2]
        for task in tasks:
            level = ("default", "vague", "default", "specific")[occurrences[task] % 4]
            occurrences[task] += 1
            schedule.append({
                "epoch": epoch, "batch": len(schedule) + 1, "task": task, "level": level,
                "seed": rng.randrange(1, 2**31), "max_steps": config["horizons"][task],
            })
    return schedule


def progress_gate(runs, minimum_episodes):
    outcomes = defaultdict(set)
    positive = 0
    for log, chunks in runs:
        start, end = chunks[0][1].start, chunks[-1][1].end
        success = log["success"]
        positive += success or end > start
        level, = log["instruction_fields_used"]
        count = log["native_progress"]["condition_count"]
        outcomes[log["task"], level, count].add((success, end - start))
    return len(runs) >= minimum_episodes and positive >= 2 and any(len(group) > 1 for group in outcomes.values())


def collect(worker, gpu, port, dest, batch, env, previous, version, deadline, num_envs):
    out = dest / "rollout"
    dest.mkdir()
    env = env | {
        "OUT": str(out), "SIM_GPU": gpu, "POLICY_GPU": gpu, "PORT": str(port),
        "NUM_ENVS": str(num_envs), "DREAM_INSTRUCTION": batch["level"],
        "SIM_SEED": str(batch["seed"]), "DREAM_MAX_STEPS": str(batch["max_steps"]),
        "DREAM_ROLLOUT_DEADLINE": str(deadline), "DREAM_EXTERNAL_POLICY": "1",
    }
    worker.begin_rollout(env, out / "capture", previous, version)
    started = time.monotonic()
    with measure_gpu(gpu) as gpu_stats:
        run_command(["bash", "dream_eval/run_pilot.sh", batch["task"]], env=env,
                    log=dest / "collect.log", timeout=max(0, deadline - started) + 300, peer=worker.policy)
    runs = list(episodes(out, reward_mode="native"))
    if (len(runs) != num_envs or any(
        log["task"] != batch["task"] or log["instruction_fields_used"] != [batch["level"]]
        or not all(path.is_file() for path, _, _ in chunks)
        for log, chunks in runs
    )):
        raise RuntimeError(f"Incomplete rollout or unexpected task or prompt in {dest}")
    metadata = batch | {"gpu": gpu, "gpu_samples": gpu_stats, "rollout_s": time.monotonic() - started}
    write_json(dest / "batch.json", metadata)
    shutil.copyfile(out / "rollouts/episode_results.jsonl", dest / "episodes.jsonl")
    return dest, metadata, runs


def update_actor(sources, stage, config, prior, env, previous, version, index):
    values = stage / "input_stage_values.json"
    write_json(values, prior["stage_values"])
    command = [
        sys.executable, "-m", "training.native_update", str(sources[0]), str(MODEL),
        "--output", str(stage), "--all-gen", "--rank", str(config["rank"]), "--no-checkpoint-blocks",
        "--accumulate", str(config["accumulate"]), "--learning-rate", str(config["learning_rate"]),
        "--online-policy-version", version, "--samples", "0", "--mismatch-weight", "0",
        "--values", str(values), "--reward-mode", "native", "--resume", str(previous),
        "--round", str(index), "--tensorboard", "--port", str(config["actor_port"]),
    ]
    for source in sources[1:]:
        command.extend(("--additional-rollout", str(source)))
    gpu = config["gpus"][0]
    with measure_gpu(gpu) as gpu_stats:
        run_command(command, env=env | {"CUDA_VISIBLE_DEVICES": gpu}, log=stage / "actor.log", timeout=3600)
    update = read_json(stage / "update_metrics.json")
    if update["policy_version"] != version or update["reward_mode"] != "native":
        raise RuntimeError("Actor used an unexpected rollout policy or reward mode")
    if update["steps"] and not all((stage / name).is_file() for name in ("heads.pt", "optimizer.pt")):
        raise RuntimeError("Missing resumable actor checkpoint")
    return update, gpu_stats


def run_collection(root, config, prior, plan, status, workers):
    index = prior["round"] + 1
    next_batch = prior["batches"]
    epoch = plan[next_batch]["epoch"]
    previous = root / prior["checkpoint"]
    version = hashlib.sha256((previous / "heads.pt").read_bytes()).hexdigest()
    name = f"round_{index:08d}"
    stage = root / f".{name}.pending"
    if stage.exists():
        archive = root / f".{name}.interrupted.{time.time_ns()}"
        stage.rename(archive)
    stage.mkdir()
    env = os.environ.copy()
    for key in ("DREAM_BRANCH", "REPLAY", "COSMOS_HEAD_ADAPTER"):
        env.pop(key, None)
    env.update(
        HF_HOME=str(CACHE), HF_HUB_OFFLINE="1", PYTHONPATH=f"{WORKSPACE}:/home/robot_share/cosmos-framework",
        DREAM_TASK_WORKERS="1", DREAM_ALL_LEVELS="0", DREAM_VIDEO_MODE="none", DREAM_ENDPOINT_ONLY="1",
        DREAM_LATENTS_ONLY="1", COSMOS_CAPTURE_NFT="1", COSMOS_UND_CACHE="8", COSMOS_POLICY_VERSION=version,
        DREAM_NATIVE_REWARD="1",
    )
    started = time.monotonic()
    deadline = started + config["collection_seconds"]
    reserved = 0
    active, completed, runs = {}, [], []
    gate = False
    status.update(phase="rollout", round=index, epoch=epoch, policy_version=version, actor_active=False)
    with ThreadPoolExecutor(max_workers=len(workers)) as pool:
        while True:
            if not (root / "STOP").exists() and not gate and time.monotonic() < deadline:
                for gpu_index, worker in enumerate(workers):
                    if (gpu_index in active or reserved + config["num_envs"] > config["max_episodes"]
                            or next_batch == len(plan) or plan[next_batch]["epoch"] != epoch):
                        continue
                    batch = plan[next_batch]
                    dest = stage / f"batch_{batch['batch']:04d}"
                    future = pool.submit(
                        collect, worker, config["gpus"][gpu_index], config["ports"][gpu_index], dest,
                        batch, env, previous, version, deadline, config["num_envs"],
                    )
                    active[gpu_index] = batch, future
                    next_batch += 1
                    reserved += config["num_envs"]
            status.update(active_rollouts=[batch | {"gpu": config["gpus"][i]} for i, (batch, _) in active.items()],
                          completed_episodes=len(runs), progress_gate=gate)
            write_json(root / "status.json", status)
            if not active:
                break
            done, _ = wait([future for _, future in active.values()], return_when=FIRST_COMPLETED)
            for gpu_index, (_, future) in list(active.items()):
                if future in done:
                    dest, metadata, new_runs = future.result()
                    completed.append((dest, metadata))
                    runs.extend(new_runs)
                    del active[gpu_index]
            gate = progress_gate(runs, config["min_episodes"])
    if not completed:
        stage.rmdir()
        return prior

    rollout_s = time.monotonic() - started
    gpu_stats = {}
    update = {"steps": 0, "stage_values": prior["stage_values"],
              "skip_reason": "progress gate not met", "prompt_level_rewards": summarize(runs)}
    completed.sort(key=lambda item: item[1]["batch"])
    if gate:
        status.update(phase="update", actor_active=True)
        write_json(root / "status.json", status)
        update, gpu_stats = update_actor([dest / "rollout" for dest, _ in completed], stage,
                                         config, prior, env, previous, version, index)
    state = {
        "round": index, "epoch": epoch, "epoch_complete": next_batch == len(plan) or plan[next_batch]["epoch"] != epoch,
        "batches": next_batch, "reward_mode": "native", "rollout_policy_version": version,
        "checkpoint": name if update["steps"] else prior["checkpoint"],
        "env_steps": prior["env_steps"] + sum(log["final_step"] for log, _ in runs),
        "optimizer_steps": prior["optimizer_steps"] + update["steps"], "stage_values": update["stage_values"],
        "prompt_rewards": update["prompt_level_rewards"], "rollout_s": rollout_s,
        "update_s": update.get("update_s", 0), "round_updates": update["steps"], "update_gpu": gpu_stats,
        "skip_reason": update.get("skip_reason"), "rollout_batches": [metadata for _, metadata in completed],
    }
    write_json(stage / "round.json", state)
    stage.rename(root / name)
    for dest, _ in completed:
        out = root / name / dest.name / "rollout"
        for filename in ("sim.log", "episodes_timing.jsonl"):
            (out / filename).replace(out.parent / filename)
        shutil.rmtree(out)
    record(root, state, config)
    status.update(last_completed=state, actor_active=False)
    write_json(root / "status.json", status)
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    config = read_json(root / "config.json")
    plan = schedule(config)
    train = {row["task"] for row in read_json(WORKSPACE / "training/task_split.json")["splits"]["train"]}
    if (len(set(config["gpus"])) != 2 or len(set(config["ports"])) != 2
            or not all(row["task"] in train for row in plan)):
        raise ValueError("Expected two distinct rollout GPUs and ports, and training-split tasks")
    with (root / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.resume:
            (root / "STOP").unlink(missing_ok=True)
        state = read_json(completed_rounds(root)[-1] / "round.json")
        if state["reward_mode"] != "native":
            raise ValueError("Curriculum training requires a native-reward checkpoint")
        write_json(root / "schedule.json", plan)
        workers = []
        for index in range(2):
            directory = root / f"worker_{index}"
            directory.mkdir(exist_ok=True)
            workers.append(ResidentWorkers(directory))
        status = {"pid": os.getpid(), "phase": "starting", "last_completed": state,
                  "planned_epochs": config["epochs"], "actor_active": False}
        write_json(root / "status.json", status)
        try:
            record(root, state, config)
            while state["batches"] < len(plan) and not (root / "STOP").exists():
                if shutil.disk_usage(root).free < config["min_free_gib"] * 2**30:
                    raise RuntimeError("Free disk below run threshold")
                state = run_collection(root, config, state, plan, status, workers)
            status.update(phase="complete" if state["batches"] == len(plan) else "stopped")
        except Exception as error:
            status.update(phase="error", error=str(error))
            raise
        finally:
            status.update(actor_active=False, active_rollouts=[])
            write_json(root / "status.json", status)
            for worker in workers:
                worker.close()


if __name__ == "__main__":
    main()
