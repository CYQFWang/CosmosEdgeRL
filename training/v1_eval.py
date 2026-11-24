"""Paired evaluation of two policies at once, one per GPU lane (train_v1.md §6.2, §9).

Both lanes run the same batches with the same seeds, so layouts and policy-noise seeds match and
episodes pair up by (task, batch, env). Seeds come from an "eval" stream never used in training.

Usage:
  python -m training.v1_eval <out_dir> --run training/runs/v1_T_01 --model base --model <checkpoint_dir> \
      [--tasks ...] [--level default] [--batches-per-task 3]
  python -m training.v1_eval <out_dir> --summarize
"""

import argparse
import hashlib
import json
import os
import threading
import time
from pathlib import Path

import numpy as np

from training.online import CACHE, WORKSPACE, ResidentWorkers, read_json, run_command, write_json
from training.v1_reward import episode_chunks, rollout_logs
from training.v1_run import stable_seed


def episode_rows(out, batch, model):
    rows = []
    for _, log in rollout_logs(out):
        chunks = episode_chunks(log)
        path = [chunks[0].p_start] + [c.p_end for c in chunks]
        holds = [c.end for c in chunks if c.p_end == 0.5]
        rows.append({
            "model": model, "task": log["task"], "level": batch["level"], "batch": batch["index"], "seed": batch["seed"],
            "env": log["env_id"], "success": bool(log["success"]), "steps": log["final_step"], "final_p": path[-1],
            "ever_held": any(p >= 0.5 for p in path), "time_to_first_hold": holds[0] if holds else None,
            "drops": sum(1 for a, b in zip(path, path[1:]) if a == 0.5 and b == 0.0),
            "wrong_grab_events": sum(1 for e in log.get("events", []) if e.get("name") == "WRONG_OBJECT_GRABBED_FAILURE"),
        })
    return rows


def lane(root, config, index, model, queue, slot, scratch, status, lock):
    gpu, port = config["gpus"][index], 18731 + index
    worker = ResidentWorkers(root / f"lane_{index}")
    worker.root.mkdir(exist_ok=True)
    if model == "base":
        previous, version = None, "base"
    else:
        previous = Path(model)
        version = hashlib.sha256((previous / "heads.pt").read_bytes()).hexdigest()
    try:
        while True:
            with lock:
                if not queue:
                    break
                batch = queue.pop(0)
            name = f"{batch['task']}_{batch['level']}_b{batch['index']}"
            result = root / slot / f"{name}.json"
            if result.exists():
                continue
            out = scratch / slot / name
            if out.exists():
                import shutil
                shutil.rmtree(out)
            env = os.environ.copy()
            for key in ("DREAM_BRANCH", "REPLAY", "COSMOS_HEAD_ADAPTER", "DREAM_MAX_STEPS", "DREAM_NATIVE_REWARD"):
                env.pop(key, None)
            env.update(
                HF_HOME=str(CACHE), HF_HUB_OFFLINE="1", PYTHONPATH=f"{WORKSPACE}:/home/robot_share/cosmos-framework",
                DREAM_TASK_WORKERS="1", DREAM_ALL_LEVELS="0", DREAM_VIDEO_MODE="none", DREAM_ENDPOINT_ONLY="1",
                DREAM_LATENTS_ONLY="1", COSMOS_CAPTURE_NFT="1", COSMOS_UND_CACHE="8", DREAM_V1_PROGRESS="1",
                DREAM_INIT_POSE_NOISE=str(config["init_pose_noise"]), DREAM_EXTERNAL_POLICY="1",
                NUM_ENVS=str(config["num_envs"]), OUT=str(out), SIM_GPU=gpu, POLICY_GPU=gpu, PORT=str(port),
                DREAM_INSTRUCTION=batch["level"], SIM_SEED=str(batch["seed"]), COSMOS_POLICY_VERSION=version,
            )
            out.mkdir(parents=True)
            worker.begin_rollout(env, out / "capture", previous, version)
            started = time.monotonic()
            with lock:
                status[f"lane{index}"] = {"model": model, "batch": name, "started": time.time()}
                write_json(root / "status.json", status)
            run_command(["bash", "dream_eval/run_pilot.sh", batch["task"]], env=env, log=out.parent / f"{name}.log",
                        timeout=7200, peer=worker.policy)
            rows = episode_rows(out, batch, model)
            if len(rows) != config["num_envs"]:
                raise RuntimeError(f"Expected {config['num_envs']} episodes in {out}")
            result.parent.mkdir(exist_ok=True)
            write_json(result, {"rows": rows, "rollout_s": time.monotonic() - started, "version": version})
            with lock:
                status.setdefault(f"done{index}", 0)
                status[f"done{index}"] += 1
                write_json(root / "status.json", status)
    except Exception as error:  # noqa: BLE001 - surface in status.json, the other lane keeps going
        with lock:
            status[f"lane{index}_error"] = repr(error)
            write_json(root / "status.json", status)
        raise
    finally:
        worker.close()


def summarize(root):
    import pandas as pd

    plan = read_json(root / "plan.json")
    frames = []
    if plan.get("baseline") == "screening":
        run = Path(plan["run"])
        for task in plan["tasks"]:
            base = pd.read_parquet(run / "counters" / f"episodes_screen_{task}.parquet")
            frames.append(base[["task", "env", "success", "final_p", "steps", "drops"]].assign(
                level="default", batch=0, model_index=0))
    for index, model in enumerate(plan["models"]):
        if model == "screening":
            continue
        for path in sorted((root / f"model{index}").glob("*.json")):
            frames.append(pd.DataFrame(read_json(path)["rows"]).assign(model_index=index))
    if not frames:
        print("no finished batches yet")
        return
    data = pd.concat(frames, ignore_index=True)
    names = [Path(m).name if m not in ("base", "screening") else m for m in plan["models"]]
    print(f"models: 0={names[0]}  1={names[1]}")
    per = data.groupby(["task", "model_index"]).agg(n=("success", "size"), success=("success", "mean"),
                                                    final_p=("final_p", "mean")).round(3).unstack("model_index")
    print(per.to_string())
    a, b = (data[data.model_index == i].set_index(["task", "level", "batch", "env"]) for i in (0, 1))
    paired = a.join(b, lsuffix="_0", rsuffix="_1", how="inner")
    if paired.empty:
        return
    rng = np.random.default_rng(0)
    out = {}
    for metric in ("success", "final_p"):
        d = paired[f"{metric}_1"].astype(float) - paired[f"{metric}_0"].astype(float)
        draws = rng.choice(d.values, (10000, len(d))).mean(axis=1)
        out[metric] = (d.mean(), np.percentile(draws, 2.5), np.percentile(draws, 97.5))
        print(f"paired Δ{metric} (model1 − model0), n={len(d)}: {d.mean():+.3f} "
              f"[{np.percentile(draws, 2.5):+.3f}, {np.percentile(draws, 97.5):+.3f}]")
    discordant = paired[paired.success_0 != paired.success_1]
    print(f"discordant pairs: model1 only {int((discordant.success_1).sum())}, model0 only {int((discordant.success_0).sum())}")
    write_json(root / "summary.json", {"models": plan["models"], "n_pairs": len(paired),
                                       "paired": {k: list(map(float, v)) for k, v in out.items()},
                                       "per_task": json.loads(per.to_json())})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("--run", type=Path, default=WORKSPACE / "training/runs/v1_T_01")
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--tasks", nargs="*")
    parser.add_argument("--level", default="default")
    parser.add_argument("--batches-per-task", type=int, default=3)
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--pair-screening", action="store_true",
                        help="Run one --model on both lanes with the screening seeds; base = stored screening episodes")
    args = parser.parse_args()
    root = args.out.resolve()
    if args.summarize:
        return summarize(root)
    run = args.run.resolve()
    config = read_json(run / "config.json")
    if args.pair_screening:
        if len(args.model) != 1 or not args.tasks or args.level != "default":
            parser.error("--pair-screening needs one --model, explicit --tasks, and the default level")
        screened = read_json(run / "state.json")["screened"]
        tasks = sorted(args.tasks, key=lambda t: -screened[t]["rollout_s"])  # longest first balances the lanes
        batches = [{"task": t, "level": "default", "index": 0, "seed": screened[t]["seed"]} for t in tasks]
        models = ["screening", args.model[0]]
    else:
        if len(args.model) != 2:
            parser.error("Give exactly two --model values (base or a checkpoint directory)")
        tasks = args.tasks or read_json(run / "selection.json")["selected"]
        batches = [{"task": t, "level": args.level, "index": i,
                    "seed": stable_seed(config["seed"], "eval", t, args.level, i)}
                   for i in range(args.batches_per_task) for t in tasks]
        models = args.model
    root.mkdir(parents=True, exist_ok=True)
    plan = {"models": models, "tasks": tasks, "level": args.level, "batches": batches, "run": str(run),
            "baseline": "screening" if args.pair_screening else "run",
            "init_pose_noise": config["init_pose_noise"], "num_envs": config["num_envs"]}
    if (root / "plan.json").exists() and read_json(root / "plan.json") != plan:
        raise ValueError("Output directory holds a different evaluation plan")
    write_json(root / "plan.json", plan)
    scratch = Path("/var/tmp/vis_rl") / f"eval_{root.parent.name}_{root.name}"  # unique per run and eval
    status, lock = {"pid": os.getpid(), "total_batches_per_lane": len(batches)}, threading.Lock()
    if args.pair_screening:  # both lanes serve the same checkpoint and share one queue
        queue = list(batches)
        threads = [threading.Thread(target=lane, args=(root, config, i, models[1], queue, "model1", scratch, status,
                                                       lock)) for i in range(2)]
    else:  # one model per lane, each lane runs every batch
        threads = [threading.Thread(target=lane, args=(root, config, i, m, list(batches), f"model{i}", scratch,
                                                       status, lock)) for i, m in enumerate(models)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    status["finished"] = time.time()
    write_json(root / "status.json", status)
    summarize(root)


if __name__ == "__main__":
    main()
