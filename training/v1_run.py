"""train_v1 orchestrator (train_v1.md §3, §6, §8, §9).

Phases: screen (26 candidates, `default`, tracked seeds, frozen base policy)
        → select (≤ max_train_tasks)
        → round0 (`specific` batches under the frozen base; one update on all round-0 data)
        → epochs (selected × {default tracked seed, specific fresh seed}, collections of 4 batches).
Each collection: collect on both lanes → decode dream frames (actor ranks) → VLM judge (GPU 0)
→ rewards (v1_reward) → DDP update on both GPUs → publish the adapter to both policy servers.
Phases never overlap on a GPU (Q8 test). Latents live in scratch and are deleted after use.

Usage: python -m training.v1_run <run_dir> [--resume]      (config.json in run_dir)
"""

import argparse
import fcntl
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
import zlib
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import numpy as np

from training.online import CACHE, MODEL, WORKSPACE, ResidentWorkers, measure_gpu, read_json, run_command, write_json
from training.v1_reward import HORIZON, as_row, dream_reward, episode_chunks, judged_stage, preference, rollout_logs

PYTHON = sys.executable


def stable_seed(*parts):
    return zlib.crc32(":".join(map(str, parts)).encode()) & 0x7FFFFFFF or 1


class Actors:
    """Two resident actor ranks; jobs are written to both and both done files awaited."""

    def __init__(self, root, config):
        self.root, self.config, self.processes, self.jobs = root, config, [], 0
        env = os.environ | {"HF_HOME": str(CACHE), "HF_HUB_OFFLINE": "1",
                            "PYTHONPATH": f"{WORKSPACE}:/home/robot_share/cosmos-framework"}
        for rank, gpu in enumerate(config["gpus"]):
            log = (root / f"actor_rank{rank}.log").open("a")
            self.processes.append(subprocess.Popen(
                [PYTHON, "-u", "-m", "training.v1_actor", "--rank", str(rank), "--world", "2",
                 "--store-port", str(config["actor_store_port"]), "--vllm-port", str(config["actor_vllm_ports"][rank]),
                 "--model", str(MODEL), "--lora-rank", str(config["lora_rank"]),
                 "--learning-rate", str(config["learning_rate"])],
                cwd=WORKSPACE, env=env | {"CUDA_VISIBLE_DEVICES": gpu}, stdin=subprocess.PIPE, stdout=log,
                stderr=subprocess.STDOUT, text=True, start_new_session=True,
            ))

    def run(self, job, directory, timeout):
        self.jobs += 1
        done = directory / f"job_{self.jobs:05d}_{job['kind']}"
        job = job | {"done": str(done)}
        for process in self.processes:
            process.stdin.write(json.dumps(job) + "\n")
            process.stdin.flush()
        deadline = time.monotonic() + timeout
        results = []
        for rank, process in enumerate(self.processes):
            path = Path(f"{done}.rank{rank}")
            while not path.exists():
                if process.poll() is not None:
                    raise RuntimeError(f"Actor rank {rank} exited; see actor_rank{rank}.log")
                if time.monotonic() > deadline:
                    raise TimeoutError(f"Actor job {job['kind']} timed out")
                time.sleep(1)
            result = json.loads(path.read_text())
            path.unlink()
            if not result.get("ok"):
                raise RuntimeError(f"Actor rank {rank} failed {job['kind']}:\n{result.get('error')}")
            results.append(result)
        return results

    def close(self):
        for process in self.processes:
            try:
                process.stdin.close()
            except BrokenPipeError:
                pass
        for process in self.processes:
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, 15)


class Run:
    def __init__(self, root):
        self.root = root
        self.config = read_json(root / "config.json")
        self.state = read_json(root / "state.json", {
            "phase": "screen", "screened": {}, "selected": [], "round": 0, "epoch": 0, "collection": 0,
            "checkpoint": None, "version": "base", "env_steps": 0, "train_started": None,
            "round0_batches": [], "pending_batches": [],
        })
        if self.config.get("start") == "train" and self.state["phase"] == "screen":
            # train_v2: tasks and tracked seeds come from a finished screening; start epochs from the base policy.
            self.state.update(phase="train", selected=list(self.config["selected"]), epoch=1, collection=0)
        self.scratch = Path(self.config["scratch"])
        self.scratch.mkdir(parents=True, exist_ok=True)
        self.counters = root / "counters"
        self.counters.mkdir(exist_ok=True)
        self.lanes = []
        self.actors = None
        self.status = {"pid": os.getpid()}
        if self.config.get("arm") == "AA" and self.config.get("alignment_weight") is None:
            raise ValueError("Arm AA needs alignment_weight (lambda) in config.json")
        if self.config.get("arm") in ("AV", "AA") and not (self.config.get("alignment") and self.config.get("real16")):
            raise ValueError("Alignment arms need alignment=true and real16=true")

    # ---------- bookkeeping ----------
    def save(self):
        write_json(self.root / "state.json", self.state)

    def note(self, **fields):
        self.status.update(fields, time=time.time())
        write_json(self.root / "status.json", self.status)

    def history(self, row):
        with (self.root / "history.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")

    def stop_requested(self):
        return (self.root / "STOP").exists()

    def check_disk(self):
        for path in (self.root, self.scratch):
            if shutil.disk_usage(path).free < self.config["min_free_gib"] * 2**30:
                raise RuntimeError(f"Free disk below {self.config['min_free_gib']} GiB at {path}")

    # ---------- rollout ----------
    def base_env(self):
        env = os.environ.copy()
        for key in ("DREAM_BRANCH", "REPLAY", "COSMOS_HEAD_ADAPTER", "DREAM_MAX_STEPS", "DREAM_NATIVE_REWARD",
                    "DREAM_ROLLOUT_DEADLINE"):
            env.pop(key, None)
        env.update(
            HF_HOME=str(CACHE), HF_HUB_OFFLINE="1", PYTHONPATH=f"{WORKSPACE}:/home/robot_share/cosmos-framework",
            DREAM_TASK_WORKERS="1", DREAM_ALL_LEVELS="0", DREAM_VIDEO_MODE="none",
            DREAM_ENDPOINT_ONLY="0" if self.config.get("real16") else "1",
            DREAM_LATENTS_ONLY="1", COSMOS_CAPTURE_NFT="1", COSMOS_UND_CACHE="8", DREAM_V1_PROGRESS="1",
            DREAM_INIT_POSE_NOISE=str(self.config["init_pose_noise"]), DREAM_EXTERNAL_POLICY="1",
            NUM_ENVS=str(self.config["num_envs"]),
        )
        return env

    def collect(self, lane, batch, directory):
        gpu, port = self.config["gpus"][lane], self.config["policy_ports"][lane]
        out = directory / "rollout"
        directory.mkdir(parents=True)
        version = self.state["version"]
        previous = Path(self.state["checkpoint"]) if self.state["checkpoint"] else None
        env = self.base_env() | {
            "OUT": str(out), "SIM_GPU": gpu, "POLICY_GPU": gpu, "PORT": str(port),
            "DREAM_INSTRUCTION": batch["level"], "SIM_SEED": str(batch["seed"]),
            "COSMOS_POLICY_VERSION": version,
        }
        if self.config.get("debug_max_steps"):
            env["DREAM_MAX_STEPS"] = str(self.config["debug_max_steps"])
        self.lanes[lane].begin_rollout(env, out / "capture", previous, version)
        started = time.monotonic()
        with measure_gpu(gpu) as gpu_stats:
            run_command(["bash", "dream_eval/run_pilot.sh", batch["task"]], env=env, log=directory / "collect.log",
                        timeout=self.config["batch_timeout_s"], peer=self.lanes[lane].policy)
        logs = list(rollout_logs(out))
        if len(logs) != self.config["num_envs"]:
            raise RuntimeError(f"Expected {self.config['num_envs']} episodes in {out}, got {len(logs)}")
        for _, log in logs:
            if log["task"] != batch["task"] or log["instruction_fields_used"] != [batch["level"]]:
                raise RuntimeError(f"Unexpected task or prompt in {out}")
            for chunk in episode_chunks(log, gamma=self.config["gamma"]):
                if not (out / "capture" / f"{chunk.key}_nft.pt").is_file():
                    raise RuntimeError(f"Missing training trace {chunk.key}")
        metadata = batch | {"gpu": gpu, "gpu_samples": gpu_stats, "rollout_s": time.monotonic() - started,
                            "policy_version": version, "directory": str(directory)}
        write_json(directory / "batch.json", metadata)
        return metadata

    def collect_all(self, batches, name):
        """Run batches over both lanes (pull model). Returns metadata in batch order."""
        directory = self.scratch / name
        directory.mkdir(parents=True, exist_ok=True)
        queue = [b for b in batches if not (directory / b["name"] / "batch.json").exists()]
        done = [read_json(directory / b["name"] / "batch.json") for b in batches
                if (directory / b["name"] / "batch.json").exists()]
        for b in queue:
            stale = directory / b["name"]
            if stale.exists():
                shutil.rmtree(stale)
        active = {}
        with ThreadPoolExecutor(max_workers=2) as pool:
            while queue or active:
                for lane in (0, 1):
                    if lane not in active and queue and not self.stop_requested():
                        batch = queue.pop(0)
                        active[lane] = batch, pool.submit(self.collect, lane, batch, directory / batch["name"])
                self.note(phase=f"collect:{name}", active=[b["name"] for b, _ in active.values()],
                          remaining=len(queue))
                if not active:
                    break
                finished, _ = wait([future for _, future in active.values()], return_when=FIRST_COMPLETED)
                for lane, (batch, future) in list(active.items()):
                    if future in finished:
                        done.append(future.result())
                        del active[lane]
        if self.stop_requested() and len(done) < len(batches):
            return None
        order = {b["name"]: i for i, b in enumerate(batches)}
        return sorted(done, key=lambda m: order[m["name"]])

    # ---------- per-collection processing ----------
    def chunks_for(self, metadata_list):
        """Score every chunk; returns list of (chunk, meta, capture_dir, log)."""
        items = []
        for meta in metadata_list:
            out = Path(meta["directory"]) / "rollout"
            for _, log in rollout_logs(out):
                for chunk in episode_chunks(log, gamma=self.config["gamma"]):
                    items.append((chunk, meta, out / "capture", log))
        return items

    def real_end(self, chunk, capture, log):
        if not chunk.terminal:
            return {"npz": str(capture / f"{chunk.task}_run{chunk.run}_env{chunk.env}_step{chunk.end:04}_request.npz"),
                    "key": "image"}
        frame = log["v1_progress"]["terminal_frame"]
        if not frame or frame["executed"] != chunk.executed or not frame["file"].startswith(chunk.key):
            raise RuntimeError(f"Terminal frame mismatch for {chunk.key}")
        return {"npy": str(capture / frame["file"])}

    def decode(self, items, name, work):
        """Decode dream frames on both actor ranks: npz frames = [0, t_k] (+ frame 16 when alignment is on)."""
        decode_items, paths = [], {}
        for chunk, meta, capture, log in items:
            cid = f"{name}_{meta['name']}_{chunk.key}"
            frames = [0, chunk.executed]
            if self.config.get("alignment") and chunk.executed >= 16:
                frames.append(16)
            paths[cid] = work / f"{cid}_dream.npz"
            decode_items.append({"trace": str(capture / f"{chunk.key}_nft.pt"), "frames": frames,
                                 "out": str(paths[cid])})
        self.note(phase=f"decode:{name}", chunks=len(decode_items))
        started = time.monotonic()
        self.actors.run({"kind": "decode", "items": decode_items}, work, timeout=3 * 3600)
        return paths, time.monotonic() - started

    def alignment(self, items, decoded, name):
        """Side-view SSIM gain of dream frame 16 over the 'nothing moves' baseline (training/align.py)."""
        from training.align import gain

        result = {}
        for chunk, meta, capture, log in items:
            cid = f"{name}_{meta['name']}_{chunk.key}"
            real16 = capture / f"{chunk.key}_real16.npy"
            if chunk.executed < 16 or not real16.exists():
                continue
            with np.load(decoded[cid]) as dream, np.load(capture / f"{chunk.key}_request.npz") as request:
                result[cid] = gain(dream["frames"][2], request["image"], np.load(real16))
        return result

    def judge(self, items, decoded, name, work):
        """Score decoded dreams and matched real frames with the VLM on GPU 0 (logging only in train_v2)."""
        manifest = []
        for chunk, meta, capture, log in items:
            cid = f"{name}_{meta['name']}_{chunk.key}"
            manifest.append({"id": cid, "task": chunk.task, "dream": str(decoded[cid]),
                             "real_start": {"npz": str(capture / f"{chunk.key}_request.npz"), "key": "image"},
                             "real_end": self.real_end(chunk, capture, log)})
        path = work / "judge_manifest.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in manifest))
        out = work / "judge.jsonl"
        self.note(phase=f"judge:{name}")
        started = time.monotonic()
        env = os.environ | {"CUDA_VISIBLE_DEVICES": self.config["gpus"][0], "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
                            "HF_HOME": str(CACHE), "HF_HUB_OFFLINE": "1", "PYTHONPATH": str(WORKSPACE)}
        run_command([PYTHON, "-m", "training.v1_judge", str(path), str(out),
                     "--samples-dir", str(self.root / "judge_samples" / name),
                     "--sample-fraction", str(self.config["judge_sample_fraction"]),
                     "--seed", str(stable_seed(self.config["seed"], name)),
                     "--gpu-memory-utilization", str(self.config["judge_gpu_memory_utilization"])],
                    env=env, log=work / "judge.log", timeout=2 * 3600)
        judged = {row["id"]: row for row in map(json.loads, out.read_text().splitlines())}
        return judged, time.monotonic() - started

    def score(self, items, judged, aligned, name):
        arm = self.config["arm"]
        bound = self.config.get("alignment_bound", 0.2)
        rows, samples = [], []
        for chunk, meta, capture, log in items:
            cid = f"{name}_{meta['name']}_{chunk.key}"
            answers = judged.get(cid, {})
            dream_stage = judged_stage(answers.get("held_dream"), answers.get("placedpair_dream"))
            real_stage = judged_stage(answers.get("held_real"), answers.get("placedpair_real"))
            r_v1, gate = dream_reward(chunk, dream_stage, real_stage, "v1")
            r_v2, _ = dream_reward(chunk, dream_stage, real_stage, "v2")
            align_gain, align_own, align_copy = aligned.get(cid, (None, None, None))
            p_action = chunk.p_action
            if arm in ("T", "V1", "V2"):
                used = {"T": None, "V1": r_v1, "V2": r_v2}[arm]
                p_video = 0.5 if used is None else preference(used, 1.0)
            elif arm == "AV":   # alignment gain to video tokens, current chunk only
                p_video = 0.5 if align_gain is None else preference(align_gain, bound)
            elif arm == "AA":   # alignment added after the task preference (probability units), current chunk only
                weight = self.config["alignment_weight"]   # c: largest shift alignment can cause
                p_video = 0.5
                if align_gain is not None:
                    p_action = min(1.0, max(0.0, chunk.p_action + weight * max(-1.0, min(1.0, align_gain / bound))))
            else:
                raise ValueError(f"Unknown arm {arm}")
            with np.load(capture / f"{chunk.key}_request.npz") as request:
                command = request["command"][:chunk.executed]
            joints = float(np.abs(np.diff(command[:, :7], axis=0)).mean()) if len(command) > 1 else 0.0
            flips = int(np.count_nonzero(np.diff(command[:, 7]))) if len(command) > 1 else 0
            row = as_row(chunk) | {
                "id": cid, "collection": name, "batch": meta["name"], "arm": arm, "epoch": self.state["epoch"],
                "policy_version": meta["policy_version"], "seed": meta["seed"],
                "dream_stage": dream_stage, "real_judged_stage": real_stage, "gate": gate,
                "r_v1": r_v1, "r_v2": r_v2, "p_video": p_video, "p_action_used": p_action,
                "align_gain": align_gain, "align_own": align_own, "align_copy": align_copy,
                "cmd_joint_step": joints, "cmd_gripper_flips": flips,
            } | {f"judge_{key}": value for key, value in answers.items() if key != "id"}
            rows.append(row)
            if p_action != 0.5 or p_video != 0.5:
                samples.append({"id": cid, "trace": str(capture / f"{chunk.key}_nft.pt"), "executed": chunk.executed,
                                "p_action": p_action, "p_video": p_video,
                                "policy_version": meta["policy_version"]})
        return rows, samples

    def episode_rows(self, metadata_list, name):
        rows = []
        for meta in metadata_list:
            for _, log in rollout_logs(Path(meta["directory"]) / "rollout"):
                chunks = episode_chunks(log, gamma=self.config["gamma"])
                path = [chunks[0].p_start] + [c.p_end for c in chunks]
                holds = [i for i, c in enumerate(chunks) if c.p_end == 0.5]
                wrong = sum(1 for e in log.get("events", []) if e.get("name") == "WRONG_OBJECT_GRABBED_FAILURE")
                rows.append({
                    "collection": name, "batch": meta["name"], "task": log["task"], "level": meta["level"],
                    "seed": meta["seed"], "env": log["env_id"], "policy_version": meta["policy_version"],
                    "epoch": self.state["epoch"], "arm": self.config["arm"],
                    "success": bool(log["success"]), "steps": log["final_step"], "final_p": path[-1],
                    "ever_held": any(p >= 0.5 for p in path), "time_to_first_hold": chunks[holds[0]].end if holds else None,
                    "grasps": sum(1 for a, b in zip(path, path[1:]) if a == 0.0 and b == 0.5),
                    "drops": sum(1 for a, b in zip(path, path[1:]) if a == 0.5 and b == 0.0),
                    "wrong_grab_events": wrong,
                    "wrong_held_boundaries": sum(1 for row in log["v1_progress"]["boundaries"] if row[4]),
                    "return_0": chunks[0].ret,
                })
        return rows

    def write_parquet(self, rows, path):
        import pandas as pd

        if rows:
            pd.DataFrame(rows).to_parquet(path, index=False)

    def process(self, metadata_list, name, *, update=True):
        """Judge, score, write counters and (optionally) update. Returns the history row."""
        work = self.scratch / f"{name}_work"
        work.mkdir(exist_ok=True)
        items = self.chunks_for(metadata_list)
        judged, aligned, decode_s, judge_s = {}, {}, 0.0, 0.0
        if self.config["judge"] or self.config.get("alignment"):
            decoded, decode_s = self.decode(items, name, work)
            if self.config.get("alignment"):
                aligned = self.alignment(items, decoded, name)
            if self.config["judge"]:
                judged, judge_s = self.judge(items, decoded, name, work)
        rows, samples = self.score(items, judged, aligned, name)
        episodes = self.episode_rows(metadata_list, name)
        self.write_parquet(rows, self.counters / f"chunks_{name}.parquet")
        self.write_parquet(episodes, self.counters / f"episodes_{name}.parquet")
        result = {"steps": 0}
        update_s = 0.0
        if update and samples:
            path = work / "samples.jsonl"
            path.write_text("".join(json.dumps(s) + "\n" for s in samples))
            output = self.root / "checkpoints" / name
            self.note(phase=f"update:{name}", samples=len(samples))
            started = time.monotonic()
            results = self.actors.run({
                "kind": "train", "samples": str(path), "resume": self.state["checkpoint"], "output": str(output),
                "seed": stable_seed(self.config["seed"], "update", name), "batch_chunks": self.config["batch_chunks"],
                "beta": self.config["beta"], "reference_weight": self.config["reference_weight"],
                "learning_rate": self.config["learning_rate"], "passes": self.config.get("passes", 1),
            }, work, timeout=4 * 3600)
            update_s = time.monotonic() - started
            result = results[0]
            for rank in (0, 1):
                src = output / f"update_samples_rank{rank}.parquet"
                if src.exists():
                    src.replace(self.counters / f"update_samples_{name}_rank{rank}.parquet")
            if (output / "steps.parquet").exists():
                shutil.copyfile(output / "steps.parquet", self.counters / f"steps_{name}.parquet")
            self.state["checkpoint"] = str(output)
            self.state["version"] = result["new_version"]
            self.prune_checkpoints()
        p_a = [r["p_action"] for r in rows]
        p_v = [r["p_video"] for r in rows]
        history = {
            "collection": name, "arm": self.config["arm"], "epoch": self.state["epoch"], "time": time.time(),
            "batches": [m["name"] for m in metadata_list], "episodes": len(episodes),
            "successes": sum(e["success"] for e in episodes), "mean_final_p": float(np.mean([e["final_p"] for e in episodes])) if episodes else None,
            "env_steps": sum(e["steps"] for e in episodes), "chunks": len(rows), "update_chunks": len(samples),
            "p_action_pos": sum(p > 0.5 for p in p_a), "p_action_neg": sum(p < 0.5 for p in p_a),
            "p_video_pos": sum(p > 0.5 for p in p_v), "p_video_neg": sum(p < 0.5 for p in p_v),
            "abstain": sum(r["gate"] != "ok" for r in rows) if self.config["judge"] else None, "clip": 0,
            "align_coverage": len(aligned) / max(len(rows), 1),
            "align_mean": float(np.mean([g for g, _, _ in aligned.values()])) if aligned else None,
            "align_clip": sum(abs(g) > self.config.get("alignment_bound", 0.2) for g, _, _ in aligned.values()),
            "p_action_clip": sum(r["p_action_used"] in (0.0, 1.0) and r["p_action"] not in (0.0, 1.0) for r in rows),
            "rollout_s": sum(m["rollout_s"] for m in metadata_list), "decode_s": decode_s, "judge_s": judge_s,
            "update_s": update_s, "optimizer_steps": result.get("steps", 0),
            "actor_peak_gib": result.get("peak_allocated_gib"), "version": self.state["version"],
            "per_task": {t: {"episodes": sum(e["task"] == t for e in episodes),
                             "successes": sum(e["success"] for e in episodes if e["task"] == t),
                             "held": sum(e["ever_held"] for e in episodes if e["task"] == t)}
                         for t in sorted({e["task"] for e in episodes})},
        }
        self.state["env_steps"] += history["env_steps"]
        self.history(history)
        return history

    def prune_checkpoints(self):
        directory = self.root / "checkpoints"
        keep = set(self.state.get("epoch_checkpoints", []))
        checkpoints = sorted((p for p in directory.iterdir() if (p / "heads.pt").exists()), key=lambda p: p.stat().st_mtime)
        for path in checkpoints[:-3]:
            if str(path) not in keep:
                shutil.rmtree(path)

    def drop_scratch(self, metadata_list, name):
        for meta in metadata_list:
            shutil.rmtree(meta["directory"], ignore_errors=True)
        shutil.rmtree(self.scratch / f"{name}_work", ignore_errors=True)
        shutil.rmtree(self.scratch / name, ignore_errors=True)

    # ---------- phases ----------
    def screen(self):
        tasks = [t for t in self.config["candidates"] if t not in self.state["screened"]]
        batches = [{"name": f"screen_{t}", "task": t, "level": "default",
                    "seed": stable_seed(self.config["seed"], "track", t, "default"), "epoch": 0} for t in tasks]
        metadata = self.collect_all(batches, "screen")
        if metadata is None:
            return False
        for meta in metadata:
            episodes = self.episode_rows([meta], "screen")
            self.write_parquet(episodes, self.counters / f"episodes_screen_{meta['task']}.parquet")
            self.state["screened"][meta["task"]] = {
                "episodes": len(episodes), "successes": sum(e["success"] for e in episodes),
                "held": sum(e["ever_held"] for e in episodes), "directory": meta["directory"],
                "seed": meta["seed"], "rollout_s": meta["rollout_s"],
                "env_steps": sum(e["steps"] for e in episodes),
            }
        self.state["env_steps"] += sum(s["env_steps"] for s in self.state["screened"].values())
        self.save()
        return True

    def select(self):
        rule = self.config["selection"]
        table = []
        for task, stats in self.state["screened"].items():
            n, s, h = stats["episodes"], stats["successes"], stats["held"]
            group = ("retention" if s >= rule["retention_min_success"] else
                     "dead" if h < rule["min_held"] else
                     "train" if s <= rule["max_success"] else "retention")
            table.append({"task": task, "episodes": n, "successes": s, "held": h, "group": group,
                          "contrast": (s / n) * (1 - s / n)})
        eligible = sorted((r for r in table if r["group"] == "train"), key=lambda r: (-r["contrast"], -r["held"], r["task"]))
        selected = [r["task"] for r in eligible[:self.config["max_train_tasks"]]]
        if not selected:
            raise RuntimeError("No task passed the selection rule")
        for r in table:
            r["selected"] = r["task"] in selected
        write_json(self.root / "selection.json", {"rule": rule, "table": table, "selected": selected})
        self.state["selected"] = selected
        for task, stats in self.state["screened"].items():
            if task not in selected:
                shutil.rmtree(stats["directory"], ignore_errors=True)
        self.state["phase"] = "round0"
        self.save()

    def round0(self):
        batches = [{"name": f"round0_{t}_specific", "task": t, "level": "specific",
                    "seed": stable_seed(self.config["seed"], "round0", t, "specific"), "epoch": 0}
                   for t in self.state["selected"]]
        metadata = self.collect_all(batches, "round0")
        if metadata is None:
            return False
        screened = [read_json(Path(self.state["screened"][t]["directory"]) / "batch.json") for t in self.state["selected"]]
        everything = screened + metadata
        self.check_disk()
        self.process(everything, "round0")
        self.drop_scratch(everything, "round0")
        for task in self.state["selected"]:
            shutil.rmtree(self.state["screened"][task]["directory"], ignore_errors=True)
        shutil.rmtree(self.scratch / "screen", ignore_errors=True)
        self.state.update(phase="train", epoch=1, collection=0, train_started=time.time())
        self.save()
        return True

    def epoch_plan(self, epoch):
        batches = []
        # train_v2: explicit (task, level) pairs; v1: every selected task at both levels.
        pairs = self.config.get("pairs") or [[t, level] for t in self.state["selected"] for level in ("default", "specific")]
        for task, level in pairs:
            if level == "default":   # tracked layouts: the screening seed, identical every epoch
                seed = stable_seed(self.config["seed"], "track", task, "default")
            else:                    # fresh layouts every epoch
                seed = stable_seed(self.config["seed"], "fresh", task, level, epoch)
            batches.append({"task": task, "level": level, "seed": seed})
        random.Random(stable_seed(self.config["seed"], "order", epoch)).shuffle(batches)
        size = self.config["batches_per_collection"]
        collections = [batches[i:i + size] for i in range(0, len(batches), size)]
        for c, group in enumerate(collections):
            for b in group:
                b["name"] = f"e{epoch:02d}c{c}_{b['task']}_{b['level']}"
                b["epoch"] = epoch
        return collections

    def train(self):
        hours = self.config["train_hours"]
        if not self.state.get("train_started"):
            self.state["train_started"] = time.time()
            self.save()
        while not self.stop_requested():
            if self.state["epoch"] > self.config["max_epochs"]:
                return "done"
            if time.time() - self.state["train_started"] > hours * 3600:
                return "done"
            collections = self.epoch_plan(self.state["epoch"])
            index = self.state["collection"]
            if index >= len(collections):
                self.state["epoch_checkpoints"] = self.state.get("epoch_checkpoints", []) + [self.state["checkpoint"]]
                self.state.update(epoch=self.state["epoch"] + 1, collection=0)
                self.save()
                continue
            name = f"e{self.state['epoch']:02d}c{index}"
            self.check_disk()
            metadata = self.collect_all(collections[index], name)
            if metadata is None:
                return "stopped"
            self.process(metadata, name)
            self.drop_scratch(metadata, name)
            self.state["collection"] = index + 1
            self.save()
        return "stopped"

    def main(self):
        self.lanes = [ResidentWorkers(self.root / f"lane_{i}") for i in range(2)]
        for lane in self.lanes:
            lane.root.mkdir(exist_ok=True)
        try:
            if self.state["phase"] == "screen":
                if not self.screen():
                    return self.note(phase="stopped")
                self.state["phase"] = "select"
                self.save()
            if self.state["phase"] == "select":
                self.select()
            self.actors = Actors(self.root, self.config)
            if self.state["phase"] == "round0":
                if not self.round0():
                    return self.note(phase="stopped")
            if self.state["phase"] == "train":
                outcome = self.train()
                self.note(phase=outcome)
        except Exception as error:
            self.note(phase="error", error=repr(error))
            raise
        finally:
            self.save()
            if self.actors:
                self.actors.close()
            for lane in self.lanes:
                lane.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--resume", action="store_true", help="Clear a previous graceful STOP request.")
    args = parser.parse_args()
    root = args.root.resolve()
    with (root / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.resume:
            (root / "STOP").unlink(missing_ok=True)
        Run(root).main()


if __name__ == "__main__":
    main()
