"""Summarize matched training and rollout throughput from benchmark artifacts."""

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean


def gpu_samples(path, windows):
    grouped = defaultdict(list)
    with path.open() as stream:
        for row in csv.DictReader(stream, skipinitialspace=True):
            timestamp = datetime.fromisoformat(row["timestamp"].replace("/", "-")).timestamp()
            if any(start <= timestamp <= stop for start, stop in windows):
                grouped[row["uuid"]].append((
                    int(row["utilization.gpu [%]"].split()[0]),
                    int(row["memory.used [MiB]"].split()[0]),
                ))
    return {uuid: {"mean_util_pct": mean(utilization for utilization, _ in values),
                   "peak_mib": max(memory for _, memory in values),
                   "samples": len(values)} for uuid, values in grouped.items()}


def ranks(root):
    rows = []
    for directory in sorted(root.glob("rank_*")):
        for arm in ("control", "mismatch"):
            path = directory / arm / "update_metrics.json"
            if not path.exists():
                continue
            data = json.loads(path.read_text())
            rows.append({"rank": data["lora_rank"], "checkpoint_blocks": data["checkpoint_blocks"],
                         "arm": arm, "chunks_per_s": len(data["chunks"]) / data["update_s"],
                         "update_s": data["update_s"], "peak_allocated_gib": data["peak_allocated_gib"],
                         "steps": data["steps"], "parameters": data["trainable_parameters"],
                         "nonzero_gradients": sum(value > 0 for value in data["gradient_norms"].values()),
                         "gpu": gpu_samples(directory / "gpu.csv", [
                             (data["update_start_unix"], data["update_end_unix"]),
                         ])})
    return sorted(rows, key=lambda row: (row["rank"], row["checkpoint_blocks"], row["arm"]))


def interval_union(intervals):
    merged = []
    for start, stop in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(stop, merged[-1][1]))
        else:
            merged.append((start, stop))
    return merged


def jsonlines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def rollouts(root):
    rows = []
    for directory in sorted(root.iterdir()):
        if not (directory / "cold_wall_s").is_file():
            continue
        workers = list((directory / "workers").glob("*")) or [directory]
        intervals, episodes = [], []
        task_times = defaultdict(float)
        for worker in workers:
            for row in jsonlines(worker / "episodes_timing.jsonl"):
                intervals.append((row["start_unix"], row["end_unix"]))
                task_times[row["task"]] += row["end_unix"] - row["start_unix"]
            episodes.extend(jsonlines(worker / "rollouts/episode_results.jsonl"))
        windows = interval_union(intervals)
        active = sum(stop - start for start, stop in windows)
        overlap = sum(stop - start for start, stop in intervals) - active
        cold = float((directory / "cold_wall_s").read_text())
        steps = sum(row["episode_step"] for row in episodes)
        task_steps = defaultdict(int)
        for episode in episodes:
            task_steps[episode["task_name"]] += episode["episode_step"]
        rows.append({"layout": directory.name, "episodes": len(episodes), "env_steps": steps,
                     "active_s": active, "overlap_s": overlap, "cold_s": cold,
                     "env_steps_per_active_s": steps / active, "env_steps_per_cold_s": steps / cold,
                     "env_steps_per_task_s": {task: count / task_times[task]
                                              for task, count in sorted(task_steps.items())},
                     "gpu": gpu_samples(directory / "gpu.csv", windows)})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("ranks", "rollouts"))
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = ranks(args.root) if args.mode == "ranks" else rollouts(args.root)
    output = json.dumps(rows, indent=2) + "\n"
    if args.output:
        args.output.write_text(output)
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
