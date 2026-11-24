"""Publish rollout, critic, and GPU telemetry to TensorBoard."""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


def write_rewards(writer, report, step):
    for level, stats in report.items():
        prefix = f"prompt/{level}"
        for key, value in stats.items():
            if isinstance(value, (float, int)):
                writer.add_scalar(f"{prefix}/{key}", value, step)
        for index, episode in enumerate(stats["rollouts"], 1):
            for key in ("sim_reward_sum", "stage_delta", "wrong_grabs", "env_steps", "success", "honesty_scored",
                        "progress_gain", "progress_loss", "condition_count", "progress_fallback"):
                if key in episode:
                    writer.add_scalar(f"{prefix}/episode/{key}", episode[key], index)


def write_gpu(writer, source):
    with source.open() as stream:
        first = None
        for row in csv.DictReader(stream, skipinitialspace=True):
            timestamp = datetime.fromisoformat(row["timestamp"].replace("/", "-")).timestamp()
            if first is None:
                first = timestamp
            step = round((timestamp - first) * 1000)
            for field, tag in (
                ("memory.used [MiB]", "memory_mib"), ("utilization.gpu [%]", "utilization_pct"),
                ("utilization.memory [%]", "memory_utilization_pct"), ("power.draw [W]", "power_w"),
            ):
                value = row[field].split()[0]
                if value not in ("[N/A]", "N/A"):
                    writer.add_scalar(f"gpu/{row['uuid']}/{tag}", float(value), step, walltime=timestamp)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logdir", type=Path)
    parser.add_argument("--rewards", type=Path)
    parser.add_argument("--gpu", type=Path)
    parser.add_argument("--judge", type=Path)
    parser.add_argument("--cache", type=Path)
    args = parser.parse_args()
    with SummaryWriter(str(args.logdir)) as writer:
        if args.rewards:
            write_rewards(writer, json.loads(args.rewards.read_text()), 0)
        if args.gpu:
            write_gpu(writer, args.gpu)
        if args.judge:
            for row in json.loads(args.judge.read_text()):
                prefix = f"judge/{row['evidence']}/{row['task']}/stage_{row['start_stage']}"
                for key, value in row.items():
                    if isinstance(value, (float, int)) and key != "start_stage":
                        writer.add_scalar(f"{prefix}/{key}", value, 0)
        if args.cache:
            data = json.loads(args.cache.read_text())
            for key, value in data.items():
                if isinstance(value, (float, int)):
                    writer.add_scalar(f"und/{key}", value, 0)


if __name__ == "__main__":
    main()
