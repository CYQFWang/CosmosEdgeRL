"""Parse the simulator reward once for both training and prompt-level reports."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from training.reward import Chunk, observed_visual_stage, score_episode
from training.task_split import LEVELS
from vlm_critic_eval.prepare import stage_at, stage_transitions


def prompt_level(log):
    fields = log.get("instruction_fields_used", [])
    return fields[0] if len(fields) == 1 and fields[0] in LEVELS else "unknown"


def critic_key(log):
    progress = log.get("native_progress")
    if progress is not None:
        return f"{log['task']}/native/{progress['condition_count']}"
    return log["task"]


def native_chunks(log):
    progress = log.get("native_progress")
    final = log["final_step"]
    success = log["success"]
    if not isinstance(progress, dict) or type(final) is not int or final < 1 or type(success) is not bool:
        raise ValueError("Native reward requires a complete episode with native progress metadata")
    count = progress["condition_count"]
    if type(count) is not int or count < 1:
        raise ValueError("Invalid native condition count")
    steps = [*range(0, final, 32), final]
    if count == 1:
        stages = [0] * (len(steps) - 1) + [int(success)]
    else:
        recorded_steps, stages = zip(*progress["boundaries"], strict=True)
        if (list(recorded_steps) != steps
                or any(type(stage) is not int or not 0 <= stage <= count for stage in stages)
                or (stages[-1] == count) != success):
            raise ValueError("Native predicate counts do not align with episode boundaries or success")
    # Keep raw flags visible in reports. Native reward and GAE ignore them.
    wrong_steps = [event["step"] for event in log["events"] if event["name"].startswith("WRONG_OBJECT_GRABBED")]
    return [
        Chunk(stage, stages[i + 1], end - start, sum(start < step <= end for step in wrong_steps),
              success and end == final, condition_count=count)
        for i, (start, end, stage) in enumerate(zip(steps[:-1], steps[1:], stages[:-1], strict=True))
    ]


def episodes(root, labels=None, values=None, *, reward_mode="legacy"):
    labels = labels or {}
    values = values or {}
    for path in sorted((root / "rollouts").glob("*/log_*_env*.json")):
        log = json.loads(path.read_text())
        if reward_mode == "native":
            chunks = native_chunks(log)
            traces = [root / "capture" / f"{log['task']}_run{log['run']}_env{log['env_id']}_step{start:04}_nft.pt"
                      for start in range(0, log["final_step"], 32)]
            scored = score_episode(chunks, values.get(critic_key(log), {}))
            yield log, list(zip(traces, chunks, scored, strict=True))
            continue
        if "native_progress" in log:
            raise ValueError(f"Use --reward-mode native for {path}")
        cfg = json.loads((path.parent / "env_cfg.json").read_text())
        subtasks = cfg["subtasks"]
        if len(subtasks) != 1 or len(subtasks[0]["conditions"]) != 1:
            raise ValueError(f"{log['task']} needs a multi-object or ordered stage parser")
        ((target, chain),) = subtasks[0]["conditions"].items()
        if len(chain) != 2 or "object_grabbed" not in chain[0][0] or not any(
            predicate in chain[1][0] for predicate in ("object_in_container", "object_on_top")
        ):
            raise ValueError(f"{log['task']} is outside the calibrated grasp/place reward contract")
        transitions = stage_transitions(log, target)
        chunks, traces = [], []
        for start in range(0, log["final_step"], 32):
            end = min(start + 32, log["final_step"])
            terminal = log["success"] and end == log["final_step"]
            start_stage = stage_at(transitions, start)
            end_stage = 2 if terminal else stage_at(transitions, end)
            key = f"{log['task']}_run{log['run']}_env{log['env_id']}_step{start:04}"
            visual = [observed_visual_stage(labels.get(f"{key}_{side}32"), start_stage)
                      for side in ("real", "dream")]
            wrong = sum(start < event["step"] <= end and event["name"].startswith("WRONG_OBJECT_GRABBED")
                        and f"'{target}'" not in event["info"] for event in log["events"])
            chunks.append(Chunk(start_stage, end_stage, end-start, wrong, terminal, *visual))
            traces.append(root / "capture" / f"{key}_nft.pt")
        scored = score_episode(chunks, values.get(log["task"], {}))
        yield log, list(zip(traces, chunks, scored, strict=True))


def summarize(rows):
    groups = defaultdict(list)
    for log, chunks in rows:
        rewards = [row["reward"] for _, _, row in chunks]
        honesty = [row["honesty"] for _, _, row in chunks if row["honesty"] is not None]
        episode = {
            "task": log["task"], "env": log["env_id"], "run": log["run"],
            "env_steps": log["final_step"], "success": log["success"],
            "sim_reward_sum": sum(rewards), "chunks": len(chunks),
            "stage_delta": sum(chunk.end - chunk.start for _, chunk, _ in chunks),
            "wrong_grabs": sum(chunk.wrong_grabs for _, chunk, _ in chunks),
            "honesty_sum": sum(honesty), "honesty_scored": len(honesty),
        }
        if progress := log.get("native_progress"):
            count = progress["condition_count"]
            changes = [(chunk.end - chunk.start) / count for _, chunk, _ in chunks]
            episode.update(
                condition_count=count, progress_gain=sum(change for change in changes if change > 0),
                progress_loss=-sum(change for change in changes if change < 0),
                progress_fallback=bool(progress.get("fallback")),
            )
        groups[prompt_level(log)].append(episode)
    report = {}
    for level, group in sorted(groups.items()):
        report[level] = stats = {
            "episodes": len(group), "success_rate": mean(row["success"] for row in group),
            "return_mean": mean(row["sim_reward_sum"] for row in group),
            "reward_per_chunk": sum(row["sim_reward_sum"] for row in group) / sum(row["chunks"] for row in group),
            "honesty_scored": sum(row["honesty_scored"] for row in group),
            "rollouts": group,
        }
        if "condition_count" in group[0]:
            stats.update(
                partial_credit_fraction=mean(row["condition_count"] > 1 for row in group),
                progress_fallbacks=sum(row["progress_fallback"] for row in group),
            )
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reward-mode", choices=("legacy", "native"), default="legacy")
    args = parser.parse_args()
    report = summarize(row for run in args.runs for row in episodes(run, reward_mode=args.reward_mode))
    if not report:
        parser.error("No supported episodes found")
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded)
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
