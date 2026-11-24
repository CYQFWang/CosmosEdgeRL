"""Read-only progress report for a train_v1 run.

Usage: python -m training.v1_status [run_dir] [--last N]
"""

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

DEFAULT = Path(__file__).resolve().parent / "runs" / "v1_T_01"


def read(path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return default


def ago(seconds):
    seconds = int(seconds)
    return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m" if seconds >= 3600 else f"{seconds // 60}m{seconds % 60:02d}s"


def alive(pid):
    return pid and Path(f"/proc/{pid}").exists()


def batch_outcomes(directory):
    """Success / held counts from a finished batch's RoboLab logs (no scoring needed)."""
    success = held = episodes = 0
    for path in (directory / "rollout" / "rollouts").glob("*/log_*_env*.json"):
        log = read(path, {})
        episodes += 1
        success += bool(log.get("success"))
        held += any(row[1] for row in log.get("v1_progress", {}).get("boundaries", []))
    return episodes, success, held


def pair_report(root, config):
    """train_v2: successes per (task, level) and epoch, and the paired change on tracked `default` layouts."""
    try:
        import numpy as np
        import pandas as pd
    except ImportError:
        print("\n-- per pair: pandas unavailable")
        return
    files = sorted(root.glob("counters/episodes_e*.parquet"))
    if not files:
        return
    eps = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    eps["ep"] = eps.collection.str[1:3].astype(int)
    table = eps.groupby(["task", "level", "ep"]).success.sum().unstack("ep")
    base_run = Path(config.get("base_run") or (root.parent / "v1_T_01"))
    screen = sorted(base_run.glob("counters/episodes_screen_*.parquet"))
    base = pd.concat([pd.read_parquet(f) for f in screen], ignore_index=True) if screen else None
    round0 = base_run / "counters" / "episodes_round0.parquet"
    spec = pd.read_parquet(round0) if round0.exists() else None
    if base is not None:
        def base_value(task, level):
            frame = base if level == "default" else (spec[spec.level == "specific"] if spec is not None else None)
            return None if frame is None else int(frame[frame.task == task].success.sum())
        table.insert(0, "base", [base_value(t, l) for t, l in table.index])
    print(f"\n-- successes per pair (of 12) by epoch; base: `default` = screening (same layouts), "
          f"`specific` = {base_run.name} round 0 (different layouts)")
    print("  " + table.to_string().replace("\n", "\n  "))
    if base is None:
        return
    rng = np.random.default_rng(0)
    for ep in sorted(eps.ep.unique()):
        merged = eps[(eps.ep == ep) & (eps.level == "default")].merge(base, on=["task", "env"], suffixes=("_t", "_b"))
        if merged.empty:
            continue
        diff = (merged.success_t.astype(float) - merged.success_b.astype(float)).values
        draws = rng.choice(diff, (4000, len(diff))).mean(axis=1)
        partial = " (partial epoch)" if len(merged) < 12 * sum(level == "default" for _, level in config["pairs"]) else ""
        print(f"  epoch {ep}: Δsuccess vs base on tracked `default` layouts {diff.mean():+.3f} "
              f"[{np.percentile(draws, 2.5):+.3f}, {np.percentile(draws, 97.5):+.3f}] n={len(diff)}{partial}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path, nargs="?", default=DEFAULT)
    parser.add_argument("--last", type=int, default=6)
    args = parser.parse_args()
    root = args.run.resolve()
    config, status = read(root / "config.json", {}), read(root / "status.json", {})
    state = read(root / "state.json") or ({"phase": "not started (queued)"} if config.get("start") == "train"
                                          else {"phase": "screen (state.json is written when screening finishes)"})
    now = time.time()

    print(f"== {root.name}  arm={config.get('arm')}  now={time.strftime('%Y-%m-%d %H:%M:%S')}")
    pid = status.get("pid")
    print(f"process {pid}: {'RUNNING' if alive(pid) else 'NOT RUNNING'}   phase={status.get('phase')}  "
          f"(status updated {ago(now - status.get('time', now))} ago)")
    if "error" in status:
        print(f"!! error: {status['error']}")
    if (root / "STOP").exists():
        print("!! STOP requested")
    if status.get("active") and alive(pid):
        print(f"active batches: {', '.join(status['active'])}   remaining in this collection: {status.get('remaining')}")
    print(f"state: phase={state.get('phase')} epoch={state.get('epoch')} collection={state.get('collection')} "
          f"env_steps={state.get('env_steps')} version={str(state.get('version'))[:12]}")
    history_rows = [json.loads(line) for line in (root / "history.jsonl").read_text().splitlines()] \
        if (root / "history.jsonl").exists() else []
    if state.get("train_started"):
        end = now if alive(pid) or not history_rows else history_rows[-1]["time"]
        limit = (f"limit {config.get('max_epochs')} epochs" if config.get("train_hours", 0) >= 999
                 else f"of {config.get('train_hours')}h budget")
        print(f"training wall time: {ago(end - state['train_started'])} ({limit})"
              + ("" if alive(pid) else ", until the last collection"))
    queue = root.parent / "v2_queue.log"
    if config.get("pairs") and queue.exists():
        print("queue: " + " | ".join(queue.read_text().splitlines()[-2:]))

    # Screening progress (finished batches in scratch, or the frozen summary in state.json).
    scratch = Path(config.get("scratch", "/nonexistent"))
    screened = state.get("screened") or {}
    rows = []
    if screened:
        rows = [(task, s["episodes"], s["successes"], s["held"], s["rollout_s"]) for task, s in screened.items()]
    elif (scratch / "screen").exists():
        for directory in sorted((scratch / "screen").iterdir()):
            meta = read(directory / "batch.json")
            if meta:
                rows.append((meta["task"], *batch_outcomes(directory), meta["rollout_s"]))
    candidates = config.get("candidates", [])
    if rows or str(state.get("phase")).startswith("screen"):
        print(f"\n-- screening: {len(rows)}/{len(candidates)} tasks done")
        for task, n, s, h, seconds in sorted(rows, key=lambda r: (-r[3], r[0])):
            print(f"  {task:34s} success {s:2d}/{n}  held {h:2d}/{n}  ({ago(seconds)})")

    selection = read(root / "selection.json")
    if selection:
        print(f"\n-- selection (rule {selection['rule']}): {', '.join(selection['selected'])}")
        groups = {}
        for row in selection["table"]:
            groups.setdefault(row["group"], []).append(row["task"])
        for group, tasks in groups.items():
            print(f"  {group:9s} {len(tasks):2d}: {', '.join(sorted(tasks))}")

    history = [json.loads(line) for line in (root / "history.jsonl").read_text().splitlines()] \
        if (root / "history.jsonl").exists() else []
    if history:
        print(f"\n-- collections ({len(history)} total, last {args.last})")
        judge = bool(config.get("judge"))
        print("  name        eps  succ  meanP  chunks  upd  p_a+/-     p_v+/-    "
              + ("abst " if judge else "align  cov  ") + " roll    dec  " + ("judge  " if judge else "") + " upd   steps")
        for h in history[-args.last:]:
            extra = (f"{h.get('abstain') or 0:5d} " if judge else
                     f"{h['align_mean']:+.3f} {h.get('align_coverage') or 0:.2f} " if h.get("align_mean") is not None
                     else "  -     -   ")
            print(f"  {h['collection']:10s} {h['episodes']:4d} {h['successes']:5d}  {h['mean_final_p'] or 0:.2f}  "
                  f"{h['chunks']:6d} {h['update_chunks']:4d}  {h['p_action_pos']:4d}/{h['p_action_neg']:<4d} "
                  f"{h['p_video_pos']:4d}/{h['p_video_neg']:<4d} {extra} {ago(h['rollout_s']):>6s} "
                  f"{ago(h['decode_s']):>5s} " + (f"{ago(h['judge_s']):>6s} " if judge else "")
                  + f"{ago(h['update_s']):>6s} {h['optimizer_steps']:4d}")
        # Per-task success on the tracked `default` seeds by epoch is in counters/episodes_*.parquet.
        last = history[-1]
        print("  last per-task:", "; ".join(f"{t} {v['successes']}/{v['episodes']} (held {v['held']})"
                                          for t, v in last["per_task"].items()))

    if config.get("pairs"):
        pair_report(root, config)

    print("\n-- GPUs")
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout
        print("  " + out.strip().replace("\n", "\n  "))
    except (OSError, subprocess.TimeoutExpired):
        print("  nvidia-smi unavailable")
    for path in (root, scratch):
        if path.exists():
            usage = shutil.disk_usage(path)
            print(f"-- disk {path}: {usage.free / 2**30:.0f} GiB free")

    print("\n-- recent problems (supervisor.log, actor logs)")
    found = False
    for log in [root / "supervisor.log", *sorted(root.glob("actor_rank*.log"))]:
        if not log.exists():
            continue
        lines = log.read_text(errors="replace").splitlines()[-400:]
        hits = [line for line in lines if any(word in line for word in ("Traceback", "Error", "error:", "Killed"))
                and "Warning" not in line]
        if hits:
            found = True
            print(f"  {log.name}: " + " | ".join(hits[-3:])[:600])
    if not found:
        print("  none")


if __name__ == "__main__":
    main()
