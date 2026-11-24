"""Offline review of a train_v1 run from its counters (read-only).

Usage: python -m training.v1_review [run_dir] [--json out.json]
Episode sources are deduplicated: screening batches are the base `default` reference (epoch 0) and are
not re-counted from the round-0 counters; round-0 `specific` batches are the base `specific` reference.
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT = Path(__file__).resolve().parent / "runs" / "v1_T_01"


def load(root):
    selected = json.loads((root / "selection.json").read_text())["selected"]
    order = [json.loads(line)["collection"] for line in (root / "history.jsonl").read_text().splitlines()]
    episodes = []
    for path in glob.glob(str(root / "counters" / "episodes_*.parquet")):
        frame = pd.read_parquet(path)
        name = Path(path).stem.removeprefix("episodes_")
        if name.startswith("screen"):
            frame = frame.assign(ep=0, level="default", source="screen")
        elif name == "round0":
            frame = frame[~frame.batch.str.startswith("screen_")].assign(ep=0, source="round0")
        else:
            frame = frame.assign(ep=int(name[1:3]), source="train")
        episodes.append(frame)
    episodes = pd.concat(episodes, ignore_index=True)
    episodes = episodes[episodes.task.isin(selected)]
    chunks = pd.concat([pd.read_parquet(root / "counters" / f"chunks_{c}.parquet") for c in order], ignore_index=True)
    chunks["ep"] = chunks.collection.map(lambda c: 0 if c == "round0" else int(c[1:3]))
    steps = [pd.read_parquet(p).assign(collection=c) for c in order
             if (p := root / "counters" / f"steps_{c}.parquet").exists()]
    steps = pd.concat(steps, ignore_index=True) if steps else pd.DataFrame()
    history = pd.DataFrame([json.loads(line) for line in (root / "history.jsonl").read_text().splitlines()])
    return selected, episodes, chunks, steps, history


def bootstrap(values, rng, n=5000):
    values = np.asarray(values, dtype=float)
    draws = rng.choice(values, (n, len(values))).mean(axis=1)
    return values.mean(), np.percentile(draws, 2.5), np.percentile(draws, 97.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path, nargs="?", default=DEFAULT)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    root = args.run.resolve()
    selected, episodes, chunks, steps, history = load(root)
    rng = np.random.default_rng(0)
    pd.set_option("display.width", 220)
    report = {}

    epochs = sorted(episodes.ep.unique())
    complete = [e for e in epochs if e == 0 or
                len(episodes[(episodes.ep == e)].groupby(["task", "level"])) == 2 * len(selected)]
    print(f"== {root.name}: tasks {', '.join(selected)}")
    print(f"epochs with all {2 * len(selected)} (task, level) batches: {complete}")

    print("\n-- success / 12 by task, level, epoch")
    for level in ("default", "specific"):
        table = episodes[episodes.level == level].pivot_table(index="task", columns="ep", values="success",
                                                              aggfunc="sum")
        print(f"[{level}]\n{table.to_string()}")
    summary = episodes.groupby(["level", "ep"]).agg(
        n=("success", "size"), success=("success", "mean"), final_p=("final_p", "mean"),
        held=("ever_held", "mean"), drops=("drops", "mean"), wrong_grabs=("wrong_grab_events", "mean"),
        steps=("steps", "mean")).round(3)
    print(f"\n-- pooled by level and epoch\n{summary.to_string()}")
    report["pooled"] = summary.reset_index().to_dict("records")

    print("\n-- paired Δ vs base on tracked `default` layouts (same seed → same layout per env); 95% bootstrap CI")
    base = episodes[(episodes.source == "screen")][["task", "env", "success", "final_p", "steps"]]
    rows = []
    for ep in [e for e in epochs if e > 0]:
        trained = episodes[(episodes.ep == ep) & (episodes.level == "default")]
        merged = base.merge(trained, on=["task", "env"], suffixes=("_b", "_t"))
        if merged.empty:
            continue
        ds = bootstrap(merged.success_t.astype(float) - merged.success_b.astype(float), rng)
        dp = bootstrap(merged.final_p_t - merged.final_p_b, rng)
        rows.append({"epoch": ep, "n": len(merged), "d_success": ds[0], "lo": ds[1], "hi": ds[2],
                     "d_final_p": dp[0], "p_lo": dp[1], "p_hi": dp[2],
                     "tasks": len(merged.task.unique())})
    paired = pd.DataFrame(rows).round(3)
    print(paired.to_string(index=False))
    report["paired_default"] = paired.to_dict("records")

    print("\n-- per task: paired Δ success on tracked layouts, epochs ≥ 1 pooled (descriptive)")
    trained = episodes[(episodes.ep > 0) & (episodes.level == "default")]
    merged = base.merge(trained, on=["task", "env"], suffixes=("_b", "_t"))
    per_task = merged.groupby("task").apply(
        lambda g: pd.Series({"base": g.success_b.mean(), "trained": g.success_t.mean(),
                             "d": (g.success_t.astype(float) - g.success_b.astype(float)).mean()}),
        include_groups=False).round(3)
    print(per_task.to_string())

    print("\n-- efficiency and failure modes (successful episodes: steps to success)")
    wins = episodes[episodes.success].groupby(["level", "ep"]).steps.median().unstack(0)
    print(wins.to_string())

    if not steps.empty:
        print("\n-- update health per collection")
        health = steps.groupby("collection", sort=False).agg(
            opt_steps=("step", "size"), grad=("grad_norm", "mean"), action_d_ref=("action_d_ref", "mean"),
            video_d_ref=("video_d_ref", "mean"), adapter_norm=("adapter_norm", "last")).round(4)
        print(health.to_string())
    hist = history[["collection", "episodes", "successes", "update_chunks", "chunks", "rollout_s", "update_s"]]
    print(f"\n-- wall time: rollout {hist.rollout_s.sum() / 3600:.1f} h, update {hist.update_s.sum() / 3600:.1f} h, "
          f"update share of chunks {hist.update_chunks.sum() / hist.chunks.sum():.2f}")

    print("\n-- action preference distribution by epoch (all chunks)")
    print(chunks.groupby("ep").p_action.describe(percentiles=[0.1, 0.5, 0.9]).round(3).to_string())

    print("\n-- dream–reality diagnostics (judge logged only in arm T)")
    chunks["agree"] = chunks.dream_stage == chunks.p_end
    readable = chunks[chunks.dream_stage.notna()]
    print(f"dream stage readable on {len(readable) / len(chunks):.2f} of chunks; gate ok on {(chunks.gate == 'ok').mean():.2f}")
    print(readable.groupby("ep").agree.mean().round(3).rename("dream==sim stage").to_string())
    progress = readable.assign(progressed=readable.p_end > readable.p_start)
    print(progress.groupby("agree").progressed.mean().round(3).rename("P(progress | dream agrees)").to_string())

    if args.json:
        args.json.write_text(json.dumps(report, indent=2, default=float) + "\n")


if __name__ == "__main__":
    main()
