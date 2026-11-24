"""Validate that paired option-4 arms differ only in mismatch routing."""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean


def audit(root):
    arms = {name: json.loads((root / name / "update_metrics.json").read_text())
            for name in ("control", "mismatch")}
    control, mismatch = arms.values()
    assert control["mismatch_weight"] == 0 and mismatch["mismatch_weight"] > 0
    for key in ("steps", "stage_values", "adapter_groups", "trainable_parameters"):
        assert control[key] == mismatch[key], key
    assert control["adapter_groups"] == 30
    deltas = []
    for left, right in zip(control["chunks"], mismatch["chunks"], strict=True):
        assert {key: value for key, value in left.items() if key != "reward_prob"} == {
            key: value for key, value in right.items() if key != "reward_prob"
        }, left["trace"]
        assert left["reward_prob"]["action"] == right["reward_prob"]["action"]
        expected = 0.5 + 0.5 * max(-1, min(1, (
            right["advantage"] + mismatch["mismatch_weight"] * (right["honesty"] or 0)
        ) / 2))
        assert abs(expected - right["reward_prob"]["video"]) < 1e-6
        deltas.append(right["reward_prob"]["video"] - left["reward_prob"]["video"])
    assert any(delta < 0 for delta in deltas), "Mismatch does not change reward probabilities"
    for arm in arms.values():
        assert len(arm["gradient_norms"]) == 30 and min(arm["gradient_norms"].values()) > 0
    by_gpu = {}
    with (root / "gpu.csv").open() as source:
        for row in csv.DictReader(source, skipinitialspace=True):
            by_gpu.setdefault(row["uuid"], []).append((
                int(row["memory.used [MiB]"].split()[0]), int(row["utilization.gpu [%]"].split()[0]),
            ))
    return {
        "matched_chunks": len(deltas),
        "optimizer_steps_per_arm": control["steps"],
        "changed_video_rewards": sum(delta < 0 for delta in deltas),
        "min_video_reward_delta": min(deltas),
        "missing_honesty": sum(row["honesty"] is None for row in control["chunks"]),
        "gpu": {uuid: {
            "peak_device_mib": max(memory for memory, _ in rows),
            "mean_util_pct_full_process": mean(util for _, util in rows),
            "peak_util_pct": max(util for _, util in rows),
        } for uuid, rows in by_gpu.items()},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.root), indent=2))


if __name__ == "__main__":
    main()
