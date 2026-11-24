"""Build a reproducible, scene-disjoint split of the RoboLab benchmark."""

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

LEVELS = ("vague", "default", "specific")
METADATA = Path("/home/robot_share/RoboLab/robolab/tasks/_metadata/task_metadata.json")
PILOT = {"BananaInBowlTask", "YogurtInBowlTask", "ThrowAwayAppleTask", "WhiteMugsInBinTask"}


def subset(scenes, groups, size):
    hits = {0: ()}
    for scene in scenes:
        count = len(groups[scene])
        for total, previous in tuple(hits.items()):
            if total + count <= size:
                hits.setdefault(total + count, (*previous, scene))
    return hits.get(size)


def build(rows, seed):
    groups = defaultdict(list)
    for row in sorted(rows, key=lambda item: item["task_name"]):
        if any(not row["instruction_variants"].get(level) for level in LEVELS):
            raise ValueError(f"Missing prompt level for {row['task_name']}")
        groups[row["scene"]].append(row)
    if len(rows) != 120 or len({row["task_name"] for row in rows}) != 120:
        raise ValueError("Expected 120 unique benchmark tasks")
    reserved = {row["scene"] for row in rows if row["task_name"] in PILOT}
    prompt_scenes = defaultdict(set)
    for scene, tasks in groups.items():
        for row in tasks:
            for level in LEVELS:
                prompt = " ".join(row["instruction_variants"][level].casefold().split())
                prompt_scenes[prompt].add(scene)
    shared_prompts = [scenes for scenes in prompt_scenes.values() if len(scenes) > 1]
    total = Counter(row["difficulty_label"] for row in rows)
    rng = random.Random(seed)
    pool = sorted(set(groups) - reserved)
    best = None
    for _ in range(512):
        rng.shuffle(pool)
        valid = subset(pool, groups, 12)
        test = subset([scene for scene in pool if scene not in valid], groups, 12) if valid else None
        if not test:
            continue
        valid_scenes, test_scenes = set(valid), set(test)
        if any(
            len({("validation" if scene in valid_scenes else "test" if scene in test_scenes else "train")
                 for scene in scenes}) > 1 for scenes in shared_prompts
        ):
            continue
        counts = (Counter(row["difficulty_label"] for scene in selection for row in groups[scene])
                  for selection in (valid, test))
        error = sum(abs(split_count[key] - count / 10) for split_count in counts for key, count in total.items())
        if best is None or error < best[0]:
            best = error, valid, test
    if best is None:
        raise ValueError("Unable to make scene-disjoint holdouts")
    _, valid, test = best
    splits = {"train": [], "validation": [], "test": []}
    for scene, tasks in sorted(groups.items()):
        name = "validation" if scene in valid else "test" if scene in test else "train"
        splits[name].extend({
            "task": row["task_name"], "scene": scene, "difficulty": row["difficulty_label"],
            "prompts": {level: row["instruction_variants"][level] for level in LEVELS},
        } for row in tasks)
    return splits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, default=METADATA)
    parser.add_argument("--seed", type=int, default=418)
    args = parser.parse_args()
    source = args.metadata.read_bytes()
    print(json.dumps({"seed": args.seed, "metadata_sha256": hashlib.sha256(source).hexdigest(),
                      "splits": build(json.loads(source), args.seed)}, indent=2))


if __name__ == "__main__":
    main()
