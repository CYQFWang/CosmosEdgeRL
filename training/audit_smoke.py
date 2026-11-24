"""Check the split, critic abstention, and terminal reward contract."""

import json
from pathlib import Path

from training.reward import Chunk, observed_visual_stage, score_episode
from training.task_split import LEVELS, METADATA, build


def main():
    manifest = json.loads(Path(__file__).with_name("task_split.json").read_text())
    splits = manifest["splits"]
    assert splits == build(json.loads(METADATA.read_text()), manifest["seed"])
    assert {name: len(tasks) for name, tasks in splits.items()} == {"train": 96, "validation": 12, "test": 12}
    tasks, scenes, prompts = set(), set(), set()
    for rows in splits.values():
        assert all(set(row["prompts"]) == set(LEVELS) for row in rows)
        names = {row["task"] for row in rows}
        layouts = {row["scene"] for row in rows}
        instructions = {" ".join(prompt.casefold().split()) for row in rows for prompt in row["prompts"].values()}
        assert len(names) == len(rows)
        assert not (tasks & names or scenes & layouts or prompts & instructions)
        tasks.update(names)
        scenes.update(layouts)
        prompts.update(instructions)
    assert observed_visual_stage({"next": {"yes": None}}, 0) is None
    assert observed_visual_stage({"current": {"yes": True}, "next": {"yes": True}}, 1) is None
    assert observed_visual_stage({"current": {"yes": False}, "next": {"yes": False}}, 1) == 0
    scores = score_episode([
        Chunk(0, 1, 32, real_stage=1, dream_stage=0),
        Chunk(1, 2, 7, success=True, real_stage=2, dream_stage=1),
    ], {2: 99})
    assert scores[0]["honesty"] == -1 and scores[1]["honesty"] is None
    assert scores[1]["reward"] == scores[1]["advantage"] == 3
    print("PASS: scene and prompt holdouts, discrete abstention, and terminal reward")


if __name__ == "__main__":
    main()
