"""train_v1 reward, return and NFT preferences (train_v1.md §1–§2, §4.2–§4.3).

Action branch:  r_k = ΔP_k + S_k,  G_k = r_k + γ G_{k+1},  p^a = ½ + ½ clip(G_k / (2 - P_k), -1, 1).
Video branch:   r^v_k under V1 (imagined progress) or V2 (realised imagined progress), γ_v = 0,
                p^v = ½ + ½ clip(r^v_k / 1, -1, 1); abstain (p^v = ½) unless the judge reads the
                matched real frame as the simulator stage P_{k+1}.
"""

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

GAMMA = 0.95
HORIZON = 32
STAGES = (0.0, 0.5, 1.0)


@dataclass
class Chunk:
    task: str
    level: str
    run: int
    env: int
    k: int
    start: int
    end: int
    executed: int
    p_start: float
    p_end: float
    success: int
    reward: float = 0.0
    ret: float = 0.0
    g_max: float = 2.0
    p_action: float = 0.5
    terminal: bool = False
    boundary_start: list = field(default_factory=list)
    boundary_end: list = field(default_factory=list)

    @property
    def key(self):
        return f"{self.task}_run{self.run}_env{self.env}_step{self.start:04}"


def boundary_progress(row, *, terminal_success=False):
    """P at a boundary row [step, held, contact, lift, wrong, d_tr, d_gt]."""
    if terminal_success:
        return 1.0
    return 0.5 if row[1] else 0.0


def preference(value, scale):
    return 0.5 + 0.5 * max(-1.0, min(1.0, value / scale))


def episode_chunks(log, *, gamma=GAMMA):
    """Turn one RoboLab episode log with v1_progress metadata into scored chunks."""
    progress = log["v1_progress"]
    if progress["horizon"] != HORIZON:
        raise ValueError("Unexpected policy horizon")
    final = log["final_step"]
    rows = {}
    for row in progress["boundaries"]:
        rows[row[0]] = row  # a terminal row at a multiple of 32 is the same boundary
    starts = list(range(0, final, HORIZON))
    missing = [s for s in starts + [final] if s not in rows]
    if missing:
        raise ValueError(f"Missing progress boundaries {missing} in {log['task']} env {log['env_id']}")
    level, = log["instruction_fields_used"]
    chunks = []
    for k, start in enumerate(starts):
        end = min(start + HORIZON, final)
        terminal = end == final
        success = int(bool(log["success"]) and terminal)
        chunks.append(Chunk(
            task=log["task"], level=level, run=log["run"], env=log["env_id"], k=k, start=start, end=end,
            executed=end - start, p_start=boundary_progress(rows[start]),
            p_end=boundary_progress(rows[end], terminal_success=bool(success)), success=success,
            terminal=terminal, boundary_start=rows[start], boundary_end=rows[end],
        ))
    following = 0.0
    for chunk in reversed(chunks):
        chunk.reward = chunk.p_end - chunk.p_start + chunk.success
        chunk.ret = chunk.reward + gamma * following
        following = chunk.ret
        chunk.g_max = 2.0 - chunk.p_start
        if abs(chunk.ret) > chunk.g_max + 1e-9:
            raise AssertionError("Return exceeds its per-state bound; the clip must never fire")
        chunk.p_action = preference(chunk.ret, chunk.g_max)
    return chunks


def rollout_logs(root):
    for path in sorted((Path(root) / "rollouts").glob("*/log_*_env*.json")):
        log = json.loads(path.read_text())
        if "v1_progress" not in log:
            raise ValueError(f"{path} lacks v1_progress; collect with DREAM_V1_PROGRESS=1")
        yield path, log


def judged_stage(held, placed, *, threshold=0.5):
    """Map judge P(yes) for held/placed to a stage, or None to abstain."""
    if held is None or placed is None or math.isnan(held) or math.isnan(placed):
        return None
    held_yes, placed_yes = held >= threshold, placed >= threshold
    if held_yes and placed_yes:
        return None  # contradictory: held and released at once
    return 1.0 if placed_yes else 0.5 if held_yes else 0.0


def dream_reward(chunk, dream_stage, real_stage, mode):
    """Return (r_v, gate) for V1/V2; r_v None means abstain (p^v = ½)."""
    if mode not in ("v1", "v2"):
        raise ValueError(mode)
    if dream_stage is None or real_stage is None:
        return None, "unreadable"
    if real_stage != chunk.p_end:
        return None, "judge_misreads_real"
    if mode == "v1":
        return dream_stage - chunk.p_start, "ok"
    if dream_stage == chunk.p_end:
        return dream_stage - chunk.p_start, "ok"
    return -abs(dream_stage - chunk.p_end), "ok"


def as_row(chunk):
    row = asdict(chunk)
    for side in ("start", "end"):
        values = row.pop(f"boundary_{side}")
        for name, value in zip(("step", "held", "contact", "lift_m", "wrong_object", "target_receptacle_m",
                                "gripper_target_m"), values, strict=True):
            row[f"{side}_{name}"] = value
    return row
