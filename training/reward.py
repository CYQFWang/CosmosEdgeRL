from collections import defaultdict
from dataclasses import dataclass
from statistics import mean


@dataclass(frozen=True)
class Chunk:
    start: int
    end: int
    executed: int
    wrong_grabs: int = 0
    success: bool = False
    real_stage: int | None = None
    dream_stage: int | None = None
    condition_count: int | None = None

    def __post_init__(self):
        if not 1 <= self.executed <= 32 or min(self.start, self.end, self.wrong_grabs) < 0:
            raise ValueError("Invalid simulator chunk")
        if self.condition_count is not None and (
            self.condition_count < 1 or max(self.start, self.end) > self.condition_count
            or self.success and self.end != self.condition_count
        ):
            raise ValueError("Invalid native condition counts")


@dataclass(frozen=True)
class Settings:
    stage_weight: float = 1.0
    success_weight: float = 2.0
    wrong_grab_weight: float = 1.0
    gamma: float = 0.95
    gae_lambda: float = 0.8
    honesty_weight: float = 0.25


def reward(chunk, settings):
    if chunk.condition_count is not None:
        return 2 * (chunk.end - chunk.start) / chunk.condition_count + settings.success_weight * chunk.success
    return (settings.stage_weight * (chunk.end - chunk.start)
            + settings.success_weight * chunk.success
            - settings.wrong_grab_weight * chunk.wrong_grabs)


def score_episode(chunks, values, settings=Settings()):
    if any(left.success or left.end != right.start for left, right in zip(chunks, chunks[1:])):
        raise ValueError("Expected one contiguous episode with success only at the end")
    scored = []
    carry = 0.0
    for chunk in reversed(chunks):
        observed = reward(chunk, settings)
        successor = 0.0 if chunk.success else values.get(chunk.end, 0.0)
        carry = observed + settings.gamma * successor - values.get(chunk.start, 0.0) + (
            0.0 if chunk.success else settings.gamma * settings.gae_lambda * carry
        )
        honesty = None
        if chunk.executed == 32 and chunk.real_stage is not None and chunk.dream_stage is not None:
            real, dream = chunk.real_stage, chunk.dream_stage
            gap = settings.stage_weight * (dream - real) + settings.gamma * (
                values.get(dream, 0.0) - values.get(real, 0.0)
            )
            honesty = -abs(gap)
        scored.append({"reward": observed, "advantage": carry, "honesty": honesty})
    return scored[::-1]


def update_values(episodes, values, settings=Settings(), rate=0.1):
    samples = defaultdict(list)
    for chunks in episodes:
        if not chunks:
            continue
        future = values.get(chunks[-1].end, 0.0)
        for chunk in reversed(chunks):
            future = reward(chunk, settings) + settings.gamma * (0.0 if chunk.success else future)
            samples[chunk.start].append(future)
    return {**values, **{stage: (1 - rate) * values.get(stage, 0.0) + rate * mean(returns)
                         for stage, returns in samples.items()}}


def visual_stage(current, next_done, start):
    if not isinstance(next_done, bool) or (start == 1 and not isinstance(current, bool)):
        raise ValueError("Stage requires discrete judge answers")
    if start == 0:
        return int(next_done)
    if start == 1:
        return 2 if next_done else int(current)
    raise ValueError("Only single-object two-subgoal tasks are calibrated")


def observed_visual_stage(answers, start):
    if not answers:
        return None
    next_done = answers["next"]["yes"]
    current = answers.get("current", {}).get("yes")
    if next_done is None or (start == 1 and (current is None or (current and next_done))):
        return None
    return visual_stage(current, next_done, start)
