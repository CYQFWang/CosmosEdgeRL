import unittest
from types import SimpleNamespace

import torch

from training.native_progress import independent_success_checks, record_native_progress
from training.reward import score_episode, update_values
from training.rollout_rewards import native_chunks


def object_in_container(env, object, *, container, require_gripper_detached, logical="all", K=1):
    assert container == "bin" and require_gripper_detached
    objects = [object] if isinstance(object, str) else object
    count = sum(env.state[:, ["mug1", "mug2"].index(obj)].long() for obj in objects)
    return count == (K if logical == "choose" else len(objects))


CONDITIONALS = SimpleNamespace(
    object_in_container=object_in_container, object_inside=None, object_on_top=None,
    object_outside_of=None, object_outside_of_and_on_surface=None, object_upright=None,
)


class FakeEnv:
    num_envs = 2

    def __init__(self):
        self.state = torch.tensor([[True, False], [False, False]])
        self.index = 0
        self.finished = {}
        self.term = SimpleNamespace(func=object_in_container, time_out=False, params={
            "object": ["mug1", "mug2"], "container": "bin", "require_gripper_detached": True,
        })
        self.termination_manager = SimpleNamespace(
            active_terms=["timeout", "success"],
            get_term_cfg=lambda name: SimpleNamespace(time_out=True) if name == "timeout" else self.term,
            get_term=lambda name: self.term.func(self, **self.term.params),
        )

    @property
    def active_env_ids(self):
        return [i for i in range(self.num_envs) if i not in self.finished]

    def step(self, _):
        self.index += 1
        if self.index == 32:
            self.state[0, 0] = False
            self.state[1, 0] = True
        if self.index == 37:
            self.state[0] = True
            self.finished[0] = {"step": 37, "success": True}
        if self.index == 64:
            self.finished[1] = {"step": 64, "success": False}

    def get_env_results(self):
        return [dict(env_id=i, **self.finished.get(i, {"step": None, "success": None}))
                for i in range(self.num_envs)]


def rollout(env):
    while env.active_env_ids:
        env.step(None)
    return env.get_env_results(), [], {}


class NativeProgressTest(unittest.TestCase):
    def test_collect_and_score_live_boundaries(self):
        env = FakeEnv()
        rows, _, _ = record_native_progress(rollout, env, conditionals=CONDITIONALS, horizon=32)
        for row, boundaries, expected in zip(
            rows,
            ([[0, 1], [32, 0], [37, 2]], [[0, 0], [32, 1], [64, 1]]),
            ([-1, 4], [1, 0]),
            strict=True,
        ):
            progress = row["episode_metadata"]["native_progress"]
            self.assertEqual(progress["boundaries"], boundaries)
            log = dict(native_progress=progress, success=row["success"], final_step=row["step"], events=[])
            chunks = native_chunks(log)
            values = {0: 0, 1: 2, 2: 100}
            scores = score_episode(chunks, values)
            self.assertEqual([item["reward"] for item in scores], expected)
            expected_advantage = 4 if row["success"] else 0.95 * 2 - 2
            self.assertAlmostEqual(scores[-1]["advantage"], expected_advantage)
            log["events"].append({"name": "WRONG_OBJECT_GRABBED_FAILURE", "step": 32})
            flagged = native_chunks(log)
            self.assertEqual(flagged[0].wrong_grabs, 1)
            self.assertEqual(score_episode(flagged, values), scores)
            self.assertEqual(update_values([flagged], values), update_values([chunks], values))

    def test_counted_and_multi_term_success_use_binary_credit(self):
        env = FakeEnv()
        env.term.params.update(logical="choose", K=1)
        self.assertEqual(independent_success_checks(env.termination_manager, CONDITIONALS), (None, ()))
        env.state[1] = True
        self.assertEqual(env.termination_manager.get_term("success").tolist(), [True, False])
        env.term.params["logical"] = "all"
        env.termination_manager.active_terms.append("another_success")
        self.assertEqual(independent_success_checks(env.termination_manager, CONDITIONALS), (None, ()))
        log = dict(native_progress={"condition_count": 1}, success=True, final_step=32, events=[])
        self.assertEqual([row["reward"] for row in score_episode(native_chunks(log), {})], [4])

    def test_disagreement_falls_back_for_the_whole_batch(self):
        env = FakeEnv()
        env.termination_manager.get_term = lambda name: torch.zeros(2, dtype=torch.bool)
        rows, _, _ = record_native_progress(rollout, env, conditionals=CONDITIONALS, horizon=32)
        for row in rows:
            progress = row["episode_metadata"]["native_progress"]
            self.assertEqual(progress["condition_count"], 1)
            self.assertIn("fallback", progress)


if __name__ == "__main__":
    unittest.main()
