"""train_v1 simulator progress for single-object pick-and-place tasks (train_v1.md §1.2, §4.3, §9.6).

Runs inside the RoboLab process. Every simulator step it evaluates, for every environment,
whether the target object touches the gripper and is lifted at least LIFT_M above its reset
height. At each 32-step policy boundary and at termination it records one row per environment:

    [step, held, contact, lift_m, wrong_object, target_receptacle_m, gripper_target_m]

``held`` is debounced: contact and lift must both have held for the last DEBOUNCE steps.
At termination it also saves the real policy-input composite so the terminal chunk's dream
can be compared with reality (dream frame t_k vs real frame t_k).
"""

import math

import numpy as np
import torch

LIFT_M = 0.03
DEBOUNCE = 8
HORIZON = 32
SCHEMA = 1


def task_objects(manager):
    """Return (target, receptacle, success_term_name) for a single-object placement task."""
    terms = [(name, manager.get_term_cfg(name)) for name in manager.active_terms
             if not manager.get_term_cfg(name).time_out]
    if len(terms) != 1:
        raise ValueError(f"Expected one success term, found {[name for name, _ in terms]}")
    name, term = terms[0]
    target = term.params.get("object")
    if isinstance(target, (list, tuple)):
        if len(target) != 1:
            raise ValueError(f"train_v1 needs one target object, got {target}")
        target = target[0]
    receptacle = term.params.get("container") or term.params.get("reference_object")
    if not isinstance(target, str) or not isinstance(receptacle, str):
        raise ValueError(f"Unsupported success term parameters: {term.params}")
    if term.params.get("logical", "all") not in ("all", None):
        raise ValueError("train_v1 needs logical='all'")
    return target, receptacle, name


def _gripper_body(env):
    try:
        robot = env.scene["robot"]
        ids, names = robot.find_bodies("base_link")
        return (robot, ids[0]) if len(ids) == 1 else None
    except Exception:  # noqa: BLE001 - geometry counter only; NaN when unavailable
        return None


def record_v1_progress(run_episode, env, *, save_terminal, **kwargs):
    from robolab.core.task.conditionals import get_wrong_object_grabbed
    from robolab.core.task.predicate_logic import in_contact
    from robolab.core.world.world_state import get_world

    target, receptacle, _ = task_objects(env.termination_manager)
    world = get_world(env)
    gripper = _gripper_body(env)
    client = kwargs["client"]
    original_step = env.step
    n = env.num_envs
    state = {"step": 0, "z0": None, "run": torch.zeros(n, dtype=torch.long, device=env.device),
             "contact": torch.zeros(n, dtype=torch.bool, device=env.device),
             "lift": torch.zeros(n, device=env.device)}
    boundaries = [[] for _ in range(n)]
    finished = {}
    terminal_frames = {}

    def measure():
        position = world.get_pose(target)[0]
        if state["z0"] is None:
            state["z0"] = position[:, 2].clone()
        contact = in_contact(world, target, "gripper")
        contact = torch.as_tensor(contact, device=env.device).bool().reshape(n)
        lift = position[:, 2] - state["z0"]
        raw = contact & (lift >= LIFT_M)
        state["run"] = torch.where(raw, state["run"] + 1, torch.zeros_like(state["run"]))
        state["contact"], state["lift"] = contact, lift
        return position

    def row(env_id, position):
        receptacle_position = world.get_pose(receptacle, env_id=env_id)[0]
        target_position = position[env_id]
        gripper_target = math.nan
        if gripper is not None:
            robot, body = gripper
            ee = robot.data.body_pos_w[env_id, body] - env.scene.env_origins[env_id]
            gripper_target = float(torch.linalg.norm(ee - target_position))
        wrong = get_wrong_object_grabbed(env, target, env_id=env_id)
        return [state["step"], bool(state["run"][env_id] >= DEBOUNCE), bool(state["contact"][env_id]),
                round(float(state["lift"][env_id]), 4), wrong,
                round(float(torch.linalg.norm(target_position - receptacle_position)), 4),
                round(gripper_target, 4) if not math.isnan(gripper_target) else None]

    def step(action):
        if state["step"] == 0:
            position = measure()
            for env_id in range(n):
                boundaries[env_id].append(row(env_id, position))
        result = original_step(action)
        state["step"] += 1
        position = measure()
        terminal = [r for r in env.get_env_results() if r["step"] is not None and r["env_id"] not in finished]
        finished.update((r["env_id"], state["step"]) for r in terminal)
        ids = [r["env_id"] for r in terminal]
        if state["step"] % HORIZON == 0:
            ids.extend(i for i in env.active_env_ids if i not in ids)
        for env_id in ids:
            boundaries[env_id].append(row(env_id, position))
        for r in terminal:
            env_id = r["env_id"]
            start = HORIZON * ((state["step"] - 1) // HORIZON)
            # Every terminal chunk needs its real end frame; full chunks also have it as the next
            # request image, but the next request never happens after termination.
            terminal_frames[env_id] = save_terminal(client, result[0], env_id, start, state["step"] - start)
        return result

    env.step = step
    try:
        result = run_episode(env=env, **kwargs)
    finally:
        env.step = original_step
    for r in result[0]:
        env_id = r["env_id"]
        if r["step"] is None or finished.get(env_id) != r["step"]:
            raise RuntimeError(f"RoboLab episode steps and executed policy actions disagree for env {env_id}")
        r.setdefault("episode_metadata", {})["v1_progress"] = {
            "schema": SCHEMA, "target": target, "receptacle": receptacle, "lift_m": LIFT_M,
            "debounce": DEBOUNCE, "horizon": HORIZON,
            "fields": ["step", "held", "contact", "lift_m", "wrong_object", "target_receptacle_m",
                       "gripper_target_m"],
            "boundaries": boundaries[env_id], "terminal_frame": terminal_frames.get(env_id),
        }
    return result


def terminal_frame_saver(capture_dir, base_class):
    """Build the callback that stores the real composite at an environment's termination step."""

    def save(client, obs, env_id, start, executed):
        extracted = base_class._extract_observation(client, obs, env_id=env_id)
        image = base_class._pack_request(client, extracted, client.last_instruction)["observation/image"]
        name = f"{client.capture_key(env_id, start)}_realend.npy"
        np.save(capture_dir / name, np.asarray(image), allow_pickle=False)
        return {"file": name, "executed": executed}

    return save
