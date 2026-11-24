"""Capture progress from RoboLab's configured success conditions."""

from functools import partial
from time import monotonic


def independent_success_checks(manager, conditionals):
    success_terms = []
    for name in manager.active_terms:
        term = manager.get_term_cfg(name)
        if not term.time_out:
            success_terms.append((name, term))
    if len(success_terms) != 1:
        return None, ()
    name, term = success_terms[0]
    # These native callables delegate to evaluate_spatial_condition with an
    # independent check per object. Grouped, counted and coupled terms use
    # the official terminal flag.
    independent = (
        conditionals.object_in_container, conditionals.object_inside,
        conditionals.object_on_top, conditionals.object_outside_of,
        conditionals.object_outside_of_and_on_surface, conditionals.object_upright,
    )
    objects = term.params.get("object")
    if (term.func not in independent or term.params.get("logical", "all") != "all"
            or not isinstance(objects, (list, tuple)) or len(objects) < 2
            or len(set(objects)) != len(objects)):
        return None, ()
    return name, tuple(partial(term.func, **(term.params | {"object": obj})) for obj in objects)


def record_native_progress(run_episode, env, *, conditionals, horizon, deadline=None, **kwargs):
    manager = env.termination_manager
    name, checks = independent_success_checks(manager, conditionals)
    term = manager.get_term_cfg(name) if checks else None
    original_step = env.step
    step_index = 0
    boundaries = [[] for _ in range(env.num_envs)]
    finished = {}
    parity = True

    def capture(env_ids, official):
        nonlocal parity
        counts = checks[0](env).long()
        for check in checks[1:]:
            counts += check(env)
        for env_id, count, success in zip(env_ids, counts[env_ids].tolist(), official[env_ids].tolist(), strict=True):
            parity &= (count == len(checks)) == success
            boundaries[env_id].append([step_index, count])

    def step(action):
        nonlocal step_index
        if checks and step_index == 0:
            capture(list(range(env.num_envs)), term.func(env, **term.params))
        if deadline is not None and (step_index + 1) % horizon == 0 and monotonic() >= deadline:
            env.cfg.episode_length_s = min(env.cfg.episode_length_s, (step_index + 1 - 0.01) * env.step_dt)
        result = original_step(action)
        step_index += 1
        # RoboLab preserves the terminal state for newly frozen environments.
        # Stop sampling those slots once their final action has been recorded.
        terminal = [row for row in env.get_env_results()
                    if row["step"] is not None and row["env_id"] not in finished]
        finished.update((row["env_id"], step_index) for row in terminal)
        if checks and parity:
            env_ids = [row["env_id"] for row in terminal]
            if step_index % horizon == 0:
                env_ids.extend(env.active_env_ids)
            if env_ids:
                capture(env_ids, manager.get_term(name))
        return result

    env.step = step
    try:
        result = run_episode(env=env, **kwargs)
    finally:
        env.step = original_step
    for row in result[0]:
        env_id = row["env_id"]
        if row["step"] is None or finished.get(env_id) != row["step"]:
            raise RuntimeError(f"RoboLab episode steps and executed policy actions disagree for env {env_id}")
        progress = {"condition_count": len(checks) if checks and parity else 1}
        if checks and parity:
            progress["boundaries"] = boundaries[env_id]
        elif checks:
            progress["fallback"] = "native condition parity failed"
        row.setdefault("episode_metadata", {})["native_progress"] = progress
    return result
