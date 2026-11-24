"""Replay observations to test resident worker lifecycle, not policy learning."""

import argparse
import hashlib
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from training.online import CACHE, MODEL, WORKSPACE, ResidentWorkers, measure_gpu, read_json, write_json

sys.path.insert(0, "/home/robot_share/RoboLab")
from robolab.eval.websocket_transport import MsgPackWebSocketTransport


def replay(policy, source, version):
    uri = "ws://127.0.0.1:18144/v1/realtime/robot/openpi"
    deadline = time.monotonic() + 180
    client = MsgPackWebSocketTransport(uri)
    while True:
        try:
            client.connect()
            break
        except ConnectionRefusedError:
            if policy.poll() is not None or time.monotonic() > deadline:
                raise
            time.sleep(1)
    actions = {}
    try:
        for path in sorted((source / "capture").glob("*_request.npz")):
            key = path.stem.removesuffix("_request")
            with np.load(path, allow_pickle=False) as trace:
                response = client.request({
                    "observation/image": trace["image"],
                    "observation/joint_position": trace["joint"],
                    "observation/gripper_position": trace["gripper"],
                    "prompt": str(trace["prompt"]), "seed": int(trace["seed"]),
                    "session_id": str(trace["session_id"]), "dream_capture_key": key,
                    "dream_policy_version": version,
                })
            if isinstance(response, dict):
                response = response.get("action", response.get("actions", response))
            actions[key] = np.asarray(response)
            if actions[key].shape != (32, 8):
                raise RuntimeError(f"Invalid action response for {key}")
    finally:
        client.close()
    return actions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("initial_checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--gpu", required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir()
    env = os.environ | {
        "HF_HOME": str(CACHE), "HF_HUB_OFFLINE": "1",
        "PYTHONPATH": f"{WORKSPACE}:/home/robot_share/cosmos-framework",
        "POLICY_GPU": args.gpu, "CUDA_VISIBLE_DEVICES": args.gpu, "PORT": "18144",
        "COSMOS_CAPTURE_NFT": "1", "DREAM_LATENTS_ONLY": "1", "COSMOS_UND_CACHE": "8",
    }
    previous = args.initial_checkpoint.resolve()
    actions, reports = [], []
    workers = ResidentWorkers(root)
    try:
        with measure_gpu(args.gpu) as memory:
            for index in (1, 2):
                stage = root / f"pass_{index}"
                capture = stage / "rollout/capture"
                capture.mkdir(parents=True)
                shutil.copytree(args.source / "rollouts", stage / "rollout/rollouts", dirs_exist_ok=True,
                                ignore=shutil.ignore_patterns("*.mp4", "*.hdf5"))
                version = hashlib.sha256((previous / "heads.pt").read_bytes()).hexdigest()
                workers.begin_rollout(env, capture, previous, version)
                actions.append(replay(workers.policy, args.source, version))
                workers.update([
                    sys.executable, "-m", "training.native_update", str(stage / "rollout"), str(MODEL),
                    "--output", str(stage), "--all-gen", "--rank", "16", "--no-checkpoint-blocks",
                    "--learning-rate", "0.0001", "--accumulate", "4", "--samples", "0",
                    "--mismatch-weight", "0", "--online-policy-version", version,
                    "--round", str(index), "--port", "18244", "--resume", str(previous),
                ], env, stage, timeout=180)
                report = read_json(stage / "update_metrics.json")
                if not report["steps"] or len(report["rollout_worker_pids"]) != 1:
                    raise RuntimeError("Missing optimizer update or policy worker audit")
                reports.append(report)
                previous = stage
        first, second = reports
        assert first["actor_pid"] == second["actor_pid"] and second["actor_reused"]
        assert first["rollout_worker_pids"] == second["rollout_worker_pids"]
        assert first["policy_version"] != second["policy_version"]
        delta = max(float(np.abs(actions[0][key] - value).max()) for key, value in actions[1].items())
        assert delta > 0
        write_json(root / "summary.json", {
            "actor_pid": second["actor_pid"], "policy_pid": second["rollout_worker_pids"][0],
            "actor_first_setup_s": first["load_s"], "actor_reused_setup_s": second["load_s"],
            "max_action_delta": delta, "gpu_total_including_live_run": memory,
            "updates": [report["steps"] for report in reports],
        })
    finally:
        workers.close()


if __name__ == "__main__":
    main()
