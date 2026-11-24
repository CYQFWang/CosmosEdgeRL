"""Measure a warm UND cache against uncached native RoboLab policy forwards."""

import argparse
import itertools
import json
import time
from dataclasses import replace
from pathlib import Path
from statistics import median

import numpy as np
import torch
from vllm.config import set_current_vllm_config
from vllm_omni.diffusion.data import OmniDiffusionConfig, TransformerConfig
from vllm_omni.diffusion.forward_context import set_forward_context
from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

from training.und_cache import FrozenUNDCache


def benchmark(pipeline, traces, repeats):
    transformer = pipeline.transformer
    transformer.requires_grad_(False)
    reasoner = transformer.language_model
    cache = FrozenUNDCache(reasoner, capacity=8)
    diffuse = pipeline.diffuse
    dream = None

    def capture(**kwargs):
        nonlocal dream
        outputs = diffuse(**kwargs)
        dream = outputs[0]
        return outputs

    def run(path, use_cache, request_id):
        with np.load(path, allow_pickle=False) as trace:
            obs = {
                "observation/image": trace["image"],
                "observation/joint_position": trace["joint"],
                "observation/gripper_position": trace["gripper"],
                "prompt": str(trace["prompt"]),
            }
            seed = int(trace["seed"])
        params = OmniDiffusionSamplingParams(extra_args={"robot_obs": obs, "format_prompt_as_json": True})
        reasoner.forward = cache.forward if use_cache else cache.original
        torch.cuda.synchronize()
        started = time.perf_counter()
        inputs = pipeline._build_robolab_policy_inputs(params)
        output = pipeline._forward_robolab_policy(params, replace(inputs, seed=seed), time.time(), request_id)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        return elapsed, output.output["payload"]["actions"], dream

    timings = []
    pipeline.diffuse = capture
    try:
        for path in traces:
            run(path, False, "warm-base")
            run(path, True, "warm-cache")
        for index, path in enumerate(itertools.islice(itertools.cycle(traces), repeats)):
            order = (False, True) if index % 2 == 0 else (True, False)
            results = {flag: run(path, flag, f"paired-{index}-{flag}") for flag in order}
            uncached, base_action, base_dream = results[False]
            cached, cached_action, cached_dream = results[True]
            torch.testing.assert_close(base_action, cached_action, rtol=0, atol=0)
            torch.testing.assert_close(base_dream, cached_dream, rtol=0, atol=0)
            timings.append({"trace": path.name, "uncached_ms": uncached * 1000, "cached_ms": cached * 1000})
    finally:
        reasoner.forward = cache.original
        del pipeline.diffuse
    base = median(row["uncached_ms"] for row in timings)
    cached = median(row["cached_ms"] for row in timings)
    return {"pairs": repeats, "exact_action_and_dream_parity": True, "uncached_median_ms": base,
            "cached_median_ms": cached, "median_paired_saved_ms": median(
                row["uncached_ms"] - row["cached_ms"] for row in timings
            ), "median_speedup": base / cached, "cache": cache.stats(), "timings": timings}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("capture", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=16)
    args = parser.parse_args()
    traces = sorted(args.capture.glob("*_request.npz"))[:4]
    if not traces or args.repeats <= 0:
        parser.error("Need real request traces and a positive repeat count")
    architecture = json.loads((args.model / "transformer/config.json").read_text())
    config = OmniDiffusionConfig(model=str(args.model), model_class_name="Cosmos3OmniDiffusersPipeline",
                                 tf_model_config=TransformerConfig.from_dict(architecture),
                                 enforce_eager=True, model_config={"guardrails": False})
    worker = DiffusionWorker(0, 0, config, distributed_init_method="tcp://127.0.0.1:18239", set_proc_title=False)
    try:
        with (torch.inference_mode(), set_current_vllm_config(worker.vllm_config),
              set_forward_context(vllm_config=worker.vllm_config, omni_diffusion_config=config)):
            result = benchmark(worker.model_runner.pipeline, traces, args.repeats)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({key: value for key, value in result.items() if key != "timings"}))
    finally:
        worker.shutdown()


if __name__ == "__main__":
    main()
