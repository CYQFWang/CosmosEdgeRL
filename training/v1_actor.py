"""Resident train_v1 actor rank (train_v1.md §1.5, §9.2–§9.4, §9.6).

One process per GPU. Both ranks load the Cosmos3 Edge pipeline once and keep it resident;
gradients are averaged over a Gloo group that is independent of vLLM's own process group.
Jobs arrive as JSON lines on stdin; each job ends by writing its ``done`` file (JSON).

decode: {"kind": "decode", "items": [{"trace": nft.pt, "frames": [0, t], "out": npz}], "done": path}
train:  {"kind": "train", "samples": jsonl, "resume": dir|null, "output": dir, "seed": int,
         "batch_chunks": 16, "beta": 0.1, "reference_weight": 0.01, "done": path}
"""

import argparse
import hashlib
import json
import os
import random
import sys
import time
import traceback
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from vllm.config import set_current_vllm_config
from vllm_omni.diffusion.data import OmniDiffusionConfig, TransformerConfig
from vllm_omni.diffusion.forward_context import set_forward_context
from vllm_omni.diffusion.worker.diffusion_worker import DiffusionWorker

from training.cosmos_nft import joint_velocity, prepare_joint_forward
from training.head_adapters import HeadAdapters

MODALITIES = ("video", "action")


def write_json(path, value):
    temporary = Path(f"{path}.tmp")
    temporary.write_text(json.dumps(value, allow_nan=True) + "\n")
    temporary.replace(path)


class Actor:
    def __init__(self, args):
        self.args = args
        self.rank, self.world = args.rank, args.world
        architecture = json.loads((Path(args.model) / "transformer" / "config.json").read_text())
        self.config = OmniDiffusionConfig(
            model=args.model, model_class_name="Cosmos3OmniDiffusersPipeline",
            tf_model_config=TransformerConfig.from_dict(architecture), enforce_eager=True,
            model_config={"guardrails": False}, diffusion_attention_config={"default": "TORCH_SDPA"},
        )
        self.worker = DiffusionWorker(0, 0, self.config, distributed_init_method=f"tcp://127.0.0.1:{args.vllm_port}",
                                      set_proc_title=False)
        self.device = self.worker.device
        self.pipeline = self.worker.model_runner.pipeline
        self.transformer = self.pipeline.transformer
        self.transformer.requires_grad_(False)
        torch.manual_seed(37)
        self.adapter = HeadAdapters(self.transformer, rank=args.lora_rank, all_gen=True, checkpoint_blocks=False)
        self.old = HeadAdapters(self.transformer, rank=args.lora_rank, all_gen=True)
        self.old.requires_grad_(False)
        self.old.active = False
        self.optimizer = torch.optim.AdamW(self.adapter.parameters(), lr=args.learning_rate)
        store = dist.TCPStore("127.0.0.1", args.store_port, self.world, is_master=self.rank == 0,
                              timeout=timedelta(hours=3))
        self.group = dist.ProcessGroupGloo(store, self.rank, self.world, timedelta(hours=3))
        self.params = [p for p in self.adapter.parameters()]
        self.fresh = True

    # ---- collectives over the Gloo group (CPU tensors) ----
    def allreduce(self, tensor):
        cpu = tensor.detach().float().cpu().contiguous()
        self.group.allreduce([cpu]).wait()
        return cpu.to(tensor.device)

    def broadcast_parameters(self):
        for param in self.params:
            cpu = param.detach().float().cpu().contiguous()
            self.group.broadcast(cpu, 0).wait()
            param.data.copy_(cpu.to(param.device, param.dtype))

    def checksum(self):
        return float(sum(p.detach().double().sum().item() for p in self.params))

    # ---- jobs ----
    def decode(self, job):
        count = 0
        with torch.inference_mode():
            for item in job["items"][self.rank::self.world]:
                trace = torch.load(item["trace"], weights_only=True, map_location="cpu")
                latents = trace["video_latents_clean"].to(self.device)
                video = self.pipeline._decode_latents(latents)[0].float().clamp(-1, 1)
                frames = video.permute(1, 2, 3, 0).add(1).mul(127.5).round().to(torch.uint8).cpu().numpy()
                if frames.shape[0] != 33:
                    raise ValueError(f"Expected 33 dream frames, got {frames.shape}")
                picks = [min(int(index), 32) for index in item["frames"]]
                np.savez(item["out"], frames=frames[picks], indices=np.asarray(picks))
                count += 1
        torch.cuda.empty_cache()
        return {"decoded": count}

    def load_policy(self, job):
        resume = job.get("resume")
        if resume:
            weights = torch.load(Path(resume) / "heads.pt", weights_only=True, map_location="cpu")
            self.adapter.load_state_dict(weights)
            optimizer_path = Path(resume) / "optimizer.pt"
            if optimizer_path.exists():
                self.optimizer.load_state_dict(torch.load(optimizer_path, weights_only=True, map_location="cpu"))
            version = hashlib.sha256((Path(resume) / "heads.pt").read_bytes()).hexdigest()
        else:
            if not self.fresh:
                raise RuntimeError("Base-policy update requested after the adapter was trained")
            self.broadcast_parameters()
            version = "base"
        for group in self.optimizer.param_groups:
            group["lr"] = job["learning_rate"]
        self.old.load_state_dict(self.adapter.state_dict())
        self.old.active = False
        return version

    def velocities(self, trace, inputs, denoise):
        self.adapter.active = False
        with torch.no_grad():
            reference = joint_velocity(self.transformer, trace, *inputs, denoise)
            self.old.active = True
            old = joint_velocity(self.transformer, trace, *inputs, denoise)
            self.old.active = False
        self.adapter.active = True
        current = joint_velocity(self.transformer, trace, *inputs, denoise)
        return current, old, reference

    def train(self, job):
        version = self.load_policy(job)
        samples = [json.loads(line) for line in Path(job["samples"]).read_text().splitlines() if line]
        for sample in samples:
            if sample["policy_version"] != version:
                raise RuntimeError(f"Sample from policy {sample['policy_version'][:12]} but resume is {version[:12]}")
        size = job["batch_chunks"]
        steps = []
        for epoch in range(job.get("passes", 1)):  # reshuffled every pass; each pass sees every sample once
            order = list(range(len(samples)))
            random.Random(job["seed"] + 7919 * epoch).shuffle(order)
            steps += [order[i:i + size] for i in range(0, len(order), size)]
        generator = torch.Generator(device=self.device).manual_seed(job["seed"] * 7919 + self.rank)
        beta, reference_weight = job["beta"], job["reference_weight"]
        output = Path(job["output"])
        output.mkdir(parents=True, exist_ok=True)
        unit_rows, step_rows = [], []
        started = time.monotonic()
        torch.cuda.reset_peak_memory_stats(self.device)
        with (set_current_vllm_config(self.worker.vllm_config),
              set_forward_context(vllm_config=self.worker.vllm_config, omni_diffusion_config=self.config)):
            for step_index, step in enumerate(steps):
                traces = {}
                # Global denominators: every rank knows the whole step's composition.
                sigma_counts = {}
                for index in step:
                    trace = torch.load(samples[index]["trace"], weights_only=True, map_location="cpu")
                    sigma_counts[index] = len(trace["train_sigmas"])
                    if index % self.world == self.rank:
                        traces[index] = trace
                active = {"action": sum(sigma_counts[i] for i in step if samples[i]["p_action"] != 0.5),
                          "video": sum(sigma_counts[i] for i in step)}
                sums = torch.zeros(8, dtype=torch.float64)  # pref_a, pref_v, ref_a, ref_v, dold_a, dold_v, dref_a, dref_v
                for index in step:
                    if index % self.world != self.rank:
                        continue
                    sample, trace = samples[index], traces[index]
                    probabilities = {"action": sample["p_action"], "video": sample["p_video"]}
                    for denoise in range(sigma_counts[index]):
                        sigma, batch = prepare_joint_forward(trace, denoise, sample["executed"], self.device, generator)
                        current, old, reference = self.velocities(trace, (batch["video"][0], batch["action"][0]), denoise)
                        loss = 0.0
                        row = {"id": sample["id"], "sigma": float(sigma.item()), "rank": self.rank, "step": step_index}
                        for name, velocity, previous, frozen in zip(batch, current, old, reference, strict=True):
                            noisy, clean, mask = batch[name]
                            velocity, previous, frozen = velocity.float(), previous.detach().float(), frozen.detach().float()
                            mask = mask.bool().expand_as(velocity)
                            count = mask.sum().clamp_min(1)
                            p = probabilities[name]
                            positive = previous + beta * (velocity - previous)
                            negative = previous - beta * (velocity - previous)
                            error = noisy.float() - clean.float()
                            preference = p * (error - sigma * positive).square() + (1 - p) * (error - sigma * negative).square()
                            preference = preference.masked_fill(~mask, 0).sum() / count
                            regular = (velocity - frozen).square().masked_fill(~mask, 0).sum() / count
                            term = preference + reference_weight * regular
                            weight = 0.0 if name == "action" and p == 0.5 else 1.0 / max(active[name], 1)
                            loss = loss + weight * term
                            with torch.no_grad():
                                d_old = ((velocity - previous).square().masked_fill(~mask, 0).sum() / count).sqrt()
                                d_ref = regular.sqrt()
                            row.update({f"{name}_p": p, f"{name}_preference": float(preference),
                                        f"{name}_reference": float(regular), f"{name}_d_old": float(d_old),
                                        f"{name}_d_ref": float(d_ref), f"{name}_weight": weight})
                            slot = 0 if name == "action" else 1
                            if weight:
                                sums[slot] += float(preference)
                                sums[2 + slot] += float(regular)
                            sums[4 + slot] += float(d_old)
                            sums[6 + slot] += float(d_ref)
                        loss.backward()
                        unit_rows.append(row)
                # Average gradients across ranks (each rank's loss already carries the global 1/N weights).
                norms = {"heads": 0.0, "gen": 0.0}
                for name, module in self.adapter.layers.items():
                    for param in module.parameters():
                        grad = param.grad if param.grad is not None else torch.zeros_like(param)
                        reduced = self.allreduce(grad)
                        if not torch.isfinite(reduced).all():
                            raise RuntimeError(f"Nonfinite gradient in {name}")
                        param.grad = reduced.to(param.dtype)
                        norms["gen" if name.startswith("gen_") else "heads"] += float(reduced.double().square().sum())
                totals = self.allreduce(sums)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                units = sum(sigma_counts.values())
                step_rows.append({
                    "step": step_index, "chunks": len(step), "units": units,
                    "action_units": active["action"], "video_units": active["video"],
                    "lr": self.optimizer.param_groups[0]["lr"],
                    "grad_norm_heads": norms["heads"] ** 0.5, "grad_norm_gen": norms["gen"] ** 0.5,
                    "grad_norm": (norms["heads"] + norms["gen"]) ** 0.5,
                    "action_preference": float(totals[0]) / max(active["action"], 1),
                    "video_preference": float(totals[1]) / max(active["video"], 1),
                    "action_reference": float(totals[2]) / max(active["action"], 1),
                    "video_reference": float(totals[3]) / max(active["video"], 1),
                    "action_d_old": float(totals[4]) / units, "video_d_old": float(totals[5]) / units,
                    "action_d_ref": float(totals[6]) / units, "video_d_ref": float(totals[7]) / units,
                    "adapter_norm": float(sum(p.detach().double().square().sum() for p in self.params) ** 0.5),
                    "elapsed_s": time.monotonic() - started,
                })
        checksum = self.checksum()
        checks = self.allreduce(torch.tensor([checksum, checksum ** 2], dtype=torch.float64))
        if abs(float(checks[0]) / self.world - checksum) > 1e-6 * max(1.0, abs(checksum)):
            raise RuntimeError("Adapter replicas diverged across ranks")
        self.fresh = False
        import pandas as pd

        pd.DataFrame(unit_rows).to_parquet(output / f"update_samples_rank{self.rank}.parquet", index=False)
        result = {"steps": len(steps), "samples": len(samples), "update_s": time.monotonic() - started,
                  "peak_allocated_gib": torch.cuda.max_memory_allocated(self.device) / 2**30,
                  "policy_version": version}
        if self.rank == 0:
            pd.DataFrame(step_rows).to_parquet(output / "steps.parquet", index=False)
            torch.save({name: tensor.cpu() for name, tensor in self.adapter.state_dict().items()}, output / "heads.pt")
            torch.save(self.optimizer.state_dict(), output / "optimizer.pt")
            result["new_version"] = hashlib.sha256((output / "heads.pt").read_bytes()).hexdigest()
        torch.cuda.empty_cache()
        return result

    def serve(self):
        print(json.dumps({"ready": True, "rank": self.rank}), flush=True)
        for line in sys.stdin:
            job = json.loads(line)
            try:
                result = getattr(self, job["kind"])(job)
                result["ok"] = True
            except Exception:  # noqa: BLE001 - report to the orchestrator, which stops the run
                result = {"ok": False, "error": traceback.format_exc()}
                print(result["error"], file=sys.stderr, flush=True)
            write_json(f"{job['done']}.rank{self.rank}", result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world", type=int, default=2)
    parser.add_argument("--store-port", type=int, required=True)
    parser.add_argument("--vllm-port", type=int, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    args = parser.parse_args()
    Actor(args).serve()


if __name__ == "__main__":
    main()
