"""Offline learning-rate probe for the train_v1 NFT update (arm T settings), one GPU per process.

Data: a labelled, on-policy capture set (v1_progress logs + *_nft.pt traces). Episodes are split per task into
train / held-out. For each learning rate the adapter and AdamW state restart from the checkpoint, train on 16 steps
of 16 non-neutral chunks (all captured sigmas, one pass, p_video = 1/2), and are evaluated on held-out chunks at
fixed noise after 8 and 16 steps:

  signal = mean Δerr on held-out non-neutral chunks weighted by w = 2p-1 (normalised by Σ|w|) minus mean Δerr on
           held-out neutral chunks; negative = moved toward preferred samples beyond generic drift.
  drift  = mean relative Δerr on held-out neutral chunks.
  size   = RMS(v_new - v_old) / RMS(v_old) per modality on held-out tokens.
where err = masked mean ((x_sigma - x0) - sigma * v)^2, the NFT fitting error.

Usage: python -m training.lr_probe <capture_root> <checkpoint_dir> <out.json> --lrs 1e-4 1e-3 --store-port P --vllm-port Q
"""

import argparse
import json
import random
import time
import zlib
from pathlib import Path

import torch
from vllm.config import set_current_vllm_config
from vllm_omni.diffusion.forward_context import set_forward_context

from training.cosmos_nft import prepare_joint_forward
from training.v1_reward import episode_chunks, rollout_logs


def build(root, checkpoint_version, seed):
    episodes = []
    for batch in sorted(p for p in root.iterdir() if (p / "rollouts").is_dir()):
        for _, log in rollout_logs(batch):
            chunks = episode_chunks(log)
            episodes.append([{"trace": str(batch / "capture" / f"{c.key}_nft.pt"), "executed": c.executed,
                              "p_action": c.p_action, "p_video": 0.5, "task": c.task, "policy_version": checkpoint_version,
                              "id": f"{batch.name}_{c.key}"} for c in chunks])
    rng = random.Random(seed)
    by_task = {}
    for episode in episodes:
        by_task.setdefault(episode[0]["task"], []).append(episode)
    train, held = [], []
    for task in sorted(by_task):
        group = by_task[task]
        rng.shuffle(group)
        half = len(group) // 2
        train += [c for e in group[:half] for c in e]
        held += [c for e in group[half:] for c in e]
    informative = [c for c in train if c["p_action"] != 0.5]
    rng.shuffle(informative)
    held_pos = [c for c in held if c["p_action"] != 0.5]
    held_neu = [c for c in held if c["p_action"] == 0.5]
    rng.shuffle(held_pos)
    rng.shuffle(held_neu)
    return informative[:256], held_pos[:64], held_neu[:64]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--lrs", type=float, nargs="+", required=True)
    parser.add_argument("--store-port", type=int, required=True)
    parser.add_argument("--vllm-port", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--passes", type=int, default=1, help="Passes over the 256 training chunks (reshuffled each pass)")
    parser.add_argument("--save", type=Path, help="Directory to save the adapter/optimizer after the last step")
    args = parser.parse_args()
    import hashlib
    from types import SimpleNamespace

    from training.online import MODEL
    from training.v1_actor import Actor

    version = hashlib.sha256((args.checkpoint / "heads.pt").read_bytes()).hexdigest()
    train, held_pos, held_neu = build(args.root, version, args.seed)
    actor = Actor(SimpleNamespace(rank=0, world=1, store_port=args.store_port, vllm_port=args.vllm_port,
                                  model=str(MODEL), lora_rank=16, learning_rate=args.lrs[0]))
    device, beta, ref_w = actor.device, 0.1, 0.01
    traces = {}

    def trace(path):
        if path not in traces:
            traces[path] = torch.load(path, weights_only=True, map_location="cpu")
        return traces[path]

    def evaluate():
        sums = {"pos_w": 0.0, "pos_wd": 0.0, "neu_rel": [], "pos_rel": [], "size_action": [], "size_video": []}
        with torch.no_grad():
            for group, sample in [("pos", s) for s in held_pos] + [("neu", s) for s in held_neu]:
                t = trace(sample["trace"])
                w = 2 * sample["p_action"] - 1
                for d in range(len(t["train_sigmas"])):
                    gen = torch.Generator(device=device).manual_seed(zlib.crc32(f"{sample['id']}:{d}".encode()))
                    sigma, batch = prepare_joint_forward(t, d, sample["executed"], device, gen)
                    _, old, _ = actor.velocities(t, (batch["video"][0], batch["action"][0]), d)
                    actor.adapter.active = True
                    from training.cosmos_nft import joint_velocity
                    new = joint_velocity(actor.transformer, t, batch["video"][0], batch["action"][0], d)
                    for name, v_new, v_old in zip(batch, new, old, strict=True):
                        noisy, clean, mask = batch[name]
                        mask = mask.bool().expand_as(v_new)
                        n = mask.sum().clamp_min(1)
                        err = lambda v: (((noisy.float() - clean.float()) - sigma * v.float()).square()
                                         .masked_fill(~mask, 0).sum() / n)
                        e_old, e_new = float(err(v_old)), float(err(v_new))
                        diff = (v_new.float() - v_old.float()).square().masked_fill(~mask, 0).sum() / n
                        base = v_old.float().square().masked_fill(~mask, 0).sum() / n
                        sums[f"size_{name}"].append(float((diff / base.clamp_min(1e-12)).sqrt()))
                        if name != "action":
                            continue
                        rel = (e_new - e_old) / max(e_old, 1e-12)
                        if group == "pos":
                            sums["pos_w"] += abs(w)
                            sums["pos_wd"] += w * rel
                            sums["pos_rel"].append(rel)
                        else:
                            sums["neu_rel"].append(rel)
        drift = sum(sums["neu_rel"]) / len(sums["neu_rel"])
        weighted = sums["pos_wd"] / max(sums["pos_w"], 1e-12)
        return {"signal": weighted - drift, "weighted_pos_rel_derr": weighted, "drift_neutral_rel_derr": drift,
                "size_action": sum(sums["size_action"]) / len(sums["size_action"]),
                "size_video": sum(sums["size_video"]) / len(sums["size_video"])}

    results = {"checkpoint": str(args.checkpoint), "train_chunks": len(train), "held_pos": len(held_pos),
               "held_neutral": len(held_neu), "runs": {}}
    with (set_current_vllm_config(actor.worker.vllm_config),
          set_forward_context(vllm_config=actor.worker.vllm_config, omni_diffusion_config=actor.config)):
        for lr in args.lrs:
            actor.fresh = False
            actor.load_policy({"resume": str(args.checkpoint), "learning_rate": lr})
            generator = torch.Generator(device=device).manual_seed(args.seed)
            steps = []
            for epoch in range(args.passes):
                order = list(range(len(train)))
                random.Random(args.seed if epoch == 0 else args.seed * 1000 + epoch).shuffle(order)
                steps += [order[i:i + 16] for i in range(0, len(order), 16)]
            run = {"evals": {}, "grad_norm": []}
            started = time.monotonic()
            for index, step in enumerate(steps, 1):
                units = sum(len(trace(train[i]["trace"])["train_sigmas"]) for i in step)
                for i in step:
                    sample, t = train[i], trace(train[i]["trace"])
                    for d in range(len(t["train_sigmas"])):
                        sigma, batch = prepare_joint_forward(t, d, sample["executed"], device, generator)
                        current, old, reference = actor.velocities(t, (batch["video"][0], batch["action"][0]), d)
                        loss = 0.0
                        for name, v, prev, frozen in zip(batch, current, old, reference, strict=True):
                            noisy, clean, mask = batch[name]
                            v, prev, frozen = v.float(), prev.detach().float(), frozen.detach().float()
                            mask = mask.bool().expand_as(v)
                            count = mask.sum().clamp_min(1)
                            p = sample["p_action"] if name == "action" else sample["p_video"]
                            pos, neg = prev + beta * (v - prev), prev - beta * (v - prev)
                            error = noisy.float() - clean.float()
                            pref = (p * (error - sigma * pos).square() + (1 - p) * (error - sigma * neg).square())
                            term = (pref.masked_fill(~mask, 0).sum()
                                    + ref_w * (v - frozen).square().masked_fill(~mask, 0).sum()) / count
                            loss = loss + term / units
                        loss.backward()
                norm = torch.sqrt(sum(p.grad.float().square().sum() for p in actor.params if p.grad is not None))
                run["grad_norm"].append(float(norm))
                actor.optimizer.step()
                actor.optimizer.zero_grad(set_to_none=True)
                if index in (8,) or index % 16 == 0:
                    run["evals"][index] = evaluate()
                    print(json.dumps({"lr": lr, "step": index, **run["evals"][index]}), flush=True)
            run["train_s"] = time.monotonic() - started
            results["runs"][str(lr)] = run
            args.out.write_text(json.dumps(results, indent=2) + "\n")
            if args.save:
                directory = args.save / f"lr{lr:g}_steps{len(steps)}"
                directory.mkdir(parents=True, exist_ok=True)
                torch.save({k: v.cpu() for k, v in actor.adapter.state_dict().items()}, directory / "heads.pt")
                torch.save(actor.optimizer.state_dict(), directory / "optimizer.pt")
    actor.worker.shutdown()


if __name__ == "__main__":
    main()
