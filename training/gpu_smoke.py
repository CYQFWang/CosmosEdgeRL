"""Synthetic DDP update benchmark; not a Cosmos3 throughput benchmark."""

import json
import os
import time

import torch
import torch.distributed as dist
from torch import nn
from verl.protocol import DataProto

from training.actor_batch import prepare_actor_batch


def main():
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    if world > 1:
        dist.init_process_group("nccl", device_id=device)
    torch.manual_seed(37)
    actor = nn.Sequential(nn.Linear(1024, 4096), nn.GELU(), nn.Linear(4096, 1024)).to(device)
    model = nn.parallel.DistributedDataParallel(actor, device_ids=[rank]) if world > 1 else actor
    optimizer = torch.optim.AdamW(model.parameters(), fused=True)
    batch = DataProto.from_dict(tensors={
        "video_latents_clean": torch.randn(2, 16, 9, 8, 8, device=device),
        "action_latents_clean": torch.randn(2, 33, 1024, device=device),
    })
    initial = actor[0].weight.detach().clone()
    targets = batch.batch["video_latents_clean"].transpose(1, 2).reshape(2, 9, 1024), batch.batch["action_latents_clean"]
    torch.manual_seed(42 + rank)
    noises = [torch.randn_like(target) for target in targets]
    counts = (32, 17)
    rounds = 100
    for step in range(rounds + 15):
        if step == 15:
            if world > 1:
                dist.barrier()
            torch.cuda.synchronize()
            start = time.perf_counter()
        prepare_actor_batch(batch, [0.5, -0.5], [-1.0, None], counts, (1, .94, .83, .62), honesty_weight=.25)
        sigma_index = step % 4
        sigma = batch.batch["train_sigmas"][:, sigma_index, None, None]
        inputs = [(1 - sigma) * target + sigma * noise for target, noise in zip(targets, noises)]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(torch.cat(inputs, dim=1)).float().split((9, 33), dim=1)
        losses = []
        for index, modality in enumerate(("video", "action")):
            probability = batch.batch["reward_prob"][:, sigma_index, index, None]
            mask = batch.batch[f"{modality}_loss_mask"]
            velocity = outputs[index]
            error = inputs[index] - targets[index]
            loss = probability * (error - sigma * velocity).square().mean(-1)
            loss += (1 - probability) * (error + sigma * velocity).square().mean(-1)
            losses.append((loss * mask).sum() / mask.sum())
        objective = sum(losses)
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        optimizer.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    assert torch.isfinite(objective) and not torch.equal(initial, actor[0].weight)
    stats = torch.tensor([elapsed, torch.cuda.max_memory_allocated() / 2**20], device=device)
    if world > 1:
        reference = actor[0].weight.detach().clone()
        dist.broadcast(reference, src=0)
        assert torch.equal(reference, actor[0].weight)
        dist.all_reduce(stats, op=dist.ReduceOp.MAX)
    if rank == 0:
        seconds, memory = stats.tolist()
        print(json.dumps({"gpus": world, "updates": rounds, "ms_per_update": 1000 * seconds / rounds,
                          "synthetic_chunks_per_second": 2 * world * rounds / seconds,
                          "peak_allocated_mib_per_gpu": memory}))
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
