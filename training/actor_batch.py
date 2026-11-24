import torch


def prepare_actor_batch(batch, advantage, honesty, executed, sigmas, *, honesty_weight=0.0, clip=2.0):
    video = batch.batch["video_latents_clean"]
    action = batch.batch["action_latents_clean"]
    if video.ndim != 5 or video.shape[2] != 9 or action.ndim != 3 or action.shape[1] != 33:
        raise ValueError("Expected joint Cosmos3 latents: clean prefix plus 8 video / 32 action rows")
    size = video.shape[0]
    if action.shape[0] != size or any(len(item) != size for item in (advantage, honesty, executed)):
        raise ValueError("Batch fields must have the same leading dimension")
    if clip <= 0 or any(not 1 <= steps <= 32 for steps in executed):
        raise ValueError("Invalid advantage clip or executed horizon")
    device = video.device
    real = torch.as_tensor(advantage, dtype=torch.float32, device=device)
    agree = real.new_tensor([0.0 if score is None else score for score in honesty])
    routed = torch.stack((real + honesty_weight * agree, real), dim=-1)
    noise = real.new_tensor(sigmas)
    if noise.shape != (4,) or not bool(((noise > 0) & (noise <= 1)).all()):
        raise ValueError("Use four recorded rollout sigmas, not synthetic scheduler steps")
    steps = torch.as_tensor(executed, device=device)
    batch.batch["reward_prob"] = (0.5 + 0.5 * (routed / clip).clamp(-1, 1))[:, None, :].expand(-1, 4, -1)
    batch.batch["train_sigmas"] = noise.expand(size, -1)
    batch.batch["action_loss_mask"] = torch.arange(33, device=device)[None, :].le(steps[:, None])
    batch.batch["action_loss_mask"][:, 0] = False
    frames = 4 * torch.arange(9, device=device)
    batch.batch["video_loss_mask"] = (frames[None, :] > 0) & (frames[None, :] <= steps[:, None])
    return batch
