"""Registered video/action DiffusionNFT objective for Cosmos3 policy samples."""

import torch
from verl_omni.trainer.diffusion.diffusion_algos import DiffusionLossFn, register_diffusion_loss


@register_diffusion_loss("cosmos3_edge_nft")
class Cosmos3EdgeNFT(DiffusionLossFn):
    uses_old_adapter = True
    required_model_output_keys = ("modalities",)

    @classmethod
    def compute_loss(cls, *, modalities, beta=0.1, reference_weight=0.01):
        if not 0 < beta <= 1 or reference_weight < 0:
            raise ValueError("Invalid NFT coefficients")
        losses = {}
        for name in ("video", "action"):
            noisy, clean, sigma, velocity, old, reference, probability, mask = modalities[name]
            velocity, old, reference = velocity.float(), old.detach().float(), reference.detach().float()
            mask = mask.bool().expand_as(velocity)
            probability = probability.reshape((-1,) + (1,) * (velocity.ndim - 1))
            sigma = sigma.reshape((-1,) + (1,) * (velocity.ndim - 1))
            positive = old + beta * (velocity - old)
            negative = old - beta * (velocity - old)
            error = noisy.float() - clean.float()
            weighted = probability * (error - sigma * positive).square()
            weighted = weighted + (1 - probability) * (error - sigma * negative).square()
            weighted = weighted + reference_weight * (velocity - reference).square()
            losses[name] = weighted.masked_fill(~mask, 0).sum() / mask.sum().clamp_min(1)
        return sum(losses.values()), {name: loss.detach() for name, loss in losses.items()}

    def __call__(self, *, config, model_output, data):
        from verl_omni.trainer.diffusion.diffusion_algos import DiffusionLossResult

        self.validate_inputs(loss_name="cosmos3_edge_nft", model_output=model_output, data=data)
        loss, metrics = self.compute_loss(**model_output)
        return DiffusionLossResult(loss=loss, metrics=metrics)


def joint_velocity(transformer, trace, video, action, denoise_index):
    """Run the native Cosmos3 Edge transformer on one captured policy sample."""
    if video.shape[0] != 1 or action.shape[0] != 1:
        raise ValueError("Native policy forward currently requires microbatch size 1")
    device = video.device
    dtype = next(transformer.parameters()).dtype
    shared = {key: value.to(device) if torch.is_tensor(value) else value
              for key, value in trace["shared_kwargs"].items()}
    for key in ("noisy_frame_mask", "action_noisy_mask"):
        shared[key] = shared[key].to(dtype)
    timestep = trace["train_timesteps"][denoise_index:denoise_index + 1].to(device)
    if timestep.numel() != 1:
        raise ValueError("Denoise index is outside the captured schedule")

    def predict(branch):
        transformer.cached_kv = transformer.cached_freqs_gen = None
        return transformer(
            hidden_states=video.to(dtype), action_latents=action.to(dtype), sound_latents=None,
            timestep=timestep, text_ids=trace[f"{branch}_ids"].to(device),
            text_mask=trace[f"{branch}_mask"].to(device), **shared,
        )

    try:
        positive = predict("cond")
        scale = trace["guidance_scale"]
        if scale <= 1:
            return positive
        negative = predict("uncond")
        return tuple(base.float() + scale * (guided.float() - base.float())
                     for guided, base in zip(positive, negative, strict=True))
    finally:
        transformer.cached_kv = transformer.cached_freqs_gen = None


def prepare_joint_forward(trace, denoise_index, executed, device, generator=None):
    if not 1 <= executed <= 32 or not 0 <= denoise_index < len(trace["train_sigmas"]):
        raise ValueError("Invalid executed horizon or schedule index")
    sigma = trace["train_sigmas"][denoise_index:denoise_index + 1].to(device=device, dtype=torch.float32)
    batch = {}
    for name, mask_name in (("video", "velocity_mask"), ("action", "action_velocity_mask")):
        clean = trace[f"{name}_latents_clean"].to(device)
        generation_mask = trace[mask_name].to(device).bool()
        noise = torch.randn(clean.shape, device=device, dtype=clean.dtype, generator=generator)
        noisy = torch.where(generation_mask, (1 - sigma) * clean + sigma * noise, clean)
        if name == "video":
            executed_mask = (4 * torch.arange(clean.shape[2], device=device) <= executed)[None, None, :, None, None]
        else:
            rows = torch.arange(clean.shape[1], device=device)[None, :, None] <= executed
            columns = torch.arange(clean.shape[2], device=device)[None, None, :] < trace["raw_action_dim"]
            executed_mask = rows & columns
        batch[name] = (noisy, clean, generation_mask & executed_mask)
    return sigma, batch
