import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class HeadAdapters(nn.Module):
    def __init__(self, transformer, *, rank=4, all_gen=False, checkpoint_blocks=False):
        super().__init__()
        if rank < 1:
            raise ValueError("LoRA rank must be positive")
        self.active = True
        self.layers = nn.ModuleDict()
        self.handles = []
        device = transformer.proj_out.weight.device
        targets = [
            ("video", transformer.proj_out, transformer.proj_out.in_features, transformer.proj_out.out_features),
            ("action", transformer.action_proj_out, transformer.action_proj_out.input_size,
             transformer.action_proj_out.output_size),
        ]
        if all_gen:
            for index, block in enumerate(transformer.gen_layers):
                projection = block.mlp.down_proj
                targets.append((f"gen_{index}", projection, projection.input_size_per_partition,
                                projection.output_size))
                if checkpoint_blocks:
                    original = block.forward
                    block.forward = self._checkpoint(original)
        for name, head, width, output in targets:
            adapter = nn.Sequential(nn.Linear(width, rank, bias=False), nn.Linear(rank, output, bias=False)).to(
                device=device, dtype=torch.bfloat16 if name.startswith("gen_") else torch.float32,
            )
            nn.init.zeros_(adapter[1].weight)
            self.layers[name] = adapter
            self.handles.append(head.register_forward_hook(self._hook(adapter)))

    def _hook(self, adapter):
        def apply(module, args, output):
            return output + adapter(args[0].to(adapter[0].weight.dtype)).to(output.dtype) if self.active else output
        return apply

    @staticmethod
    def _checkpoint(forward):
        def checked(*args, **kwargs):
            if torch.is_grad_enabled():
                return checkpoint(forward, *args, use_reentrant=False, **kwargs)
            return forward(*args, **kwargs)
        return checked
