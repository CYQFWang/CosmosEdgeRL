import time
from collections import OrderedDict

import torch


class FrozenUNDCache:
    """Worker-local cache for one immutable UND instance."""

    def __init__(self, model, capacity=8, verify=False):
        if capacity < 1 or any(param.requires_grad for param in model.parameters()):
            raise ValueError("UND cache requires frozen weights and a positive capacity")
        self.original = model.forward
        self.items = OrderedDict()
        self.capacity = capacity
        self.verify = verify
        self.hits = self.misses = self.verified = 0
        self.lookup_s = self.encode_s = 0.0
        model.forward = self.forward

    @torch.no_grad()
    def forward(self, ids, frequencies):
        start = time.perf_counter()
        key = (ids.device, tuple(ids.shape), tuple(ids.flatten().tolist()),
               tuple((part.dtype, tuple(part.shape), part.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
                     for part in frequencies))
        self.lookup_s += time.perf_counter() - start
        if key in self.items:
            self.hits += 1
            previous_frequencies, output = self.items[key]
            self.items.move_to_end(key)
            if self.verify and not self.verified:
                actual = self.original(ids, frequencies)
                for left, right in zip(frequencies, previous_frequencies, strict=True):
                    torch.testing.assert_close(left, right, rtol=0, atol=0)
                for left, right in zip(actual, output, strict=True):
                    for left_tensor, right_tensor in zip(left, right, strict=True):
                        torch.testing.assert_close(left_tensor, right_tensor, rtol=0, atol=0)
                self.verified += 1
            return output
        self.misses += 1
        start = time.perf_counter()
        output = self.original(ids, frequencies)
        self.encode_s += time.perf_counter() - start
        self.items[key] = (frequencies, output)
        if len(self.items) > self.capacity:
            self.items.popitem(last=False)
        return output

    def stats(self):
        return {name: getattr(self, name) for name in ("hits", "misses", "verified", "lookup_s", "encode_s")} | {
            "entries": len(self.items),
            "bytes": sum(tensor.numel() * tensor.element_size()
                         for frequencies, output in self.items.values()
                         for tensor in (*frequencies, *(value for layer in output for value in layer))),
        }
