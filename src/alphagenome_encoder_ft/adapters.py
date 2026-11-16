"""Low-rank convolution adapters (LoCon) for fine-tuning the encoder.

``LoConConv1d`` wraps a frozen ``Conv1d`` and adds a trainable low-rank path in parallel:
``y = base(x) + up(down(x))``, with ``down`` a ``kernel_size`` convolution to ``rank``
channels and ``up`` a 1x1 convolution back to ``out_channels``. ``up`` starts at zero, so a
freshly wrapped model computes exactly what the base did. The update adds ``up @ down`` to the
convolution's *effective* kernel, which keeps it well defined for AlphaGenome's
weight-standardized convolutions, whose raw weights pass through a non-linear standardization.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class LoConConv1d(nn.Module):
    """A frozen ``Conv1d`` plus a trainable rank-``rank`` update (stride 1, 'same' padding)."""

    def __init__(self, base: nn.Conv1d, rank: int) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("rank must be >= 1")
        if base.stride != (1,) or base.dilation != (1,) or base.groups != 1:
            raise ValueError("LoCon supports stride-1, undilated, ungrouped convolutions only")
        self.base = base
        self.rank = rank
        kernel_size = base.kernel_size[0]
        device, dtype = base.weight.device, base.weight.dtype
        self.down = nn.Parameter(torch.empty(rank, base.in_channels, kernel_size, device=device, dtype=dtype))
        self.up = nn.Parameter(torch.zeros(base.out_channels, rank, 1, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.down, a=math.sqrt(5))
        self.pad = ((kernel_size - 1) // 2, kernel_size - 1 - (kernel_size - 1) // 2)

    def forward(self, x: Tensor) -> Tensor:
        low_rank = F.conv1d(F.pad(x, self.pad), self.down.to(x.dtype))
        return self.base(x) + F.conv1d(low_rank, self.up.to(x.dtype))


def locon_state_dict_from_plain(state_dict: dict[str, Tensor], module: nn.Module, prefix: str = "") -> dict[str, Tensor]:
    """Map a state dict saved without adapters onto ``module``, which has them.

    Each wrapped convolution's saved weights move to ``<conv>.base.*``; its adapter takes
    ``down`` from the module and a zero ``up``, a no-op update, so the loaded model computes
    what the saved one did. ``prefix`` is ``module``'s name inside ``state_dict``'s owner.
    """

    out = dict(state_dict)
    for name, sub in module.named_modules():
        if not isinstance(sub, LoConConv1d):
            continue
        path = f"{prefix}{name}."
        for key in [k for k in state_dict if k.startswith(path) and not k.startswith(path + "base.")]:
            out[path + "base." + key[len(path):]] = out.pop(key)
        out[path + "down"] = sub.down.detach().clone()
        out[path + "up"] = torch.zeros_like(sub.up)
    return out


def add_locon(module: nn.Module, rank: int) -> list[str]:
    """Wrap every ``conv`` child that is a ``Conv1d`` in ``LoConConv1d``; return the wrapped names.

    In the AlphaGenome encoder these are the convolutions of the ``ConvBlock``s (the DNA
    embedder's block and both blocks of every down block); the embedder's first
    convolution (``conv1``) is left alone. Freezing is left to the caller.
    """

    wrapped = []
    for name, child in list(module.named_modules()):
        conv = getattr(child, "conv", None)
        if isinstance(conv, nn.Conv1d) and not isinstance(conv, LoConConv1d):
            child.conv = LoConConv1d(conv, rank)
            wrapped.append(f"{name}.conv" if name else "conv")
    if not wrapped:
        raise ValueError("no 'conv' Conv1d children to wrap")
    return wrapped
