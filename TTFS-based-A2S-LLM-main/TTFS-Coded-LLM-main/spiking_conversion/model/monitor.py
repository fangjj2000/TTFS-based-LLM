from typing import Callable

import torch
import torch.nn as nn

from ..util import BaseMonitor


class MonitoredLayerNorm(nn.Module):

    def __init__(self, ann: nn.LayerNorm):
        super().__init__()
        assert isinstance(ann, nn.LayerNorm)
        self.normalized_shape = ann.normalized_shape
        self.dim = tuple(range(-len(self.normalized_shape), 0))
        self.eps = ann.eps
        self.weight = ann.weight
        self.bias = ann.bias
        self.monitor_0 = BaseMonitor()
        self.monitor_1 = BaseMonitor()
        self.monitor_2 = BaseMonitor()
        self.monitor_o = BaseMonitor()

    def forward(self, x: torch.Tensor):
        x = self.monitor_0(x)
        mean = x.mean(self.dim, keepdim=True)
        x = x - mean
        x = self.monitor_1(x)
        # Layer 2: Invert
        inv_std = (x.var(self.dim, keepdim=True) + self.eps) ** (-1 / 2)
        inv_std = self.monitor_2(inv_std)
        # Layer 3/4: Multiply
        out = x * inv_std
        out = out * self.weight + self.bias
        inv_std = self.monitor_o(inv_std)
        return out


class MonitoredSoftmax(nn.Module):

    def __init__(self, ann: nn.Softmax):
        super().__init__()
        assert isinstance(ann, nn.Softmax)
        self.dim = ann.dim
        self.monitor_1 = BaseMonitor()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x - torch.max(x, dim=self.dim, keepdim=True).values
        lse = x.exp().sum(dim=self.dim, keepdim=True).log()
        lse = self.monitor_1(lse)
        x -= lse
        x.exp_()
        return x


def set_monitor(model: nn.Module, monitor: Callable[[], BaseMonitor]):
    for _, mod in model.named_modules():
        for child_name, child_mod in mod.named_children():
            if isinstance(child_mod, nn.LayerNorm):
                setattr(mod, child_name, MonitoredLayerNorm(child_mod))
            elif isinstance(child_mod, nn.Softmax):
                setattr(mod, child_name, MonitoredSoftmax(child_mod))

    for _, mod in model.named_modules():
        for child_name, child_mod in mod.named_children():
            if isinstance(child_mod, BaseMonitor):
                setattr(mod, child_name, monitor())
