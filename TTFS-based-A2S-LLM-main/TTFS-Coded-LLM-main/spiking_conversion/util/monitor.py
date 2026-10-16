from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

__all__ = ["BaseMonitor", "PercentileMonitor", "RangeMonitor", "MinMaxMonitor"]


class BaseMonitor(nn.Module):

    def __init__(self) -> None:
        r"""
        Monitor 是一个 透明的“探针”，插在网络中，不影响数值流动，但能记录信息。
        """
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        r"""
        默认直接返回输入，不改变计算图。
        :param x:
        :type: Tensor
        :return:
        """
        return x

    def monitor_data(self) -> Any:
        r"""
        留给子类实现，返回收集到的统计结果。
        :return:
        """
        raise NotImplementedError()


class PercentileMonitor(BaseMonitor):
    def __init__(self, percentile: list[float]) -> None:
        r"""
        统计数据分布的典型位置（中位数、四分位等），了解数据大致落在哪些区间。
        :param percentile:
        :type percentile: list[float]
        """
        super().__init__()
        self.num = 0
        self.percentile = Tensor(percentile, device="cuda")
        self.value = torch.zeros(len(percentile), device="cuda")

    def forward(self, x: Tensor) -> Tensor:
        r"""
        对 x 按指定分位数计算（例如 0.25, 0.5, 0.75 分位），累计到 self.value。
        :param x:
        :type x: Tensor
        :return: 平均分位数（运行多次 forward 后取平均）。
        :rtype: Tensor
        """
        self.value += torch.quantile(x, self.percentile)
        self.num += 1
        return x

    def monitor_data(self):
        return (self.value / self.num).tolist()


class RangeMonitor(BaseMonitor):
    def __init__(self) -> None:
        r"""
        确定某个张量在训练或推理中，可能达到的全局数值范围。
        全局极值（受一次异常波动影响可能特别大）。
        """
        super().__init__()
        self.min = torch.inf
        self.max = -torch.inf

    def forward(self, x: Tensor) -> Tensor:
        r"""
        记录见过的最小值、最大值（运行期间全局 min/max）。
        :param x:
        :type x: Tensor
        :return: 最终的 (min, max)。
        :rtype: Tensor
        """
        self.min = min(self.min, torch.min(x).item())
        self.max = max(self.max, torch.max(x).item())
        return x

    def monitor_data(self):
        return self.min, self.max


class MinMaxMonitor(BaseMonitor):
    def __init__(self):
        r"""
        平均极值，更平滑，减少极端 outlier 的影响。
        """
        super().__init__()
        self.min = 0.0
        self.max = 0.0
        self.num = 0

    def forward(self, x: Tensor) -> Tensor:
        r"""
        每次 forward 的 min/max 先算出来，再进行累计平均。
        :param x:
        :type x: Tensor
        :return:平均 min/max。
        :rtype: Tensor
        """
        self.min += torch.min(x).item()
        self.max += torch.max(x).item()
        self.num += 1
        return x

    def monitor_data(self):
        return (self.min / self.num, self.max / self.num) if self.num else None


def get_monitor_data(model: nn.Module) -> dict[str, Any]:
    r"""

    :param model:
    :type model: nn.Module
    :return:
    :rtype: dict
    """
    data = {}
    for name, child in model.named_modules():
        if isinstance(child, BaseMonitor):
            info = child.monitor_data()
            if info is not None:
                data[name] = info
    return data
