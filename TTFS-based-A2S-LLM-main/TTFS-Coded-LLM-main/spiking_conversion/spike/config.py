import logging

import torch
from torch import nn, Tensor


def value_to_ttfs(x: torch.Tensor, interval: tuple[float, float]) -> torch.Tensor:
    r"""
    仅仅是归一化，不代表 值域 -> 时域。
    :param x:
    :param interval:
    :return:
    """
    a, b = interval
    return (x - a) / (b - a)


def ttfs_to_value(t: torch.Tensor, interval: tuple[float, float]) -> torch.Tensor:
    r"""
    to be abandoned
    :param t:
    :param interval:
    :return:
    """
    a, b = interval
    return t * (b - a) + a


class SpikeConfig(nn.Module):
    def __init__(self,
                 nameLogger: str,
                 precision: int,
                 scale: Tensor,
                 T_ref: Tensor,
                 group_size: int,
                 debug: bool = False) -> None:
        r"""

        :param nameLogger:
        :type nameLogger: str
        :param precision:
        :type precision: int
        :param scale:
        :type scale: Tensor
        :param T_ref:
        :type T_ref: Tensor
        :param debug:
        :type debug: bool
        """
        super().__init__()
        self.logger = logging.getLogger(nameLogger)
        self.register_buffer("scale", scale)
        self.register_buffer("tau", 1.0 / scale)
        self.register_buffer("T_ref", T_ref)
        self.precision = precision
        self.debug = debug
        self.group_size = group_size

    def extra_repr(self) -> str:
        repr_info = (f"precision = {self.precision}, "
                     f"group_size = {self.group_size}")
        return repr_info

    def emit(self,
             value: Tensor
             ) -> Tensor:
        r"""
        Convert the theoretical emitting time to the real emitting time.
        Simulate the precision loss.
        :param value:
        :type value: Tensor
        :return:
        :rtype: Tensor
        """
        if self.debug:
            bs, n, dim1 = value.shape
            value = value.reshape(bs, n, -1, self.group_size)
            value = value * self.tau
            self.logger.debug(f"value after x scale ({self.scale}):{value}")
            value = self.T_ref - value
            self.logger.debug(f"value after - from Tref:{value}")
            value.round_()
            self.logger.debug(f"value after round:{value}")
            value = value.clip(0, self.precision - 1)
            self.logger.debug(f"value after clip tp [0, {self.precision - 1}]:{value}")
            value = self.T_ref - value
            self.logger.debug(f"value after - from Tref again:{value}")
            value = value * self.scale
            self.logger.debug(f"value after mul by scale({self.scale}):{value}")
            out = value.reshape(bs, n, dim1)
        else:
            bs, n, d = value.shape
            value = value.view(bs, n, -1, self.group_size)
            q = torch.round(self.T_ref - value * self.tau)
            q = torch.clamp(q, 0, self.precision - 1)
            out = (self.T_ref - q) * self.scale
            out = out.view(bs, n, d)
        return out


class SpikeConfigTTFSFormer(nn.Module):
    def __init__(self,
                 nameLogger: str,
                 precision: int,
                 debug: bool = False):
        r"""
        All SpikingLayers share the same SpikeConfig.
        :param nameLogger:
        :type nameLogger: str
        :param precision:
        :type precision: int
        """
        super().__init__()
        self.logger = logging.getLogger(nameLogger)
        self.debug = debug
        self.precision = precision
        self.inv_precision = 1 / precision

    @torch.no_grad()
    def emit(self,
             value: Tensor) -> Tensor:
        r"""
        Convert the theoretical emitting time to the real emitting time.
        Simulate the precision loss.
        :param value:
        :type value: Tensor
        :return:
        :rtype: Tensor
        """
        if self.debug:
            v1 = value * self.precision
            self.logger.debug(f"after mul: {v1}")
            v2 = torch.round(v1)
            self.logger.debug(f"after round: {v2}")
            v3 = v2 * self.inv_precision
            self.logger.debug(f"after div: {v3}")
            v4 = torch.clamp(v3, 0.0, 1.0)
            self.logger.debug(f"after clamp: {v4}")
            y = v4
        else:
            y = torch.clamp(torch.round(value * self.precision) * self.inv_precision, 0.0, 1.0)
        return y
