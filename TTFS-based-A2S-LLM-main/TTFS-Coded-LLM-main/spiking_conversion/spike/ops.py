import math

import torch
from torch import Tensor

from . import ttfs_to_value
from .config import SpikeConfig, value_to_ttfs, SpikeConfigTTFSFormer
from .kernel import spiking_matmul_core


def add(x: Tensor,
        y: Tensor,
        out_range: tuple[float, float],
        config: SpikeConfig
        ) -> Tensor:
    r"""
    Abandoned。
    时域进、时域出。因为用不上，所以没有修改为我们的 值域进、值域出。
    :param x:
    :type x: Tensor
    :param y:
    :type y: Tensor
    :param out_range:
    :type out_range: tuple[float, float]
    :param config:
    :type config: SpikeConfig
    :return:
    :rtype: Tensor
    """
    out = value_to_ttfs(x + y, out_range)  # 值域  -> 时域
    out = config.emit(out)
    out = ttfs_to_value(out, out_range)  # 时域  -> 值域
    return out


def add_scalar(x: Tensor,
               b: Tensor,
               out_range: tuple[float, float],
               config: SpikeConfig
               ) -> Tensor:
    r"""
    Abandoned。
    时域进、时域出。因为用不上，所以没有修改为我们的 值域进、值域出。
    :param x:
    :type x: Tensor
    :param b:
    :type b: Tensor
    :param out_range:
    :type out_range: tuple[float, float]
    :param config:
    :type config: SpikeConfig
    :return:
    :rtype: Tensor
    """
    out = value_to_ttfs(x + b, out_range)  # 值域  -> 时域
    out = config.emit(out)
    out = ttfs_to_value(out, out_range)  # 时域  -> 值域
    return out


def matmul(x: Tensor,
           y: Tensor,
           x_range: tuple[float, float],
           y_range: tuple[float, float],
           out_range: tuple[float, float],
           config: SpikeConfigTTFSFormer,
           scale: float = 1.0
           ) -> Tensor:
    r"""

    :param x:
    :type x: Tensor
    :param y:
    :type y: Tensor
    :param x_range:
    :type x_range: tuple[float, float]
    :param y_range:
    :type y_range: tuple[float, float]
    :param out_range:
    :type out_range: tuple[float, float]
    :param config:
    :type config: SpikeConfig
    :param scale: 作用相当于，Attention中的head_dim.
    :type scale: float
    :return:
    :rtype: Tensor
    """
    max_abs = max(x_range[1], -x_range[0], y_range[1], -y_range[0])
    out = spiking_matmul_core(x, y, max_abs, config.precision or 0)

    out = value_to_ttfs(out * scale, out_range)  # 值域  -> 时域
    out = config.emit(out)
    out = ttfs_to_value(out, out_range)  # 时域  -> 值域
    return out


def hadamard(x: Tensor,
             y: Tensor,
             x_range: tuple[float, float],
             y_range: tuple[float, float],
             out_range: tuple[float, float],
             config: SpikeConfigTTFSFormer,
             scale: float | torch.Tensor = 1.0,
             bias: float | torch.Tensor = 0.0,
             ) -> Tensor:
    r"""

    :param x:
    :type x: Tensor
    :param y:
    :type y: Tensor
    :param x_range:
    :type x_range: tuple[float, float]
    :param y_range:
    :type y_range: tuple[float, float]
    :param out_range:
    :type out_range: tuple[float, float]
    :param config:
    :type config: SpikeConfig
    :param scale:
    :type scale: float | torch.Tensor
    :param bias:
    :type bias: float | torch.Tensor
    :return:
    :rtype: Tensor
    """
    mid = torch.log(x.abs() + 1) + torch.log(y.abs() + 1)

    # mid_range = (0, math.log(max(x_range[1], -x_range[0]) + 1) + math.log(max(y_range[1], -y_range[0]) + 1))
    #
    # t_mid = value_to_ttfs(mid,
    #                       mid_range)
    # t_mid = config.emit(t_mid)
    # mid = ttfs_to_value(t_mid,
    #                     mid_range)

    out = mid.exp()
    out -= y.abs()
    out -= x.abs()
    out -= 1
    out *= torch.sign(x) * torch.sign(y)

    out = out * scale + bias

    t_out = value_to_ttfs(out, out_range)
    t_out = config.emit(t_out)
    out = ttfs_to_value(t_out, out_range)

    return out


def hadamard_scalar(x: Tensor,
                    b: Tensor,
                    out_range: tuple[float, float],
                    config: SpikeConfig,
                    ) -> Tensor:
    r"""
    Abandoned。
    时域进、时域出。因为用不上，所以没有修改为我们的 值域进、值域出。
    :param x:
    :type x: Tensor
    :param b:
    :type b: Tensor
    :param out_range:
    :type out_range: tuple[float, float]
    :param config:
    :type config: SpikeConfig
    :return:
    :rtype: Tensor
    """
    out = value_to_ttfs(x * b, out_range)  # 值域  -> 时域
    out = config.emit(out)
    out = ttfs_to_value(out, out_range)  # 时域  -> 值域
    return out
