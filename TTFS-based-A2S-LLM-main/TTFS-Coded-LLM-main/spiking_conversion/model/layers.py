from functools import partial

import torch
import torch.nn as nn
from torch import Tensor

from ..util import BaseMonitor


def drop_path(x: Tensor, drop_prob: float = 0.0, training: bool = False) -> Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):

    def __init__(
            self,
            img_size: int = 224,
            patch_size: int = 16,
            in_c: int = 3,
            embed_dim: int = 768,
            norm_layer: type[nn.Module] | None = None,
    ):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.monitor_0 = BaseMonitor()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.monitor_1 = BaseMonitor()
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        self.monitor_o = BaseMonitor()

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        assert (
                H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = self.monitor_0(x)
        x = self.proj(x)
        x = self.monitor_1(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = self.monitor_o(x)
        return x


class Mlp(nn.Module):
    def __init__(
            self,
            in_features: int,
            hidden_features: int | None = None,
            out_features: int | None = None,
            act_layer: type[nn.Module] = nn.GELU,
            norm_layer: type[nn.Module] | None = None,
            drop: float = 0.0,
    ) -> None:
        r"""
        最常见的前馈网络结构（FFN），
        常见于 Transformer、ViT 等模型的 Feed-Forward 层
        :param in_features:
        :type in_features: int
        :param hidden_features:
        :type hidden_features: int
        :param out_features:
        :type out_features: int
        :param act_layer:
        :type act_layer:
        :param norm_layer
        :type norm_layer:
        :param drop:
        :type drop: float
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.monitor_0 = BaseMonitor()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.monitor_1 = BaseMonitor()
        self.act = act_layer()
        self.monitor_2 = BaseMonitor()
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.monitor_3 = BaseMonitor()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.monitor_o = BaseMonitor()
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        r"""
        逐点非线性：输入特征通过一次扩张 (fc1) → 激活 → 再投影回原维度 (fc2)。
        :param x:
        :type x: Tensor
        :return:
        :rtype: Tensor
        """
        x = self.monitor_0(x)
        x = self.fc1(x)
        x = self.monitor_1(x)
        x = self.act(x)
        x = self.monitor_2(x)
        x = self.drop(x)
        x = self.norm(x)
        x = self.monitor_3(x)
        x = self.fc2(x)
        x = self.monitor_o(x)
        x = self.drop(x)
        return x


class GluMlp(nn.Module):
    def __init__(
            self,
            in_features: int,
            hidden_features: int | None = None,
            out_features: int | None = None,
            act_layer: type[nn.Module] = nn.Sigmoid,
            norm_layer: type[nn.Module] | None = None,
            bias: bool = True,
            drop: float = 0.0,
            use_conv: bool = False,
            gate_last: bool = True,
    ) -> None:
        r"""
        基于门控线性单元 Gated Linear Units (GLU) 思想的 MLP。
        相比普通 MLP 的 "激活再线性"，GLU 在信息流上增加了乘法交互，表达力更强。
        :param in_features:
        :type in_features: int
        :param hidden_features:
        :type hidden_features: int
        :param out_features:
        :type out_features: int
        :param act_layer:
        :type act_layer:
        :param norm_layer:
        :type norm_layer:
        :param bias:
        :type bias: bool
        :param drop:
        :type drop: float
        :param use_conv:
        :type use_conv: bool
        :param gate_last:
        :type gate_last: bool
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        assert hidden_features % 2 == 0
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear
        self.chunk_dim = 1 if use_conv else -1
        self.gate_last = gate_last  # use second half of width for gate
        self.fc1 = linear_layer(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.norm = norm_layer(hidden_features // 2) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features // 2, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

        self.monitor_0 = BaseMonitor()
        self.monitor_1 = BaseMonitor()
        self.monitor_2 = BaseMonitor()
        self.monitor_3 = BaseMonitor()
        self.monitor_4 = BaseMonitor()
        self.monitor_o = BaseMonitor()

    def forward(self, x: Tensor) -> Tensor:
        r"""
        第一层输出被切分为两半：
        :param x:
        :type x: Tensor
        :return:
        :rtype: Tensor
        """
        x = self.monitor_0(x)
        x = self.fc1(x)
        x = self.monitor_1(x)
        x1, x2 = x.chunk(2, dim=self.chunk_dim)

        # 一半特征经过激活函数当作 gate，去控制另一半特征的通道流量。
        if self.gate_last:
            x2 = self.monitor_2(self.act(x2))
        else:
            x1 = self.monitor_2(self.act(x1))

        x = x1 * x2
        x = self.drop1(x)
        x = self.monitor_3(x)
        x = self.norm(x)
        x = self.monitor_4(x)
        x = self.fc2(x)
        x = self.drop2(x)
        x = self.monitor_o(x)
        return x


class SwiGLU(nn.Module):

    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.SiLU,
            norm_layer=None,
            bias=True,
            drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1_g = nn.Linear(in_features, hidden_features, bias=bias)
        self.fc1_x = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

        self.monitor_0 = BaseMonitor()
        self.monitor_1 = BaseMonitor()
        self.monitor_2 = BaseMonitor()
        self.monitor_3 = BaseMonitor()
        self.monitor_4 = BaseMonitor()
        self.monitor_5 = BaseMonitor()
        self.monitor_o = BaseMonitor()

    def forward(self, x):
        x = self.monitor_0(x)
        x_gate = self.monitor_1(self.fc1_g(x))
        x_gate = self.monitor_2(self.act(x_gate))
        x = self.monitor_3(self.fc1_x(x))
        x = x_gate * x
        x = self.drop1(x)
        x = self.monitor_4(x)
        x = self.norm(x)
        x = self.monitor_5(x)
        x = self.fc2(x)
        x = self.drop2(x)
        x = self.monitor_o(x)
        return x


def rot(x: Tensor) -> Tensor:
    return torch.stack([-x[..., 1::2], x[..., ::2]], -1).reshape(x.shape)


def apply_rot_embed_cat(x: Tensor, emb: Tensor) -> Tensor:
    sin_emb, cos_emb = emb.tensor_split(2, -1)
    if sin_emb.ndim == 3:
        return x * cos_emb.unsqueeze(1).expand_as(x) + rot(x) * sin_emb.unsqueeze(1).expand_as(x)
    return x * cos_emb + rot(x) * sin_emb


def apply_keep_indices_nlc(x: Tensor, pos_embed: Tensor, keep_indices: Tensor) -> Tensor:
    pos_embed = pos_embed.unsqueeze(0).expand(x.shape[0], -1, -1)
    pos_embed = pos_embed.gather(1, keep_indices.unsqueeze(-1).expand(-1, -1, pos_embed.shape[-1]))
    return pos_embed
