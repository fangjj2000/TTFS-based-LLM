import logging
import math
from copy import deepcopy
from typing import Optional, cast
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import Cache
from transformers.models.llama.modeling_llama import LlamaAttention, rotate_half, apply_rotary_pos_emb, repeat_kv
from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaMLP

from quantize.quantizer import UniformAffineQuantizer, clamp_ste, round_ste
from ..spike import SpikeConfig, ttfs_to_value, value_to_ttfs, ops, SpikeParamDict
from ..spike.config import SpikeConfigTTFSFormer


# region TTFSFormer 中实现的算子、网络层

class SpikingModule(nn.Module):

    def __init__(self, config: SpikeConfigTTFSFormer):
        super().__init__()
        self.spikeConfig = config


class SpikingWrapped(SpikingModule):
    def __init__(self,
                 inner: nn.Module,
                 out_range: Tuple[float, float],
                 config: SpikeConfigTTFSFormer,
                 ):
        r"""

        :param inner:
        :type inner: nn.Module
        :param config:
        :type config: SpikeConfig
        """
        super().__init__(config)
        self.inner = inner
        self.out_range = out_range

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""

        :param x: 就是值域的特征，不再是时域的时刻值。
        :return:
        """
        x = self.inner(x)
        t = value_to_ttfs(x, interval=self.out_range)
        t = self.spikeConfig.emit(value=t)
        x = ttfs_to_value(t, interval=self.out_range)
        return x


class SpikingLayerNorm(SpikingModule):

    def __init__(
            self,
            ann: nn.LayerNorm,
            config: SpikeConfigTTFSFormer,
            in_range: tuple[float, float],
            out_range: tuple[float, float],
            param: SpikeParamDict,
    ):
        r"""
        Abandoned，用不上。
        TTFSFormer 的代码，假设算子、网络层的输入、输出都是 时域，
        而我们假设算子、网络层的输入、输出都是时域。
        而且我们的模型中，并没有用到 LayerNorm 类。
        :param ann:
        :param config:
        :param in_range:
        :param out_range:
        :param param:
        """
        super().__init__(config)
        assert isinstance(ann, nn.LayerNorm)
        self.normalized_shape = ann.normalized_shape
        self.dim = tuple(range(-len(self.normalized_shape), 0))
        self.eps = ann.eps
        self.weight = ann.weight
        self.bias = ann.bias
        self.in_range = in_range
        self.x_range = param.at("monitor_1")
        self.inv_std_range = param.at("monitor_2")
        self.out_range = out_range

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        x = ttfs_to_value(t, self.in_range)
        # Layer 1: substract
        mean = x.mean(self.dim, keepdim=True)
        x.sub_(mean)
        t = self.spikeConfig.emit(value_to_ttfs(x, self.x_range))
        # Layer 2: Invert
        x = ttfs_to_value(t, self.x_range)
        inv_std = (x.var(self.dim, keepdim=True) + self.eps) ** (-1 / 2)
        inv_std = self.spikeConfig.emit(value_to_ttfs(inv_std, self.inv_std_range))
        # Layer 3/4: Multiply
        # out = x * inv_std
        # out = out * self.weight + self.bias
        out = ops.hadamard(
            t, inv_std, self.x_range, self.inv_std_range, self.out_range, self.spikeConfig, self.weight, self.bias
        )
        return out


# endregion

# region === 转换工具函数 ===


def rot(x: Tensor) -> Tensor:
    return torch.stack([-x[..., 1::2], x[..., ::2]], -1).reshape(x.shape)


def spiking_apply_rot_embed_cat(
        t: Tensor,
        emb: Tensor,
        in_range: tuple[float, float],
        out_range: tuple[float, float],
        config: SpikeConfig,
) -> Tensor:
    sin_emb, cos_emb = emb.tensor_split(2, -1)
    x = ttfs_to_value(t, in_range)
    if sin_emb.ndim == 3:
        out = x * cos_emb.unsqueeze(1).expand_as(x) + rot(x) * sin_emb.unsqueeze(1).expand_as(x)
    else:
        out = x * cos_emb + rot(x) * sin_emb
    out = value_to_ttfs(out, out_range)
    return config.emit(out)


# endregion

# region ==== 新增的脉冲算子 by G.t. ====

# region 用于我们的方法的算子、网络层。

class TTFS_Spiker(nn.Module):
    def __init__(self,
                 nameLogger: str,
                 quantizer: UniformAffineQuantizer,
                 debug: bool = False):
        r"""

        :param nameLogger:
        :type nameLogger: str
        :param quantizer:
        :type quantizer: UniformAffineQuantizer
        :param debug:
        :type debug: bool
        """
        super().__init__()
        precision = 2 ** cast(int, quantizer.n_bits)
        scale = (clamp_ste(quantizer.scale.detach(), 5e-4, 1e4))
        zero_point = clamp_ste(round_ste(quantizer.zero_point.detach()),
                               cast(int, quantizer.qmin),
                               cast(int, quantizer.qmax))

        self.configSpike = SpikeConfig(nameLogger=nameLogger,
                                       precision=precision,
                                       scale=scale.clone(),
                                       T_ref=(precision - zero_point.clone() - 1),
                                       group_size=cast(int, quantizer.group_size),
                                       debug=debug)

        self.group_size = quantizer.group_size

    def forward(self, x: Tensor) -> Tensor:
        r"""

        :param x:
        :type x: Tensor
        :return:
        :rtype: Tensor
        """
        x = self.configSpike.emit(value=x)
        return x


# endregion


# region 用于将浮点 Llama 按照 TTFSFormer 的方式转换为脉冲模型的算子、网络层。
class SpikingBaseModule(nn.Module):
    def __init__(self,
                 out_range: Tuple[float, float],
                 ann_module: nn.Module,
                 spikeConfig: SpikeConfigTTFSFormer) -> None:
        r"""
        作为 不能简单被 SpikingWrapper 封装的算子转换过程的父类。
        :param out_range:
        :type out_range: Tuple[float, float]
        :param ann_module:
        :type ann_module: nn.Module
        :param spikeConfig:
        :type spikeConfig: SpikeConfig
        """
        super().__init__()
        self.spikeConfig = spikeConfig
        self.inner = deepcopy(ann_module)
        self.out_range = out_range

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""

        :param x: 就是值域的特征，不再是时域的时刻值。
        :return:
        """
        x = self.inner(x)
        t = value_to_ttfs(x, interval=self.out_range)
        t = self.spikeConfig.emit(value=t)
        x = ttfs_to_value(t, interval=self.out_range)
        return x


class SpikingLinear(SpikingBaseModule):
    def __init__(self,
                 out_range: Tuple[float, float],
                 inner: nn.Linear,
                 configSpike: SpikeConfigTTFSFormer) -> None:
        super().__init__(out_range, inner, configSpike)


class SpikingSiLU(SpikingBaseModule):
    def __init__(self,
                 out_range: Tuple[float, float],
                 inner: nn.SiLU,
                 configSpike: SpikeConfigTTFSFormer) -> None:
        super().__init__(out_range, inner, configSpike)


class SpikingSoftmax(SpikingBaseModule):
    def __init__(
            self,
            vanilla_softmax: nn.Softmax,
            lse_range: Tuple[float, float],
            out_range: Tuple[float, float],
            spikeConfig: SpikeConfigTTFSFormer):
        r"""

        :param vanilla_softmax:
        :type vanilla_softmax: nn.Softmax
        :param lse_range:
        :type lse_range: Tuple[float, float]
        :param out_range:
        :type out_range: Tuple[float, float]
        :param spikeConfig:
        :type spikeConfig: SpikeConfig
        """
        super().__init__(out_range=out_range,
                         ann_module=vanilla_softmax,
                         spikeConfig=spikeConfig)
        assert isinstance(vanilla_softmax, nn.Softmax), \
            f"{self.__class__.__name__} 只能封装 {nn.Softmax.__name__}"
        self.dim = vanilla_softmax.dim
        self.lse_range = lse_range

    def forward(self, x: Tensor) -> Tensor:
        # 数值稳定的常规 softmax trick，避免 exp 爆炸。
        x = x - torch.max(x, dim=self.dim, keepdim=True).values
        lse = x.exp().sum(dim=self.dim, keepdim=True).log()

        t = value_to_ttfs(lse, self.lse_range)
        t = self.spikeConfig.emit(value=t)
        lse = ttfs_to_value(t, self.lse_range)

        x = x - lse
        x.exp_()

        # t = value_to_ttfs(x, self.out_range)
        # t = self.spikeConfig.emit(value=t)
        # x = ttfs_to_value(t, self.out_range)

        return x


class SpikingRMSNorm(SpikingBaseModule):
    def __init__(
            self,
            rmsn: LlamaRMSNorm,
            configSpike: SpikeConfigTTFSFormer,
            paramSpike: SpikeParamDict
    ) -> None:
        r"""
        :param rmsn:
        :type rmsn: LlamaRMSNorm
        :param configSpike:
        :type configSpike: SpikeConfig
        :param param: RMSNorm 层的监控值字典。
        :type param: SpikeParamDict
        """
        super().__init__(out_range=paramSpike.at(f"monitor_o"),
                         ann_module=rmsn,
                         spikeConfig=configSpike)
        assert isinstance(rmsn, LlamaRMSNorm), f"{self.__class__.__name__} 只能封装 {LlamaRMSNorm.__name__}"
        self.in_range = paramSpike.at(f"monitor_0")
        self.std_range = paramSpike.at(f"monitor_2")
        # RMSNorm 只在最后一维做归一
        self.dim = -1
        self.keepdim = True

        # 数值参数
        self.variance_epsilon = rmsn.variance_epsilon

        # 逐元素权重（等价于 gamma）
        # 与 SpikingLayerNorm 保持一致：直接引用（若需冻结/拷贝，可改为 .detach().clone()）
        self.register_buffer("weight", rmsn.weight)

    def forward(self, x: Tensor) -> Tensor:
        r"""
        数学一致性：QuantRMSNorm.forward 做的是 x = x * rsqrt(mean(x^2)+eps) * weight，
                  本实现严格复现该次序，只是把缩放因子 inv_rms 作为第二层 spike 序列来编码并在第三层做哈达玛积。

        精度路径：遵循 QuantRMSNorm 的思路，在求均方与 rsqrt 时用 fp32（x32），再交由编码器以 inv_rms_range 输出；
                这与 SpikingLayerNorm 中的数值稳健习惯兼容。

        监控量（range）复用：保持与 SpikingLayerNorm 同样约定：

        ops.hadamard：沿用已有签名（传入 weight、bias=None），从而把两路 spike（x 与 inv_rms）在硬件/算子侧做融合，并施加逐元素权重。
        :param x:
        :type x:Tensor
        :return:
        :rtype: Tensor
        """
        var = x.pow(2).mean(self.dim, keepdim=self.keepdim)
        inv_rms = torch.rsqrt(var + self.variance_epsilon)

        t = value_to_ttfs(inv_rms,
                          self.std_range)
        t = self.spikeConfig.emit(value=t)
        inv_rms = ttfs_to_value(t,
                                self.std_range)

        out = ops.hadamard(x=x,
                           y=inv_rms,
                           x_range=self.in_range,
                           y_range=self.std_range,
                           out_range=self.out_range,
                           config=self.spikeConfig,
                           scale=self.weight)
        return out


class SpikingLlamaMLP(LlamaMLP):
    def __init__(self,
                 mlp: LlamaMLP,
                 param: SpikeParamDict,
                 spikeConfig: SpikeConfigTTFSFormer):
        r"""

        :param mlp:
        :param param:
        :param spikeConfig:
        """
        super().__init__(config=mlp.config)

        # 确保 ann 是 LlamaMLP 实例
        assert isinstance(mlp, LlamaMLP), f"{self.__class__.__name__} 只能封装 {LlamaMLP.__name__}"

        # 将 MLP 中的 fc1, gate_proj, up_proj, down_proj, act 转换为脉冲形式.
        self.gate_proj = SpikingLinear(inner=mlp.gate_proj,
                                       out_range=param.at(f"monitor_1g"),
                                       configSpike=spikeConfig)
        self.up_proj = SpikingLinear(inner=mlp.up_proj,
                                     out_range=param.at(f"monitor_1u"),
                                     configSpike=spikeConfig)
        self.down_proj = SpikingLinear(inner=mlp.down_proj,
                                       out_range=param.at(f"monitor_o"),
                                       configSpike=spikeConfig)

        self.act_fn = SpikingSiLU(inner=mlp.act_fn,
                                  out_range=param.child(
                                      f"act_fn").at(f"monitor_o"),
                                  configSpike=spikeConfig)

        self.spikeConfig = spikeConfig
        self.param = param

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""

        :param x:
        :type x: Tensor
        :return:
        :rtype: Tensor
        """
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = ops.hadamard(self.act_fn(gate_proj),
                                               up_proj,
                                               x_range=self.act_fn.out_range,
                                               y_range=self.up_proj.out_range,
                                               out_range=self.param.at(f"monitor_2"),
                                               config=self.spikeConfig).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(ops.hadamard(x=self.act_fn(self.gate_proj(x)),
                                                    y=self.up_proj(x),
                                                    x_range=self.act_fn.out_range,
                                                    y_range=self.up_proj.out_range,
                                                    out_range=self.param.at(f"monitor_2"),
                                                    config=self.spikeConfig))

        return down_proj


class SpikingLlamaAttention(LlamaAttention):
    def __init__(self,
                 attn: LlamaAttention,
                 configSpike: SpikeConfigTTFSFormer,
                 paramSpike: SpikeParamDict,
                 nameLogger: str,
                 layer_idx: Optional[int] = None) -> None:
        r"""
        :param attn:
        :type attn:LlamaAttention
        :param configSpike: 来自于 TTFSFormer。
        :type configSpike: SpikeConfig
        :param paramSpike:
        :type paramSpike: SpikeParamDict
        :param nameLogger:
        :type nameLogger: str
        :param layer_idx:
        :type layer_idx: int
        """
        super().__init__(config=attn.config, layer_idx=layer_idx)
        self.logger = logging.getLogger(nameLogger)
        self.configSpike = configSpike
        self.paramSpike = paramSpike

        self.q_proj = SpikingLinear(inner=attn.q_proj,
                                    out_range=paramSpike.at(f"monitor_1q"),
                                    configSpike=self.configSpike)
        self.k_proj = SpikingLinear(inner=attn.k_proj,
                                    out_range=paramSpike.at(f"monitor_1k"),
                                    configSpike=self.configSpike)
        self.v_proj = SpikingLinear(inner=attn.v_proj,
                                    out_range=paramSpike.at(f"monitor_1v"),
                                    configSpike=self.configSpike)
        self.o_proj = SpikingLinear(inner=attn.o_proj,
                                    out_range=paramSpike.at(f"monitor_o"),
                                    configSpike=self.configSpike)

        self.softmax = SpikingSoftmax(vanilla_softmax=nn.Softmax(dim=-1),
                                      lse_range=self.paramSpike.child(f"softmax").at(f"monitor_1"),
                                      out_range=self.paramSpike.at(f"monitor_3"),
                                      spikeConfig=configSpike)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        ori_dtype = hidden_states.dtype
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states,
                                                        key_states,
                                                        cos,
                                                        sin)

        past_key_value = getattr(self, "past_key_value", past_key_value)
        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        B, H, Q, D = query_states.shape
        K = key_states.size(2)
        is_causal = causal_mask is None and q_len > 1
        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(D)

        # attn_weights = ops.matmul(x=query_states.to(torch.float32),
        #                           y=key_states.transpose(2, 3).to(torch.float32),
        #                           x_range=self.paramSpike.at(f"monitor_1q"),
        #                           y_range=self.paramSpike.at(f"monitor_1k"),
        #                           out_range=self.paramSpike.at(f"monitor_2"),
        #                           config=self.configSpike).to(ori_dtype) / math.sqrt(D)

        if causal_mask is not None:
            if causal_mask.dtype == torch.bool:
                attn_weights = attn_weights.masked_fill(~causal_mask, float('-inf'))
            else:
                attn_weights = attn_weights + causal_mask

        # 应用 causal mask（上三角部分设为 -inf）
        if is_causal:
            causal_mask = torch.triu(
                torch.ones(Q, K, dtype=torch.bool, device=query_states.device), diagonal=1
            )
            attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))

        # attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = self.softmax(attn_weights)

        dropout_p = self.attention_dropout if self.training else 0.0
        if dropout_p > 0.0:
            attn_weights = F.dropout(attn_weights, p=dropout_p)

        # attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        attn_output = torch.matmul(attn_weights, value_states)
        # attn_output = ops.matmul(x=attn_weights.to(torch.float32),
        #                          y=value_states.to(torch.float32),
        #                          x_range=self.paramSpike.at(f"monitor_3"),
        #                          y_range=self.paramSpike.at(f"monitor_1v"),
        #                          out_range=self.paramSpike.at(f"monitor_4"),
        #                          config=self.configSpike).to(ori_dtype)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


def ann_to_snn(
        module: nn.Module,
        configTTFSFormer: SpikeConfigTTFSFormer,
        in_range: Tuple[float, float],
        out_range: Tuple[float, float],
        param: SpikeParamDict | None = None,
) -> SpikingBaseModule | SpikingModule | nn.Identity | SpikingLlamaMLP:
    r"""
    没用上，目前选择逐个算子网络层一对一定义脉冲版本的，而不是调用这个通用函数。不过这个函数可以正常使用。

    1. `nn.Conv2d`, `nn.Linear`, `nn.ReLU`, `nn.GELU`, `nn.SiLU`, `nn.Sigmoid`
    这些模型都被统一封装在同一个 `SpikingWrapped` 中，原因如下：
    共性：逐元素操作或简单的线性变换
    这些层的共同特点是：
    逐元素操作：它们要么是线性变换（`nn.Conv2d`, `nn.Linear`），
               要么是逐元素的激活函数（`nn.ReLU`, `nn.GELU`, `nn.SiLU`, `nn.Sigmoid`）。
    在脉冲神经网络（SNN）中，这类操作通常可以通过**简单的逐点转换**来模拟，而不需要额外复杂的时序依赖或者其他特征。

    `nn.Conv2d` 和 `nn.Linear`：它们执行线性变换，通常都可以直接通过简单的脉冲计算来模仿。
    `nn.ReLU`, `nn.GELU`, `nn.SiLU`, `nn.Sigmoid`：这些都是激活函数，按元素操作，
                                                   可通过脉冲（spike）域来直接实现，通常不涉及复杂的跨时间步的依赖。

    脉冲转换：直接映射到 `SpikingWrapped`

    这些层的处理模式非常类似，都能通过在脉冲神经网络中将输入脉冲直接转换为输出脉冲来模拟：

    `SpikingWrapped` 的封装，主要通过 `ttfs_to_value` 和 `value_to_ttfs` 做脉冲与数值之间的映射，确保输入输出脉冲的时间序列变换有效。

    ------

    2. `nn.LayerNorm`
    `nn.LayerNorm` 被单独封装成 `SpikingLayerNorm`。原因如下：
    共性：全局归一化和跨维度操作

    `LayerNorm`与激活函数或线性变换的不同之处在于，它是一种 **全局归一化** 操作，
    通常是按**特定维度**计算均值和方差进行规范化。这种操作需要考虑**多个维度的交互**，并且依赖于整个输入的数据。

    在 `SpikingLayerNorm` 中，需要计算均值和方差，并对其进行归一化。该操作不仅仅是逐元素的，因此需要一种专门的机制来处理输入输出的数值关系，涉及到跨时间步的统计信息。

    `SpikingLayerNorm` 封装的复杂性

    相比简单的逐元素操作，`LayerNorm` 需要计算均值、方差，并对它们进行规范化，这意味着：
        更多的参数和操作：归一化不仅涉及输入的值，还可能涉及不同维度的**统计量**，这会增加其计算复杂度。
        非逐点：`LayerNorm` 操作依赖于输入的一些统计量（如均值和方差），这种计算方式与简单的激活函数或线性变换的逐元素操作不同，因此需要特定的封装方式。

    3. `nn.Softmax`
    `nn.Softmax` 也被单独封装为 `SpikingSoftmax`。原因如下：

    **共性：归一化且跨维度的操作**

    Softmax 是一种跨维度的操作，它计算每个输入向量元素的指数，然后归一化，使得所有元素的和为 1。
    与 `LayerNorm` 类似，Softmax 涉及跨维度的操作，而不仅仅是逐元素计算。

    `SpikingSoftmax` 的封装

    由于 **Softmax** 需要在一维上做归一化处理，因此它的计算方式与逐元素的线性变换或激活函数有本质不同。
    为了准确模拟这种操作，作者选择了**专门的封装** `SpikingSoftmax`，并提供了额外的参数（如 `monitor_1`）来控制 Softmax 输出的范围。


    4. `nn.Identity`
    `nn.Identity` 是一个直接返回输入的层，因此它在脉冲神经网络中不需要做额外的封装，可以直接通过 `nn.Identity()` 来处理。
                  这里的条件 `assert in_range == out_range` 确保了输入和输出的范围是相同的。

    **为什么选择这样封装？**

    **逐元素的操作**（如 `nn.Linear`, `nn.Conv2d`, `nn.ReLU`, `nn.GELU`, `nn.SiLU`, `nn.Sigmoid`）
                    可以直接通过简单的输入输出脉冲映射来模拟。因此，作者将它们统一用 **`SpikingWrapped`** 封装。
    **全局归一化和跨维度的操作**（如 `nn.LayerNorm` 和 `nn.Softmax`）需要考虑数据的**整体统计量**，并且它们的计算方式不同，
                    因此需要专门的封装类来处理，如 **`SpikingLayerNorm`** 和 **`SpikingSoftmax`**。

     **总结**
    这些条件分支的依据是**操作的类型和复杂度**：
    * 逐元素的层（如线性变换、激活函数）被统一封装在 `SpikingWrapped` 中。
    * 需要考虑跨维度的统计量（如归一化）的操作被单独封装成 **`SpikingLayerNorm`** 或 **`SpikingSoftmax`**。
    :param configTTFSFormer:
    :param module:
    :param config:
    :param in_range:
    :param out_range:
    :param param:
    :return:
    """

    if isinstance(module, (nn.Identity, nn.Conv2d, nn.Linear, nn.ReLU, nn.GELU, nn.SiLU, nn.Sigmoid, nn.Embedding)):
        return SpikingWrapped(inner=module,
                              out_range=out_range,
                              config=configTTFSFormer)
    elif isinstance(module, nn.LayerNorm):
        return SpikingLayerNorm(ann=module,
                                config=configTTFSFormer,
                                in_range=in_range,
                                out_range=out_range,
                                param=param)
    elif isinstance(module, LlamaRMSNorm):
        return SpikingRMSNorm(rmsn=module, configSpike=configTTFSFormer, paramSpike=param)
    elif isinstance(module, nn.Softmax):
        lse_range = param.at(f"monitor_1")
        return SpikingSoftmax(module,
                              lse_range,
                              out_range,
                              configTTFSFormer)
    elif isinstance(module, LlamaMLP):
        return SpikingLlamaMLP(module,
                               param,
                               configTTFSFormer)
    else:
        raise NotImplementedError(f"Not implemented for module of type {type(module)}")


# endregion

def spiking_apply_rotary_pos_emb(q: Tensor,
                                 k: Tensor,
                                 cos: Tensor,
                                 sin: Tensor,
                                 out_range_q: Tuple[float, float],
                                 out_range_k: Tuple[float, float],
                                 config: SpikeConfig,
                                 unsqueeze_dim=1) -> Tuple[Tensor, Tensor]:
    r"""
    Applies Rotary Position Embedding to the query and key tensors.
    :param q: The query tensor.
    :type q: Tensor
    :param k: The key tensor.
    :type k: Tensor
    :param cos: The cosine part of the rotary embedding.
    :type cos: Tensor
    :param sin: The sine part of the rotary embedding.
    :type sin: Tensor
    :param out_range_q:
    :type out_range_q: Tuple[float,float]
    :param out_range_k:
    :type out_range_k: Tuple[float,float]
    :param config:
    :type config: SpikeConfig
    :param unsqueeze_dim: (`int`, *optional*, defaults to 1):
                            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
                            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
                            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
                            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
                            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
                            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    :type unsqueeze_dim: int
    :return: comprising of the query and key tensors rotated using the Rotary Position Embedding.
    :rtype: Tuple[Tensor, Tensor]
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    q_embed = value_to_ttfs(q_embed, out_range_q)
    k_embed = value_to_ttfs(k_embed, out_range_k)
    q_embed = config.emit(q_embed)
    k_embed = config.emit(k_embed)
    q_embed = ttfs_to_value(q_embed, out_range_q)
    k_embed = ttfs_to_value(k_embed, out_range_k)
    return q_embed, k_embed

# endregion
