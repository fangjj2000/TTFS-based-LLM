import logging
from typing import cast, Tuple, Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, LongTensor
from transformers import Cache
from transformers.models.llama.modeling_llama import LlamaMLP, LlamaAttention, repeat_kv, \
    LlamaRMSNorm, apply_rotary_pos_emb, rotate_half
from transformers.models.mistral.modeling_mistral import MistralRMSNorm
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

import utils.hadamard_utils as hadamard_utils
from utils.model_utils import RMSN


# region ==== 新增的量化算子 by G.t ====
class QuantizedBaseModule(nn.Module):
    def __init__(self,
                 org_module: Optional[nn.Module] = None) -> None:
        r"""
        :param org_module:
        :type org_module: nn.Module
        """
        super().__init__()
        # 默认所有模块都不执行量化操作。
        self.use_act_quant = False
        self.use_weight_quant = False

        # 默认将所有的量化比特数都设值为 16，也是为了将量化操作默认关闭，除非在外部更改 bit 数，并分配量化器，来开启量化。
        self.wbits = 16
        self.input_bits = 16
        self.output_bits = 16
        if org_module is not None:
            self.inner_repr = repr(org_module)
        else:
            self.inner_repr = None

    def set_quant_state(self,
                        weight_quant: bool = False,
                        act_quant: bool = False) -> None:
        r"""
        用于从外部设置量化版本的算子是否进行量化操作。
        :param weight_quant:
        :type weight_quant: bool
        :param act_quant:
        :type act_quant: bool
        :return:
        """
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant

    def extra_repr(self) -> str:
        info = (f"input_bits = {self.input_bits}, output_bits = {self.output_bits}, weight_bits = {self.wbits}"
                f"\nquantA = {self.use_act_quant}, quantW = {self.use_weight_quant}")
        if self.inner_repr is not None:
            info = info + f"\noriginal module = {self.inner_repr}, "
        return info

    def forward(self,
                *args):
        NotImplementedError("需要子类分别实现各自的forward函数!")


class QuantizedIdentity(QuantizedBaseModule):
    def __init__(
            self,
            num_features: int) -> None:
        r"""

        :param num_features:
        :type num_features: int
        """
        super().__init__()
        self.num_features = num_features

    def extra_repr(self) -> str:
        super_repr = super().extra_repr()
        curr_repr = f"\nnum_features = {self.num_features}"
        return super_repr + curr_repr

    def forward(self,
                x: Tensor) -> Tensor:
        r"""

        :param x:
        :type x: Tensor
        :return:
        :rtype: Tensor
        """
        if self.use_act_quant and self.output_bits < 16:
            x = self.output_quantizer(x)
        return x


class QuantizedSiLU(QuantizedBaseModule):
    def __init__(self,
                 org_silu: nn.SiLU):
        super().__init__(org_module=org_silu)
        assert isinstance(org_silu, nn.SiLU), f"QuantizedSiLU 只支持 SiLU 算子，实际为 {type(org_silu)}."
        self.silu = org_silu

    def forward(self, x: Tensor) -> Tensor:
        r"""

        :param x:
        :type x: Tensor
        :param y:
        :type y: Tensor
        :return:
        :rtype: Tensor
        """
        x = self.silu(x)
        if self.use_act_quant and self.output_bits < 16:
            x = self.output_quantizer(x)
        return x


class QuantizedEmbedding(QuantizedBaseModule):
    def __init__(self,
                 org_embedding: nn.Embedding) -> None:
        r"""
        :param org_embedding:
        :type org_embedding: nn.Module
        """
        super().__init__(org_module=org_embedding)
        assert isinstance(org_embedding, nn.Embedding), \
            f"传给 {self.__class__.__name__} 的必须是 nn.Embedding，实际为 {type(org_embedding)}"

        self.register_buffer('weight', org_embedding.weight)
        self.out_features = org_embedding.embedding_dim
        self.padding_idx = org_embedding.padding_idx
        self.max_norm = org_embedding.max_norm
        self.norm_type = org_embedding.norm_type
        self.scale_grad_by_freq = org_embedding.scale_grad_by_freq
        self.sparse = org_embedding.sparse

    def forward(self, x: Tensor) -> Tensor:
        r"""

        :param x:
        :type x: Tensor
        :param y:
        :type y: Tensor
        :return:
        :rtype: Tensor
        """
        x = F.embedding(x,
                        self.weight,
                        self.padding_idx,
                        self.max_norm,
                        self.norm_type,
                        self.scale_grad_by_freq,
                        self.sparse)
        if self.use_act_quant and self.output_bits < 16:
            x = self.output_quantizer(x)
        return x


class QuantizedLinear(QuantizedBaseModule):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """

    def __init__(self,
                 org_linear: nn.Linear) -> None:
        r"""
        :param org_linear:
        :type org_linear: nn.Linear
        """
        super().__init__(org_module=org_linear)
        assert isinstance(org_linear, nn.Linear), \
            f"传给 {self.__class__.__name__} 的必须是 nn.Linear，实际为 {type(org_linear)}"

        self.fwd_func = F.linear
        self.register_parameter('weight', org_linear.weight)  # trainable
        if org_linear.bias is not None:
            self.register_buffer('bias', org_linear.bias)
        else:
            self.bias = None

        self.in_features = org_linear.in_features
        self.out_features = org_linear.out_features
        self.online_full_had = False
        self.use_temporary_parameter = False

    def forward(self,
                inputTensor: Tensor) -> Tensor:
        r"""

        :param inputTensor:
        :type inputTensor: Tensor
        :return:
        :rtype: Tensor
        """
        input_dtype = inputTensor.dtype

        # Rotate, if needed
        if self.online_full_had:
            if self.fp32_had:  # Full Hadamard in FP32
                inputTensor = hadamard_utils.matmul_hadU_cuda(inputTensor.float(), self.had_K, self.K).to(input_dtype)
            else:  # Full Hadamard in FP16
                inputTensor = hadamard_utils.matmul_hadU_cuda(inputTensor, self.had_K, self.K)

        if self.use_temporary_parameter:
            weight = self.temp_weight
        else:
            weight = self.weight

        bias = self.bias

        if self.use_weight_quant and self.wbits < 16:
            weight = self.weight_quantizer(weight)

        if self.use_act_quant and self.input_bits < 16:
            inputTensor = self.input_quantizer(inputTensor)

        out = self.fwd_func(inputTensor, weight, bias)

        if self.use_act_quant and self.output_bits < 16:
            out = self.output_quantizer(out)

        return out


class QuantizedHadamard(QuantizedBaseModule):
    def __init__(self,
                 num_features: int,
                 attachIdentity: bool, ) -> None:
        super().__init__()
        self.attachIdentity = attachIdentity
        if self.attachIdentity:
            self.identity = QuantizedIdentity(num_features=num_features)

    def forward(self,
                x: Tensor,
                y: Tensor):
        mid = torch.log1p(x.abs()) + torch.log1p(y.abs())
        if self.attachIdentity:
            mid = self.identity(mid)

        sign = torch.sign(x) * torch.sign(y)
        out = torch.exp(mid) - x.abs() - y.abs() - 1.0  # 非原地链式构造
        out = out * sign  # 非原地
        return out


class QuantizedSoftmax(QuantizedBaseModule):
    def __init__(self,
                 dim: int,
                 dtype: torch.dtype,
                 num_features: int,
                 attachIdentity: bool) -> None:
        r"""

        :param dim:
        :type dim: int
        :param dtype:
        :type dtype: torch.dtype
        :param num_features:
        :type num_features: int
        :param attachIdentity:

        """
        super().__init__()
        self.dim = dim
        self.dtype = dtype
        self.attachIdentity = attachIdentity
        if self.attachIdentity:
            self.identityLSE = QuantizedIdentity(num_features)

    def forward(self,
                x: Tensor,
                num_heads: int) -> Tensor:
        r"""

        :param x:
        :type x: Tensor
        :param num_heads:
        :type num_heads: int
        :return:
        :rtype: Tensor
        """
        x = x.to(self.dtype)
        # 更稳：直接用 logsumexp
        x_shifted = x - x.max(dim=self.dim, keepdim=True).values
        lse = torch.logsumexp(x_shifted, dim=self.dim, keepdim=True)

        if self.attachIdentity:
            b, head, d1, d2 = lse.shape
            lse_ = lse.view(b, num_heads, -1).transpose(1, 2).contiguous()
            lse_ = self.identityLSE(lse_)
            lse = lse_.transpose(1, 2).contiguous().view(b, head, d1, d2)

        # 非原地：概率 = exp(shifted - lse)
        probs = torch.exp(x_shifted - lse)
        return probs


class QuantizedLlamaRMSNorm(QuantizedBaseModule):
    def __init__(self,
                 org_norm: LlamaRMSNorm | RMSN):
        r"""

        :param org_norm:
        :type org_norm: LlamaRMSNorm | RMSN
        """
        super().__init__(org_module=org_norm)
        assert isinstance(org_norm, (LlamaRMSNorm, RMSN)), \
            f"传给 {self.__class__.__name__} 的必须是 LlamaRMSNorm 或者 RMSN，实际为 {type(org_norm)}"
        self.register_buffer('weight', org_norm.weight)
        self.variance_epsilon = org_norm.variance_epsilon
        self.use_temporary_parameter = False
        self.out_features = org_norm.weight.shape[-1]
        self.hidden_shape = org_norm.weight.shape
        self.inner_repr += f"\nhidden_shape = {self.hidden_shape}"

    def forward(self, x):
        if self.use_temporary_parameter:
            weight = self.temp_weight
        else:
            weight = self.weight

        input_dtype = x.dtype
        if x.dtype == torch.float16:
            x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        variance = torch.rsqrt(variance + self.variance_epsilon)
        x = x * variance
        x = x.to(input_dtype) * weight

        if self.use_act_quant and self.output_bits < 16:
            x = self.output_quantizer(x)

        return x


class QuantizedQwen2RMSNorm(QuantizedBaseModule):
    def __init__(self,
                 org_norm: Qwen2RMSNorm | RMSN):
        r"""
        Qwen2RMSNorm 的量化版本。

        :param org_norm: 原始的 Qwen2RMSNorm 或通用 RMSNorm 模块
        :type org_norm: Qwen2RMSNorm | RMSN
        """
        super().__init__(org_module=org_norm)
        assert isinstance(org_norm, (Qwen2RMSNorm, RMSN)), \
            f"传给 {self.__class__.__name__} 的必须是 Qwen2RMSNorm 或 RMSN，实际为 {type(org_norm)}"

        # 与 QuantizedLlamaRMSNorm 一致，权重使用 buffer 方式注册（避免被优化器更新）
        self.register_buffer('weight', org_norm.weight)
        self.variance_epsilon = org_norm.variance_epsilon

        # 临时参数控制，与 QuantizedLlamaRMSNorm 保持一致接口
        self.use_temporary_parameter = False
        self.out_features = org_norm.weight.shape[-1]
        self.hidden_shape = org_norm.weight.shape
        self.inner_repr += f"\nhidden_shape = {self.hidden_shape}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 选择使用临时权重还是原始权重
        if self.use_temporary_parameter:
            weight = self.temp_weight
        else:
            weight = self.weight

        input_dtype = x.dtype
        # 与原始 Qwen2RMSNorm / LlamaRMSNorm 一致：在 float32 中计算方差
        if x.dtype == torch.float16:
            x = x.to(torch.float32)

        variance = x.pow(2).mean(-1, keepdim=True)
        variance = torch.rsqrt(variance + self.variance_epsilon)
        x = x * variance
        x = x.to(input_dtype) * weight

        # 激活量化（输出量化），逻辑与 QuantizedLlamaRMSNorm 保持一致
        if self.use_act_quant and self.output_bits < 16:
            x = self.output_quantizer(x)

        return x


class QuantizedMistralRMSNorm(QuantizedBaseModule):
    def __init__(self,
                 org_norm: MistralRMSNorm | RMSN):
        r"""
        MistralRMSNorm 的量化版本。

        :param org_norm: 原始的 MistralRMSNorm 或通用 RMSNorm 模块
        :type org_norm: MistralRMSNorm | RMSN
        """
        super().__init__(org_module=org_norm)
        assert isinstance(org_norm, (MistralRMSNorm, RMSN)), \
            f"传给 {self.__class__.__name__} 的必须是 MistralRMSNorm 或 RMSN，实际为 {type(org_norm)}"

        # 与前面保持一致：把权重注册成 buffer，避免被优化器更新
        self.register_buffer("weight", org_norm.weight)
        self.variance_epsilon = org_norm.variance_epsilon

        # 与其他量化 RMSNorm 保持一致的接口
        self.use_temporary_parameter = False
        self.out_features = org_norm.weight.shape[-1]
        self.hidden_shape = org_norm.weight.shape
        self.inner_repr += f"\nhidden_shape = {self.hidden_shape}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 选择是否使用临时权重
        if self.use_temporary_parameter:
            weight = self.temp_weight
        else:
            weight = self.weight

        input_dtype = x.dtype
        if x.dtype == torch.float16:
            x = x.to(torch.float32)

        variance = x.pow(2).mean(-1, keepdim=True)
        variance = torch.rsqrt(variance + self.variance_epsilon)
        x = x * variance
        x = x.to(input_dtype) * weight

        # 输出量化：逻辑与 QuantizedLlamaRMSNorm / QuantizedQwen2RMSNorm 保持一致
        if self.use_act_quant and self.output_bits < 16:
            x = self.output_quantizer(x)

        return x


class QuantizedLlamaMLP(LlamaMLP):
    def __init__(self,
                 org_mlp: LlamaMLP) -> None:
        r"""

        :param org_mlp:
        :type org_mlp: LlamaMLP
        """
        super().__init__(config=org_mlp.config)
        assert isinstance(org_mlp, LlamaMLP), \
            f"传给 {self.__class__.__name__} 的必须为 LlamaMLP 对象, 实际为 {type(org_mlp)}。"

        self.gate_proj = QuantizedLinear(org_linear=cast(nn.Linear, org_mlp.gate_proj))
        self.up_proj = QuantizedLinear(org_linear=cast(nn.Linear, org_mlp.up_proj))
        self.down_proj = QuantizedLinear(org_linear=cast(nn.Linear, org_mlp.down_proj))

        self.act_fn = QuantizedSiLU(org_silu=cast(nn.SiLU, org_mlp.act_fn))


class QuantizedLlamaAttentionOld(LlamaAttention):
    def __init__(self,
                 org_attn: LlamaAttention,
                 nameLogger: str,
                 attachIdentity: bool = True) -> None:
        r"""
        :param org_attn:
        :type org_attn: LlamaAttention
        :param nameLogger:
        :type nameLogger: str
        :param attachIdentity:
        :type attachIdentity: bool
        """
        super().__init__(config=org_attn.config, layer_idx=org_attn.layer_idx)
        self.logger = logging.getLogger(nameLogger)
        assert isinstance(org_attn, LlamaAttention), \
            f"传给 {self.__class__.__name__} 的必须为 LlamaAttention 对象, 实际为 {type(org_attn)}。"

        # 这几个 nn.Linear 类型的 proj 层的封装去掉了，因为已经通过 Linear 对象的变量来全局替换了所有的 Linear 对象。
        # self.q_proj = QuantizedLinear(cast(nn.Linear, org_attn.q_proj))
        # self.k_proj = QuantizedLinear(cast(nn.Linear, org_attn.k_proj))
        # self.v_proj = QuantizedLinear(cast(nn.Linear, org_attn.v_proj))
        # self.o_proj = QuantizedLinear(cast(nn.Linear, org_attn.o_proj))

        self.attachIdentity = attachIdentity
        if self.attachIdentity:
            self.identityQ = QuantizedIdentity(org_attn.q_proj.out_features)
            self.identityK = QuantizedIdentity(org_attn.q_proj.out_features)
            self.identityV = QuantizedIdentity(org_attn.q_proj.out_features)
            self.identitySoftmax = QuantizedIdentity(cast(int, self.num_heads))

        # self.softmax = QuantizedSoftmax(dim=-1,
        #                                 dtype=torch.float32,
        #                                 num_features=org_attn.v_proj.out_features,
        #                                 attachIdentity=self.attachIdentity)

    def forward(self,
                hidden_states: Tensor,
                attention_mask: Optional[Tensor] = None,
                position_ids: Optional[LongTensor] = None,
                past_key_value: Optional[Cache] = None,
                output_attentions: bool = False,
                use_cache: bool = False,
                cache_position: Optional[LongTensor] = None,
                **kwargs
                ) -> Tuple[Tensor, Optional[Tensor], Optional[Tuple[Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            self.logger.warning(
                "LlamaModel is using LlamaSdpaAttention, "
                "but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`."
                " Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. '
                'This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )

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

        if self.attachIdentity:  # 对 施加了位置编码后的 Q, K, V 进行量化。
            B, H, Q, D = key_states.shape
            query_states = query_states.transpose(1, 2).contiguous().view(bsz, q_len, -1)
            key_states = key_states.transpose(1, 2).contiguous().view(bsz, Q, -1)
            value_states = value_states.transpose(1, 2).contiguous().view(bsz, Q, -1)

            query_states = self.identityQ(query_states)
            key_states = self.identityK(key_states)
            value_states = self.identityV(value_states)

            query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = key_states.view(B, Q, H, D).transpose(1, 2)
            value_states = value_states.view(B, Q, H, D).transpose(1, 2)

        B, H, Q, D = query_states.shape
        K = key_states.size(2)
        is_causal = causal_mask is None and q_len > 1
        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(D)

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
        attn_weights = F.softmax(attn_weights, dim=-1)

        if self.attachIdentity:  # 对 softmax 之后的注意力 score 进行量化。
            b, head, d1, d2 = attn_weights.shape
            attn_weights = attn_weights.view(bsz, self.num_heads, -1).transpose(1, 2)
            attn_weights = self.identitySoftmax(attn_weights)
            attn_weights = attn_weights.transpose(1, 2).contiguous().view(b, head, d1, d2)

        dropout_p = self.attention_dropout if self.training else 0.0
        if dropout_p > 0.0:
            attn_weights = F.dropout(attn_weights, p=dropout_p)

        # attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        attn_output = torch.matmul(attn_weights, value_states)

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


class QuantizedLlamaAttention(LlamaAttention):
    def __init__(self,
                 org_attn: LlamaAttention,
                 nameLogger: str,
                 attachIdentity: bool = True) -> None:
        r"""
        :param org_attn:
        :type org_attn: LlamaAttention
        :param nameLogger:
        :type nameLogger: str
        :param attachIdentity:
        :type attachIdentity: bool
        """
        super().__init__(config=org_attn.config, layer_idx=org_attn.layer_idx)
        self.logger = logging.getLogger(nameLogger)
        assert isinstance(org_attn, LlamaAttention), \
            f"传给 {self.__class__.__name__} 的必须为 LlamaAttention 对象, 实际为 {type(org_attn)}。"

        # 这几个 nn.Linear 类型的 proj 层的封装去掉了，因为已经通过 Linear 对象的变量来全局替换了所有的 Linear 对象。
        # self.q_proj = QuantizedLinear(cast(nn.Linear, org_attn.q_proj))
        # self.k_proj = QuantizedLinear(cast(nn.Linear, org_attn.k_proj))
        # self.v_proj = QuantizedLinear(cast(nn.Linear, org_attn.v_proj))
        # self.o_proj = QuantizedLinear(cast(nn.Linear, org_attn.o_proj))

        self.attachIdentity = attachIdentity
        if self.attachIdentity:
            self.identityQ = QuantizedIdentity(org_attn.q_proj.out_features)
            self.identityK = QuantizedIdentity(org_attn.q_proj.out_features)
            self.identityV = QuantizedIdentity(org_attn.q_proj.out_features)
            self.identitySoftmax = QuantizedIdentity(cast(int, self.num_heads))

        # 已经通过 identitySoftmax 对 softmax 算子进行了量化。
        # self.softmax = QuantizedSoftmax(dim=-1,
        #                                 dtype=torch.float32,
        #                                 num_features=org_attn.v_proj.out_features,
        #                                 attachIdentity=self.attachIdentity)

    def forward(self,
                hidden_states: Tensor,
                attention_mask: Optional[Tensor] = None,
                position_ids: Optional[LongTensor] = None,
                past_key_value: Optional[Cache] = None,
                output_attentions: bool = False,
                use_cache: bool = False,
                cache_position: Optional[LongTensor] = None,
                **kwargs
                ) -> Tuple[Tensor, Optional[Tensor], Optional[Tuple[Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            self.logger.warning(
                "LlamaModel is using LlamaSdpaAttention, "
                "but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`."
                " Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. '
                'This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        cos_unsq = cos.unsqueeze(1)
        sin_unsq = sin.unsqueeze(1)
        query_states = (query_states * cos_unsq) + (rotate_half(query_states) * sin_unsq)
        key_states = (key_states * cos_unsq) + (rotate_half(key_states) * sin_unsq)

        # In case static cache is used, it is an instance attribute.
        past_key_value = getattr(self, "past_key_value", past_key_value)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos,
                            "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states,
                                                             value_states,
                                                             self.layer_idx,
                                                             cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        if self.attachIdentity:  # 对 施加了位置编码后的 Q, K, V 进行量化。
            B, H, Q, D = key_states.shape
            query_states = query_states.transpose(1, 2).contiguous().view(bsz, q_len, -1)
            key_states = key_states.transpose(1, 2).contiguous().view(bsz, Q, -1)
            value_states = value_states.transpose(1, 2).contiguous().view(bsz, Q, -1)

            query_states = self.identityQ(query_states)
            key_states = self.identityK(key_states)
            value_states = self.identityV(value_states)

            query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = key_states.view(B, Q, H, D).transpose(1, 2)
            value_states = value_states.view(B, Q, H, D).transpose(1, 2)

        B, H, Q, D = query_states.shape
        K = key_states.size(2)
        is_causal = causal_mask is None and q_len > 1

        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(D)

        if causal_mask is not None:
            if causal_mask.dtype == torch.bool:
                attn_weights = attn_weights.masked_fill(
                    ~causal_mask, float('-inf'))
            else:
                attn_weights = attn_weights + causal_mask

        if is_causal:
            causal_mask = torch.triu(
                torch.ones(Q, K, dtype=torch.bool, device=query_states.device), diagonal=1
            )
            attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        if self.attachIdentity:  # 对 softmax 之后的注意力 score 进行量化。
            b, head, d1, d2 = attn_weights.shape
            attn_weights = attn_weights.view(bsz, self.num_heads, -1).transpose(1, 2)
            attn_weights = self.identitySoftmax(attn_weights)
            attn_weights = attn_weights.transpose(1, 2).contiguous().view(b, head, d1, d2)

        dropout_p = self.attention_dropout if self.training else 0.0
        if dropout_p > 0.0:
            attn_weights = F.dropout(attn_weights, p=dropout_p)

        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value
# endregion
