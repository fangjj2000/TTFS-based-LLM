'''
Copyright (C) 2022 Guangyao Chen - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 * You should have received a copy of the MIT license with
 * this file. If not visit https://opensource.org/licenses/MIT
'''

import math
from typing import Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import Attention
from torch import Tensor
from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaRotaryEmbedding, LlamaMLP, LlamaAttention, \
    LlamaSdpaAttention

from quantize.int_linear_fake import QuantizedLinear, QuantizedLlamaAttention, QuantizedLlamaRMSNorm, \
    QuantizedSiLU, QuantizedLlamaMLP
from quantize.quant_norm import QuantRMSNorm
from spiking_conversion.model.spiking_layers import SpikingRMSNorm, SpikingLlamaAttention, SpikingBaseModule, \
    SpikingLinear, SpikingLlamaMLP, SpikingSiLU
from utils.model_utils import RMSN

try:
    from spikingjelly.clock_driven.neuron import MultiStepIFNode, MultiStepLIFNode, IFNode, LIFNode, \
        MultiStepParametricLIFNode, ParametricLIFNode
except:
    from spikingjelly.activation_based.neuron import MultiStepIFNode, MultiStepLIFNode, IFNode, LIFNode, \
        MultiStepParametricLIFNode, ParametricLIFNode

r"""
对不同的逐元素运算相对于ac的倍数，进行如下的假设。
add/mul/div = 1

exp/log/sqrt = 5

tanh/sigmoid = 5（因为内部也有 exp）

SiLU = 6（exp+add+reciprocal+mul+mul）

"""

DEFAULT_WEIGHTS: Dict[str, int] = {
    "mul": 1,
    "add": 1,
    "neg": 1,
    "exp": 5,
    "reciprocal": 2,
    "sigmoid": 5,  # fused 模式用得到
}

ANN: bool = False

precison=32

def spike_rateZIP(inp):
    # 改变判断前层输入是否为脉冲的方法，假定unique元素个数小于“某个整数”（一个shortcut会使得脉冲矩阵的最大值+1）
    # Nspks_max = 30  # for spikformer-4-384, real Nspks_max is 9 (2*4+1=9)；for spikformer-8-512, real Nspks_max is 17 (2*8+1=17)；
    # Nspks_max = 1  # similar to the original version by ChenGY 只有真正全0-1矩阵才作为event-driven，计算AC，否则均计算为MAC
    # SpikeZIP: 输出是-threshold,0,threshold三值的矩阵，因此这个值需要修改成 2
    Nspks_max = 32
    if torch.abs(inp).max() == 0:
        spkhistc = None
        spike = True
        spike_rate = 0
        return spike, spike_rate, spkhistc
    inp = inp / torch.abs(inp).max()  # SpikeZIP: 将feature除以threshold转化为-1,0,1的三值矩阵，threshold通过inp.max()求得
    num = inp.unique()
    # print(len(num))
    # if len(num) <= Nspks_max+1 and inp.max() <= Nspks_max and inp.min() >= 0:
    if len(num) <= Nspks_max + 1 and inp.max() <= Nspks_max - 1 and inp.min() >= -(
            Nspks_max - 1):  # SpikeZIP: 这里由于有负值，因为最大值应该是>=-1
        # 将[0,1,2...16]这种累积脉冲的矩阵，转化为只含有[0,1]的矩阵，把“整数*浮点数”作为一个AC
        # inp = torch.where(inp<1.0, 0.0, 1.0)

        # calculate module.__spkhistc__ （较为耗时，且对多个batch的数据进行统计较麻烦（类似于发放率，逐个batch累加，最后求平均），因此此处仅用一个batch的测试获取此信息）
        # https://numpy.org/doc/stable/reference/generated/numpy.histogram.html
        # spkhistc, _ = np.histogram(np.array(inp.cpu()), bins=np.arange(21))
        spkhistc = None

        spike = True
        spike_rate = (torch.abs(inp).sum() / inp.numel()).item()  # 此种计算方法已计入了时长T的影响，因为inp包含T这一维度。（注意：分母也包含了T这一维度）
        # neg_rate = ((inp<0).sum() / inp.numel()).item()
        # pos_rate = ((inp>0).sum() / inp.numel()).item()
        # print("spike_rate",spike_rate,"neg_rate",neg_rate,"pos_rate",pos_rate)
        # print(len(num), spike_rate, num)
        # print(len(num), spike_rate, num, inp.unique())
        # print(len(num), spike_rate, num, spkhistc)
    else:
        spkhistc = None

        spike = False
        spike_rate = 1
        # print(len(num), spike_rate, num.max())
        # print(len(num), spike_rate, num.max(), spkhistc)

    return spike, spike_rate, spkhistc


def spike_rate(inp: torch.Tensor):
    """
    判断输入是否为脉冲（spike）数据。
    - 若为 {-1, 0, 1} 离散三值，返回 True + 实际 firing rate
    - 否则返回 False + rate=1.0，表示密集 ANN 激活
    """
    # raise NotImplementedError(f"")
    HistogramSpikes = None

    if inp.abs().max() == 0:
        return True, 0.0, HistogramSpikes

    inp = inp.float()  # 保证安全的数值运算
    nonzero_count = torch.count_nonzero(inp)
    total_count = inp.numel()
    rateSpike = float(nonzero_count / total_count)
    if rateSpike >= 1.0:
        HistogramSpikes = None
        spike = False
    else:
        HistogramSpikes = None
        spike = True
    return spike, rateSpike, HistogramSpikes


def spike_rate_new(inp, unique_threshold=6, value_range=10, int_tolerance=1e-3, sparsity_threshold=0.9):
    """
    更强大的 spike 判断：
    - 允许任意低稀疏、低值域的输入视为脉冲
    - 考虑唯一值个数、是否接近整数、是否稀疏、最大最小值范围等
    """

    spkhistc = None

    if torch.abs(inp).max() == 0:
        return True, 0.0, spkhistc

    unique_vals = inp.unique()
    num_unique = len(unique_vals)

    min_val = unique_vals.min().item()
    max_val = unique_vals.max().item()

    # 判断是否近似为整数值
    all_close_to_int = torch.allclose(unique_vals, unique_vals.round(), atol=int_tolerance)

    # 稀疏度判断（输入为 0 的比例）
    zero_ratio = (inp == 0).sum().item() / inp.numel()

    # 满足以下任意组合，认为是脉冲编码
    if (num_unique <= unique_threshold and
            abs(min_val) <= value_range and abs(max_val) <= value_range and
            all_close_to_int and
            zero_ratio >= sparsity_threshold):
        spike = True
        spike_rate = inp.abs().sum().item() / inp.numel()
        return spike, spike_rate, spkhistc

    # 否则按普通激活处理
    return False, 1.0, spkhistc


# region ==== Vanilla Hooks ====
def empty_syops_counter_hook(module, input, output):
    module.__syops__ += np.array([0.0, 0.0, 0.0, 0.0])


def upsample_syops_counter_hook(module, input, output):
    output_size = output[0]
    batch_size = output_size.shape[0]
    output_elements_count = batch_size
    for val in output_size.shape[1:]:
        output_elements_count *= val
    module.__syops__[0] += int(output_elements_count)

    # spike, rate = spike_rate(output[0])
    spike, rate, _ = spike_rate(output)
    assert rate <= 1, f"rate {rate}"

    if spike:
        module.__syops__[1] += int(output_elements_count) * rate
    else:
        module.__syops__[2] += int(output_elements_count)

    module.__syops__[3] += rate * 100


def relu_syops_counter_hook(module, input, output):
    active_elements_count = output.numel()
    module.__syops__[0] += int(active_elements_count)

    # spike, rate = spike_rate(output[0])
    spike, rate, _ = spike_rate(output)
    assert rate <= 1, f"rate {rate}"
    if spike:
        module.__syops__[1] += int(active_elements_count) * rate
    else:
        module.__syops__[2] += int(active_elements_count)

    module.__syops__[3] += rate * 100


def IF_syops_counter_hook(module, input, output):
    active_elements_count = input[0].numel()
    module.__syops__[0] += int(active_elements_count)

    # spike, rate = spike_rate(output[0])
    spike, rate, spkhistc = spike_rate(output)
    assert rate <= 1, f"rate {rate}"
    module.__syops__[1] += int(active_elements_count)
    module.__syops__[3] += rate * 100
    module.__spkhistc__ = spkhistc


def LIF_syops_counter_hook(module, input, output):  # output is <class 'torch.Tensor'>
    active_elements_count = input[
        0].numel()  # input is tuple, input[0].shape = torch.Size([4, 1, 48, 32, 32]) [T, B, C, H, W]
    module.__syops__[0] += int(
        active_elements_count)  # 输入的元素个数作为操作数，不管是否有脉冲？似乎有问题？？ input[0].numel() = np.prod(input[0].shape)

    # spike, rate = spike_rate(output[0])  # output.shape = torch.Size([4, 1, 48, 32, 32]) [T, B, C, H, W] # 神经元层计算的是本身的发放率。但只用了第一个时间步的脉冲输入来计算发放率，代表性不足？？ 考虑使用spike, rate = spike_rate(output)。猜测作者是误认为output与input一样是tuple
    spike, rate, spkhistc = spike_rate(output)
    assert rate <= 1, f"rate {rate}"
    module.__syops__[1] += int(active_elements_count)
    # module.__syops__[2] += int(active_elements_count)
    module.__syops__[3] += rate * 100
    module.__spkhistc__ = spkhistc


def cal_linear_sparsity(input, weight_mask):
    input_mask = (input != 0).float()
    output_mask = F.linear(input_mask, weight_mask, bias=None)
    input_full = torch.ones(size=input.shape).to(input.device)
    weight_full = torch.ones(size=weight_mask.shape).to(weight_mask.device)
    output_full = F.linear(input_full, weight_full, bias=None)

    return torch.sum(output_mask) / torch.sum(output_full)


def linear_syops_counter_hook(module, input, output):
    input = input[0]

    # 计算了前一层的发放率  # input.unique --> [0,1,2]  spike=False 不把该层作为spike-triggered
    spike, rate, spkhistc = spike_rate(input)
    # print(f"spike? {spike}, rate = {rate}")
    # if spike == True:
    #     if hasattr(module,"weight_mask"):
    #         real_rate = cal_linear_sparsity(input,module.weight_mask)
    #     else:
    #         real_rate = cal_linear_sparsity(input,(module.weight != 0).float())
    # else:
    #     real_rate = 1.0
    # pytorch checks dimensions, so here we don't care much
    assert rate <= 1, f"rate {rate}"
    batch_size = input.shape[0]
    output_last_dim = output.shape[-1]
    # bias_syops = output_last_dim if module.bias is not None else 0
    bias_syops = output_last_dim * batch_size if module.bias is not None else 0  # need to take batch_size into account, as in conv_syops_counter_hook
    module.__syops__[0] += int(np.prod(input.shape) * output_last_dim + bias_syops)
    if spike:
        module.__syops__[1] += int(np.prod(input.shape) * output_last_dim + bias_syops) * rate
    else:
        module.__syops__[2] += int(np.prod(input.shape) * output_last_dim + bias_syops)

    module.__syops__[3] += rate * 100
    module.__spkhistc__ = spkhistc


def pool_syops_counter_hook(module, input, output):
    input = input[
        0]  # input is tuple, input[0].shape = torch.Size([4, 192, 32, 32]) [TB, C, H, W]  # output.shape = torch.Size([4, 192, 16, 16])
    spike, rate, spkhistc = spike_rate(input)
    assert rate <= 1, f"rate {rate}"
    module.__syops__[0] += int(np.prod(input.shape))  # 直接加上元素个数，对吗？？

    if spike:
        module.__syops__[1] += int(np.prod(input.shape)) * rate
    else:
        module.__syops__[2] += int(np.prod(input.shape))

    module.__syops__[3] += rate * 100
    module.__spkhistc__ = spkhistc


def bn_syops_counter_hook(module, input, output):
    input = input[0]  # input is tuple, input[0].shape = torch.Size([4, 48, 32, 32]) [TB, C, H, W]
    spike, rate, spkhistc = spike_rate(input)
    assert rate <= 1, f"rate {rate}"
    batch_syops = np.prod(input.shape)
    if module.affine:
        batch_syops *= 2
    module.__syops__[0] += int(batch_syops)

    if spike:
        module.__syops__[1] += int(batch_syops) * rate
    else:
        module.__syops__[2] += int(batch_syops)

    module.__syops__[3] += rate * 100
    module.__spkhistc__ = spkhistc


def ln_syops_counter_hook(module, input, output):
    input = input[0]  # input is tuple, input[0].shape = torch.Size([4, 48, 32, 32]) [TB, C, H, W]
    spike, rate, spkhistc = spike_rate(input)
    assert rate <= 1, f"rate {rate}"
    # batch_syops = np.prod(input.shape)
    batch_syops = np.prod(input.shape)
    if module.elementwise_affine:
        batch_syops *= 2
    module.__syops__[0] += int(batch_syops)

    if spike:
        module.__syops__[1] += int(batch_syops) * rate
    else:
        module.__syops__[2] += int(batch_syops)

    module.__syops__[3] += rate * 100
    module.__spkhistc__ = spkhistc


def cal_conv_sparsity(input, weight_mask, stride, padding, groups):
    input_mask = (input != 0).float()
    output_mask = F.conv2d(input_mask, weight_mask, bias=None, stride=stride, padding=padding, groups=groups)
    input_full = torch.ones(size=input.shape).to(input.device)
    weight_full = torch.ones(size=weight_mask.shape).to(weight_mask.device)
    output_full = F.conv2d(input_full, weight_full, bias=None, stride=stride, padding=padding, groups=groups)
    return torch.sum(output_mask) / torch.sum(output_full)


def conv_syops_counter_hook(conv_module, input, output):
    # Can have multiple inputs, getting the first one
    input = input[0]  # input is tuple, input[0].shape = torch.Size([4, 3, 32, 32]) [TB, C, H, W]
    spike, rate, spkhistc = spike_rate(input)
    assert rate <= 1, f"rate {rate}"
    batch_size = input.shape[0]
    output_dims = list(output.shape[2:])  # output.shape = torch.Size([4, 48, 32, 32])

    kernel_dims = list(conv_module.kernel_size)
    in_channels = conv_module.in_channels
    out_channels = conv_module.out_channels
    groups = conv_module.groups
    padding = conv_module.padding[0]
    stride = conv_module.stride[0]
    # if spike==True:
    #     if hasattr(conv_module,"weight_mask"):
    #         real_rate = cal_conv_sparsity(input, conv_module.weight_mask, stride,padding,groups)
    #     else:
    #         real_rate = cal_conv_sparsity(input, (conv_module.weight != 0).float(), stride,padding,groups)
    # else:
    #     real_rate = 1.0

    filters_per_channel = out_channels // groups
    conv_per_position_syops = int(np.prod(kernel_dims)) * \
                              in_channels * filters_per_channel

    active_elements_count = batch_size * int(np.prod(output_dims))

    overall_conv_syops = conv_per_position_syops * active_elements_count

    bias_syops = 0

    if conv_module.bias is not None:
        bias_syops = out_channels * active_elements_count

    # overall_syops = overall_conv_syops + bias_syops
    overall_syops = overall_conv_syops + bias_syops

    conv_module.__syops__[0] += int(overall_syops)

    if spike:
        conv_module.__syops__[1] += int(overall_syops) * rate
    else:
        conv_module.__syops__[2] += int(overall_syops)

    conv_module.__syops__[3] += rate * 100
    conv_module.__spkhistc__ = spkhistc


def rnn_syops(syops, rnn_module, w_ih, w_hh, input_size):
    # matrix matrix mult ih state and internal state
    syops += w_ih.shape[0] * w_ih.shape[1]
    # matrix matrix mult hh state and internal state
    syops += w_hh.shape[0] * w_hh.shape[1]
    if isinstance(rnn_module, (nn.RNN, nn.RNNCell)):
        # add both operations
        syops += rnn_module.hidden_size
    elif isinstance(rnn_module, (nn.GRU, nn.GRUCell)):
        # hadamard of r
        syops += rnn_module.hidden_size
        # adding operations from both states
        syops += rnn_module.hidden_size * 3
        # last two hadamard product and add
        syops += rnn_module.hidden_size * 3
    elif isinstance(rnn_module, (nn.LSTM, nn.LSTMCell)):
        # adding operations from both states
        syops += rnn_module.hidden_size * 4
        # two hadamard product and add for C state
        syops += rnn_module.hidden_size + rnn_module.hidden_size + rnn_module.hidden_size
        # final hadamard
        syops += rnn_module.hidden_size + rnn_module.hidden_size + rnn_module.hidden_size
    return syops


def rnn_syops_counter_hook(rnn_module, input, output):
    """
    Takes into account batch goes at first position, contrary
    to pytorch common rule (but actually it doesn't matter).
    If sigmoid and tanh are hard, only a comparison syops should be accurate
    """
    syops = 0
    # input is a tuple containing a sequence to process and (optionally) hidden state
    inp = input[0]
    batch_size = inp.shape[0]
    seq_length = inp.shape[1]
    num_layers = rnn_module.num_layers

    for i in range(num_layers):
        w_ih = rnn_module.__getattr__('weight_ih_l' + str(i))
        w_hh = rnn_module.__getattr__('weight_hh_l' + str(i))
        if i == 0:
            input_size = rnn_module.input_size
        else:
            input_size = rnn_module.hidden_size
        syops = rnn_syops(syops, rnn_module, w_ih, w_hh, input_size)
        if rnn_module.bias:
            b_ih = rnn_module.__getattr__('bias_ih_l' + str(i))
            b_hh = rnn_module.__getattr__('bias_hh_l' + str(i))
            syops += b_ih.shape[0] + b_hh.shape[0]

    syops *= batch_size
    syops *= seq_length
    if rnn_module.bidirectional:
        syops *= 2
    rnn_module.__syops__[0] += int(syops)


def rnn_cell_syops_counter_hook(rnn_cell_module, input, output):
    syops = 0
    inp = input[0]
    batch_size = inp.shape[0]
    w_ih = rnn_cell_module.__getattr__('weight_ih')
    w_hh = rnn_cell_module.__getattr__('weight_hh')
    input_size = inp.shape[1]
    syops = rnn_syops(syops, rnn_cell_module, w_ih, w_hh, input_size)
    if rnn_cell_module.bias:
        b_ih = rnn_cell_module.__getattr__('bias_ih')
        b_hh = rnn_cell_module.__getattr__('bias_hh')
        syops += b_ih.shape[0] + b_hh.shape[0]

    syops *= batch_size
    rnn_cell_module.__syops__[0] += int(syops)


def multihead_attention_counter_hook(multihead_attention_module, input, output):
    syops = 0

    # print(input[0].shape)
    q = k = v = input[0]

    batch_first = True

    if batch_first:
        batch_size = q.shape[0]
        len_idx = 1
    else:
        batch_size = q.shape[1]
        len_idx = 0

    dim_idx = 2

    qdim = q.shape[dim_idx]
    kdim = k.shape[dim_idx]
    vdim = v.shape[dim_idx]

    qlen = q.shape[len_idx]
    klen = k.shape[len_idx]
    vlen = v.shape[len_idx]

    num_heads = multihead_attention_module.num_heads
    assert qdim == multihead_attention_module.num_heads * multihead_attention_module.head_dim

    if hasattr(multihead_attention_module, "kdim") is None:
        assert kdim == qdim
    if hasattr(multihead_attention_module, "vdim") is None:
        assert vdim == qdim

    syops = 0

    # Q scaling
    syops += qlen * qdim

    # Initial projections
    syops += (
            (qlen * qdim * qdim)  # QW
            + (klen * kdim * kdim)  # KW
            + (vlen * vdim * vdim)  # VW
    )

    if hasattr(multihead_attention_module, "in_proj_bias") is not None:
        syops += (qlen + klen + vlen) * qdim

    # attention heads: scale, matmul, softmax, matmul
    qk_head_dim = qdim // num_heads
    v_head_dim = vdim // num_heads

    head_syops = (
            (qlen * klen * qk_head_dim)  # QK^T
            + (qlen * klen)  # softmax
            + (qlen * klen * v_head_dim)  # AV
    )

    syops += num_heads * head_syops

    # final projection, bias is always enabled
    syops += qlen * vdim * (vdim + 1)

    syops *= batch_size
    multihead_attention_module.__syops__[0] += int(syops)
    multihead_attention_module.__syops__[2] += int(syops)


def lora_syops_hook(lora_module, input, output):
    if not hasattr(lora_module, "__syops__"):
        lora_module.__syops__ = [0, 0, 0, 0]

    x = input[0]  # input: [B, in_features]
    B = x.shape[0]
    in_features = output.shape[-1]
    out_features = lora_module.out_features
    rank = lora_module.rank
    bias = lora_module.bias is not None

    spike, rate, spkhistc = spike_rate(x)
    assert rate <= 1, f"rate {rate}"
    # 1. 原始层
    ops_base = B * in_features * out_features
    if bias:
        ops_base += B * out_features

    # 2. LoRA 路径
    ops_lora_A = B * in_features * rank
    ops_lora_B = B * rank * out_features

    # 3. 缩放与加法
    ops_scale = B * out_features
    ops_add = B * out_features

    total_ops = ops_base + ops_lora_A + ops_lora_B + ops_scale + ops_add

    lora_module.__syops__[0] += total_ops
    if spike:
        lora_module.__syops__[1] += int(total_ops * rate)
    else:
        lora_module.__syops__[2] += total_ops
    lora_module.__syops__[3] += rate * 100
    lora_module.__spkhistc__ = spkhistc


# endregion

# region ==== Newly Added Hooks ====

# region === Hooks in Llama ===
def nn_embedding_syops_counter_hook(module: nn.Embedding, input, output):
    r"""
    ✅
    :param module:
    :param input:
    :param output:
    :return:
    """
    input = input[0]
    num_lookups = input.numel()
    acs = 0.
    if module.max_norm:
        acs += num_lookups * (module.embedding_dim - 1)
    module.__syops__[0] += acs
    module.__syops__[1] += acs
    module.__syops__[2] += 0
    module.__syops__[3] += 100

    # elif module.model_type=="spiking_ttfs":
    #     input = input[0]
    #     num_lookups = input.numel()
    #     acs = 0.
    #     if module.oriebd.max_norm:
    #         raise 'not implement'
    #         acs += num_lookups * (module.embedding_dim - 1)
    #     # module.__syops__ = [acs, acs, 0, 100] zhy修改
    #     module.__syops__[0] += acs
    #     module.__syops__[1] += acs
    #     module.__syops__[2] += 0
    #     module.__syops__[3] += 100
    # else:
    #     raise ValueError(
    #     f"Unsupported model_type: {module.model_type}. "
    #     f"Expected one of ['original', 'spiking_ttfs']")



# def rmsnorm_syops_counter_hook(module: RMSN | LlamaRMSNorm | QuantRMSNorm, input, output):
#     r"""
#     ✅
#     统计 LlamaRMSNorm 的操作数（MAC=0，全为AC），并累加到：
#         module.___syops__ = [total, ac, mac, 100]
#     计数包含：square、沿最后一维求和与除法、加 eps、rsqrt、两次逐元素乘（缩放与乘weight）。
#     不计：dtype转换、广播、view/reshape/keepdim。
#     :param module:
#     :param input:
#     :param output:
#     :return:
#     """
#     x = input[0]
#     numel = x.numel()
#     H = int(x.shape[-1])
#     rows = int(numel // H)  # B*T

#     # AC
#     ac = int(3 * rows * H  # square + scale mul + weight mul
#              + rows * (H - 1)  # sum over H
#              + rows  # divide by H
#              + rows  # add eps
#              + rows)  # rsqrt
#     mac = 0
#     total = ac

#     if not hasattr(module, "___syops__") or module.___syops__ is None:
#         module.___syops__ = [0, 0, 0, 100]

#     module.__syops__[0] += total
#     module.__syops__[1] += ac
#     module.__syops__[2] += mac
#     module.__syops__[3] += 100
def rmsnorm_syops_counter_hook(module: RMSN | LlamaRMSNorm | QuantizedLlamaRMSNorm, input, output):
    r"""
    ✅
    :param module:
    :param input:
    :param output:
    :return:
    """
    x = input[0]
    #ops_per_token = 2 * x.shape[-1]  # 平方 + 乘以 weight（逐元素）
    ops_per_token = 3 * x.shape[-1]  # zhz修改 因为还有乘以方差的倒数
    mac_ops = (x.numel() // x.shape[-1]) * ops_per_token
    ##module.__syops__ = [ac_ops, ac_ops, 0, 100]
    #module.__syops__ = [ac_ops, 0, ac_ops, 100] # zhz修改 因为认为是MAC操作
    ac_ops = (x.numel() // x.shape[-1]) * (x.shape[-1] - 1) + (x.numel() // x.shape[-1]) #zhz修改 添加计算方差的求和以及+eps的求和
    module.__syops__[0] += (mac_ops+ac_ops)
    module.__syops__[1] += ac_ops
    module.__syops__[2] += mac_ops
    module.__syops__[3] += 100


def silu_syops_counter_hook(module: nn.SiLU, input, output):
    r"""
    🤔
    nn.SiLU 的 操作数统计 hook。
    * 理论上:          SiLU = x / (1 + exp(-x)) -> neg + exp + add1 + reciprocal + mul
    * 实际 torch 中:   SiLU = x * sigmoid(x)    -> sigmoid + mul


    计数累加到 module.___syops__ = [total, ac, mac, 100]
    其中 ac 为“加权 AC”，mac 恒为 0。
    :param module:
    :param input:
    :param output:
    :return:
    """
    if module.model_type=="original":
        num_elements = output.numel()
        per_element_mac = 9  # 近似值：exp + div + mul
        per_element_ac = 7  # 近似值：neg + add + div中AC
        macs = num_elements * per_element_mac  # x * sigmoid(x)
        acs = num_elements * per_element_ac  # 1 + exp(-x)
        # module.__syops__ = [acs + macs, acs, macs, 100]
        module.__syops__[0] += (acs + macs)
        module.__syops__[1] += acs
        module.__syops__[2] += macs
        module.__syops__[3] += 100
    elif module.model_type=="spiking_ttfs":
        x = input[0]
        batch, seq_len, hidden = x.shape
        ac = precison * batch * seq_len * hidden
        mac = 0
        total = ac
        module.__syops__[0] += total
        module.__syops__[1] += ac
        module.__syops__[2] += mac
        module.__syops__[3] += 100
    else:
        raise ValueError(
        f"Unsupported model_type: {module.model_type}. "
        f"Expected one of ['original', 'spiking_ttfs']")



def nn_linear_syops_counter_hook(module: nn.Linear | QuantizedLinear, input, output):
    r"""
    ✅ Hook for nn.Linear to estimate MACs and ACs.

    Assumes input of shape (B, L, D), where:
        - D = in_features
        - output shape: (B, L, out_features)

    Computation:
        - MACs = B × L × in_features × out_features
        - ACs  = B × L × out_features (bias add per output element)

    Note:
        - Assumes bias is used. If `module.bias is None`, ACs will be 0.
    :param module:
    :param input:
    :param output:
    :return:
    """
    B, L, D = input[0].shape
    in_features = module.in_features
    out_features = module.out_features

    syopsMatMul = B * L * in_features * out_features
    syopsBias = B * L * out_features if module.bias is not None else 0
    module.__syops__[0] += (syopsMatMul + syopsBias)
    module.__syops__[1] += syopsBias
    module.__syops__[2] += syopsMatMul
    module.__syops__[3] += 100


def rotary_embedding_syops_counter_hook(module: LlamaRotaryEmbedding, input, output):
    r"""
    🤔
    :param module:
    :param input:
    :param output:
    :return:
    """
    # 目标 shape：[batch, head, seq_len, dim]
    # 统计返回 cos, sin 所产生的 ops 总数
    x = input[0]  # shape: (B, num_heads, S, D)

    batch_size = x.shape[0]
    num_heads = x.shape[1]
    seq_len = x.shape[2]
    head_dim = x.shape[3]

    num_pos = batch_size * seq_len
    num_emb_elements = num_pos * head_dim

    # 不考虑 freqs 点乘, 可缓存
    # mac_freq = num_pos * (head_dim // 2)

    # sin/cos 多项式近似
    mac_trig = num_emb_elements * 2 * 2 * num_heads  # Q和K下: 每个head_dim下，y_2i = x_2i * cos - x_2i+1 * sin 两次乘积
    ac_trig = num_emb_elements * 1 * 2 * num_heads  # Q和K下: 每个head_dim下，y_2i = x_2i * cos - x_2i+1 * sin 一次加法

    total_mac = mac_trig
    total_ac = ac_trig
    module.__syops__[0] += (total_mac + total_ac)
    module.__syops__[1] += total_ac
    module.__syops__[2] += total_mac
    module.__syops__[3] += 100


def llama_mlp_syops_counter_hook(module: LlamaMLP, input, output):
    r"""
    ✅
    统计 LlamaMLP 中，除三次 Linear 和 act_fn 之外的操作数：
      - Hadamard 积 (act_fn(gate_proj(x)) * up_proj(x)) -> AC
      - 当 pretraining_tp > 1 时，sum(down_proj_list) 的逐元素加法 -> AC
    结果累加到 module.___syops__ = [total, ac, mac, 100]
    :param module:
    :param input:
    :param output:
    :return:
    """
    if module.model_type=="original":
        x = input[0]
        batch, seq_len, hidden = x.shape
        # mac_ops = batch * seq_len * hidden  # 逐元素乘法
        mac_ops = batch * seq_len * module.intermediate_size #zhz修改 因为哈达吗积维度不对
        #module.__syops__ = [mac_ops, 0, mac_ops, 100]
        module.__syops__[0] += mac_ops
        module.__syops__[1] += 0
        module.__syops__[2] += mac_ops
        module.__syops__[3] += 100
    
    elif module.model_type=="spiking_ttfs":
        x = input[0]
        batch, seq_len, hidden = x.shape
        ac = precison * batch * seq_len * module.intermediate_size
        module.__syops__[0] += ac
        module.__syops__[1] += ac
        module.__syops__[2] += 0
        module.__syops__[3] += 100
    else:
        raise ValueError(
        f"Unsupported model_type: {module.model_type}. "
        f"Expected one of ['original', 'spiking_ttfs']")


def llama_attention_syops_counter_hook_abandoned(module: LlamaAttention, input, output):
    r"""
    ✅
    Manually estimate the MACs in scaled dot-product attention in LlamaSdpaAttention.
    :param module:
    :param input:
    :param output:
    :return:
    """
    B, L, _ = input[0].shape
    num_heads = module.num_heads
    head_dim = module.head_dim

    # Q x K^L: [B, h, L, d] x [B, h, d, L] => [B, h, L, L]
    attn_scores_mac = B * num_heads * L * head_dim * L + B * num_heads * L * L  # zhz修改 添加scale的成本(+号后面)

    # Attn x V: [B, h, L, L] x [B, h, L, d] => [B, h, L, d]
    attn_value_mac = B * num_heads * L * L * head_dim

    total_mac = attn_scores_mac + attn_value_mac

    module.__syops__[0] += total_mac
    module.__syops__[1] += 0
    module.__syops__[2] += total_mac
    module.__syops__[3] += 100


def llama_attention_syops_counter_hook(module: LlamaAttention,
                                       args, kwargs, output):
    r"""
    统计 LlamaAttention 前传中的操作数（除 q/k/v/o 线性投影外）：
      - 计入：RoPE、缩放、mask 相加、softmax、dropout、QK^T 与 A·V 的 matmul
      - 不计：q/k/v/o Linear、view/transpose/contiguous/reshape、repeat_kv 的拷贝、
             rotary_emb 里 cos/sin 的生成、cache 的读写（可选开启，见 NOTE）

    统计结果会累加到:
        module.___syops__ = [total, ac, mac, 100]
    :param module:
    :param input:
    :param output:
    :return:
    """
    # 先拿 hidden_states：优先 args[0]；拿不到用 kwargs['hidden_states']；再不行用 output[0] 兜底形状
    if module.model_type=="original":
        if isinstance(args, (tuple, list)) and len(args) > 0:
            hidden_states = args[0]
            bsz, q_len, _ = hidden_states.shape
        elif isinstance(kwargs, dict) and "hidden_states" in kwargs:
            hidden_states = kwargs["hidden_states"]
            bsz, q_len, _ = hidden_states.shape
        else:
            attn_out = output[0] if isinstance(output, (tuple, list)) else output
            if not (hasattr(attn_out, "shape") and attn_out.ndim >= 3):
                return
            bsz, q_len, _ = attn_out.shape[:3]

        # 再拿 attention_mask：优先 kwargs；其次 args[1]；否则 None
        B=bsz
        L=q_len
        num_heads = module.num_heads
        head_dim = module.head_dim

        # Q x K^L: [B, h, L, d] x [B, h, d, L] => [B, h, L, L]
        attn_scores_mac = B * num_heads * L * head_dim * L  +  B * num_heads * L * L  #zhz修改 添加scale的成本(+号后面)

        # Attn x V: [B, h, L, L] x [B, h, L, d] => [B, h, L, d]
        attn_value_mac = B * num_heads * L * L * head_dim

        total_mac = attn_scores_mac + attn_value_mac

        # module.__syops__ = [total_mac, 0, total_mac, 100]
        module.__syops__[0] += total_mac
        module.__syops__[1] += 0
        module.__syops__[2] += total_mac
        module.__syops__[3] += 100
    elif module.model_type=="spiking_ttfs":
        if isinstance(output, tuple):
            hidden_states = output[0]  # 第一个元素才是真正的 hidden_states Tensor
            bsz, q_len, _ = hidden_states.shape
        else:
            hidden_states = output
            bsz, q_len, _ = hidden_states.shape

        H = module.num_heads
        D = module.head_dim

        # 注意：KV 经过 repeat_kv 后与 H 对齐，故 matmul 按 H 统计
        # 估计 k_len：优先从 attention_mask 推断，否则假定等于 q_len
        attention_mask = None
        if attention_mask is not None and attention_mask is not False:
            # mask 形状通常为 (bsz, 1, q_len, k_len) 或 (bsz, H, q_len, k_len)
            k_len = attention_mask.shape[-1]
        else:
            # 无 cache 的常见推理/训练步
            k_len = q_len

        training = bool(getattr(module, "training", False))
        p_drop = float(getattr(module, "attention_dropout", 0.0) or 0.0)

        # ============== 1. RoPE（apply_rotary_pos_emb）- AC ==============
        # 对 q 与 k 各执行：每对维度(2D pair) -> 4 次乘法 + 2 次加法
        # => 对 D 维：mult = 2D，add = D；q 和 k 各一遍 => mult = 4D，add = 2D
        # rope_rows = bsz * H * q_len
        # rope_ac = rope_rows * (4 * D + 2 * D)

        # ============== 2. 注意力分数 S = Q @ K^T  AC ==============
        # MAC： (B, H, Q, D) @ (B, H, D, K) -> (B, H, Q, K) : B*H*Q*K*D
        ac_qkt = bsz * H * q_len * k_len * (3 * D + 2)  # 参考的 TTFSFormer 附录表格 3.

        elems_scores = bsz * H * q_len * k_len
        # ==== 2-1 1/sqrt(D) 缩放：逐元素乘 ====
        scale_ac = elems_scores
        # ==== 2-2 causal/padding mask（若有）====
        mask_ac = elems_scores if (attention_mask is not None and attention_mask is not False) else 0

        # ============== 3. softmax(S)（沿 K 维） ==============
        # 对每一行（长度 K）：max (K-1) 比较 + 减 max K + exp K + 求和 (K-1) + 除法 K
        # 合计每行： (K-1) + K + K + (K-1) + K = 5K - 2

        softmax_ac = elems_scores * 6  # 参考的 TTFSFormer 附录表格 3.

        # ===== 3-1 AC：dropout（训练且 p>0）：逐元素乘 =====
        dropout_ac = elems_scores if (training and p_drop > 0.0) else 0

        # ============== 4. 输出聚合 O = A @ V  MAC==============
        ac_av = bsz * H * q_len * (3 * k_len + 2) * D  # 参考的 TTFSFormer 附录表格 3.

        # 汇总
        mac = 0
        ac = int(scale_ac + mask_ac + precison * softmax_ac + dropout_ac+ precison * ac_av + precison * ac_qkt)
        total = mac + ac

        # 累加到 module.___syops__ = [total, ac, mac, 100]
        if not hasattr(module, "___syops__") or module.___syops__ is None:
            module.___syops__ = [0, 0, 0, 100]

        module.__syops__[0] += total
        module.__syops__[1] += ac
        module.__syops__[2] += mac
        module.__syops__[3] += 100
    else:
        raise ValueError(
        f"Unsupported model_type: {module.model_type}. "
        f"Expected one of ['original', 'spiking_ttfs']")


# endregion

# region === Hooks in Q-Llama
def quant_linear_syops_counter_hook(module: nn.Linear | QuantizedLinear, input, output):
    r"""
    ✅
    :param module:
    :param input:
    :param output:
    :return:
    """
    x = input[0]  # shape: (B, L, in_features)
    out = output  # shape: (B, L, out_features)

    # 展平除最后一维之外的所有批次
    batch_size = x.shape[0]
    in_features = x.shape[-1]
    out_features = out.shape[-1]
    num_tokens = x.numel()

    macs = num_tokens * out_features

    acs = 0
    # online_full_had 为 True 的时候，有部分 Hadamard 积计算。
    if hasattr(module, "online_full_had") and module.online_full_had:
        log2_in_features = int(math.log2(in_features))
        acs = batch_size * in_features * log2_in_features

    bias_syops = out_features * batch_size if module.bias is not None else 0
    acs += bias_syops

    module.__syops__ = [macs + acs, acs, macs, 100]


# endregion

# region === Hooks in S-Llama ===
def snn_embedding_syops_counter_hook(module, input, output):
    r"""
    ✅
    total_mac ≈ number of scalar divisions (one per output element)
    These divisions are approximated as 1 MAC each

    :param module:
    :param input:
    :param output:
    :return:
    """
    input = input[0]
    num_lookups = input.numel()
    acs = 0.
    if module.oriebd.max_norm:
        raise 'not implement'
        acs += num_lookups * (module.embedding_dim - 1)
    # module.__syops__ = [acs, acs, 0, 100] zhy修改
    module.__syops__[0] += acs
    module.__syops__[1] += acs
    module.__syops__[2] += 0
    module.__syops__[3] += 100


def snn_rmsnorm_syops_counter_hook(module: SpikingRMSNorm, input, output):
    r"""
    ✅
    Hook to estimate MAC and AC operations in snnRMSNorm forward pass.

    Operation is performed over input(shape (T*B, L, D) reshaped as (T, B, D, L), where:
        - T: time steps
        - B: batch size
        - D: feature dimension
        - L: sequence length

    For each time step (t in T), the following operations are performed:

        MAC (Multiply-Accumulate):
            1. X.pow(2) → 1 MAC per element       → N
            2. mean(-1): division step            → 1 MAC per element       → N
            3. rsqrt (x⁻¹ᐟ²) ≈ 2 MAC per element  → 2 MAC per (B × D)       → 2BD
            4. final * weight                     → 1 MAC per element       → N
            → total per step: 3N + 2BD

        AC (Add / Accumulate):
            1. X + x[t]                           → 1 AC per element        → N
            2. mean(-1): accumulation step        → 1 AC per element        → N
            3. rsqrt input + eps                  → 1 AC per (B × D)        → BD
            4. Y - Y_pre                          → 1 AC per element        → N
            → total per step: 3N + BD

    Total MACs and ACs are summed across T time steps.
    :param module:
    :param input:
    :param output:
    :return:
    """
    macs = 0
    acs = 0
    T = 2**(module.wbits)
    x = input[0]  # shape: [BT, D, L]

    BT, D, L = x.shape
    B = BT //T
    x = x.view(T, B, D, L)
    if module.use_temporary_parameter:
        weight = module.temp_weight
    else:
        weight = module.weight

    input_dtype = x.dtype
    if x.dtype == torch.float16:
        x = x.to(torch.float32)

    X = torch.zeros_like(x[0])
    Y_pre = 0
    Out = []
    for t in range(module.T):
        X = X + x[t]
        acs += torch.count_nonzero(x[t]).item()

        variance = X.pow(2).mean(-1, keepdim=True)
        macs += torch.count_nonzero(X).item()
        acs += torch.count_nonzero(X).item() - 1

        Y = X * torch.rsqrt(variance + module.variance_epsilon)
        macs += torch.count_nonzero(X).item()

        Y = Y.to(input_dtype) * weight
        macs += torch.count_nonzero(Y).item()

        Out.append(Y - Y_pre)
        if torch.is_tensor(Y_pre):
            acs += torch.count_nonzero(Y * Y_pre).item()
        # 应比较Y_pre和Y较大的非零元素个数
        Y_pre = Y

    # module.__syops__ = [macs + acs, acs, macs, 100] zhy修改认为需要累加
    module.__syops__[0] += (macs + acs)
    module.__syops__[1] += acs
    module.__syops__[2] += macs
    module.__syops__[3] += 100


def snn_llama_attention_syops_counter_hook(module: LlamaAttention,
                                           args, kwargs, output):
    r"""
    统计 LlamaAttention 前传中的操作数（除 q/k/v/o 线性投影外）：
      - 计入：RoPE、缩放、mask 相加、softmax、dropout、QK^T 与 A·V 的 matmul
      - 不计：q/k/v/o Linear、view/transpose/contiguous/reshape、repeat_kv 的拷贝、
             rotary_emb 里 cos/sin 的生成、cache 的读写（可选开启，见 NOTE）

    统计结果会累加到:
        module.___syops__ = [total, ac, mac, 100]
    :param module:
    :param input:
    :param output:
    :return:
    """
    # 先拿 hidden_states：优先 args[0]；拿不到用 kwargs['hidden_states']；再不行用 output[0] 兜底形状
    if isinstance(args, (tuple, list)) and len(args) > 0:
        hidden_states = args[0]
        bsz, q_len, _ = hidden_states.shape
    elif isinstance(kwargs, dict) and "hidden_states" in kwargs:
        hidden_states = kwargs["hidden_states"]
        bsz, q_len, _ = hidden_states.shape
    else:
        attn_out = output[0] if isinstance(output, (tuple, list)) else output
        if not (hasattr(attn_out, "shape") and attn_out.ndim >= 3):
            return
        bsz, q_len, _ = attn_out.shape[:3]

    # 再拿 attention_mask：优先 kwargs；其次 args[1]；否则 None
    attention_mask = None
    if isinstance(kwargs, dict) and "attention_mask" in kwargs:
        attention_mask = kwargs["attention_mask"]
    elif isinstance(args, (tuple, list)) and len(args) > 1:
        attention_mask = args[1]

    H = module.num_heads
    D = module.head_dim

    # 注意：KV 经过 repeat_kv 后与 H 对齐，故 matmul 按 H 统计
    # 估计 k_len：优先从 attention_mask 推断，否则假定等于 q_len

    if attention_mask is not None and attention_mask is not False:
        # mask 形状通常为 (bsz, 1, q_len, k_len) 或 (bsz, H, q_len, k_len)
        k_len = attention_mask.shape[-1]
    else:
        # 无 cache 的常见推理/训练步
        k_len = q_len

    training = bool(getattr(module, "training", False))
    p_drop = float(getattr(module, "attention_dropout", 0.0) or 0.0)

    # ============== 1. RoPE（apply_rotary_pos_emb）- AC ==============
    # 对 q 与 k 各执行：每对维度(2D pair) -> 4 次乘法 + 2 次加法
    # => 对 D 维：mult = 2D，add = D；q 和 k 各一遍 => mult = 4D，add = 2D
    rope_rows = bsz * H * q_len
    rope_ac = rope_rows * (4 * D + 2 * D)

    # ============== 2. 注意力分数 S = Q @ K^T  AC ==============
    # 这里因为我们的版本的注意力模块中，q,k,v 都是时序脉冲，所以计算操作为 AC
    # AC： (B, H, Q, D) @ (B, H, D, K) -> (B, H, Q, K) : B*H*Q*K*D
    ac_qkt = bsz * H * q_len * k_len * D

    elems_scores = bsz * H * q_len * k_len

    # ==== 2-1 1/sqrt(D) 缩放：逐元素乘 ====
    scale_ac = elems_scores

    # ==== 2-2 causal/padding mask（若有）====
    mask_ac = elems_scores if (attention_mask is not None and attention_mask is not False) else 0

    # ============== 3. softmax(S)（沿 K 维） ==============
    # 对每一行（长度 K）：max (K-1) 比较 + 减 max K + exp K + 求和 (K-1) + 除法 K
    # 合计每行： (K-1) + K + K + (K-1) + K = 5K - 2
    num_rows = bsz * H * q_len
    softmax_per_row = 5 * k_len - 2
    softmax_ac = num_rows * softmax_per_row

    # ===== 3-1 AC：dropout（训练且 p>0）：逐元素乘 =====
    dropout_ac = elems_scores if (training and p_drop > 0.0) else 0

    # ============== 4. 输出聚合 O = A @ V  AC==============
    # 因为 我们对V, 注意力概率，都经过了脉冲， 所以这部分计算也为 AC
    ac_av = bsz * H * q_len * k_len * D

    # 汇总
    mac = 0
    ac = int(rope_ac + ac_qkt + scale_ac + mask_ac + softmax_ac + dropout_ac + ac_av)
    total = mac + ac

    # 累加到 module.___syops__ = [total, ac, mac, 100]
    if not hasattr(module, "___syops__") or module.___syops__ is None:
        module.___syops__ = [0, 0, 0, 100]

    module.___syops__[0] += total
    module.___syops__[1] += ac
    module.___syops__[2] += mac
    module.___syops__[3] += 100


def snn_llama_attention_syops_counter_hook_phase(module: SpikingLlamaAttention, input, output):
    r"""
    🤔
    Hook to estimate MACs for QK^T and AttnV in snn-style attention.

    Input Tensor:
        hidden_states.shape = [B*T, L, D]
    Processing:
        hidden_states → [T, B, L, D]
        Projections → [T, B, H, Q, D]
        Output computed via:
            Q @ K^T → [T, B, H, Q, K]
            Attn @ V → [T, B, H, Q, D]

    MACs:
        1. Attention score matmul: Q x K^T → T × B × H × Q × D × K
        2. Output matmul: Attn x V → T × B × H × Q × K × D

    :param module:
    :param input:
    :param output:
    :return:
    """
    macs = 0
    acs = 0

    # K /scale 先按乘法算（视为取倒数+乘，忽略取倒数因为就对一个值取倒数）
    macs += torch.count_nonzero(module.key_states_tmp)

    # Q*K
    # 统计S_Q和S_K的计算数
    acs += torch.count_nonzero(module.query_states_tmp)
    acs += torch.count_nonzero(module.key_states_tmp)

    # 模拟法统计乘积的计算数
    S_q = module.query_states_tmp.cumsum(dim=0)
    S_k = module.key_states_tmp.cumsum(dim=0)

    S_q[S_q != 0.0] = 1.0
    S_k[S_k != 0.0] = 1.0
    module.query_states_tmp[module.query_states_tmp != 0.0] = 1.0
    module.key_states_tmp[module.key_states_tmp != 0.0] = 1.0
    # term1: S_q @ k^T
    term1 = torch.matmul(S_q, module.key_states_tmp).float()
    # term2: q @ S_k^T
    term2 = torch.matmul(module.query_states_tmp, S_k).float()
    # term3: q @ k^T
    term3 = torch.matmul(module.query_states_tmp, module.key_states_tmp).float()
    acs += int(torch.sum(term1) + torch.sum(term2) + torch.sum(term3))

    # score*v
    S_w = module.attn_weights_tmp.cumsum(dim=0)
    S_v = module.value_states_tmp.cumsum(dim=0)

    S_w[S_w != 0.0] = 1.0
    S_v[S_v != 0.0] = 1.0
    module.attn_weights_tmp[module.attn_weights_tmp != 0.0] = 1.0
    module.value_states_tmp[module.value_states_tmp != 0.0] = 1.0
    # term1: S_q @ k^T
    term1 = torch.matmul(S_w, module.value_states_tmp).float()
    # term2: q @ S_k^T
    term2 = torch.matmul(module.attn_weights_tmp, S_v).float()
    # term3: q @ k^T
    term3 = torch.matmul(module.attn_weights_tmp, module.value_states_tmp).float()
    acs += int(torch.sum(term1) + torch.sum(term2) + torch.sum(term3))

    module.__syops__[0] += macs + acs
    module.__syops__[1] += acs
    module.__syops__[2] += macs
    module.__syops__[3] += 100

    del module.query_states_tmp
    del module.key_states_tmp

    del module.attn_weights_tmp
    del module.value_states_tmp


def snn_linear_syops_counter_hook(module: SpikingLinear | nn.Linear,
                                  input: Tuple,
                                  output: Tensor):
    r"""
    :param module: 根据情况，可能会对nn.Linear 或者 SpikingLinear 对象，启用该hook。
    :param input:
    :param output:
    :return:
    """
    x = input[0]
    B, L, Di = x.shape
    Do = module.out_features
    T = 16

    mac_ops = 0

    syopsBias = T * B * L * Do if module.bias is not None else 0

    syopsMatMul = B * L * Di * Do
    ac_ops = syopsMatMul + syopsBias

    module.__syops__[0] += (mac_ops+ac_ops)
    module.__syops__[1] += ac_ops
    module.__syops__[2] += mac_ops
    module.__syops__[3] += 100  # TTFS 编码中，没有发射率的概念。


def act_swl_syops_counter_hook(module, input, output):
    r"""
    ✅
    Hook to estimate MACs and ACs for a custom SNN-style SiLU activation layer.

        Input shape: (T, B, L, D)
        For each time step:
            - Add: X += x[t]             → 1 × AC
            - Activation: SiLU(X)       → ~5 × MAC + ~2 × AC
            - Delta: Y - Y_prev         → 1 × AC

        So per element per time step:
            - MACs: 5
            - ACs: 1 (add) + 2 (SiLU) + 1 (delta) = 4

        Total:
            - MACs = 5 × T × B × L × D
            - ACs  = 4 × T × B × L × D

    :param input: Tuple[(T, B, L, D)]
    :param output: Tensor of shape (T, B, L, D)
    """
    mac_ops = 0
    ac_ops = 0

    x = input[0]  # (T, B, L, D)

    X = torch.zeros_like(x[0])
    Y_pre = 0
    Out = []
    for t in range(module.T):
        X = X + x[t]
        ac_ops += torch.count_nonzero(x[t]).item()

        Y = module.act_fn(X)

        num_elements = torch.count_nonzero(X).item()
        per_element_mac = 9  # 近似值：exp + div + mul
        per_element_ac = 7  # 近似值：neg + add + div中AC
        mac_ops += num_elements * per_element_mac  # x * sigmoid(x)
        ac_ops += num_elements * per_element_ac  # 1 + exp(-x)
        # module.__syops__ = [acs + macs, acs, macs, 100]

        Out.append(Y - Y_pre)
        if torch.is_tensor(Y_pre):
            ac_ops += torch.count_nonzero(Y * Y_pre).item()

        Y_pre = Y

    module.__syops__[0] += (mac_ops + ac_ops)
    module.__syops__[1] += ac_ops
    module.__syops__[2] += mac_ops
    module.__syops__[3] += 100


def spiking_base_module_syops_counter_hook(module: SpikingBaseModule,
                                           input,
                                           output):
    pass


# endregion

# region === Hooks in TTFSFormer Styled S-Llama ===
def ttfs_llama_attention_syops_counter_hook(module: LlamaSdpaAttention,
                                            input,
                                            output):
    r"""
    统计 TTFSFormer 风格的 LlamaAttention 前传中的操作数（除 q/k/v/o 线性投影外）：
      - 计入：RoPE、缩放、mask 相加、softmax、dropout、QK^T 与 A·V 的 matmul
      - 不计：q/k/v/o Linear、view/transpose/contiguous/reshape、repeat_kv 的拷贝、
             rotary_emb 里 cos/sin 的生成、cache 的读写（可选开启，见 NOTE）

    统计结果会累加到:
        module.___syops__ = [total, ac, mac, 100]
    :param module:
    :param input:
    :param output:
    :return:
    """
    if isinstance(output, tuple):
        hidden_states = output[0]  # 第一个元素才是真正的 hidden_states Tensor
    else:
        hidden_states = output
    bsz, q_len, _ = hidden_states.shape

    H = module.num_heads
    D = module.head_dim

    # 注意：KV 经过 repeat_kv 后与 H 对齐，故 matmul 按 H 统计
    # 估计 k_len：优先从 attention_mask 推断，否则假定等于 q_len
    attention_mask = input[1] if len(input) > 1 else None
    if attention_mask is not None and attention_mask is not False:
        # mask 形状通常为 (bsz, 1, q_len, k_len) 或 (bsz, H, q_len, k_len)
        k_len = attention_mask.shape[-1]
    else:
        # 无 cache 的常见推理/训练步
        k_len = q_len

    training = bool(getattr(module, "training", False))
    p_drop = float(getattr(module, "attention_dropout", 0.0) or 0.0)

    # ============== 1. RoPE（apply_rotary_pos_emb）- AC ==============
    # 对 q 与 k 各执行：每对维度(2D pair) -> 4 次乘法 + 2 次加法
    # => 对 D 维：mult = 2D，add = D；q 和 k 各一遍 => mult = 4D，add = 2D
    rope_rows = bsz * H * q_len
    rope_ac = rope_rows * (4 * D + 2 * D)

    # ============== 2. 注意力分数 S = Q @ K^T  AC ==============
    # MAC： (B, H, Q, D) @ (B, H, D, K) -> (B, H, Q, K) : B*H*Q*K*D
    ac_qkt = bsz * H * q_len * k_len * (3 * D + 2)  # 参考的 TTFSFormer 附录表格 3.

    elems_scores = bsz * H * q_len * k_len
    # ==== 2-1 1/sqrt(D) 缩放：逐元素乘 ====
    scale_ac = elems_scores
    # ==== 2-2 causal/padding mask（若有）====
    mask_ac = elems_scores if (attention_mask is not None and attention_mask is not False) else 0

    # ============== 3. softmax(S)（沿 K 维） ==============
    # 对每一行（长度 K）：max (K-1) 比较 + 减 max K + exp K + 求和 (K-1) + 除法 K
    # 合计每行： (K-1) + K + K + (K-1) + K = 5K - 2

    softmax_ac = elems_scores * 6  # 参考的 TTFSFormer 附录表格 3.

    # ===== 3-1 AC：dropout（训练且 p>0）：逐元素乘 =====
    dropout_ac = elems_scores if (training and p_drop > 0.0) else 0

    # ============== 4. 输出聚合 O = A @ V  MAC==============
    mac_av = bsz * H * q_len * (3 * k_len + 2) * D  # 参考的 TTFSFormer 附录表格 3.

    # 汇总
    mac = int(ac_qkt + mac_av)
    ac = int(rope_ac + scale_ac + mask_ac + softmax_ac + dropout_ac)
    total = mac + ac

    # 累加到 module.___syops__ = [total, ac, mac, 100]
    if not hasattr(module, "___syops__") or module.___syops__ is None:
        module.___syops__ = [0, 0, 0, 100]

    module.__syops__[0] += total
    module.__syops__[1] += ac
    module.__syops__[2] += mac
    module.__syops__[3] += 100

def ttfs_llama_mlp_syops_counter_hook(module: LlamaMLP, input, output):
    r"""
    Hook for snnLlamaMLP_SWL module.
    Only accounts for hadamard,
    and skips act_func and snnLinear ops which are separately hooked.
    """
    x = input[0]
    batch, seq_len, hidden = x.shape

    ac = batch * seq_len * module.intermediate_size * 5
    module.__syops__[0] += ac
    module.__syops__[1] += ac
    module.__syops__[2] += 0
    module.__syops__[3] += 100

    # total_ops = T * batch * seq_len * hidden  # 逐元素乘法

    # spike, rate, spkhistc = spike_rate(x)
    # assert rate <= 1, f"rate {rate}"
    # module.__syops__[0] += total_ops
    # if spike:
    #     module.__syops__[1] += int(total_ops * rate)
    # else:
    #     module.__syops__[2] += total_ops
    # module.__syops__[3] += rate * 100
    # module.__spkhistc__ = spkhistc

def ttfs_silu_syops_counter_hook(module: nn.SiLU, input, output):
    r"""
    🤔
    nn.SiLU 的 操作数统计 hook。
    * 理论上:          SiLU = x / (1 + exp(-x)) -> neg + exp + add1 + reciprocal + mul
    * 实际 torch 中:   SiLU = x * sigmoid(x)    -> sigmoid + mul


    计数累加到 module.___syops__ = [total, ac, mac, 100]
    其中 ac 为“加权 AC”，mac 恒为 0。
    :param module:
    :param input:
    :param output:
    :return:
    """
    x = input[0]
    batch, seq_len, hidden = x.shape
    ac = batch * seq_len * hidden
    mac = 0
    total = ac


    module.__syops__[0] += total
    module.__syops__[1] += ac
    module.__syops__[2] += mac
    module.__syops__[3] += 100


def ttfs_rmsnorm_syops_counter_hook(module: QuantizedLlamaRMSNorm, input, output):
    r"""
    🤔
    nn.SiLU 的 操作数统计 hook。
    * 理论上:          SiLU = x / (1 + exp(-x)) -> neg + exp + add1 + reciprocal + mul
    * 实际 torch 中:   SiLU = x * sigmoid(x)    -> sigmoid + mul


    计数累加到 module.___syops__ = [total, ac, mac, 100]
    其中 ac 为“加权 AC”，mac 恒为 0。
    :param module:
    :param input:
    :param output:
    :return:
    """
    x = input[0]
    batch, seq_len, hidden = x.shape
    ac = batch * seq_len * precison * (9 * hidden + 1)
    mac = 0
    total = ac


    module.__syops__[0] += total
    module.__syops__[1] += ac
    module.__syops__[2] += mac
    module.__syops__[3] += 100

def ttfs_linear_syops_counter_hook(module: QuantizedLinear,
                                  input: Tuple,
                                  output: Tensor):
    r"""
    :param module: 根据情况，可能会对nn.Linear 或者 SpikingLinear 对象，启用该hook。
    :param input:
    :param output:
    :return:
    """
    x = input[0]
    B, L, Di = x.shape
    Do = module.out_features
    T = precison

    mac_ops = 0

    syopsBias = T * B * L * Do if module.bias is not None else 0

    syopsMatMul = B * L * Di * Do
    ac_ops = syopsMatMul + syopsBias

    module.__syops__[0] += (mac_ops+ac_ops)
    module.__syops__[1] += ac_ops
    module.__syops__[2] += mac_ops
    module.__syops__[3] += 100  # TTFS 编码中，没有发射率的概念。


# endregion
# endregion




CUSTOM_MODULES_MAPPING = {}

# TODO 需要注意，量化模型和脉冲模型，结构是一样的，模块也是一样的，只是量化操作等价替换为了脉冲发射。
#  同时，能耗统计的时候，需要注意，量化操作、脉冲发射操作不计入能耗。
#  因此，量化模型和原 ANN 模型能耗和操作数相同，只选统计后者即。
#  而脉冲模型的能耗使用 Quantized* 模块前缀来统计，但是不代表量化模型的能耗。
MODULES_MAPPING = {
    # region  === Vanilla Hooks ===
    # convolutions
    nn.Conv1d: conv_syops_counter_hook,
    nn.Conv2d: conv_syops_counter_hook,
    nn.Conv3d: conv_syops_counter_hook,

    # activations
    nn.ReLU: relu_syops_counter_hook,
    nn.PReLU: relu_syops_counter_hook,
    nn.ELU: relu_syops_counter_hook,
    nn.LeakyReLU: relu_syops_counter_hook,
    nn.ReLU6: relu_syops_counter_hook,
    # poolings
    nn.MaxPool1d: pool_syops_counter_hook,
    nn.AvgPool1d: pool_syops_counter_hook,
    nn.AvgPool2d: pool_syops_counter_hook,
    nn.MaxPool2d: pool_syops_counter_hook,
    nn.MaxPool3d: pool_syops_counter_hook,
    nn.AvgPool3d: pool_syops_counter_hook,
    nn.AdaptiveMaxPool1d: pool_syops_counter_hook,
    nn.AdaptiveAvgPool1d: pool_syops_counter_hook,
    nn.AdaptiveMaxPool2d: pool_syops_counter_hook,
    nn.AdaptiveAvgPool2d: pool_syops_counter_hook,
    nn.AdaptiveMaxPool3d: pool_syops_counter_hook,
    nn.AdaptiveAvgPool3d: pool_syops_counter_hook,
    # BNs
    nn.BatchNorm1d: bn_syops_counter_hook,
    nn.BatchNorm2d: bn_syops_counter_hook,
    nn.BatchNorm3d: bn_syops_counter_hook,
    nn.LayerNorm: ln_syops_counter_hook,

    # Neuron IF
    MultiStepIFNode: IF_syops_counter_hook,
    IFNode: IF_syops_counter_hook,  # SpikeZIP: our neuron
    # Neuron LIF
    MultiStepLIFNode: LIF_syops_counter_hook,
    LIFNode: LIF_syops_counter_hook,
    # Neuron PLIF
    MultiStepParametricLIFNode: LIF_syops_counter_hook,
    ParametricLIFNode: LIF_syops_counter_hook,

    nn.InstanceNorm1d: bn_syops_counter_hook,
    nn.InstanceNorm2d: bn_syops_counter_hook,
    nn.InstanceNorm3d: bn_syops_counter_hook,
    nn.GroupNorm: bn_syops_counter_hook,
    # Upscale
    nn.Upsample: upsample_syops_counter_hook,
    # Deconvolution
    nn.ConvTranspose1d: conv_syops_counter_hook,
    nn.ConvTranspose2d: conv_syops_counter_hook,
    nn.ConvTranspose3d: conv_syops_counter_hook,
    # RNN
    nn.RNN: rnn_syops_counter_hook,
    nn.GRU: rnn_syops_counter_hook,
    nn.LSTM: rnn_syops_counter_hook,
    nn.RNNCell: rnn_cell_syops_counter_hook,
    nn.LSTMCell: rnn_cell_syops_counter_hook,
    nn.GRUCell: rnn_cell_syops_counter_hook,
    nn.MultiheadAttention: multihead_attention_counter_hook,
    Attention: multihead_attention_counter_hook,
    # endregion

    # region === Newly added Hooks ===

    # Layers in Llama
    nn.Embedding: nn_embedding_syops_counter_hook,  # ✅
    LlamaRMSNorm: rmsnorm_syops_counter_hook,  # ✅
    RMSN: rmsnorm_syops_counter_hook,  # ✅
    nn.SiLU: silu_syops_counter_hook,  # 🤔
    nn.Linear: nn_linear_syops_counter_hook,  # ✅
    LlamaRotaryEmbedding: rotary_embedding_syops_counter_hook,  # 🤔
    LlamaMLP: llama_mlp_syops_counter_hook,  # ✅
    LlamaAttention: llama_attention_syops_counter_hook,  # ✅
    LlamaSdpaAttention: llama_attention_syops_counter_hook,

    # Layers in Q-Llama
    # QuantizedLlamaMLP: llama_mlp_syops_counter_hook,  # ✅ 已经分别替换了 LlamaMLP 中的 SiLU 以及 Linear。
    # QuantizedLinear: nn_linear_syops_counter_hook if ANN else snn_linear_syops_counter_hook,  # ✅
    # QuantizedLlamaRMSNorm: rmsnorm_syops_counter_hook,  # ✅
    # QuantizedLlamaAttention: llama_attention_syops_counter_hook if ANN else snn_llama_attention_syops_counter_hook,  # ✅

    # Layers in TTFSFormer-LLama
    QuantizedSiLU: ttfs_silu_syops_counter_hook,
    # TTFSFormer 作者文中没有给 哈达玛积 的操作数公式，但是代码中自己实现了哈达玛积，MLP中就采用一般计数。
    QuantizedLlamaMLP: ttfs_llama_mlp_syops_counter_hook,
    QuantizedLlamaRMSNorm: ttfs_rmsnorm_syops_counter_hook,
    QuantizedLlamaAttention: ttfs_llama_attention_syops_counter_hook,  # 🤔
    QuantizedLinear: ttfs_linear_syops_counter_hook,  # 🤔
    # endregion
}

if hasattr(nn, 'GELU'):
    MODULES_MAPPING[nn.GELU] = relu_syops_counter_hook
