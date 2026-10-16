import logging
import math
from typing import Dict, List, Type
from typing import Tuple, Callable, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaAttention

from spiking_conversion.util import BaseMonitor

# region ==== 原 TTFSFormer 中的 monitor的定义 ===
r"""
==============================
监控点布局（参考 ViT，并映射到 LLaMA）
==============================

【模型级 (VisionTransformer)】
    monitor_0：原始输入
    monitor_1：patch_embed 输出
    monitor_2：位置编码后
    monitor_3：Transformer blocks 后
    monitor_4：分类头输入
    monitor_o：最终 logits

【PatchEmbed 子模块】
    monitor_0：图像输入
    monitor_1：卷积投影输出
    monitor_o：flatten+norm 之后

【Attention 子模块（ViT 版本）】
    monitor_0：输入 hidden_states
    monitor_1：qkv 线性层输出
    
    2、3 由softmax的旁支监控获取。
    monitor_2：softmax 前的注意力分数（logits）
    monitor_3：softmax 后的注意力权重（可能含 dropout）
        
    monitor_4：注意力上下文 attn @ v（投影前）
    monitor_o：输出投影后的结果



【Mlp 子模块（ViT 版本）】
    monitor_0：模块输入
    monitor_1：第一层全连接（fc1）输出
    monitor_2：激活后的中间态
    monitor_3：drop/norm 之后、第二层全连接前
    monitor_o：第二层全连接（fc2）输出

------------------------------
LLaMA（LlamaAttention）与上述对齐关系
------------------------------
【Embedding】
    embed_tokens.monitor_0：嵌入层输入（input_ids 或 inputs_embeds）
    embed_tokens.monitor_o：nn.Embedding 输出（[B,T,H]）    

【SelfAttention（LlamaAttention）】
    layers.{i}.self_attn.monitor_0：模块输入 hidden_states
    layers.{i}.self_attn.monitor_1q/k/v：q_proj / k_proj / v_proj 输出（合并记录）
    layers.{i}.self_attn.monitor_2：注意力分数 scores = QK^T / sqrt(d)，由 Softmax 旁路记录
    layers.{i}.self_attn.monitor_3：注意力概率 probs = softmax(scores)，由 Softmax 旁路记录
    layers.{i}.self_attn.monitor_4：注意力上下文 context = probs @ V（o_proj 之前）
    layers.{i}.self_attn.monitor_o：o_proj 输出

【Softmax】
    monitor_1: lse

【MLP（LlamaMLP）】
    layers.{i}.mlp.monitor_0：MLP 输入 x
    layers.{i}.mlp.monitor_1：gate_proj / up_proj 输出（合并记录）
    layers.{i}.mlp.monitor_2：act(gate) * up（通过 down_proj 的 pre-hook 获取）
    layers.{i}.mlp.monitor_3：对齐 ViT 命名，复用与 monitor_2 同点
    layers.{i}.mlp.monitor_o：down_proj 输出

【SiLU 激活（当激活函数为 nn.SiLU）】
    layers.{i}.mlp.act.monitor_0：SiLU 输入（act 前）
    layers.{i}.mlp.act.monitor_o：SiLU 输出（act 后）

【RMSNorm（可选，模型存在时生效）】
    {any.rmsnorm.path}.monitor_0：RMSNorm 输入
    {any.rmsnorm.path}.monitor_o：RMSNorm 输出

【模型级（LLaMA 顶层）】
    model.monitor_0：LlamaModel.forward 的输入（input_ids/inputs_embeds）
    model.monitor_1：嵌入后（embed_tokens 输出）
    model.monitor_2：全部 decoder 层之后、final norm 之前
    model.monitor_3：final norm 之后（LlamaModel 输出前）
    lm_head.monitor_4：lm_head 输入（最后隐状态）
    lm_head.monitor_o：lm_head 输出（logits）
"""


# endregion

# region ======== 工具函数：注册/路由 ========

# region ========== 通用：前向 pre_hook 模板（返回 None）==========

def pre_embed_tokens(monitors, MonitorCls):
    """
    embed_tokens 的前向 pre_hook：
    - 记录 monitor_0：输入（通常是 token ids，形状 [B, T]）
    - 必须返回 None，表示不改输入
    """

    def _hook(mod: nn.Module, args: tuple, kwargs: dict):
        # embed_tokens 一般是位置参数（input_ids）
        if len(args) > 0:
            _reg_monitor(monitors, "embed_tokens.monitor_0", MonitorCls)(args[0])
        return None

    return _hook


def pre_attn_input(monitors, layer_stack, base_name: str, MonitorCls):
    """
    LlamaAttention 的前向 pre_hook：
    - 压栈层名（给 SDPA 旁路路由 monitor_2 用）
    - 记录 monitor_0：注意力模块输入 hidden_states（[B, T, H]）
    - 返回 None（不改变输入）
    """

    def _hook(mod: nn.Module, args: tuple, kwargs: dict):
        # 关键：既兼容 kwargs 也兼容 args
        x = kwargs.get("hidden_states", args[0] if len(args) > 0 else None)
        if x is not None:
            _route_current_layer_push(layer_stack, base_name)
            _reg_monitor(monitors, f"{base_name}.monitor_0", MonitorCls)(x)
        return None

    return _hook


def pre_o_proj_input(monitors, base_name: str, MonitorCls):
    """
    o_proj 的前向 pre_hook：
    - 记录 monitor_4：o_proj 输入（attn @ v 之后），形状 [B, T, H]
    - 返回 None
    """

    def _hook(mod: nn.Module, args: tuple, kwargs: dict):
        if len(args) > 0:
            _reg_monitor(monitors, f"{base_name}.monitor_4", MonitorCls)(args[0])
        return None

    return _hook


def pre_mlp_input(monitors, base_name: str, MonitorCls):
    """
    MLP 的前向 pre_hook：
    - 记录 monitor_0：MLP 输入 x，形状 [B, T, H]
    - 返回 None
    """

    def _hook(mod: nn.Module, args: tuple, kwargs: dict):
        if len(args) > 0:
            _reg_monitor(monitors, f"{base_name}.monitor_0", MonitorCls)(args[0])
        return None

    return _hook


def pre_down_proj_input(monitors, base_name: str, MonitorCls):
    """
    down_proj 的前向 pre_hook：
    - 记录 monitor_2：中间张量 act(gate) * up
    - 返回 None
    """

    def _hook(mod: nn.Module, args: tuple, kwargs: dict):
        if len(args) > 0:
            inter = args[0]
            _reg_monitor(monitors, f"{base_name}.monitor_2", MonitorCls)(inter)
        return None

    return _hook


def pre_lm_head_input(monitors, MonitorCls):
    """
    lm_head 的前向 pre_hook：
    - 记录 lm_head.monitor_4：logits 之前的最后隐状态 [B, T, H]
    - 返回 None
    """

    def _hook(mod: nn.Module, args: tuple, kwargs: dict):
        if len(args) > 0:
            _reg_monitor(monitors, "lm_head.monitor_4", MonitorCls)(args[0])
        return None

    return _hook


# endregion

def _reg_monitor(monitors: Dict[str, BaseMonitor], name: str, MonitorCls=BaseMonitor) -> BaseMonitor:
    """
    注册或获取一个监控器实例。

    参数:
        monitors: 全局监控器字典
        name:     监控器在字典中的键（如 'layers.0.self_attn.monitor_2'）
        MonitorCls: 监控器类；默认 BaseMonitor，可传你的子类，如 MinMaxMonitor

    返回:
        对应的监控器实例
    """
    mon = monitors.get(name)
    if mon is None:
        mon = MonitorCls()
        monitors[name] = mon
    return mon


def _is_rmsnorm(mod: nn.Module) -> bool:
    r"""
    统一识别 LlamaRMSNorm / RMSNorm / QuantRMSNorm：有 variance_epsilon + weight 即视作 RMSNorm 家族

    只要有 variance_epsilon 且有 weight、而不像 LayerNorm 那样有 bias/normalized_shape，
    即可视作 RMSNorm 家族（兼容 LlamaRMSNorm / RMSN / QuantRMSNorm）。
    :param mod:
    :type mod: nn.Module
    """
    return (
            hasattr(mod, "variance_epsilon")
            and hasattr(mod, "weight")
            and not isinstance(mod, nn.LayerNorm)
    )


def _get_eps(mod: nn.Module) -> float:
    eps = getattr(mod, "variance_epsilon", None)
    if eps is None:
        # 兜底（几乎用不到）
        eps = getattr(mod, "eps", 1e-6)
    return float(eps)


def _route_current_layer_push(layer_stack: List[str], layer_name: str) -> None:
    """
    将当前层名压栈，用于 SDPA 软路由 monitor_2（softmax 概率）。

    参数:
        layer_stack: 维护当前激活注意力层名的栈
        layer_name:  本次进入的层名（如 'layers.3.self_attn'）
    """
    layer_stack.append(layer_name)


def _route_current_layer_pop(layer_stack: List[str]) -> None:
    """
    弹出当前层名，表示本层注意力模块前传结束。

    参数:
        layer_stack: 维护当前激活注意力层名的栈
    """
    if layer_stack:
        layer_stack.pop()


# region ======== SDPA 以及 Softmax 包裹器：旁路计算 softmax 概率 ========

class SdpaTap:
    """
    将 torch.nn.functional.scaled_dot_product_attention 全局包裹，
    在不改变模型输出的前提下，旁路计算 attention scores（softmax 前）与 probs（softmax 后），
    并分别送入对应层的 monitor_2（scores）与 monitor_3（probs）。

    用法:
        tap = SdpaTap(monitors, layer_stack, logger, MonitorCls)
        tap.install()
        ... 做推理/训练 ...
        tap.uninstall()
    """

    def __init__(
            self,
            monitors: Dict[str, BaseMonitor],
            layer_stack: List[str],
            nameLogger: str,
            MonitorCls=BaseMonitor,
    ) -> None:
        self.monitors = monitors
        self.layer_stack = layer_stack
        self.logger = logging.getLogger(nameLogger)
        self.MonitorCls = MonitorCls
        self._orig_sdpa = F.scaled_dot_product_attention  # 备份原函数

    def _wrapped(self, q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False):
        """
        替换后的 SDPA：旁路计算 scores/probs 并投喂 monitor_2 / monitor_3，然后调用原始 SDPA。
        """
        # 如果当前在某个 self_attn 层内（layer_stack 非空），则做旁路统计
        if self.layer_stack:
            layer_name = self.layer_stack[-1]
            try:
                with torch.no_grad():
                    # 计算注意力分数 scores = Q K^T / sqrt(d_k)  -> 形状 [B, H, Q, K]
                    scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))

                    # 加上 mask（加法掩码）
                    valid = None
                    if attn_mask is not None:
                        if attn_mask.dtype == torch.bool:
                            valid = ~attn_mask
                            add_mask = attn_mask.masked_fill(attn_mask, float('-inf')).to(scores.dtype)
                        else:
                            add_mask = attn_mask.to(scores.dtype)
                            valid = ~torch.isneginf(add_mask)

                        scores = scores + add_mask

                    # 将 scores 喂给 monitor_2（softmax 前的 logits）
                    _reg_monitor(self.monitors,
                                 f"{layer_name}.monitor_2",
                                 self.MonitorCls)(scores)

                    # lse（稳定）：logsumexp; 计算 log-sum-exp（LSE）并注册到 monitor_1
                    # lse = torch.logsumexp(scores, dim=-1, keepdim=True)
                    # _reg_monitor(self.monitors,
                    #              f"{layer_name}.softmax.monitor_1",
                    #              self.MonitorCls)(lse)

                    # 计算 log-sum-exp（LSE）并注册到 monitor_1
                    scores = scores - scores.max(dim=-1, keepdim=True).values
                    lse = scores.exp().sum(dim=-1, keepdim=True).log()
                    _reg_monitor(self.monitors,
                                 f"{layer_name}.softmax.monitor_1",
                                 self.MonitorCls)(lse)

                    # 计算 softmax 概率，且对全遮挡行置零
                    probs = (scores - lse).exp_()
                    if valid is not None:
                        row_has_valid = valid.any(dim=-1, keepdim=True)  # [B,H,Q,1]
                        probs = torch.where(row_has_valid, probs, torch.zeros_like(probs))

                    # 将 probs 喂给 monitor_3（softmax 后的概率）
                    _reg_monitor(self.monitors,
                                 f"{layer_name}.monitor_3",
                                 self.MonitorCls)(probs)

            except Exception as e:
                self.logger.warning(f"SDPA tap failed on {layer_name}: {e}")

        # 返回原始 SDPA 结果，不改变模型逻辑
        return self._orig_sdpa(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)

    def install(self) -> None:
        """安装 SDPA 包裹器（进程级替换）。"""
        F.scaled_dot_product_attention = self._wrapped

    def uninstall(self) -> None:
        """卸载 SDPA 包裹器，恢复原始实现。"""
        F.scaled_dot_product_attention = self._orig_sdpa


class SoftmaxTap:
    """
    全局包裹 torch.nn.functional.softmax：
    - 仅当 layer_stack 非空（处于某层 self_attn 内）时才记录；
    - 记录 monitor_2：softmax 的输入 logits（含缩放/掩码/上采样到 fp32 等处理之后）；
    - 记录 softmax.monitor_1：沿 TTFS 写法记录数值稳定的 log-sum-exp (LSE)；
    - 记录 monitor_3：softmax 输出的概率；
    - 不改变原始 F.softmax 的返回值。
    """

    def __init__(
            self,
            monitors: Dict[str, "BaseMonitor"],
            layer_stack: List[str],
            logger_name: str,
            MonitorCls: Type["BaseMonitor"] = None,
    ) -> None:
        self.monitors = monitors
        self.layer_stack = layer_stack
        self.logger = logging.getLogger(logger_name)
        self.MonitorCls = MonitorCls or BaseMonitor
        self._orig_softmax = nn.functional.softmax  # 备份原始实现

    def _wrapped(self, input, dim=None, _stacklevel=3, dtype=None):
        """
        F.softmax 的替换函数。仅在 self_attn 上下文中记录监控量。
        """
        # 设置默认的 dim 为 -1
        if dim is None:
            dim = input.dim() - 1  # 默认对最后一维做 softmax

        layer_name = self.layer_stack[-1] if self.layer_stack else 'N/A'

        # 检查是否在最后一维上执行 softmax
        if dim != input.dim() - 1 and dim != -1:
            return self._orig_softmax(input, dim=dim, _stacklevel=_stacklevel, dtype=dtype)

        try:
            with torch.no_grad():
                # 1) monitor_2: pre-softmax logits（已经在模块里完成缩放/掩码/可能的fp32上采样）
                _reg_monitor(self.monitors, f"{layer_name}.monitor_2", self.MonitorCls)(input)

                # 2) softmax.monitor_1: 数值稳定 LSE（对齐 TTFS MonitoredSoftmax）
                x_stab = input - input.max(dim=dim, keepdim=True).values
                lse = x_stab.exp().sum(dim=dim, keepdim=True).log()
                _reg_monitor(self.monitors, f"{layer_name}.softmax.monitor_1", self.MonitorCls)(lse)
        except Exception as e:
            self.logger.warning(f"Softmax tap pre failed on {layer_name}: {e}")

        # 调用原始 softmax 获得概率
        out = self._orig_softmax(input, dim=dim, _stacklevel=_stacklevel, dtype=dtype)

        try:
            with torch.no_grad():
                # 3) monitor_3: softmax 概率
                _reg_monitor(self.monitors, f"{layer_name}.monitor_3", self.MonitorCls)(out)
        except Exception as e:
            self.logger.warning(f"Softmax tap post failed on {layer_name}: {e}")

        return out

    def install(self) -> None:
        """安装 Softmax 包裹器（进程级覆盖 F.softmax）。"""
        nn.functional.softmax = self._wrapped

    def uninstall(self) -> None:
        """卸载 Softmax 包裹器，恢复原始实现。"""
        nn.functional.softmax = self._orig_softmax


# endregion


# region ======== 顶层 API：为 LLaMA（ LlamaAttention）挂所有监控点 ========

def attach_monitors_llama_sdpa(
        model: LlamaForCausalLM,
        MonitorCls: type[BaseMonitor],
        logger_name: str = "llama_mon",

) -> Tuple[Dict[str, BaseMonitor], Callable]:
    r"""
    按 ViT 风格，为包含 LlamaAttention 的 LLaMA 模型挂上监控器（基于 forward hooks + SDPA 旁路）。

    - Embedding:
        embed_tokens.monitor_0 / monitor_1 / monitor_o
    - SelfAttention（LlamaAttention）:
        layers.{i}.self_attn.monitor_0 / 1 / 2 / 3 / 4 / o
          * 0: 模块输入 hidden_states
          * 1: q/k/v 线性层输出（合并）
          * 2: attention scores（softmax 前，SDPA 旁路记录）
          * softmax.monitor_1: softmax 中的 lse
          * 3: attention probs（softmax 后，SDPA 旁路记录）
          * 4: o_proj 输入（context = AttnWeights @ V）
          * o: o_proj 输出
    - MLP:
        layers.{i}.mlp.monitor_0 / 1 / 2 / 3 / o
          * 0: MLP 输入 x
          * 1: gate_proj / up_proj 输出（合并）
          * 2: act(gate) * up（down_proj pre-hook）
          * 3: 对齐 ViT 命名，复用与 2 同点
          * o: down_proj 输出
    - SiLU（当激活为 nn.SiLU）:
        layers.{i}.mlp.act.monitor_0 / monitor_o（分别记录激活前/后）
    - RMSNorm（模型中存在时）:
        {name}.monitor_0 / {name}.monitor_o
    - 顶层与 lm_head:
        model.monitor_0 / 1 / 2 / 3；lm_head.monitor_4 / monitor_o

    备注:
        - SDPA 旁路仅用于记录 monitor_2/3，不改变原始计算路径。
        - 若你对模型做过 “LayerNorm/RMSNorm 融合/替换”，则 RMSNorm 探针仅在对应模块仍存在时生效。

    :param model: LLaMA 模型（如 LlamaForCausalLM 或包含 .model.layers 的实例）
    :type model: LlamaForCausalLM
    :param logger_name:logging 日志名称
    :type logger_name:str
    :param MonitorCls: 监控器类（默认 BaseMonitor；可传 MinMaxMonitor 等）
    :type MonitorCls: BaseMonitor
    :return: (monitors, teardown)：
                monitors: {name -> monitor实例}，随前传动态填充/调用
                teardown: 调用以移除所有 hooks，并恢复被替换的 SDPA
    :rtype:Tuple[Dict[str, BaseMonitor], Callable]
    """
    logger = logging.getLogger(logger_name)
    monitors: Dict[str, BaseMonitor] = {}
    hooks: List[torch.utils.hooks.RemovableHandle] = []

    # 用于在 Softmax 包裹器中路由 monitor_2 到“当前 self_attn 层”
    layer_stack: List[str] = []
    # sdpa_tap = SdpaTap(monitors, layer_stack, logger_name, MonitorCls) # 用于 SDPA 捕获特征的。常规 LlamaAttention 用不上。
    # sdpa_tap.install()
    softmax_tap = SoftmaxTap(monitors, layer_stack, logger_name, MonitorCls)
    softmax_tap.install()

    # 取核心子模块（HF 的 LLaMA 通常是 model.embed_tokens / model.layers / lm_head）
    core = getattr(model, "model", model)

    # ---- model.monitor_0：模型 forward 的“输入” ----
    def _pre_model(m, args, kwargs):
        # HF 签名：
        # forward(self,
        #         input_ids=None,
        #         attention_mask=None,
        #         position_ids=None,
        #         past_key_values=None,
        #         inputs_embeds=None,
        #         ...)
        if len(args) > 0:
            _reg_monitor(monitors, "model.monitor_0", MonitorCls)(args[0])
        return None

    def _post_embed(mod, inp, out):
        _reg_monitor(monitors, "embed_tokens.monitor_o", MonitorCls)(out)
        _reg_monitor(monitors, "model.monitor_1", MonitorCls)(out)

    hooks.append(core.register_forward_pre_hook(_pre_model, with_kwargs=True))

    # region ========== Embedding（对齐 PatchEmbed 的 0/1/o）==========
    if hasattr(core, "embed_tokens") and isinstance(core.embed_tokens, nn.Embedding):
        # pre: monitor_0 —— 输入（token ids），有些情形也可能是 embed 前的 hidden 输入
        hooks.append(core.embed_tokens.register_forward_pre_hook(
            pre_embed_tokens(monitors, MonitorCls),
            with_kwargs=True
        ))
        # post: monitor_1 —— 输出（[B,T,H] 的嵌入张量）
        hooks.append(core.embed_tokens.register_forward_hook(_post_embed))
        # hooks.append(core.embed_tokens.register_forward_hook(
        #     lambda mod, inp, out: _reg_monitor(monitors, "embed_tokens.monitor_1", MonitorCls)(out)
        # ))
    # endregion

    # region ========== 遍历每个 Decoder Layer ==========
    layers = getattr(core, "layers", None)
    if layers is None:
        raise RuntimeError("未找到 model.model.layers；请传入标准 LLaMA 模型实例。")

    for i, layer in enumerate(layers):
        attn = layer.self_attn
        if isinstance(layer.self_attn, LlamaAttention):
            base = f"layers.{i}.self_attn"

            hooks.append(attn.register_forward_pre_hook(
                pre_attn_input(monitors, layer_stack, base, MonitorCls),
                with_kwargs=True
            )
            )

            # post: 仅负责弹栈
            def _attn_post(mod, inputs, outputs, b=base):
                _route_current_layer_pop(layer_stack)

            hooks.append(attn.register_forward_hook(_attn_post))

            # monitor_1 —— 分别记录 q/k/v 线性层输出
            m1q = _reg_monitor(monitors, f"{base}.monitor_1q", MonitorCls)
            m1k = _reg_monitor(monitors, f"{base}.monitor_1k", MonitorCls)
            m1v = _reg_monitor(monitors, f"{base}.monitor_1v", MonitorCls)

            # 注册 Q、K、V 分别的线性层监控
            hooks.append(attn.q_proj.register_forward_hook(lambda mod, inp, out, m=m1q: m(out)))
            hooks.append(attn.k_proj.register_forward_hook(lambda mod, inp, out, m=m1k: m(out)))
            hooks.append(attn.v_proj.register_forward_hook(lambda mod, inp, out, m=m1v: m(out)))

            # monitor_4 / monitor_o —— o_proj 的输入/输出
            hooks.append(attn.o_proj.register_forward_pre_hook(
                pre_o_proj_input(monitors, base, MonitorCls),
                with_kwargs=True
            ))
            hooks.append(attn.o_proj.register_forward_hook(
                lambda mod, inp, out, b=base: _reg_monitor(monitors, f"{b}.monitor_o", MonitorCls)(out)
            ))

        # ---- MLP（LlamaMLP）----
        if hasattr(layer, "mlp") and isinstance(layer.mlp, nn.Module):
            mlp = layer.mlp
            base = f"layers.{i}.mlp"

            # pre: monitor_0 —— MLP 输入 x
            hooks.append(mlp.register_forward_pre_hook(
                pre_mlp_input(monitors, base, MonitorCls),
                with_kwargs=True
            ))

            # monitor_1 —— gate_proj / up_proj 输出（合并记录）
            if hasattr(mlp, "gate_proj"):
                hooks.append(mlp.gate_proj.register_forward_hook(
                    lambda mod, inp, out, b=base: _reg_monitor(monitors, f"{b}.monitor_1g", MonitorCls)(out)
                ))
            if hasattr(mlp, "up_proj"):
                hooks.append(mlp.up_proj.register_forward_hook(
                    lambda mod, inp, out, b=base: _reg_monitor(monitors, f"{b}.monitor_1u", MonitorCls)(out)
                ))

            # monitor_2 —— down_proj 的输入（即 act(gate)*up）
            if hasattr(mlp, "down_proj"):
                hooks.append(mlp.down_proj.register_forward_pre_hook(
                    pre_down_proj_input(monitors, base, MonitorCls),
                    with_kwargs=True
                ))
                # monitor_o —— down_proj 输出
                hooks.append(mlp.down_proj.register_forward_hook(
                    lambda mod, inp, out, b=base: _reg_monitor(monitors, f"{b}.monitor_o", MonitorCls)(out)
                ))

            # ---- 激活（仅当是 nn.SiLU 时监控；你的 ACT2FN['silu'/'swish'] 会返回 nn.SiLU 实例）----
            act = getattr(mlp, "act_fn", None)
            if act is not None:
                act_base = f"{base}.act_fn"  # 命名：layers.{i}.mlp.act.monitor_*

                # 输入 -> monitor_0
                def _pre_act(mod, args, kwargs, k=f"{act_base}.monitor_0"):
                    x = args[0] if len(args) else kwargs.get("input", None)
                    if x is not None:
                        _reg_monitor(monitors, k, MonitorCls)(x)
                    return None  # 必须返回 None，表示不改输入

                hooks.append(act.register_forward_pre_hook(_pre_act, with_kwargs=True))

                # 输出 -> monitor_o
                def _post_act(mod, args, out, k=f"{act_base}.monitor_o"):
                    _reg_monitor(monitors, k, MonitorCls)(out)

                hooks.append(act.register_forward_hook(_post_act))

    # ========== RMSNorm 监控(0/1/2/o)：对全部 RMSNorm 注册 pre/post 钩子 ==========
    # 兼容 transformers 不同版本的类名

    for name, mod in model.named_modules():
        if not _is_rmsnorm(mod):
            continue

        cache = {"x": None}  # pre/post 之间临时传参，不改图

        # pre：记录输入，并旁路计算 inv_rms 与 x_hat（未乘 weight）
        def _pre_rms(m, args, kwargs, key=name):
            x = args[0] if len(args) > 0 else kwargs.get("input", None)
            if x is None:
                return None
            cache["x"] = x
            _reg_monitor(monitors, f"{key}.monitor_0", MonitorCls)(x)

            try:
                with torch.no_grad():
                    x32 = x.to(torch.float32)
                    inv_rms = torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + _get_eps(m)).to(x.dtype)
                    x_hat = x * inv_rms
                _reg_monitor(monitors, f"{key}.monitor_2", MonitorCls)(inv_rms)
                _reg_monitor(monitors, f"{key}.monitor_1", MonitorCls)(x_hat)
            except Exception as e:
                logger and logger.warning(f"[RMSNorm pre] {key}: {e}")
                raise e
            return None

        # post：记录输出（乘了 weight 之后的结果）
        def _post_rms(m, args, out, key=name):
            _reg_monitor(monitors, f"{key}.monitor_o", MonitorCls)(out)

        hooks.append(mod.register_forward_pre_hook(_pre_rms, with_kwargs=True))
        hooks.append(mod.register_forward_hook(_post_rms))

    if hasattr(core, "norm") and isinstance(core.norm, nn.Module):
        # pre: model.monitor_2 —— 所有层之后、norm 之前
        def _pre_final_norm(mod, args, kwargs):
            x = args[0] if len(args) > 0 else kwargs.get("hidden_states", None)
            if x is not None:
                _reg_monitor(monitors, "model.monitor_2", MonitorCls)(x)
            return None

        hooks.append(core.norm.register_forward_pre_hook(_pre_final_norm, with_kwargs=True))

        # post: model.monitor_3 —— norm 之后（LlamaModel 输出前）
        def _post_final_norm(mod, args, out):
            _reg_monitor(monitors, "model.monitor_3", MonitorCls)(out)

        hooks.append(core.norm.register_forward_hook(_post_final_norm))
    # endregion

    # ========== lm_head（对齐 ViT head 的 4/o）==========
    if hasattr(model, "lm_head") and isinstance(model.lm_head, nn.Module):
        # monitor_4 —— lm_head 输入（最后隐状态）
        hooks.append(model.lm_head.register_forward_pre_hook(
            pre_lm_head_input(monitors, MonitorCls),
            with_kwargs=True
        ))
        # monitor_o —— lm_head 输出（logits）
        hooks.append(model.lm_head.register_forward_hook(
            lambda mod, inp, out: _reg_monitor(monitors, "lm_head.monitor_o", MonitorCls)(out)
        ))

    def _seed_expected_monitors(model: LlamaForCausalLM,
                                monitors: dict,
                                MonitorCls: BaseMonitor):
        core = getattr(model, "model", model)

        def ensure(key: str) -> None:
            # 若不存在则创建；若已存在（例如 monitor_1 在注册时已经 set），则保持原实例
            monitors.setdefault(key, MonitorCls())

        # region ---- 模型整体：与 ViT 的 forward_features 对齐 ----
        ensure("model.monitor_0")  # 模型输入（input_ids / inputs_embeds）
        ensure("model.monitor_1")  # embedding 之后（embed_tokens 输出）
        ensure("model.monitor_2")  # 全部 decoder 层之后、final norm 之前
        ensure("model.monitor_3")  # final norm 之后
        # endregion

        # region ---- Embedding ----
        if hasattr(core, "embed_tokens"):
            ensure("embed_tokens.monitor_0")
            ensure("embed_tokens.monitor_1")
            ensure("embed_tokens.monitor_o")
        # endregion

        # region ---- 遍历每层LlamaDecoderLayer ----
        layers = getattr(core, "layers", [])
        for i, layer in enumerate(layers):
            # region --- Self-Attn（只管 LlamaAttention）---
            if isinstance(layer.self_attn, LlamaAttention):
                # if getattr(layer, "self_attn", None) and layer.self_attn.__class__.__name__ == "LlamaAttention":
                b = f"layers.{i}.self_attn"
                ensure(f"{b}.monitor_0")  # 模块输入 hidden_states
                ensure(f"{b}.monitor_1")  # q/k/v 线性层输出（合并）
                ensure(f"{b}.monitor_2")  # attention scores（SDPA 旁路写入）
                ensure(f"{b}.softmax.monitor_1")  # 细粒 softmax 中的 lse（SDPA 旁路写入）
                ensure(f"{b}.monitor_3")  # attention probs  （SDPA 旁路写入）
                ensure(f"{b}.monitor_4")  # o_proj 输入（context = Attn@V）
                ensure(f"{b}.monitor_o")  # o_proj 输出
            # endregion

            # region --- MLP ---
            if getattr(layer, "mlp", None):
                b = f"layers.{i}.mlp"
                ensure(f"{b}.monitor_0")  # MLP 输入 x
                ensure(f"{b}.monitor_1g")  # gate_proj 输出
                ensure(f"{b}.monitor_1u")  # up_proj 输出
                ensure(f"{b}.monitor_2")  # down_proj 输入 = act(gate)*up
                ensure(f"{b}.monitor_3")  # （与 2 同点，对齐 ViT 命名）
                ensure(f"{b}.monitor_o")  # down_proj 输出
                # 处理激活函数 act（添加监控器）
                if hasattr(layer.mlp, 'act_fn'):
                    ensure(f"{b}.act.monitor_0")  # 激活函数输入
                    ensure(f"{b}.act.monitor_o")  # 激活函数输出
            # endregion
        # endregion

        # region ---- lm_head ----
        if hasattr(model, "lm_head"):
            ensure("lm_head.monitor_4")  # lm_head 输入（最后隐状态）
            ensure("lm_head.monitor_o")  # lm_head 输出（logits）
        # endregion

        # region ---- （可选）RMSNorm：仅当模型中真的有 RMSNorm 才 seed ----
        for name, mod in model.named_modules():
            if _is_rmsnorm(mod):
                ensure(f"{name}.monitor_0")  # 输入
                ensure(f"{name}.monitor_1")  # x_hat = x * inv_rms（未乘 weight）
                ensure(f"{name}.monitor_2")  # inv_rms
                ensure(f"{name}.monitor_o")  # 输出
        # endregion

    _seed_expected_monitors(model, monitors, MonitorCls)
    model.monitors = monitors

    # ========== 清理函数 ==========
    def teardown() -> None:
        """卸载所有 hooks，并恢复 SDPA 原实现。"""
        # sdpa_tap.uninstall()
        softmax_tap.uninstall()
        for h in hooks:
            try:
                h.remove()
            except Exception:
                pass
        hooks.clear()
        layer_stack.clear()
        del model.monitors

    return monitors, teardown


# endregion


# endregion

# region ======== 检查Monitor是否附加真确的工具 ========
class CountingMonitor(BaseMonitor):
    """
    测试/自检用监控器：
    - forward 透明；
    - 记录调用次数、最后一次张量的 shape/dtype、部分统计；
    - 便于在测试里断言。
    """

    def __init__(self):
        super().__init__()
        self.calls = 0
        self.last_shape = None
        self.last_dtype = None
        self.last_min = None
        self.last_max = None
        self.last_mean = None

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        self.calls += 1
        self.last_shape = tuple(x.shape)
        self.last_dtype = x.dtype
        # 轻量统计（避免太慢）
        if x.numel() > 0:
            self.last_min = float(x.min().item())
            self.last_max = float(x.max().item())
            self.last_mean = float(x.float().mean().item())
        return x

    def monitor_data(self):
        return dict(
            calls=self.calls,
            shape=self.last_shape,
            dtype=str(self.last_dtype),
            min=self.last_min,
            max=self.last_max,
            mean=self.last_mean,
        )


def expected_keys(model) -> List[str]:
    r"""
    根据当前挂载策略，枚举所有应存在的 monitor 键（用于覆盖度自检）：

      - Embedding:
          embed_tokens.monitor_0 / monitor_1 / monitor_o
      - SelfAttention（仅 LlamaAttention）:
          layers.{i}.self_attn.monitor_{0,1,2,3,4,o}
            * 2 = scores（softmax 前），3 = probs（softmax 后），4 = context（o_proj 前）
      - MLP:
          layers.{i}.mlp.monitor_{0,1,2,3,o}
      - SiLU（当激活为 nn.SiLU 时，可能存在）:
          layers.{i}.mlp.act.monitor_0 / layers.{i}.mlp.act.monitor_o
      - RMSNorm（全模型扫描，模型中存在才纳入）:
          {name}.monitor_0 / {name}.monitor_o
      - lm_head:
          lm_head.monitor_4 / lm_head.monitor_o
      - 模型级:
          model.monitor_{0,1,2,3}
    """

    keys: List[str] = []

    core = getattr(model, "model", model)
    layers = getattr(core, "layers", [])

    # ---- Embedding ----
    if hasattr(core, "embed_tokens"):
        keys += [
            "embed_tokens.monitor_0",
            "embed_tokens.monitor_1",
            "embed_tokens.monitor_o",
        ]

    # ---- 每层 Attention----
    for i, layer in enumerate(layers):
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue
        # if attn.__class__.__name__ != "LlamaSdpaAttention":
        # 不是 SDPA 的注意力实现则跳过（没挂对应 hooks）
        # continue
        b = f"layers.{i}.self_attn"
        keys += [
            f"{b}.monitor_0",
            f"{b}.monitor_1",
            f"{b}.monitor_2",  # scores
            f"{b}.softmax.monitor_1",  # 细粒 softmax 中的 lse
            f"{b}.monitor_3",  # probs
            f"{b}.monitor_4",  # context (o_proj 前)
            f"{b}.monitor_o",  # o_proj 后
        ]

    # ---- 每层 MLP ----
    for i, layer in enumerate(layers):
        if not hasattr(layer, "mlp"):
            continue
        b = f"layers.{i}.mlp"
        keys += [
            f"{b}.monitor_0",
            f"{b}.monitor_1g",
            f"{b}.monitor_1u",
            f"{b}.monitor_2",
            f"{b}.monitor_3",
            f"{b}.monitor_o",
        ]

    # ---- RMSNorm：全模型扫描，若存在则列出 0/1/2/o ----
    for name, mod in model.named_modules():
        if _is_rmsnorm(mod):
            keys += [
                f"{name}.monitor_0",  # 输入
                f"{name}.monitor_1",  # x_hat = x * inv_rms（未乘weight）
                f"{name}.monitor_2",  # inv_rms
                f"{name}.monitor_o",  # 输出
            ]

    # ---- lm_head ----
    if hasattr(model, "lm_head"):
        keys += ["lm_head.monitor_4", "lm_head.monitor_o"]

    # ---- SiLU ----
    for name, mod in model.named_modules():
        if isinstance(mod, nn.SiLU):
            keys += [f"{name}.monitor_0", f"{name}.monitor_o"]

    return keys


@torch.no_grad()
def self_test_monitors(model: LlamaForCausalLM,
                       attach_fn: Callable,
                       tokenizer,
                       text: str = "hello",
                       atol: float = 1e-5,
                       rtol: float = 1e-5):
    r"""
    自检流程（轻量版）：
      1) 用 CountingMonitor 挂监控；
      2) 前传一次收集 monitors；
      3) teardown 卸掉 hooks / 恢复 SDPA 后再前传一次，对比 logits 等价；
      4) 检查 key 覆盖、调用次数，以及各模块的形状是否符合约定。

    本函数进行的检查：

    【输出等价性（透明性）】
      - 挂监控前后各前传一次，比较 logits 是否 allclose。
      - 证明 monitor 透明（forward 不改值；hooks 返回 None）。

    【覆盖度】
      - expected_keys(model) 枚举应存在的监控键：
          Embedding: 0/1/o
          SelfAttention（LlamaAttention）: 0/1/2/3/4/o
          MLP（LlamaMLP）: 0/1/2/3/o
          RMSNorm（若存在）: 每处 0/o
          模型级: model.monitor_0/1/2/3
          lm_head: 4/o
      - 比较 expected vs 实际 monitors.keys()：
          missing=0 → 无漏挂； extra=0 → 无多挂。
      - 表示命名与布局对齐 ViT 风格。

    【调用与形状（全层扫描，轻量）】
      - 每个 expected 键至少被调用 1 次。
      - 形状断言（仅读取 last_shape）：
          Embedding:          monitor_1 / monitor_o → [B, T, H]
          Attention:
              monitor_2（scores） → [B, nH, Q, K]
              monitor_3（probs）  → [B, nH, Q, K]
              monitor_4（context）→ [B, T, H]
          MLP:
              monitor_0 → [B, T, H]
              monitor_1 → [B, T, I]
              monitor_2/3（down_proj 输入）→ [B, T, I]
              monitor_o（down_proj 输出）  → [B, T, H]
          RMSNorm（若存在）:
              {name}.monitor_0 / {name}.monitor_o → [B, T, H]
          模型级:
              model.monitor_1 / model.monitor_3 → [B, T, H]
          lm_head:
              monitor_4 → [B, T, H]
              monitor_o → [B, T, V]
          SiLU（若挂）:
              layers.{i}.mlp.act.monitor_0 / monitor_o → [B, T, I]

    备注：
      - 注意力概率归一（probs 在 K 维上求和≈1）未在轻量版中严格检查（CountingMonitor 不缓存张量）。
        若需启用，可换用会缓存小切片的 Monitor，并在循环中断言。
      - ViT 规则映射：
          PatchEmbed ↔ embed_tokens
          Attention(Block) ↔ LlamaAttention
          Mlp ↔ LlamaMLP
          Head ↔ lm_head
        expected 与 got 一致 + 形状正确，即视为与 ViT 规则一一对应。

    :param model:       LlamaForCausalLM（或包含 .model.layers 的兼容模型）
    :param attach_fn:   安装监控的函数（返回 monitors, teardown）
    :param tokenizer:   与 model 匹配的分词器
    :param text:        自检用的最小输入文本
    :param atol:        logits 比对的绝对误差阈值
    :param rtol:        logits 比对的相对误差阈值
    :return:            None（断言失败会抛错；成功则打印通过信息）
    """
    # 1. attach
    monitors, teardown = attach_fn(model, logger_name="selftest", MonitorCls=CountingMonitor)

    # 2. run once with monitors
    inputs = tokenizer(text, return_tensors="pt").to(next(model.parameters()).device)
    out1 = model(**inputs)
    logits1 = out1.logits if hasattr(out1, "logits") else out1[0]

    # 3. teardown and run again
    teardown()
    out2 = model(**inputs)
    logits2 = out2.logits if hasattr(out2, "logits") else out2[0]

    # 不干扰性
    ok_equal = torch.allclose(logits1, logits2, atol=atol, rtol=rtol)
    print(f"[OK] outputs equal (allclose={ok_equal}, atol={atol}, rtol={rtol})")
    assert ok_equal, "监控器不透明：前后 logits 不相等"

    # 覆盖度
    exp = set(expected_keys(model))
    got = set(monitors.keys())
    missing = sorted(list(exp - got))
    extra = sorted(list(got - exp))
    print(f"[COVER] expected={len(exp)}, got={len(got)}, missing={len(missing)}, extra={len(extra)}")
    if missing:
        print("  Missing keys:", missing)
    # 只提示 extra，不一定是错（例如临时加的别名）
    if extra:
        print("  Extra keys:", extra)

    # 调用次数（至少 1 次）
    for k in sorted(exp):
        m = monitors.get(k)
        assert m is not None, f"缺少监控器：{k}"
        assert getattr(m, "calls", 0) >= 1, f"监控器未被调用：{k}"

    # 形状与概率测试（轻量全层版：只读 last_shape，不触碰张量）
    core = getattr(model, "model", model)
    B, T = inputs["input_ids"].shape
    H = core.embed_tokens.embedding_dim if hasattr(core, "embed_tokens") else getattr(model.config, "hidden_size", None)
    nH = getattr(model.config, "num_attention_heads", None)
    no_cache = not inputs.get("past_key_values", None) and not inputs.get("use_cache", False)

    def _shape(k):
        return getattr(monitors[k], "last_shape", None)

    # ---- 模型级别：model.monitor_{0,1,2,3} ----
    # monitor_0 来自 forward 的第一个位置参数（一般是 input_ids: [B,T]），所以仅检查前两维
    assert _shape("model.monitor_0")[0:2] == (B, T), f"model.monitor_0 形状异常: {_shape('model.monitor_0')}"
    # 1/2/3 都是隐藏状态 [B,T,H]
    for k in ("model.monitor_1", "model.monitor_2", "model.monitor_3"):
        s = _shape(k)
        assert isinstance(s, tuple) and s[0:2] == (B, T), f"{k} 形状异常: {s}"
        if H is not None:
            assert s[2] == H, f"{k} hidden_size 不匹配: {s[2]} vs {H}"

    # ---- Embedding ----
    assert _shape("embed_tokens.monitor_1")[0:2] == (B, T), "embed_tokens.monitor_1 形状应为 [B, T, H] 开头"
    if H is not None:
        assert _shape("embed_tokens.monitor_1")[2] == H, "embed_tokens.monitor_1 hidden_size 不匹配"

    # ---- 遍历全部层：Attention / MLP ----
    num_layers = len(getattr(core, "layers", []))
    for i in range(num_layers):
        attn_base = f"layers.{i}.self_attn"
        mlp_base = f"layers.{i}.mlp"

        # Attention（仅当这一层有挂 monitor 时才检查）
        if f"{attn_base}.monitor_2" in monitors:
            s2 = _shape(f"{attn_base}.monitor_2")  # scores: [B, nH, Q, K]
            s3 = _shape(f"{attn_base}.monitor_3")  # probs : [B, nH, Q, K]
            s4 = _shape(f"{attn_base}.monitor_4")  # ctx   : [B, T, H]

            # 检查 monitor_2 (scores)
            assert isinstance(s2, tuple) and len(s2) == 4 and s2[0] == B, f"{attn_base}.monitor_2 形状异常: {s2}"
            assert isinstance(s3, tuple) and len(s3) == 4 and s3[0] == B, f"{attn_base}.monitor_3 形状异常: {s3}"
            if nH is not None:
                assert s2[1] == nH and s3[1] == nH, f"{attn_base} 头数不匹配: s2={s2}, s3={s3}, nH={nH}"
            if no_cache:
                assert s2[2] == T and s2[3] == T, f"{attn_base}.monitor_2 Q/K 长度异常: {s2} vs T={T}"
                assert s3[2] == T and s3[3] == T, f"{attn_base}.monitor_3 Q/K 长度异常: {s3} vs T={T}"

            lse_key = f"{attn_base}.softmax.monitor_1"
            if lse_key in monitors:
                s_lse = _shape(lse_key)
                # 允许 [B, H, Q, 1] 或 [B, H, Q, 1]-like（某些实现 squeeze 不彻底）
                assert isinstance(s_lse, tuple) and len(s_lse) == 4, f"{lse_key} 形状异常: {s_lse}"
                assert s_lse[0] == B, f"{lse_key} batch 不匹配: {s_lse[0]} vs {B}"
                if nH is not None:
                    assert s_lse[1] == nH, f"{lse_key} 头数不匹配: {s_lse[1]} vs {nH}"
                if no_cache:
                    assert s_lse[2] == T, f"{lse_key} Q长度异常(无cache应等于T): {s_lse[2]} vs {T}"
                assert s_lse[3] == 1, f"{lse_key} 最后一维应为1（keepdim=True）: {s_lse}"

        # MLP（如有）
        if f"{mlp_base}.monitor_0" in monitors:
            s0 = _shape(f"{mlp_base}.monitor_0")  # [B, T, H]
            s1g = _shape(f"{mlp_base}.monitor_1g")  # [B, T, I]
            s1u = _shape(f"{mlp_base}.monitor_1u")  # [B, T, I]
            s2 = _shape(f"{mlp_base}.monitor_2")  # [B, T, I]
            s3 = _shape(f"{mlp_base}.monitor_3")  # [B, T, I]
            so = _shape(f"{mlp_base}.monitor_o")  # [B, T, H]

            assert isinstance(s0, tuple) and s0[0:2] == (B, T), f"{mlp_base}.monitor_0 形状异常: {s0}"
            assert isinstance(s1g, tuple) and s1g[0:2] == (B, T), f"{mlp_base}.monitor_1g 形状异常: {s1g}"
            assert isinstance(s1u, tuple) and s1u[0:2] == (B, T), f"{mlp_base}.monitor_1u 形状异常: {s1u}"
            assert isinstance(s2, tuple) and s2[0:2] == (B, T), f"{mlp_base}.monitor_2 形状异常: {s2}"
            assert isinstance(s3, tuple) and s3[0:2] == (B, T), f"{mlp_base}.monitor_3 形状异常: {s3}"
            assert isinstance(so, tuple) and so[0:2] == (B, T), f"{mlp_base}.monitor_o 形状异常: {so}"
            if H is not None:
                assert s0[2] == H and so[2] == H, f"{mlp_base}: H 不匹配(s0={s0[2]}, so={so[2]}, H={H})"

    # ---- lm_head ----
    if hasattr(model, "lm_head"):
        s4 = _shape("lm_head.monitor_4")  # [B, T, H]
        so = _shape("lm_head.monitor_o")  # [B, T, V]
        assert isinstance(s4, tuple) and s4[0:2] == (B, T), f"lm_head.monitor_4 形状异常: {s4}"
        if H is not None:
            assert s4[2] == H, f"lm_head.monitor_4 hidden_size 不匹配: {s4[2]} vs {H}"
        assert isinstance(so, tuple) and so[0:2] == (B, T), f"lm_head.monitor_o 形状异常: {so}"

    # ---- RMSNorm（有则检；支持 0/1/2/o）----
    # 注意：inv_rms 的 monitor_2 可能是 [B,T,1] 或 [B,T,H]（广播形状），两者都接受
    for name, mod in model.named_modules():
        if _is_rmsnorm(mod):
            k0 = f"{name}.monitor_0"
            k1 = f"{name}.monitor_1"
            k2 = f"{name}.monitor_2"
            ko = f"{name}.monitor_o"

            if k0 in monitors:
                s0 = _shape(k0)
                assert isinstance(s0, tuple) and s0[0:2] == (B, T), f"{k0} 形状异常: {s0}"
                if H is not None:
                    assert s0[2] == H, f"{k0} hidden_size 不匹配: {s0[2]} vs {H}"

            if k1 in monitors:
                s1 = _shape(k1)
                assert isinstance(s1, tuple) and s1[0:2] == (B, T), f"{k1} 形状异常: {s1}"
                if H is not None:
                    assert s1[2] == H, f"{k1} hidden_size 不匹配: {s1[2]} vs {H}"

            if k2 in monitors:
                s2 = _shape(k2)
                assert isinstance(s2, tuple) and s2[0:2] == (B, T), f"{k2} 形状异常: {s2}"
                if H is not None:
                    assert s2[2] in (1, H), f"{k2} 最后一维应为 1 或 H，当前 {s2[2]}"

            if ko in monitors:
                so = _shape(ko)
                assert isinstance(so, tuple) and so[0:2] == (B, T), f"{ko} 形状异常: {so}"
                if H is not None:
                    assert so[2] == H, f"{ko} hidden_size 不匹配: {so[2]} vs {H}"

    print("[DONE] monitors self-test passed.")


# endregion

def get_monitor_data(model: LlamaForCausalLM) -> dict[str, Any]:
    r"""
    收集监控器中的数据。
    :param model: 一定是经过 attach_monitors_llama_sdpa 函数注册完 monitors 之后的 Llama模型，因为注册之后，才有monitors成员。
    :type model: nn.Module
    :return:
    :rtype: dict[str, Any]
    """
    data = {}
    # 遍历模型的所有模块，找出是 BaseMonitor 类型的模块
    for name, child in model.monitors.items():
        # 收集该监控器的数据
        info = child.monitor_data()
        if info is not None:
            data[name] = info
    return data


if __name__ == "__main__":
    model_path = "/ephemeral/Models/Llama-3-8b"
    # 1) 加载模型
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 torch_dtype=torch.float16)
    model.eval()

    # 2) 挂监控（可把 MonitorCls 换成你自己的子类，比如 MinMaxMonitor）
    monitors, teardown = attach_monitors_llama_sdpa(model, logger_name="llama_mon", MonitorCls=BaseMonitor)

    # 3) 前传一次，hooks 会自动调用各监控点
    tok = AutoTokenizer.from_pretrained(model_path)
    inputs = tok("你好，世界", return_tensors="pt")
    with torch.no_grad():
        _ = model(**inputs)

    # 4) 取监控器对象，调用子类里的 monitor_data() 导出统计
    #    例如：monitors["layers.0.self_attn.monitor_2"].monitor_data()

    # 5) 完成后清理
    teardown()

    self_test_monitors(model,
                       attach_monitors_llama_sdpa,
                       tok
                       )
