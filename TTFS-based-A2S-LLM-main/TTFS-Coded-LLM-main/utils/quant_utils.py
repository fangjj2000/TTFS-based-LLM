import functools
import torch.nn as nn
from argparse import Namespace
from pprint import pformat
from torch import Tensor
from tqdm import tqdm
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from typing import Dict

import quantize.int_linear_fake as int_linear_fake
import utils.rotation_utils as rotation_utils
from quantize.int_linear_fake import *
from quantize.quantizer import UniformAffineQuantizer
from utils.model_utils import get_kv_cache
from utils.rotation_utils import QKRotationWrapper


def get_act_stat(model, dataloader, accumulate_type='max', prefixed_tokens=None, online_had=False):
    model.eval()
    num_heads = model.config.num_attention_heads
    num_kv_heads = model.config.num_key_value_heads
    model_dim = model.config.hidden_size
    head_dim = model_dim // num_heads
    kv_dim = num_kv_heads * head_dim
    device = next(model.parameters()).device
    act_stat = {}
    prefixed_length = len(prefixed_tokens) if prefixed_tokens is not None else 0

    if online_had:
        had_K, K = hadamard_utils.get_hadK(model.config.intermediate_size)

    def stat_tensor(name, tensor, type):
        hidden_dim = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim).abs().detach()
        ema_factor = 0.99
        if accumulate_type == 'max':
            comming_max = torch.max(tensor, dim=0)[0].float().cpu()
        elif accumulate_type == 'mean':
            comming_max = torch.mean(tensor, dim=0).float().cpu()
        key_name = f"{name}.{type}"
        if key_name in act_stat:
            act_stat[key_name] = ema_factor * act_stat[key_name] + (1 - ema_factor) * comming_max
        else:
            act_stat[key_name] = comming_max

    def stat_input_hook(m, x, y, name):
        if 'apply_rotary_pos_emb_qk_rotation_wrapper' in name:
            input_Q = x[0].transpose(1, 2).flatten(-2)
            input_K = x[1].transpose(1, 2).flatten(-2)
            output_Q = y[0].transpose(1, 2).flatten(-2)
            output_K = y[1].transpose(1, 2).flatten(-2)
            if prefixed_length > 0:
                input_Q = input_Q[:, prefixed_length:,]
                input_K = input_K[:, prefixed_length:,]
                output_Q = output_Q[:, prefixed_length:,]
                output_K = output_K[:, prefixed_length:,]
            stat_tensor(name, input_Q, 'input_Q')
            stat_tensor(name, input_K, 'input_K')
            stat_tensor(name, output_Q, 'output_Q')
            stat_tensor(name, output_K, 'output_K')
        else:
            if isinstance(x, tuple):
                x = x[0]
            if prefixed_length > 0:
                x_ = x[:, prefixed_length:,]
                y_ = y[:, prefixed_length:,]
            else:
                x_, y_ = x, y
            if online_had and 'down_proj' in name:
                x_ = hadamard_utils.matmul_hadU_cuda(x_, had_K, K)
            stat_tensor(name, x_, 'input')
            stat_tensor(name, y_, 'output')

    hooks = []
    for name, m in model.named_modules():
        # if isinstance(m, nn.Linear):
        if isinstance(m, (nn.Linear, LlamaRMSNorm, RMSN,
                          QKRotationWrapper,
                          QuantizedLlamaRMSNorm, QuantizedIdentity,
                          QuantizedSiLU, QuantizedEmbedding,
                          QuantizedLinear)):
            hooks.append(
                m.register_forward_hook(
                    functools.partial(stat_input_hook, name=name)))

    for i in tqdm(range(len(dataloader)), desc='obtain activation stat'):
        data = dataloader[i][0]
        if prefixed_tokens is not None:
            data = torch.cat([torch.tensor([prefixed_tokens]), data], dim=1)
        model(data.to(device))

    for h in hooks:
        h.remove()

    return act_stat


def wrap_to_quant_model(model: LlamaForCausalLM,
                        args: Namespace,
                        nameLogger: str) -> None:
    """
    将部分模块替换为量化版本：
      - 跳过：lm_head、model.norm
      - 替换：embed_tokens → QuantizedEmbedding
              LlamaSdpaAttention → QuantizedLlamaAttention(attachIdentity=True)
              LlamaMLP → QuantizedLlamaMLP
              (LlamaRMSNorm | RMSN) → QuantizedLlamaRMSNorm
    :param model
    :type model: LlamaForCausalLM
    :param args
    :type args: Namespace
    """

    logger = logging.getLogger(nameLogger)

    SKIP_NAME_SUBSTR = ("lm_head", "model.norm")

    def _should_skip(name: str) -> bool:
        return any(s in name for s in SKIP_NAME_SUBSTR)

    def _swap(name: str, new_module: nn.Module, why: str):
        set_op_by_name(model, name, new_module)
        logger.info(f"✅ 已替换 {name}（{why}）→\n  {new_module} \n  类型：{type(new_module)}\n")

    # 类型→构造器的映射（除了 embed_tokens 层用名字判定）
    TYPE_MAP = (
        # (LlamaSdpaAttention, lambda m: QuantizedLlamaAttention(org_attn=m, nameLogger=nameLogger, attachIdentity=True),
        #  "LlamaSdpaAttention → QuantizedLlamaAttention"),
        # (nn.SiLU, lambda m: QuantizedSiLU(org_silu=m), "nn.SiLU → QuantizedSiLU"),

        #  # LlamaMLP 中的 Linear 层与 SiLU 算子，已经通过下面的 Linear，SiLU 的全局替换实现更改。
        # (LlamaMLP, lambda m: QuantizedLlamaMLP(org_mlp=m), "LlamaMLP → QuantizedLlamaMLP"),

        (nn.Linear, lambda m: QuantizedLinear(org_linear=m), "nn.Linear → QuantizedLinear"),
        ((LlamaRMSNorm, RMSN), lambda m: QuantizedLlamaRMSNorm(org_norm=m), "LlamaRMSNorm → QuantizedLlamaRMSNorm"),
        ((Qwen2RMSNorm, RMSN), lambda m: QuantizedQwen2RMSNorm(org_norm=m), "QwenRMSNorm → QuantizedQwenRMSNorm"),
        ((MistralRMSNorm, RMSN), lambda m: QuantizedMistralRMSNorm(org_norm=m), "MistralRMSNorm → QuantizedMistralRMSNorm"),
    )

    for name, module in model.named_modules():
        # 通用的日志记录设置为 debug，避免刷屏。。。
        logger.debug(f"\n▶️ 模块 {name}：\n实例 - {pformat(module)}\n类型 - {type(module)}")

        # 1) 跳过的层
        if _should_skip(name):
            logger.info(f"❎ 跳过 {name}（保持原始模块）：{type(module)}\n")
            continue

        # 2) embed_tokens 用名字匹配
        if name.endswith("embed_tokens") or ".embed_tokens" in name:
            continue  # PrefixQuant 中没有对输入嵌入层进行量化。
            new_m = QuantizedEmbedding(module)
            _swap(name, new_m, "Embedding 量化")
            continue

        # 3) 其余层走映射
        replaced = False
        for typ, builder, msg in TYPE_MAP:
            if isinstance(module, typ):
                new_m = builder(module)
                _swap(name, new_m, msg)
                replaced = True
                break

        if not replaced:
            logger.debug("⭕ 无需替换，跳过。\n")


def register_online_had(model):
    for name, module in model.named_modules():
        if isinstance(module, int_linear_fake.QuantizedLinear) and 'down_proj' in name:
            had_K, K = hadamard_utils.get_hadK(model.config.intermediate_size)
            module.online_full_had = True
            module.had_K = had_K
            module.K = K
            module.fp32_had = False


# region ==== 初始化 【量化器】 的工具函数 ====


def init_weight_quantizer(args: Namespace,
                          nameLogger: str,
                          model: LlamaForCausalLM,
                          minmax_init: bool = True) -> None:
    """
    初始化 QuantizedLinear 的【权重量化】（weight quantizer）。
    说明：
      - 仅对 QuantizedLinear 生效；
      - scale/zero_point 在 minmax_init=True 时用权重本身统计初始化；
      - group_size=-1 时自动等于 out_features 的列宽（weight.shape[-1]）。
    :param args:
    :type args: Namespace
    :param nameLogger:
    :type nameLogger: str
    :param model:
    :type model: LlamaForCausalLM
    :param minmax_init:
    :type minmax_init: bool
    :return:
    """
    logger = logging.getLogger(nameLogger)

    def _normalize_group_size(weight_shape: torch.Size,
                              gs: Optional[int]) -> int:
        r"""

        :param weight_shape:
        :type weight_shape: torch.Size
        :param gs:
        :type gs: int
        :return:
        :rtype: int
        """
        # 避免依赖类内部对 -1 维度的特殊处理
        if gs is None or gs <= 0 or gs == -1:
            return weight_shape[-1]
        return gs

    def _apply_weight_quant(name: str,
                            module: int_linear_fake.QuantizedLinear):
        r"""

        :param name:
        :type name: str
        :param module:
        :type module: LlamaForCausalLM
        :return:
        """
        # 统一 bit
        if getattr(module, "wbits", None) != args.wbits:
            logger.info(f"✅ 将原模块的权重量化 bit 数 {getattr(module, 'wbits', 'N/A')} 变为 {args.wbits}")
            module.wbits = args.wbits

        module.use_weight_quant = True

        # 量化形状与 group_size
        wshape = module.weight.shape  # nn.Linear 的权重 - (out_features, in_features)
        group_size = _normalize_group_size(wshape, args.w_group_size)

        # 统计初始化
        quantized_item_stat = module.weight if minmax_init else None

        module.weight_quantizer = UniformAffineQuantizer(
            n_bits=args.wbits,
            quantized_shape=wshape,
            asym=args.w_asym,
            group_size=group_size,
            quantized_item_stat=quantized_item_stat,
            quant_type="weight",
            minmax_init=minmax_init,
        )

        sym_stat = "asymmetric" if args.w_asym else "symmetric"
        zp = module.weight_quantizer.zero_point
        zp_shape = None if zp is None else tuple(zp.shape)

        logger.info(
            f"✅ 权重量化：将 {name} 量化为 "
            f"{args.wbits}-bit, group_size={group_size}, {sym_stat};"
            f"\n   scale.shape={tuple(module.weight_quantizer.scale.shape)}, zero_point.shape={zp_shape}\n"
        )

    for name, module in model.named_modules():
        logger.debug(f"\n▶️ 当前处理模块 - {name}：\n实例 - {pformat(module)}\n类型 - {type(module)}")
        if not isinstance(module, int_linear_fake.QuantizedLinear):
            logger.debug("⭕ 非 QuantizedLinear，跳过。\n")
            continue
        _apply_weight_quant(name, module)


def init_output_act_quantizer(args: Namespace,
                              nameLogger: str,
                              model: LlamaForCausalLM,
                              activation_stat: Optional[Dict[str, torch.Tensor]] = None,
                              minmax_init: bool = True) -> None:
    """
    初始化【线性层】、【RMSNorm 层】、【嵌入层】的输出量化器：
      1) 仅对白名单中的 Linear/Embedding 做“输出量化”，不做输入量化；
      2) RMSNorm 全部做输出量化；
      3) q/k/v 及 lm_head 不在白名单内，自然不会被量化。
      4) 新增：对 mlp.act_fn（如 QuantizedSiLU/QuantizedActivation）做输出量化；
    :param args:
    :type args: Namespace
    :param nameLogger:
    :type nameLogger: str
    :param model:
    :type model: LlamaForCausalLM
    :param activation_stat:
    :type activation_stat: Tensor
    :param minmax_init:
    :type minmax_init: bool
    :return:
    """
    logger = logging.getLogger(nameLogger)

    # 哪些“层名”需要对输出做量化（取模块名最后一段）
    ALLOW_LINEAR = {
        # "gate_proj",
        "up_proj",
        "down_proj",
        "o_proj"
    }  # 不包含 embed_tokens
    ALLOW_EMBED = {"embed_tokens"}  # 整个 model 的 Embedding 层
    ALLOW_ACT = {"act_fn"}  # 主要用于 mlp.act_fn

    def _find_module_by_name(root: nn.Module,
                             dotted: str) -> Optional[nn.Module]:
        cur = root
        for seg in dotted.split("."):
            if not hasattr(cur, seg):
                return None
            cur = getattr(cur, seg)
        return cur

    def _infer_out_features(name: str, module: nn.Module, out_stat: Optional[torch.Tensor]) -> int:
        # 模块自带 out_features的话，就直接用。 除了 QuantizedSiLU，別的Quantized* 都定义了 out_features
        v = getattr(module, "out_features", None)
        if isinstance(v, int) and v > 0:
            return v

        # activation_stat 里有输出统计，直接用 shape[-1]
        if isinstance(out_stat, torch.Tensor) and out_stat.ndim >= 1:
            return int(out_stat.shape[-1])

        # RMSNorm 可从 weight 推断
        w = getattr(module, "weight", None)
        if isinstance(w, torch.Tensor) and w.ndim >= 1:
            return int(w.shape[-1])

        # 针对 act_fn：从同层 gate_proj 读取
        leaf = name.split(".")[-1]
        if leaf == "act_fn":
            # act_fn 的输出维度 = gate_proj 的输出维度
            # 因为 act_fn 只是逐元素激活，不会改变张量形状，
            # 它的输入来自 gate_proj(x)，所以 out_features = gate_proj.out_features
            gate_name = name.rsplit(".", 1)[0] + ".gate_proj"
            gate_mod = _find_module_by_name(model, gate_name)
            if gate_mod is not None:
                ov = getattr(gate_mod, "out_features", None)
                if isinstance(ov, int) and ov > 0:
                    return ov
            # 兜底：用模型配置的 intermediate_size
            if hasattr(model, "config") and getattr(model.config, "intermediate_size", None):
                return int(model.config.intermediate_size)

        # 用 embedding_dim
        emb_dim = getattr(module, "embedding_dim", None)
        if isinstance(emb_dim, int) and emb_dim > 0:
            return emb_dim

        # 其它可能的表示特征维度的字段
        for attr in ("num_features", "features"):
            vv = getattr(module, attr, None)
            if isinstance(vv, int) and vv > 0:
                return vv

        raise ValueError(
            f"Cannot infer out_features for '{name}' ({type(module)}). "
            f"Consider setting module.out_features at wrap time."
        )

    def _should_quantize(name: str,
                         module: nn.Module) -> bool:
        r"""

        :param name:
        :type name: str
        :param module:
        :type module: nn.Module
        :return:
        :rtype: bool
        """
        leaf = name.split(".")[-1]
        if isinstance(module, int_linear_fake.QuantizedLinear):
            return leaf in ALLOW_LINEAR
        if isinstance(module, int_linear_fake.QuantizedEmbedding):
            return leaf in ALLOW_EMBED
        if isinstance(module, int_linear_fake.QuantizedLlamaRMSNorm):
            return True
        if isinstance(module, int_linear_fake.QuantizedSiLU):
            return leaf in ALLOW_ACT
        return False

    def _apply_output_quant(name: str,
                            module: nn.Module) -> None:
        r"""

        :param name:
        :type name: str
        :param module:
        :type module: nn.Module
        :return:
        """
        # 统计（可能为 None）
        out_stat = activation_stat.get(f"{name}.output") if activation_stat is not None else None

        # 开启“输出量化”，设置 bit。模块内的默认 bit 数为 16，是将量化关闭的。
        if getattr(module, "output_bits", None) != args.output_bits:
            logger.info(f"✅ 将原模块的输出 bit 数 {getattr(module, 'output_bits', 'N/A')} 变为 {args.output_bits}")
            module.output_bits = args.output_bits

        module.use_act_quant = True

        # 量化形状：(1, out_features)
        # 假设 Quantized* 模块里都有 out_features，所以这里可以直接用；
        # 即使有的 nn 模块，没有out_features, 也已经从 weight / embedding_dim 推断了。
        # _infer_out_features 中单独针对 量化silu 层 进行了输出特征数的推断，利用了 gate_proj 的输出特征维度。
        out_features = _infer_out_features(name, module, out_stat)
        quantized_shape = (1, out_features)

        module.output_quantizer = UniformAffineQuantizer(
            n_bits=args.output_bits,
            quantized_shape=quantized_shape,
            asym=args.output_asym,
            group_size=args.output_group_size,
            quantized_item_stat=out_stat,  # 用激活统计初始化
            quant_type="activation",
            mode=args.output_mode,
            activation_clipping=args.activation_clipping,
            minmax_init=minmax_init,
        )

        sym_stat = "asymmetric" if args.output_asym else "symmetric"
        # 打印核心摘要
        zp = module.output_quantizer.zero_point
        zp_shape = None if zp is None else tuple(zp.shape)
        logger.info(
            f"✅ 输出激活量化：将 {name} 量化为 "
            f"{args.output_bits}-bit, group_size={args.output_group_size}, mode={args.output_mode}, {sym_stat}；"
            f"\n scale 的形状为 {tuple(module.output_quantizer.scale.shape)}, zero_point 的形状为 {zp_shape}\n"
        )

    for name, module in model.named_modules():
        # 这里的日志记录级别设置为 debug，避免刷屏。。。
        logger.debug(f"\n▶️ 当前处理模块 - {name}：\n实例 - {pformat(module)}\n类型 - {type(module)}")

        if not isinstance(
                module,
                (int_linear_fake.QuantizedLinear,
                 int_linear_fake.QuantizedLlamaRMSNorm,
                 # int_linear_fake.QuantizedEmbedding,
                 int_linear_fake.QuantizedSiLU
                 ),
        ):
            logger.debug("⭕ 非目标模块，跳过。\n")
            continue

        if _should_quantize(name, module):
            _apply_output_quant(name, module)
        else:
            logger.debug("⭕ 不在白名单/无需输出量化，跳过。\n")


def init_identity_quantizer(args,
                            nameLogger: str,
                            model: nn.Module,
                            activation_stat: Optional[Dict[str, torch.Tensor]] = None,
                            minmax_init: bool = True) -> None:
    """
    初始化所有 QuantizedIdentity 的输出量化器。
    规则：
      - 对名字中包含 "identitySoftmax" 或 "identityQ" 的模块，使用 output_group_size；
      - 对名字中包含 "identityK" 或 "identityV" 的模块，使用 kv_group_size；
      - 其余名称（若命中 QuantizedIdentity 却不在上述集合），视为配置错误并报错。
    :param args:
    :type args:Namespace
    :param nameLogger:
    :type nameLogger:str
    :param model:
    :type model:
    :param activation_stat:
    :type activation_stat:
    :param minmax_init:
    :type minmax_init:
    :return:
    """
    logger = logging.getLogger(nameLogger)

    def _pick_group_size(module_name: str) -> int:
        r"""

        :param module_name:
        :type module_name: str
        :return:
        :rtype: int
        """
        if ("identitySoftmax" in module_name) or ("identityQ" in module_name):
            return args.output_group_size
        if ("identityK" in module_name) or ("identityV" in module_name):
            return args.kv_group_size
        raise NotImplementedError(
            f"为 {module_name} 选择 group_size 失败：期望名称包含 "
            f"{{identitySoftmax, identityQ}} 或 {{identityK, identityV}}。"
        )

    def _apply_identity_output_quant(name: str,
                                     module: int_linear_fake.QuantizedIdentity) -> None:
        r"""

        :param name:
        :type name: str
        :param module:
        :type module: nn.Module
        :return:
        :rtype:
        """
        group_size = _pick_group_size(name)
        out_stat = activation_stat.get(f"{name}.output") if activation_stat is not None else None

        # 将 output_bits 统一为 args.output_bits
        if getattr(module, "output_bits", None) != args.output_bits:
            logger.info(
                f"✅ 将原模块 {name} 的输出 bit 数 {getattr(module, 'output_bits', 'N/A')} 变为 {args.output_bits}")
            module.output_bits = args.output_bits

        module.use_act_quant = True

        # QuantizedIdentity 已有 num_features，这里直接使用
        quantized_shape = (1, module.num_features)

        module.output_quantizer = UniformAffineQuantizer(
            n_bits=args.output_bits,
            quantized_shape=quantized_shape,
            asym=args.output_asym,
            group_size=group_size,
            quantized_item_stat=out_stat,  # 若有激活统计则用其做 minmax_init
            quant_type="activation",
            mode=args.output_mode,
            minmax_init=minmax_init,
            activation_clipping=args.activation_clipping,
        )

        sym_stat = "asymmetric" if args.output_asym else "symmetric"
        zp = module.output_quantizer.zero_point
        zp_shape = None if zp is None else tuple(zp.shape)

        logger.info(
            f"✅ 恒等量化：将 {name} 量化为 -> "
            f"{args.output_bits}-bit, group_size={group_size}, mode={args.output_mode}, {sym_stat};"
            f"\n  scale 的形状为 {tuple(module.output_quantizer.scale.shape)}, zero_point 的形状为 {zp_shape}\n"
        )

    for name, module in model.named_modules():
        logger.debug(f"\n▶️ 当前处理模块 - {name}：\n实例 - {pformat(module)}\n类型 - {type(module)}")

        if not isinstance(module, int_linear_fake.QuantizedIdentity):
            logger.debug("⭕ 非 QuantizedIdentity，跳过。\n")
            continue

        _apply_identity_output_quant(name, module)


def init_intput_act_quantizer(args: Namespace,
                              nameLogger: str,
                              model: LlamaForCausalLM,
                              activation_stat: Optional[Dict[str, torch.Tensor]] = None,
                              minmax_init: bool = True) -> None:
    """
    仅为指定层启用【输入激活量化】：
      - self_attn.v_proj  使用 KV 配置（v_bits/kv_group_size/kv_asym/kv_mode）
      - mlp.down_proj     使用 input_* 配置
      - self_attn.o_proj  使用 input_* 配置（按你的诉求已启用）
    :param args:
    :type args: Namespace
    :param nameLogger:
    :type nameLogger: str
    :param model:
    :type model: LlamaForCausalLM
    :param activation_stat:
    :type activation_stat: Tensor
    :param minmax_init:
    :type minmax_init: bool
    :return:
    """
    logger = logging.getLogger(nameLogger)

    # 允许启用“输入量化”的层（按模块名末段匹配）
    ALLOW_IN_ACT = {
        "down_proj",
        # "o_proj"
    }

    def _pick_in_quant_cfg(leaf: str) -> Optional[Dict]:
        r"""

        :param leaf:
        :type leaf: str
        :return:
        """
        if leaf in {"down_proj", "o_proj"}:
            return {
                "bits": args.input_bits,
                "group_size": args.input_group_size,
                "asym": args.input_asym,
                "mode": args.input_mode,
            }
        return None

    def _apply_input_quant(name: str,
                           module: int_linear_fake.QuantizedLinear,
                           cfg: Dict):
        r"""

        :param name:
        :type name: str
        :param module:
        :type module: nn.Module
        :param cfg:
        :type cfg: Dict
        :return:
        """
        in_stat = activation_stat.get(f"{name}.input") if activation_stat is not None else None

        module.use_act_quant = True
        quantized_shape = (1, module.in_features)

        module.input_quantizer = UniformAffineQuantizer(
            n_bits=cfg["bits"],
            quantized_shape=quantized_shape,
            asym=cfg["asym"],
            group_size=cfg["group_size"],
            quantized_item_stat=in_stat,
            quant_type="activation",
            mode=cfg["mode"],
            minmax_init=minmax_init,
            activation_clipping=args.activation_clipping,
        )

        sym_stat = "asymmetric" if cfg["asym"] else "symmetric"
        zp = module.input_quantizer.zero_point
        zp_shape = None if zp is None else tuple(zp.shape)

        logger.info(
            f"✅ 输入激活量化：将 {name} 量化为 "
            f"{cfg['bits']}-bit, group_size={cfg['group_size']}, mode={cfg['mode']}, {sym_stat};"
            f"\n   scale.shape={tuple(module.input_quantizer.scale.shape)}, zero_point.shape={zp_shape}\n"
        )

    for name, module in model.named_modules():
        logger.debug(f"\n▶️ 当前处理模块 - {name}：\n实例 - {pformat(module)}\n类型 - {type(module)}")

        # 仅处理 QuantizedLinear
        if not isinstance(module, int_linear_fake.QuantizedLinear):
            logger.debug("⭕ 非 QuantizedLinear，跳过。\n")
            continue

        leaf = name.split(".")[-1]
        if leaf not in ALLOW_IN_ACT:
            logger.debug(f"⭕ 未在输入量化白名单（{ALLOW_IN_ACT}）内，跳过 {name}。\n")
            continue

        cfg = _pick_in_quant_cfg(leaf)
        if cfg is None:
            logger.debug(f"⭕ 未找到 {leaf} 的输入量化配置，跳过。\n")
            continue

        _apply_input_quant(name, module, cfg)


def init_k_quantizer(args, model, activation_stat=None, minmax_init=True):
    r"""
    PrefixQuant 的源代码中的函数。我们的代码中没用上。
    :param args:
    :param model:
    :param activation_stat:
    :param minmax_init:
    :return:
    """
    num_heads = model.config.num_attention_heads
    num_kv_heads = model.config.num_key_value_heads
    model_dim = model.config.hidden_size
    head_dim = model_dim // num_heads
    kv_dim = num_kv_heads * head_dim
    assert args.kv_group_size in [-1, head_dim], f'Only token-wise/{head_dim}g quantization is supported for K-cache'
    # for the quantization of k/v output (kv-cache quantization)
    if args.k_pre_rope:
        for name, module in model.named_modules():
            if isinstance(module, int_linear_fake.QuantizedLinear) and 'k_proj' in name:
                output_bits = args.k_bits
                module.output_bits = output_bits
                if output_bits >= 16:
                    continue
                output_group_size = args.kv_group_size
                output_asym = args.kv_asym
                output_mode = args.kv_mode
                output_stat = activation_stat[f'{name}.output'] if activation_stat is not None else None
                module.use_act_quant = True
                quantized_shape = (1, module.out_features)
                module.output_quantizer = UniformAffineQuantizer(output_bits, quantized_shape, output_asym,
                                                                 output_group_size,
                                                                 quantized_item_stat=output_stat,
                                                                 quant_type='activation',
                                                                 mode=output_mode,
                                                                 minmax_init=minmax_init)
                sym_stat = "asymmetric" if output_asym else 'symmetric'
                print(
                    f'k-cache quantization: set {name} as {output_bits}-bit {output_group_size} groupsize {output_mode} {sym_stat} quantization')
    else:
        for name, module in model.named_modules():
            if isinstance(module, rotation_utils.QKRotationWrapper):
                output_bits = args.k_bits
                if output_bits >= 16:
                    continue
                output_group_size = args.kv_group_size
                output_asym = args.kv_asym
                output_mode = args.kv_mode
                output_stat = activation_stat[f'{name}.output_K'] if activation_stat is not None else None
                module.use_k_quant = True
                module.k_bits = output_bits
                module.online_had = args.qk_online_had
                quantized_shape = (1, kv_dim)
                module.k_quantizer = UniformAffineQuantizer(output_bits, quantized_shape, output_asym,
                                                            output_group_size,
                                                            quantized_item_stat=output_stat,
                                                            quant_type='activation',
                                                            mode=output_mode,
                                                            minmax_init=minmax_init)
                sym_stat = "asymmetric" if output_asym else 'symmetric'
                print(
                    f'k-cache quantization: set {name} as {output_bits}-bit {output_group_size} groupsize {output_mode} {sym_stat} quantization')


# endregion


# region ==== 关于初始化校准的 mse 的工具函数 ====
@torch.no_grad()
def weight_layer_mse_init(module, input_feat, n_grid=20, max_shrink=0.5):
    # def weight_layer_mse_init(module, input_feat, n_grid=50, max_shrink=0.75):
    '''
    inspired by https://github.com/mit-han-lab/llm-awq/blob/3665e1abbf04139aa037254ff4ff3f261bd68a40/awq/quantize/auto_clip.py#L11
    only for weight
    '''
    quantizer = module.weight_quantizer
    w = module.weight
    dev = w.device
    # original_scale = quantizer.scale.clone().to(dev)       # init with MAX-MIN
    group_size = quantizer.group_size
    if isinstance(input_feat, list):
        input_feat = torch.cat(input_feat, dim=0)
    input_feat = input_feat.view(-1, input_feat.shape[-1])
    input_feat = input_feat.reshape(input_feat.shape[0], -1, group_size)

    out_c = w.shape[0]
    # oc_batch_size = 256 if out_c % 256 == 0 else 64  # prevent OOM
    oc_batch_size = 64 if out_c % 256 == 0 else 32  # prevent OOM
    assert out_c % oc_batch_size == 0

    w = w.reshape(out_c, -1, group_size)

    ori_scale = quantizer.scale.clone()

    input_feat = input_feat.to(w.device)
    oc_batch_size = 256 if w.shape[0] % 256 == 0 else 64  # prevent OOM
    assert w.shape[0] % oc_batch_size == 0

    best_scale_list = []
    min_errs_list = []
    for i_b in range(w.shape[0] // oc_batch_size):
        w_sub = w[i_b * oc_batch_size: (i_b + 1) * oc_batch_size]
        best_scale = ori_scale[i_b * quantizer.inc_groups * oc_batch_size:(
                                                                                  i_b + 1) * quantizer.inc_groups * oc_batch_size].clone()
        if quantizer.zero_point is not None:
            zero_point = quantizer.zero_point[
                i_b * quantizer.inc_groups * oc_batch_size:(i_b + 1) * quantizer.inc_groups * oc_batch_size]
        else:
            zero_point = None
        org_out = torch.einsum('ngc, gcm -> ngm', input_feat.to(w_sub.dtype), w_sub.permute(1, 2, 0))
        min_errs = torch.ones_like(best_scale) * 1e9
        for i_s in range(int(max_shrink * n_grid)):
            clip_factor = (1 - i_s / n_grid)
            cur_scale = best_scale * clip_factor
            w_sub = w_sub.view(-1, group_size)
            q_w = quantizer.custom_quant(w_sub, cur_scale, zero_point)
            q_w = q_w.view(oc_batch_size, -1, group_size)
            cur_out = torch.einsum('ngc, gcm -> ngm', input_feat.to(w_sub.dtype), q_w.permute(1, 2, 0))

            # co, 1, n_group, 1
            err = (cur_out - org_out).pow(2).mean(dim=0).view(min_errs.shape)
            del q_w
            del cur_out
            cur_best_idx = err < min_errs
            min_errs[cur_best_idx] = err[cur_best_idx]
            best_scale[cur_best_idx] = cur_scale[cur_best_idx]
        best_scale_list.append(best_scale)
        min_errs_list.append(min_errs)

    best_scale = torch.cat(best_scale_list, dim=0)
    quantizer.scale.data = best_scale
    min_errs = torch.cat(min_errs_list, dim=0)
    return min_errs.mean()


@torch.no_grad()
def k_cache_mse_init(module, kv_cache, softmax=False, n_grid=20, max_shrink=0.5):
    quantizer = module.k_quantizer
    num_key_value_groups = module.num_key_value_groups
    num_key_value_heads = module.num_key_value_heads
    head_dim = module.head_dim,
    group_size = quantizer.group_size
    assert group_size == head_dim[0]
    dev = quantizer.scale.device
    Q_list = []
    K_list = []
    for Q, K in kv_cache:
        Q_list.append(Q)
        K_list.append(K)
    Q = torch.cat(Q_list, dim=0)
    K = torch.cat(K_list, dim=0)
    (bsz, num_heads, seq_len, head_dim) = K.shape

    # head-wise init to avoid OOM
    best_scale_list = []
    min_errs_list = []
    ori_scale = quantizer.scale.clone().to(dev)
    for head_index in range(num_heads):
        K_sub = K[:, head_index:head_index + 1,]
        Q_sub = Q[:, head_index * num_key_value_groups:(head_index + 1) * num_key_value_groups,]
        best_scale = ori_scale[head_index:head_index + 1]
        rep_K = repeat_kv(K_sub, num_key_value_groups)
        org_out = torch.matmul(Q_sub, rep_K.transpose(2, 3)) / math.sqrt(head_dim)
        if softmax:
            org_out = nn.functional.softmax(org_out, dim=-1, dtype=torch.float32)
        if quantizer.zero_point is not None:
            zero_point = quantizer.zero_point[head_index:head_index + 1]
        else:
            zero_point = None
        min_errs = torch.ones_like(best_scale) * 1e9
        for i_s in range(int(max_shrink * n_grid)):
            clip_factor = (1 - i_s / n_grid)
            cur_scale = best_scale * clip_factor
            q_K = K_sub.transpose(1, 2).flatten(-2)
            q_K = quantizer.custom_quant(q_K, cur_scale, zero_point).reshape((bsz, seq_len, 1, head_dim)).transpose(1,
                                                                                                                    2).to(
                Q)
            cur_out = torch.matmul(Q_sub, repeat_kv(q_K, num_key_value_groups).transpose(2, 3)) / math.sqrt(head_dim)
            if softmax:
                cur_out = nn.functional.softmax(cur_out, dim=-1, dtype=torch.float32)
            err = (cur_out - org_out).pow(2)
            err = err.reshape((bsz, 1, num_key_value_groups, seq_len, seq_len)).mean(dim=(0, 2, 3, 4)).view(
                min_errs.shape)
            del q_K
            del cur_out
            cur_best_idx = err < min_errs
            min_errs[cur_best_idx] = err[cur_best_idx].to(min_errs)
            best_scale[cur_best_idx] = cur_scale[cur_best_idx]
        best_scale_list.append(best_scale)
        min_errs_list.append(min_errs)

    best_scale = torch.cat(best_scale_list, dim=0)
    quantizer.scale.data = best_scale
    min_errs = torch.cat(min_errs_list, dim=0)
    return min_errs.mean()


@torch.no_grad()
def tensor_mse_init_static(quantizer, data, n_grid=20, max_shrink=0.5):
    dev = quantizer.scale.device
    group_size = quantizer.group_size
    original_scale = quantizer.scale.clone().to(dev)  # init with MAX-MIN
    best_scale = quantizer.scale.clone().to(dev)
    min_errs = torch.ones_like(best_scale) * 1e9
    if isinstance(data, list):
        data = torch.cat(data, dim=0)
    bs, seq_len, dims = data.shape
    for i_s in range(int(max_shrink * n_grid)):
        clip_factor = (1 - i_s / n_grid)
        cur_scale = original_scale * clip_factor
        quantizer.scale.data = cur_scale
        q_data = quantizer(data)
        err = (data - q_data).pow(2).reshape((bs, seq_len, -1, group_size)).mean(dim=(0, 1, 3)).view(min_errs.shape)
        del q_data
        cur_best_idx = err < min_errs
        min_errs[cur_best_idx] = err[cur_best_idx].to(min_errs)
        best_scale[cur_best_idx] = cur_scale[cur_best_idx]

    quantizer.scale.data = best_scale
    return min_errs.mean()


# @torch.no_grad()
# def tensor_mse_init_dynamic(quantizer, data):
#     dev = quantizer.scale.device
#     group_size = quantizer.group_size
#     min_errs = torch.inf
#     if isinstance(data,list):
#         data = torch.cat(data, dim=0)
#     bs, seq_len, dims = data.shape
#     if quantizer.asym:
#         searched_upbound_factors = torch.linspace(0.6, 1, 9).to(dev)
#         searched_lowbound_factors = torch.linspace(0.6, 1, 9).to(dev)
#         for upbound_fator in searched_upbound_factors:
#             for lowbound_factor in searched_lowbound_factors:
#                 quantizer.upbound_factor.data = upbound_fator
#                 quantizer.lowbound_factor.data = lowbound_factor        
#         q_data = quantizer(data)
#         err = (data - q_data).pow(2).reshape((bs, seq_len, -1, group_size)).mean(dim=(0,1,3)).view(min_errs.shape)
#         del q_data
#         cur_best_idx = err < min_errs
#         min_errs[cur_best_idx] = err[cur_best_idx].to(min_errs)
#         best_scale[cur_best_idx] = cur_scale[cur_best_idx]

#     quantizer.scale.data = best_scale
#     return min_errs.mean()


@torch.no_grad()
def block_mse_init_static_old(quantizer, qblock, prefixed_key_values, dev, data_inputs, data_gts, attention_mask,
                              position_ids):
    '''
    share a clipping scale accross a layer
    '''
    loss_func = torch.nn.MSELoss()
    best_loss = torch.inf
    best_scale = None
    original_scale = quantizer.scale.clone().to(dev)
    # coarse_search_factors = torch.linspace(0.45, 0.95, 8).to(dev)
    # coarse_search_factors = torch.linspace(0.40, 0.95, 8).to(dev)
    coarse_search_factors = torch.linspace(0.19, 0.99, 16).to(dev)
    bs = data_inputs.shape[0]
    for clip_factor in coarse_search_factors:
        cur_scale = original_scale * clip_factor
        quantizer.scale.data = cur_scale
        past_key_value = get_kv_cache(prefixed_key_values, bs)
        with torch.cuda.amp.autocast():
            quant_out = qblock(data_inputs, attention_mask=attention_mask, position_ids=position_ids,
                               past_key_value=past_key_value)[0]
        cur_loss = loss_func(quant_out, data_gts)
        if cur_loss <= best_loss:
            best_loss = cur_loss
            best_scale = cur_scale
            best_clip_factor = clip_factor

    fine_search_factors = torch.linspace(best_clip_factor - 0.05, best_clip_factor + 0.05, 10).to(dev)

    for clip_factor in fine_search_factors:
        cur_scale = original_scale * clip_factor
        quantizer.scale.data = cur_scale
        past_key_value = get_kv_cache(prefixed_key_values, bs)
        with torch.cuda.amp.autocast():
            quant_out = qblock(data_inputs, position_ids=position_ids, past_key_value=past_key_value)[0]
        cur_loss = loss_func(quant_out, data_gts)
        if cur_loss <= best_loss:
            best_loss = cur_loss
            best_scale = cur_scale
            best_clip_factor = clip_factor

    quantizer.scale.data = best_scale
    return best_clip_factor, best_loss


@torch.no_grad()
def block_mse_init_static(quantizer, qblock, prefixed_key_values, dev,
                          data_inputs, data_gts, attention_mask, position_ids):
    loss_func = torch.nn.MSELoss()
    best_loss = float("inf")
    best_scale = None
    best_clip_factor = None

    original_scale = quantizer.scale.detach().clone().to(dev)
    coarse_search_factors = torch.linspace(0.19, 0.99, 16, device=dev)
    bs = data_inputs.shape[0]

    def _forward_once(cur_scale):
        with torch.no_grad():
            quantizer.scale.copy_(cur_scale)

        past_key_value = get_kv_cache(prefixed_key_values, bs)
        if isinstance(past_key_value, (list, tuple)):
            past_key_value = tuple(
                tuple(t.to(dev) for t in layer)
                for layer in past_key_value
            )

        x = data_inputs.to(dev)
        attn_mask = attention_mask.to(dev) if attention_mask is not None else None
        pos_ids = position_ids.to(dev) if position_ids is not None else None

        # 开启 AMP，让 matmul 输入/权重自动对齐到 compute_dtype（多半是 fp16）
        with torch.autocast("cuda", dtype=torch.float32):
            out = qblock(
                x,
                attention_mask=attn_mask,
                position_ids=pos_ids,
                past_key_value=past_key_value
            )[0]

        cur_loss_t = loss_func(out.float(), data_gts.to(dev, dtype=torch.float32))
        return cur_loss_t.item() if torch.isfinite(cur_loss_t) else float("inf")

    # --- 粗搜 ---
    for clip_factor in coarse_search_factors:
        cur_scale = original_scale * clip_factor
        cur_loss = _forward_once(cur_scale)
        if cur_loss <= best_loss:
            best_loss = cur_loss
            best_scale = cur_scale
            best_clip_factor = float(clip_factor)

    # 全部 NaN/Inf：回退
    if best_clip_factor is None:
        with torch.no_grad():
            quantizer.scale.copy_(original_scale)
        return 1.0, best_loss

    # --- 细搜 ---
    fine_search_factors = torch.linspace(best_clip_factor - 0.05,
                                         best_clip_factor + 0.05,
                                         10, device=dev)
    for clip_factor in fine_search_factors:
        cur_scale = original_scale * clip_factor
        cur_loss = _forward_once(cur_scale)
        if cur_loss <= best_loss:
            best_loss = cur_loss
            best_scale = cur_scale
            best_clip_factor = float(clip_factor)

    with torch.no_grad():
        quantizer.scale.copy_(best_scale)

    return best_clip_factor, best_loss


@torch.no_grad()
def block_mse_init_dynamic(quantizer, qblock, prefixed_key_values, dev, data_inputs, data_gts, attention_mask,
                           position_ids):
    '''
    search for the clipping value of dynamic quantizer
    '''
    loss_func = torch.nn.MSELoss()
    best_loss = torch.inf
    bs = data_inputs.shape[0]

    if quantizer.asym:
        searched_upbound_factors = torch.linspace(0.6, 1, 9).to(dev)
        searched_lowbound_factors = torch.linspace(0.6, 1, 9).to(dev)
        for upbound_fator in searched_upbound_factors:
            for lowbound_factor in searched_lowbound_factors:
                quantizer.upbound_factor.data = upbound_fator
                quantizer.lowbound_factor.data = lowbound_factor
                past_key_value = get_kv_cache(prefixed_key_values, bs)
                with torch.cuda.amp.autocast():
                    quant_out = qblock(data_inputs, attention_mask=attention_mask, position_ids=position_ids,
                                       past_key_value=past_key_value)[0]
                cur_loss = loss_func(quant_out, data_gts)
                if cur_loss <= best_loss:
                    best_loss = cur_loss
                    best_upbound_factor = upbound_fator
                    best_lowbound_factor = lowbound_factor
        quantizer.upbound_factor.data = best_upbound_factor
        quantizer.lowbound_factor.data = best_lowbound_factor
        best_clip_factor = (best_lowbound_factor.item(), best_upbound_factor.item())
    else:
        searched_bound_factors = torch.linspace(0.6, 1, 9).to(dev)
        for bound_factor in searched_bound_factors:
            quantizer.bound_factor.data = bound_factor
            past_key_value = get_kv_cache(prefixed_key_values, bs)
            with torch.cuda.amp.autocast():
                quant_out = qblock(data_inputs, attention_mask=attention_mask, position_ids=position_ids,
                                   past_key_value=past_key_value)[0]
            cur_loss = loss_func(quant_out, data_gts)
            # print(cur_loss)
            # import pdb;pdb.set_trace()
            if cur_loss <= best_loss:
                best_loss = cur_loss
                best_bound_factor = bound_factor
        quantizer.bound_factor.data = best_bound_factor
        best_clip_factor = best_bound_factor.item()
    return best_clip_factor, best_loss


@torch.no_grad()
def mse_init(qblock, prefixed_key_values, dev, data_inputs, attention_mask, position_ids, logger, args,
             data_gt_asym=None):
    # def mse_init(qblock, prefixed_key_values, dev, data_inputs, position_ids, logger, args):
    # part1: obtain the intermediate input/output of each linear layer
    set_quant_state(qblock, weight_quant=False, act_quant=False)
    input_activation_dict = {}
    output_activation_dict = {}

    def get_activation_hook(layer_name, input_activation_dict, output_activation_dict):
        def hook(model, input, output):
            if not 'qk_rotation_wrapper' in layer_name:
                input = input[0]

            if layer_name in input_activation_dict:
                input_activation_dict[layer_name].append(input)
            else:
                input_activation_dict[layer_name] = [input]
            if layer_name in output_activation_dict:
                output_activation_dict[layer_name].append(output)
            else:
                output_activation_dict[layer_name] = [output]

        return hook

    hooks = []
    for name, layer in qblock.named_modules():
        if isinstance(layer, (QKRotationWrapper, QuantizedLinear,
                # QuantizedLlamaRMSNorm, QuantizedIdentity, QuantizedSiLU, QuantizedEmbedding,
                              )):
            hooks.append(
                layer.register_forward_hook(get_activation_hook(name, input_activation_dict, output_activation_dict)))
    data_gts = torch.zeros_like(data_inputs)
    for i in range(len(data_inputs)):
        with torch.amp.autocast("cuda"):
            # data_gts[i:i+1] = qblock(data_inputs[i:i+1],position_ids=position_ids,past_key_value=get_kv_cache(prefixed_key_values))[0]    
            data_gts[i:i + 1] = qblock(data_inputs[i:i + 1], attention_mask=attention_mask, position_ids=position_ids,
                                       past_key_value=get_kv_cache(prefixed_key_values))[0]
    if data_gt_asym is not None:
        data_gts = data_gt_asym
    for h in hooks:
        h.remove()
    # end of part 1
    # part2: mse init of quantizer
    batch_attention_mask = None if attention_mask is None else attention_mask.repeat(data_gts.shape[0], 1, 1, 1)
    for name, module in qblock.named_modules():
        if isinstance(module, (QKRotationWrapper,
                               QuantizedLinear,
                               QuantizedLlamaRMSNorm,
                               # QuantizedEmbedding,
                               QuantizedSiLU,
                               QuantizedIdentity)):
            module.set_quant_state(weight_quant=False, act_quant=True)
            # init input quantizer
            if hasattr(module, 'input_quantizer') and module.input_quantizer.n_bits < 16:
                module.input_quantizer.activate()
                if module.input_quantizer.mode == 'static':
                    best_clip_factor, best_loss = block_mse_init_static(module.input_quantizer, qblock,
                                                                        prefixed_key_values, dev, data_inputs, data_gts,
                                                                        batch_attention_mask, position_ids)
                    logger.info(f"[{name}.input_quantizer] "
                                f"clipping factor: ({best_clip_factor}); "
                                f"best_loss:{best_loss} ")
                elif module.input_quantizer.mode == 'dynamic' and module.input_quantizer.activation_clipping:
                    best_clip_factor, best_loss = block_mse_init_dynamic(module.input_quantizer, qblock,
                                                                         prefixed_key_values, dev, data_inputs,
                                                                         data_gts, batch_attention_mask, position_ids)
                    logger.info(
                        f"[{name}.input_quantizer] clipping factor: ({best_clip_factor}); best_loss:{best_loss} ")

            # init output quantizer        
            if hasattr(module, 'output_quantizer') and module.output_quantizer.n_bits < 16:
                module.output_quantizer.activate()
                # V cache quantization
                if module.output_quantizer.mode == 'static':
                    if 'v_proj' in name and module.output_quantizer.inc_groups > 1:
                        best_loss = tensor_mse_init_static(module.output_quantizer, output_activation_dict[name])
                        logger.info(f"[{name}.output_quantizer] best_loss:{best_loss} ")
                    else:
                        best_clip_factor, best_loss = block_mse_init_static(module.output_quantizer, qblock,
                                                                            prefixed_key_values, dev, data_inputs,
                                                                            data_gts, batch_attention_mask,
                                                                            position_ids)
                        logger.info(f"[{name}.output_quantizer] "
                                    f"clipping factor: ({best_clip_factor}); "
                                    f"best_loss:{best_loss} ")
                elif module.output_quantizer.mode == 'dynamic' and module.output_quantizer.activation_clipping:
                    # if 'v_proj' in name and module.output_quantizer.inc_groups > 1:
                    #     pass
                    # else:
                    #     best_clip_factor, best_loss = block_mse_init_dynamic(module.output_quantizer, qblock, prefixed_key_values, dev, data_inputs, data_gts, batch_attention_mask, position_ids)
                    #     logger.info(f"[{name}.output_quantizer] clipping factor: ({best_clip_factor}); best_loss:{best_loss} ")
                    best_clip_factor, best_loss = block_mse_init_dynamic(module.output_quantizer, qblock,
                                                                         prefixed_key_values, dev, data_inputs,
                                                                         data_gts, batch_attention_mask, position_ids)
                    logger.info(
                        f"[{name}.output_quantizer] clipping factor: ({best_clip_factor}); best_loss:{best_loss} ")

            # init k quantizer
            if hasattr(module,
                       'k_quantizer') and module.k_quantizer.mode == 'static' and module.k_quantizer.n_bits < 16:
                module.k_quantizer.activate()
                if module.k_quantizer.inc_groups > 1:
                    best_loss = k_cache_mse_init(module, output_activation_dict[name], softmax=True)
                    logger.info(f"[{name}_k_quantizer] best_loss:{best_loss} ")
                else:
                    best_clip_factor, best_loss = block_mse_init_static(module.k_quantizer, qblock, prefixed_key_values,
                                                                        dev, data_inputs, data_gts,
                                                                        batch_attention_mask, position_ids)
                    logger.info(
                        f"[{name}_k_quantizer] clipping factor: ({best_clip_factor:.2f}); best_loss:{best_loss} ")
            module.set_quant_state(weight_quant=True, act_quant=False)

            # init weight quantizer
            if hasattr(module, 'weight_quantizer') and module.weight_quantizer.n_bits < 16:
                module.weight_quantizer.activate()
                if 'q_proj' in name or 'k_proj' in name:
                    if args.skip_qk_weight_init:
                        continue
                    elif args.block_qk_weight_init:
                        best_clip_factor, best_loss = block_mse_init_static(module.weight_quantizer, qblock,
                                                                            prefixed_key_values, dev, data_inputs,
                                                                            data_gts, batch_attention_mask,
                                                                            position_ids)
                        logger.info(
                            f"[{name}.weight_quantizer] "
                            f"clipping factor: ({best_clip_factor:.2f}); "
                            f"best_loss:{best_loss} ")
                    else:
                        best_loss = weight_layer_mse_init(module, input_activation_dict[name])
                        logger.info(f"[{name}.weight_quantizer] best_loss:{best_loss} ")
                else:
                    best_loss = weight_layer_mse_init(module, input_activation_dict[name])
                    logger.info(f"[{name}.weight_quantizer] best_loss:{best_loss} ")
            module.set_quant_state(weight_quant=False, act_quant=False)
    # end of part 2


# endregion


class MultiBlock(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.block_list = nn.ModuleList([])

    def add_block(self, block):
        self.block_list.append(block)

    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None):
        for block in self.block_list:
            hidden_states = block(hidden_states, attention_mask=attention_mask, position_ids=position_ids)[0]
        return (hidden_states,)


# region ==== 设置、设置、统计 模型权重可学习性（是否反传更新）的工具函数====
def set_weight_parameters(model, requires_grad):
    params = []
    for n, m in model.named_parameters():
        # if n.find('weight') > -1 and not (n.find('scale') > -1 or n.find('zero_point') > -1):
        if n.find('weight') > -1 and not (n.find('weight_quantizer') > -1):
            m.requires_grad = requires_grad
    return iter(params)


def weight_parameters(model):
    params = []
    for n, m in model.named_parameters():
        # if n.find('weight') > -1 and not (n.find('scale') > -1 or n.find('zero_point') > -1):
        if n.find('weight') > -1 and not (n.find('weight_quantizer') > -1):
            params.append(m)
    return iter(params)


def set_scale_parameters(model, requires_grad):
    params = []
    for n, m in model.named_parameters():
        if n.find('in_scale') > -1 or n.find('out_scale') > -1:
            m.requires_grad = requires_grad
    return iter(params)


def scale_parameters(model):
    params = []
    for n, m in model.named_parameters():
        if n.find('in_scale') > -1 or n.find('out_scale') > -1:
            params.append(m)
    return iter(params)


def set_quant_parameters(model, requires_grad):
    params = []
    for n, m in model.named_parameters():
        # if (n.find('scale') > -1 or n.find('zero_point') > -1) and (not n.find('smooth_scale') > -1):
        if (n.find('scale') > -1 or n.find('zero_point') > -1 or n.find('bound_factor') > -1) and (
                not (n.find('in_scale') > -1 or n.find('out_scale') > -1)):
            m.requires_grad = requires_grad
    return iter(params)


def quant_parameters(model):
    params = []
    for n, m in model.named_parameters():
        if (n.find('scale') > -1 or n.find('zero_point') > -1 or n.find('bound_factor') > -1) and (
                not (n.find('in_scale') > -1 or n.find('out_scale') > -1)):
            params.append(m)
    return iter(params)


def trainable_parameters(model):
    params = []
    for n, m in model.named_parameters():
        if m.requires_grad:
            params.append(m)
    return iter(params)


def trainable_parameters_num(model):
    params = []
    total = 0
    for n, m in model.named_parameters():
        if m.requires_grad:
            total += m.numel()
    return total


# endregion


# region ==== 设置量化器工作状态（是否进行量化）的工具函数 ====
def set_quant_state(model, weight_quant: bool = False, act_quant: bool = False):
    for m in model.modules():
        if isinstance(m, (QKRotationWrapper,
                          QuantizedLinear, QuantizedSiLU, QuantizedEmbedding, QuantizedLlamaRMSNorm)):
            m.set_quant_state(weight_quant, act_quant)


def activate_quantizer(model):
    for m in model.modules():
        if isinstance(m, UniformAffineQuantizer):
            m.activate()


def deactivate_quantizer(model):
    for m in model.modules():
        if isinstance(m, UniformAffineQuantizer):
            m.deactivate()


# endregion


# region ====  -他工具函数 ====
@torch.no_grad()
def quant_inplace(model):
    for name, module in model.named_modules():
        if isinstance(module, QuantizedLinear):
            if module.wbits < 16:
                module.weight.data = module.weight_quantizer(module.weight.data)


class TruncateFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold):
        truncated_tensor = input.clone()
        truncated_tensor[truncated_tensor.abs() < threshold] = truncated_tensor[
                                                                   truncated_tensor.abs() < threshold].sign() * threshold
        return truncated_tensor

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None


def truncate_number(number, threshold=1e-2):
    # avoid overflow with AMP training
    return TruncateFunction.apply(number, threshold)


def get_named_linears(module, type):
    # return {name: m for name, m in module.named_modules() if isinstance(m, torch.nn.Linear)}
    return {name: m for name, m in module.named_modules() if isinstance(m, type)}


def set_op_by_name(layer, name, new_module):
    levels = name.split('.')
    if len(levels) > 1:
        mod_ = layer
        for l_idx in range(len(levels) - 1):
            if levels[l_idx].isdigit():
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], new_module)
    else:
        setattr(layer, name, new_module)


def combine_linear_layers(*linears):
    if len(linears) < 2:
        raise ValueError("at least two linear layers")

    in_features = linears[0].in_features
    for linear in linears:
        assert linear.in_features == in_features, "input dims must be the same"

    combined_out_features = sum(linear.out_features for linear in linears)

    combined_linear = nn.Linear(in_features, combined_out_features, bias=None)

    # combine the original weights
    with torch.no_grad():
        start = 0
        for linear in linears:
            end = start + linear.out_features
            combined_linear.weight[start:end, :] = linear.weight
            # combined_linear.bias[start:end] = linear.bias
            start = end

    return combined_linear


def check_quantizer(model):
    for name, module in model.named_modules():
        if isinstance(module, UniformAffineQuantizer):
            bits = module.n_bits
            sym = 'asymmetric' if module.asym else 'symmetric'
            if module.enable:
                print(f'{name}: {bits}-bit {sym} quantization')


def get_quant_config(args):
    quantization_config = {"wbits": args.wbits,
                           "w_group_size": args.w_group_size,
                           "w_asym": args.w_asym,
                           "input_bits": args.input_bits,
                           "input_group_size": args.input_group_size,
                           "input_asym": args.input_asym,
                           "input_mode": args.input_mode,
                           "output_bits": args.output_bits,
                           "output_group_size": args.output_group_size,
                           "output_asym": args.output_asym,
                           "output_mode": args.output_mode,
                           "k_bits": args.k_bits,
                           "v_bits": args.v_bits,
                           "kv_group_size": args.kv_group_size,
                           "kv_asym": args.kv_asym,
                           "k_pre_rope": args.k_pre_rope,
                           "kv_mode": args.kv_mode,
                           "down_online_had": args.down_online_had and args.pre_rotate,
                           "qk_online_had": args.qk_online_had and args.pre_rotate,
                           "real_quant": args.real_quant,
                           "set_prefixed_tokens": args.set_prefixed_tokens,
                           "activation_clipping": args.activation_clipping}
    return quantization_config
# endregion
