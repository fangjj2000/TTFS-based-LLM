import logging
from argparse import Namespace
from pprint import pformat

from torch import nn
from torch.nn import SiLU
from transformers import LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaSdpaAttention, LlamaRMSNorm, LlamaMLP, LlamaAttention

from quantize.int_linear_fake import QuantizedLlamaMLP
from quantize.quant_norm import QuantRMSNorm
from quantize.quantizer import UniformAffineQuantizer
from spiking_conversion.model.spiking_layers import SpikingLlamaAttention, SpikingLlamaMLP, TTFS_Spiker, SpikingRMSNorm
from spiking_conversion.spike import SpikeConfig, SpikeParamDict
from spiking_conversion.spike.config import SpikeConfigTTFSFormer
from utils.model_utils import RMSN
from utils.quant_utils import set_op_by_name


def wrap_to_snn_model_old(model: LlamaForCausalLM,
                          convert_from: str,
                          args: Namespace,
                          nameLogger: str,
                          ) -> None:
    r"""

    :param model:
    :type model: LlamaForCausalLM
    :param convert_from: "ANN", "QANN".
    :type convert_from:str
    :param args:
    :type args: Namespace
    :param nameLogger:
    :type nameLogger: str
    :return:
    """
    logger = logging.getLogger(nameLogger)

    # if convert_from == "ANN":
    #     precision = args.precision
    # elif convert_from == "QANN":
    #     precision = args.input_bits ** 2
    # else:
    #     raise ValueError(f"convert_from must be 'ANN' or 'QANN', but got {convert_from}!!")
    # logger.info(f"Converting from {convert_from}...")

    # TODO 增加 args noise
    configSpike: SpikeConfig = SpikeConfig(nameLogger=nameLogger)  # 是递归所有位置共享的。noise参数我们用不上。
    # TODO 单独对 ANN 或 QANN 跑一遍，获取 monitors。

    logger.info(f"🛠️脉冲发射配置为：{configSpike}, 全局生效。")

    for name, module in model.named_modules():
        logger.info(f"\n▶️当前处理模块名 - {name}：\n实例 - {pformat(module)}, \n原始类型 - {type(module)}")

        if 'lm_head' in name:
            # TODO 后续考虑是否对 lm_head 也脉冲化。
            logger.info(f"❎  保留 lm_head： {module}, 类型 {type(module)}, 不替换.")
            continue

        elif 'model.norm' in name:
            # TODO 后续考虑是否对 Llama.norm 也脉冲化。
            logger.info(f"❎  保留 model.norm： {module}, 类型 {type(module)}, 不替换.")
            continue

        elif isinstance(module, UniformAffineQuantizer) and module.quant_type == 'activation':
            # continue
            spikeEmitter = TTFS_Spiker(configSpike=configSpike,
                                       quantizer=module)
            set_op_by_name(model, name, spikeEmitter)
            logger.info(f"✅  已经替换为：\n{spikeEmitter}, \n类型：{type(spikeEmitter)}")
            del module

        # region ✅ === 替换 embed_tokens 层 ===
        elif "embed_tokens" in name:  # ✅
            continue
            param = paramsSpike.child(key=name.removeprefix("model."))
            SpikedEmbedModule = quant_to_spike(module=module, config=configSpike)
            set_op_by_name(model, name, SpikedEmbedModule)
            logger.info(f"✅  已经替换为：\n{SpikedEmbedModule}, \n类型：{type(SpikedEmbedModule)}")
            del module
        # endregion

        # region === 替换 LlamaSdpaAttention 层 ===
        elif isinstance(module, LlamaSdpaAttention):
            continue
            param = paramsSpike.child(key=name.removeprefix("model."))
            SpikedAttnModule = SpikingLlamaAttention(configLlama=module.config,
                                                     configSpike=configSpike,
                                                     layer_idx=module.layer_idx,
                                                     param=param,
                                                     nameLogger=nameLogger)
            set_op_by_name(model, name, SpikedAttnModule)
            logger.info(f"✅  已经替换为：\n{SpikedAttnModule}, \n类型：{type(SpikedAttnModule)}")
            del module
        # endregion

        # region  ✅ === 替换 LlamaMLP 层 ===
        # 注意，LLAMA 中的 SiLU只出现在 MLP 中，
        # 所以，SpikingLlamaMLP 处理好 SiLU 之后，该循环内后面就不需要针对SiLU进行单独处理。
        elif isinstance(module, QuantizedLlamaMLP):
            continue
            spikedLlamaMLP = SpikingLlamaMLP(mlp=module, spikeConfig=configSpike)
            set_op_by_name(model, name, spikedLlamaMLP)
            logger.info(f"✅  已经替换为：\n{spikedLlamaMLP}, \n类型：{type(spikedLlamaMLP)}")
            del module
        # endregion

        # region ✅ === 替换 SiLU 操作 ===
        # 注意，LLAMA 中的 SiLU 只出现在 MLP 中，
        # 所以，SpikingLlamaMLP 处理好 SiLU 之后，该循环内后面就不需要针对 SiLU 进行单独处理。
        # 这留着注释，是方便测试单独替换 SiLU 的等价性。
        elif isinstance(module, SiLU):  # ✅
            continue
            param = paramsSpike.child(key=name.removeprefix("model."))
            spikingSiLU = quant_to_spike(module=module, config=configSpike)
            set_op_by_name(model, name, spikingSiLU)
            logger.info(f"✅  已经替换为：\n{spikingSiLU}, \n类型：{type(spikingSiLU)}")
            del module
        # endregion

        # region === 替换 RMS 归一化 ===
        # 注意，LLAMA 中的 RMS 出现在 LlamaDecoderLayer 中，但是不在 Attention 中，
        # Attention 和 RMS 都是 LlamaDecoderLayer的成员。
        elif isinstance(module, (QuantRMSNorm, LlamaRMSNorm, RMSN)):
            continue
            param = paramsSpike.child(key=name)
            spikingNorm = quant_to_spike(module=module, config=configSpike, param=param)
            set_op_by_name(model, name, spikingNorm)
            logger.info(f"✅  已经替换为：\n{spikingNorm}, \n类型：{type(spikingNorm)}")
            del module
        # endregion

        else:
            logger.info(f"⭕  未进行任何操作!!!")


def wrap_to_snn_model(model: LlamaForCausalLM,
                      convert_from: str,
                      args: Namespace,
                      nameLogger: str,
                      ) -> None:
    """
    将量化模型替换为 SNN/脉冲版本：
      - 跳过：lm_head、model.norm（可在 SKIP_NAME_SUBSTR 调整）
      - 已启用：将 activation 量化器(UniformAffineQuantizer, quant_type='activation') → TTFS_Spiker
      - 保留分支（默认关闭，FEATURE_TOGGLES 可一键开启）：
          * embed_tokens → spiking 版本
          * LlamaSdpaAttention → SpikingLlamaAttention
          * QuantizedLlamaMLP → SpikingLlamaMLP
          * SiLU → spiking 版本
          * (QuantRMSNorm / LlamaRMSNorm / RMSN) → spiking 版本
    :param model:
    :type model: LlamaForCausalLM
    :param convert_from: "ANN", "QANN".
    :type convert_from:str
    :param args:
    :type args: Namespace
    :param nameLogger:
    :type nameLogger: str
    :return:
    """
    logger = logging.getLogger(nameLogger)

    # --- 各个网络层、算子，是否转换（可按需修改为 True ） ---
    # 🎉我们通过最细粒度的 （仅针对特征量化）【量化器 -> 脉冲发放】 逐一替换，来实现脉冲化，避免了逐个模块的替换。
    FEATURE_TOGGLES = {
        "convert_activation_quantizer": True,  # UniformAffineQuantizer(activation) → TTFS_Spiker
        "convert_embed_tokens": False,  # embed_tokens → spiking
        "convert_attention": False,  # LlamaSdpaAttention → SpikingLlamaAttention
        "convert_mlp": False,  # QuantizedLlamaMLP → SpikingLlamaMLP
        "convert_silu": False,  # SiLU → spiking
        "convert_rmsnorm": False,  # (QuantRMSNorm/LlamaRMSNorm/RMSN) → spiking
    }

    # --- 跳过名单（按名字匹配） ---
    SKIP_NAME_SUBSTR = ("lm_head", "model.norm")

    def _should_skip(name: str) -> bool:
        r"""

        :param name:
        :type name: str
        :return:
        :rtype: bool
        """
        return any(s in name for s in SKIP_NAME_SUBSTR)

    # --- 统一替换封装 ---
    def _swap(name: str,
              new_module: nn.Module,
              why: str):
        r"""

        :param name:
        :type name: str
        :param new_module:
        :type new_module: nn.Module
        :param why:
        :type why: str
        :return:
        :rtype:
        """
        set_op_by_name(model, name, new_module)
        logger.info(f"✅ 已替换 {name}（{why}）→\n  {new_module}\n  类型：{type(new_module)}\n")

    # --- 主循环 ---
    for name, module in model.named_modules():
        logger.debug(f"\n▶️ 模块 {name}：\n实例 - {pformat(module)}\n类型 - {type(module)}")

        # 跳过清单
        if _should_skip(name):
            logger.info(f"❎ 跳过 {name}（保持原模块）：{type(module)}")
            continue

        # 已启用：将“激活量化器”替换为脉冲发射器（TTFS）
        if FEATURE_TOGGLES["convert_activation_quantizer"]:
            if isinstance(module, UniformAffineQuantizer) and getattr(module, "quant_type", None) == "activation":
                # 量化器 → 发射器（注意：TTFS_Spiker 使用量化器里学到的 scale/zero_point）
                spiker = TTFS_Spiker(nameLogger=nameLogger,
                                     quantizer=module,
                                     debug=False)
                _swap(name, spiker, "Activation Quantizer → TTFS_Spiker")
                del module
                continue

        # region 保留分支：将来需要时开启即可
        # embed_tokens → spiking
        if FEATURE_TOGGLES["convert_embed_tokens"]:
            if name.endswith("embed_tokens") or ".embed_tokens" in name:
                # SpikedEmbedModule = quant_to_spike(module=module, config=configSpike)  # 依据你的实现
                # _swap(name, SpikedEmbedModule, "Embedding → SpikingEmbedding")
                # continue
                pass  # 预留

        # LlamaSdpaAttention → SpikingLlamaAttention
        if FEATURE_TOGGLES["convert_attention"]:
            if isinstance(module, LlamaSdpaAttention):
                # SpikedAttnModule = SpikingLlamaAttention(
                #     quant_attn=...? , configLlama=module.config, configSpike=configSpike,
                #     layer_idx=module.layer_idx, nameLogger=nameLogger
                # )
                # _swap(name, SpikedAttnModule, "LlamaSdpaAttention → SpikingLlamaAttention")
                # continue
                pass  # 预留

        # QuantizedLlamaMLP → SpikingLlamaMLP
        if FEATURE_TOGGLES["convert_mlp"]:
            if isinstance(module, QuantizedLlamaMLP):
                # spikedLlamaMLP = SpikingLlamaMLP(quant_mlp=module, spikeConfig=configSpike)
                # _swap(name, spikedLlamaMLP, "QuantizedLlamaMLP → SpikingLlamaMLP")
                # continue
                pass  # 预留

        # SiLU → SpikingSiLU（如果你需要单独替换）
        # if FEATURE_TOGGLES["convert_silu"]:
        #     if isinstance(module, SiLU):
        #         spikingSiLU = quant_to_spike(module=module, config=configSpike)
        #         _swap(name, spikingSiLU, "SiLU → SpikingSiLU")
        #         continue

        # (QuantRMSNorm / LlamaRMSNorm / RMSN) → SpikingRMSNorm
        if FEATURE_TOGGLES["convert_rmsnorm"]:
            if isinstance(module, (QuantRMSNorm, LlamaRMSNorm, RMSN)):
                # spikingNorm = quant_to_spike(module=module, config=configSpike)
                # _swap(name, spikingNorm, "RMSNorm → SpikingRMSNorm")
                # continue
                pass  # 预留
        # endregion

        # 默认不处理
        logger.debug("⭕ 无需替换，跳过。")


def wrap_to_TTFSFormer_style(model: LlamaForCausalLM,
                             monitoredParams: SpikeParamDict,
                             configSpike: SpikeConfigTTFSFormer,
                             nameLogger: str):
    """
        将浮点模型替换为 TTFSFormer的计算版本的 SNN/脉冲版本：
          - 跳过：lm_head、model.norm（可在 SKIP_NAME_SUBSTR 调整）
          - 已启用：将 activation 量化器(UniformAffineQuantizer, quant_type='activation') → TTFS_Spiker
          - 保留分支（默认关闭，FEATURE_TOGGLES 可一键开启）：
              * embed_tokens → spiking 版本
              * LlamaSdpaAttention → SpikingLlamaAttention
              * QuantizedLlamaMLP → SpikingLlamaMLP
              * SiLU → spiking 版本
              * (QuantRMSNorm / LlamaRMSNorm / RMSN) → spiking 版本
        :param model:
        :type model: LlamaForCausalLM
        :param monitoredParams:
        :type monitoredParams: SpikeParamDict
        :param configSpike:
        :type configSpike: SpikeConfigTTFSFormer
        :param nameLogger:
        :type nameLogger: str
        :return:
        """
    logger = logging.getLogger(nameLogger)
    # --- 各个网络层、算子，是否转换（可按需修改为 True ） ---
    FEATURE_TOGGLES = {
        "convert_attention": True,  # LlamaAttention → SpikingLlamaAttention
        "convert_mlp": True,  # LlamaMLP → SpikingLlamaMLP
        "convert_silu": False,  # SiLU → spiking ❗ SiLU 已经在 MLP 中替换.
        "convert_rmsnorm": True,  # (LlamaRMSNorm/RMSN) → spiking
    }

    # --- 跳过名单（按名字匹配） ---
    SKIP_NAME_SUBSTR = ("lm_head", "model.norm", "embed_tokens")

    def _should_skip(name: str) -> bool:
        r"""

        :param name:
        :type name: str
        :return:
        :rtype: bool
        """
        return any(s in name for s in SKIP_NAME_SUBSTR)

    # --- 统一替换封装 ---
    def _swap(name: str,
              new_module: nn.Module,
              how: str):
        r"""

        :param name:
        :type name: str
        :param new_module:
        :type new_module: nn.Module
        :param how:
        :type how: str
        :return:
        :rtype:
        """
        set_op_by_name(model, name, new_module)
        logger.info(f"✅ 已替换 {name}（{how}）→\n  {new_module}\n  类型：{type(new_module)}\n")

    # --- 主循环 ---
    for name, module in model.named_modules():
        logger.debug(f"\n▶️ 模块 {name}：\n实例 - {pformat(module)}\n类型 - {type(module)}")

        # 跳过清单
        if _should_skip(name):
            logger.info(f"❎ 跳过 {name}（保持原模块）：{type(module)}")
            continue

        # LlamaSdpaAttention → SpikingLlamaAttention
        if FEATURE_TOGGLES["convert_attention"]:
            if isinstance(module, LlamaAttention):
                paramAttn = monitoredParams.child(f"{name.removeprefix('model.')}")
                SpikedAttnModule = SpikingLlamaAttention(attn=module,
                                                         configSpike=configSpike,
                                                         paramSpike=paramAttn,
                                                         nameLogger=nameLogger,
                                                         layer_idx=module.layer_idx)
                _swap(name, SpikedAttnModule, "LlamaAttention → SpikingLlamaAttention")
                del module
                continue

        # LlamaMLP → SpikingLlamaMLP
        if FEATURE_TOGGLES["convert_mlp"]:
            if isinstance(module, LlamaMLP):
                paramMlp = monitoredParams.child(f"{name.removeprefix('model.')}")
                spikedLlamaMLP = SpikingLlamaMLP(mlp=module,
                                                 param=paramMlp,
                                                 spikeConfig=configSpike)
                _swap(name, spikedLlamaMLP, "LlamaMLP → SpikingLlamaMLP")
                del module
                continue

        # SiLU → SpikingSiLU !!!已经不用了，在 SpikingMLP 中，已经替换了 SiLU。
        # if FEATURE_TOGGLES["convert_silu"]:
        #     if isinstance(module, SiLU):
        #         spikingSiLU = quant_to_spike(module=module, config=configSpike)
        #         _swap(name, spikingSiLU, "SiLU → SpikingSiLU")
        #         continue

        # (QuantRMSNorm / LlamaRMSNorm / RMSN) → SpikingRMSNorm
        if FEATURE_TOGGLES["convert_rmsnorm"]:
            if isinstance(module, (LlamaRMSNorm, RMSN)):
                paramNorm = monitoredParams.child(f"{name}")
                spikingNorm = SpikingRMSNorm(rmsn=module,
                                             configSpike=configSpike,
                                             paramSpike=paramNorm)
                _swap(name, spikingNorm, "LlamaRMSNorm → SpikingRMSNorm")
                del module
                continue

        # 默认不处理
        logger.debug("⭕ 无需替换，跳过。")
