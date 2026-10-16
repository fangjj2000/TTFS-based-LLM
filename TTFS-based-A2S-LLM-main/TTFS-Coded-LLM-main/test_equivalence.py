import argparse
from typing import Dict

import torch
from torch import nn

from emit_and_quantizer_equivalence import compare_tensors
from spiking_conversion.model.spiking_layers import SpikingWrapped
from spiking_conversion.spike import SpikeParamDict
from spiking_conversion.spike.config import SpikeConfigTTFSFormer
from spiking_conversion.util import BaseMonitor, MinMaxMonitor
from utils.monitor_tools import get_monitor_data

MonitorCls = MinMaxMonitor

if __name__ == "__main__":
    # region ==== 参数 ====
    parser = argparse.ArgumentParser()
    # -----------------model setting------------------------------------
    parser.add_argument("--model_path", type=str, help="model path",
                        default="/mnt/swl/Models/Llama-2-7b-hf")
    parser.add_argument("--model_name", type=str, default="Llama-2-7B",
                        help="model name, for the saving of corresponding data cache")
    parser.add_argument("--cache_dir", default="./cache", type=str,
                        help="cache dir of dataset, leading to faster debug")
    parser.add_argument("--output_dir", default="./log/temp",
                        type=str, help="direction of logging file")
    parser.add_argument("--save_quant_dir", default="./log/temp",
                        type=str, help="direction for saving quantization model")
    parser.add_argument("--real_quant", default=False, action="store_true",
                        help="use real quantization instead of fake quantization, can reduce memory footprint")
    parser.add_argument("--resume_quant", type=str, default=None,
                        help="model path of resumed quantized model")
    # -----------------quantization setting------------------------------------
    parser.add_argument("--wbits", type=int, default=8,
                        help="quantization bits")
    parser.add_argument("--w_group_size", type=int,
                        default=-1, help="quantization group size")
    parser.add_argument("--w_asym", dest="w_asym",
                        action="store_true", help="Set w_asym to True")
    parser.add_argument("--w_sym", dest="w_asym",
                        action="store_false", help="Set w_asym to False")
    parser.set_defaults(w_asym=False)
    parser.add_argument("--input_bits", type=int,
                        default=8, help="quantization bits")
    parser.add_argument("--input_group_size", type=int,
                        default=-1, help="quantization group size")
    parser.add_argument("--input_mode", type=str,
                        default='dynamic', help="quantization type")
    parser.add_argument("--input_asym", dest="input_asym", default=True,
                        action="store_true", help="Set input_asym to True")
    parser.add_argument("--input_sym", dest="input_asym",
                        action="store_false", help="Set input_asym to False")
    parser.set_defaults(input_asym=False)
    parser.add_argument("--k_bits", type=int, default=16, help="")
    parser.add_argument("--v_bits", type=int, default=16, help="")
    parser.add_argument("--kv_group_size", type=int,
                        default=32, help="default as head-wise")
    parser.add_argument("--k_pre_rope", action="store_true", default=True)
    parser.add_argument("--kv_mode", type=str,
                        default='static', help="quantization type")
    parser.add_argument("--kv_asym", dest="kv_asym",
                        action="store_true", help="Set kv_asym to True")
    parser.add_argument("--kv_sym", dest="kv_asym",
                        action="store_false", help="Set kv_asym to False")
    parser.set_defaults(kv_asym=False)
    parser.add_argument("--mse_init", action="store_true", default=True,
                        help="init step size through MSE instead of MIN-MAX")
    parser.add_argument("--asym_mse_init", action="store_true",
                        help="init step size through MSE instead of MIN-MAX")
    parser.add_argument("--skip_qk_weight_init", action="store_true")
    parser.add_argument("--block_qk_weight_init", action="store_true")
    parser.add_argument("--mse_init_size", type=int, default=8,
                        help="sample number used in mse_init; actually, even 4 or 2 is enough")
    parser.add_argument("--fp_mse_init", action="store_true",
                        help="use full-precision block input during the mse init process")
    # ----------------- rotation and prefix setting------------------------------------
    parser.add_argument("--pre_rotate", action="store_true")
    parser.add_argument("--rotate_mode", type=str, default='hadamard')
    parser.add_argument("--down_online_had", action="store_true", default=True)
    parser.add_argument("--qk_online_had", action="store_true", default=True)
    parser.add_argument("--set_prefixed_tokens", action="store_true", default=True)
    parser.add_argument("--outlier_threshold", type=int, default=64,
                        help="\eta in Eq.(3), indicating the oitlier threshold ratio detect outlier tokens, ")
    parser.add_argument("--activation_clipping", action="store_true",
                        help="layer-wise activation clipping for dynamic quantization")
    # -----------------training setting------------------------------------
    parser.add_argument("--quant_lr", type=float, default=5e-5,
                        help="lr of quantization parameters (s and z)")
    parser.add_argument("--weight_lr", type=float,
                        default=5e-6, help="lr of fp weights")
    parser.add_argument("--min_lr_factor", type=float,
                        default=10, help="min_lr = lr/min_lr_factor")
    parser.add_argument("--clip_grad", type=float, default=0.3)
    parser.add_argument("--wd", type=float, default=0, help="weight decay")
    parser.add_argument("--off_load_to_disk", action="store_true", default=False,
                        help="save training dataset to disk, saving CPU memory but may reduce training speed")
    parser.add_argument("--use_fp32", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--early_stop", type=int, default=0,
                        help="early stoping after validation loss do not decrease")
    parser.add_argument("--constant_wlr", action="store_true")
    parser.add_argument("--train_size", type=int, default=512,
                        help="Number of calibration data samples.")
    parser.add_argument("--val_size", type=int, default=64,
                        help="Number of validation data samples.")
    parser.add_argument("--training_seqlen", type=int,
                        default=1024, help="lenth of the training sequence.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--calib_dataset", type=str, default="pile",
                        choices=["wikitext2", "ptb", "c4",
                                 "mix", "redpajama", "pile"],
                        help="Where to extract calibration data from.")
    parser.add_argument("--batch_size", type=int,
                        default=4, help="batch size.")
    parser.add_argument("--loss_type", type=str, default="mse", help="")
    parser.add_argument("--training_target", type=str, default="fp_input",
                        choices=["fp_input", "quant_input", "both"],
                        help="what is the source of the input to obatin the training target")
    # -----------------evaluation setting------------------------------------
    parser.add_argument("--ppl_seqlen", type=int, default=2048,
                        help="lenth of the training sequence.")
    parser.add_argument("--seed", type=int, default=2,
                        help="Seed for sampling the calibration data.")
    parser.add_argument("--eval_ppl", action="store_true", default=True,
                        help="evaluate perplexity on wikitext2 and c4 with 2048 context length")
    parser.add_argument("--eval_tasks", type=str, default="winogrande",
                        help="exampe:piqa,arc_easy,arc_challenge,hellaswag,winogrande")
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--max_memory", type=str,
                        default="70GiB", help="The maximum memory of each GPU")
    # ------------------ others ------------------------------------------
    parser.add_argument("--max_outlier", type=float, default=5, help="")
    parser.add_argument("--max_item_index", type=int, default=5, help="")
    parser.add_argument("--set_outlier_zero", action="store_true")
    parser.add_argument("--modified_index", type=int, default=0, help="")
    # ------------------ ablation ------------------------------------------
    parser.add_argument("--ablate_prefix_number",
                        type=int, default=None, help="")
    # endregion

    args = parser.parse_args()

    inputTensor = torch.randn(size=(1, 16, 32)).cuda()
    # act = nn.SiLU().cuda()
    fc = nn.Linear(in_features=32, out_features=64, bias=False).cuda()
    # fcQ = int_linear_fake.QuantizedLinear(fc)
    # init_weight_quantizer(args, fcQ)
    # init_output_act_quantizer(args, fcQ)
    # act_base = f"act_fn"
    fc_base = f"fc"
    # fcQ_base = f"fcQ"
    level = 20
    configSpike = SpikeConfigTTFSFormer(nameLogger="temp test",
                                        precision=2 ** level,
                                        debug=False)

    hooks = []
    monitors = {}


    # region 关于 silu 的hooks
    def _reg_monitor(monitors: Dict[str, BaseMonitor],
                     name: str,
                     MonitorCls: type[BaseMonitor]) -> BaseMonitor:
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


    # 输入 -> monitor_0
    # def _pre_act(mod, args, kwargs, k=f"{act_base}.monitor_0"):
    #     x = args[0] if len(args) else kwargs.get("input", None)
    #     if x is not None:
    #         _reg_monitor(monitors, k, MonitorCls)(x)
    #     return None  # 必须返回 None，表示不改输入
    #
    #
    # # 输出 -> monitor_o
    # def _post_act(mod, args, out, k=f"{act_base}.monitor_o"):
    #     _reg_monitor(monitors, k, MonitorCls)(out)

    # endregion

    # region 关于 linear 的hooks
    def pre_input(monitors, base_name: str, MonitorCls):
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


    # endregion
    # hooks.append(act.register_forward_pre_hook(_pre_act, with_kwargs=True))
    # hooks.append(act.register_forward_hook(_post_act))

    hooks.append(fc.register_forward_pre_hook(
        pre_input(monitors, fc_base, MonitorCls),
        with_kwargs=True
    ))

    hooks.append(fc.register_forward_pre_hook(
        pre_input(monitors, fc_base, MonitorCls),
        with_kwargs=True
    ))

    hooks.append(fc.register_forward_hook(
        lambda mod, inp, out, b=fc_base: _reg_monitor(monitors, f"{fc_base}.monitor_o", MonitorCls)(out)
    ))
    # hooks.append(fcQ.register_forward_hook(
    #     lambda mod, inp, out, b=fcQ_base: _reg_monitor(monitors, f"{fcQ_base}.monitor_o", MonitorCls)(out)
    # ))
    outputTensor = fc(inputTensor)
    # outputTensorQ = fcQ(inputTensor)
    fc.monitors = monitors
    # fcQ.monitors = monitors
    # print(f"monitors: {act.monitors}")
    monitor_data = get_monitor_data(fc)
    # monitor_data = get_monitor_data(fcQ)
    print(f"monitor data:{monitor_data}")

    print(f"input:{inputTensor}")
    print(f"output:{outputTensor}")
    # print(f"output from Qfc:{outputTensorQ}")

    param = SpikeParamDict(value=monitor_data)
    param = param.child(fc_base)
    # spikingAct = ann_to_snn(act,
    #                         config=configSpike,
    #                         in_range=param.at("monitor_0"),
    #                         out_range=param.at("monitor_o"))
    spikeNet = SpikingWrapped(inner=fc,
                              out_range=param.at(f"monitor_o"),
                              config=configSpike)
    # spikeInput = value_to_ttfs(inputTensor, param.at("monitor_0"))
    spikeInput = inputTensor
    # print(f"Spike Input:{spikeInput}")
    SpikeOutput = spikeNet(spikeInput)
    print(f"Spike Output:{SpikeOutput}")
    compare_tensors(outputTensor, SpikeOutput)