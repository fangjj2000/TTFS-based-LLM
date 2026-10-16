import logging
import re
import sys
from pprint import pformat
from typing import Dict, Optional

import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from .engine import get_syops_pytorch
from .utils import syops_to_string, params_to_string
import numpy as np

# ssa_info = {'depth': 8, 'Nheads': 8, 'embSize': 384, 'patchSize': 14, 'Tsteps': 4}  # lifconvbn-8-384
# ssa_info = {'depth': 8, 'Nheads': 8, 'embSize': 512, 'patchSize': 14, 'Tsteps': 4}  # lifconvbn-8-512
# ssa_info = {'depth': 12, 'Nheads': 12, 'embSize': 768, 'patchSize': 16, 'Tsteps': 64}  # base
# ssa_info = {'depth': 12, 'Nheads': 6, 'embSize': 384, 'patchSize': 16, 'Tsteps': 16}  # small


# ssa_info = {'depth': 24, 'Nheads': 16, 'embSize': 1024, 'patchSize': 16, 'Tsteps': 32}  # large

def replace_decimal_strings(input_string):
    pattern = r'\.(\d+)'
    replaced_string = re.sub(pattern, r'[\1]', input_string)
    return replaced_string


def get_energy_cost(model, nameLogger: str, verbose: bool = False) -> Dict[str, float]:
    logger = logging.getLogger(nameLogger)
    logger.info(f'开始计算能耗...')
    # 递归聚合整个模型的 syops（包括复合模块和子模块）
    if hasattr(model, 'accumulate_syops'):
        syops = model.accumulate_syops()
        syops = np.array([item / model.__batch_counter__ for item in syops])
    else:
        raise RuntimeError("模型未注册 accumulate_syops，可能未执行 add_syops_counting_methods() 或 syops hook 未生效")
    # syops = model

    total_syops, total_ac, total_mac, _ = syops

    if verbose:
        logger.info(f"统计得到：Total = {total_syops}, AC = {total_ac}, MAC = {total_mac}")

    # 单位转换（G）
    total_mac_G = total_mac / 1e9
    total_ac_G = total_ac / 1e9

    # 能耗计算
    E_mac = total_mac_G * 4.6  # mJ
    E_ac = total_ac_G * 0.9  # mJ
    E_total = E_mac + E_ac

    logger.info(f"Number of operations: {total_mac_G:.3f} G MACs, {total_ac_G:.3f} G ACs")
    logger.info(f"Energy consumption: MAC = {E_mac:.3f} mJ, AC = {E_ac:.3f} mJ, Total = {E_total:.3f} mJ")

    return dict(
        Nmac=total_mac_G,
        Nac=total_ac_G,
        E_mac=E_mac,
        E_ac=E_ac,
        E_total=E_total
    )


def get_model_complexity_info(model,
                              input_res,
                              ssa_info,
                              nameLogger: str,
                              dataloader=None,
                              numBatches_for_calculate_energy: Optional[int] = 10,
                              print_per_layer_stat=True,
                              as_strings=True,
                              input_constructor=None,
                              ost=sys.stdout,
                              verbose=False,
                              ignore_modules=[],
                              custom_modules_hooks={},
                              backend='pytorch',
                              syops_units=None,
                              param_units=None,
                              output_precision=2):
    assert not isinstance(model, DistributedDataParallel), f"为了最好的兼容性，传入的model不能是分布式的模型。"
    logger = logging.getLogger(nameLogger)
    logger.info(f"ssa_info: {ssa_info}")
    logger.info(f"model:\n{pformat(model)}")
    # assert type(input_res) is tuple
    # assert len(input_res) >= 1
    assert isinstance(model, nn.Module)
    real_Tsteps = ssa_info.get("Tsteps", 1)  # 从 ssa_info 中读取时间步（默认为 1）

    model.__Tsteps__ = real_Tsteps  # 设置主模型
    model.apply(lambda m: setattr(m, "__Tsteps__", real_Tsteps))  # 向所有子模块传播

    if backend == 'pytorch':
        syops_count, params_count, syops_model = get_syops_pytorch(model,
                                                                   input_res,
                                                                   nameLogger,
                                                                   dataloader,
                                                                   numBatches_for_calculate_energy,
                                                                   print_per_layer_stat,
                                                                   input_constructor,
                                                                   ost,
                                                                   verbose,
                                                                   ignore_modules,
                                                                   custom_modules_hooks,
                                                                   output_precision=output_precision,
                                                                   syops_units=syops_units,
                                                                   param_units=param_units)
        # calculate energy consumption according to E_mac = 4.6 pJ and E_ac = 0.9 pJ
        get_energy_cost(syops_model, nameLogger, verbose)


    else:
        errMsg = ValueError('Wrong backend name')
        logger.error(f"计算复杂度遇到错误：{errMsg}")
    if as_strings:
        syops_string = syops_to_string(
            syops_count[0],
            units=syops_units,
            precision=output_precision
        )
        ac_syops_string = syops_to_string(
            syops_count[1],
            units=syops_units,
            precision=output_precision
        )
        mac_syops_string = syops_to_string(
            syops_count[2],
            units=syops_units,
            precision=output_precision
        )
        params_string = params_to_string(
            params_count,
            units=param_units,
            precision=output_precision
        )
        return [syops_string, ac_syops_string, mac_syops_string], params_string

    return syops_count, params_count
