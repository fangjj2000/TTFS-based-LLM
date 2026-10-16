import inspect
import logging
import sys
from functools import partial
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from progress.bar import Bar as Bar
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaMLP

from quantize.int_linear_fake import QuantizedLlamaAttention, QuantizedLlamaMLP
from spiking_conversion.model.spiking_layers import SpikingLlamaMLP, SpikingLlamaAttention

try:
    from spikingjelly.clock_driven import surrogate, neuron, functional
except:
    from spikingjelly.activation_based import surrogate, neuron, functional

from .ops import MODULES_MAPPING
from .utils import syops_to_string, params_to_string


def get_syops_pytorch(model,
                      input_res,
                      nameLogger: str,
                      dataloader=None,
                      numBatches_for_calculate_energy: Optional[int] = 10,
                      print_per_layer_stat=True,
                      input_constructor=None,
                      ost=sys.stdout,
                      verbose=False,
                      ignore_modules=[],
                      custom_modules_hooks={},
                      output_precision=3,
                      syops_units='GMac',
                      param_units='M'):
    logger = logging.getLogger(nameLogger)
    global CUSTOM_MODULES_MAPPING
    CUSTOM_MODULES_MAPPING = custom_modules_hooks
    syops_model = add_syops_counting_methods(model)
    check_syops_hooks(model, nameLogger)
    logger.info(f"初始状态，model 的 __batch_counter__ 为： {syops_model.__batch_counter__}")
    syops_model.apply(lambda m: setattr(m, '__batch_counter__', 0))  # 向所有子模块传播
    syops_model.eval()
    syops_model.start_syops_count(ost=ost,
                                  verbose=verbose,
                                  ignore_list=ignore_modules)
    if dataloader is not None:
        bar = Bar('Processing', max=len(dataloader))
        batch_idx = 0
        for batch, target in dataloader:
            batch_idx += 1
            torch.cuda.empty_cache()
            param = next(syops_model.parameters())
            batch = batch.to(device=param.device)
            with torch.no_grad():
                _ = syops_model(batch)
                logger.info(f"第 {batch_idx} 个批次，model 的 __batch_counter__ 为： {syops_model.__batch_counter__}")
                syops_model.apply(lambda m: setattr(m, '__batch_counter__', syops_model.__batch_counter__))  # 向所有子模块传播

            # functional.reset_net(syops_model) # 用于 Spikingjelly 中算子的时间重置。

            bar.suffix = '({batch}/{size})'.format(batch=batch_idx, size=len(dataloader))

            bar.next()
            if numBatches_for_calculate_energy is not None and batch_idx >= numBatches_for_calculate_energy:
                logger.info(f"能耗统计只进行 {numBatches_for_calculate_energy} 个批次的计算，现在已经达到，退出！")
                break
        bar.finish()
        # 整个网络的操作数加和除以累积的batchsize、总的参数和。未对网络中子模块操作。
        syops_count, params_count = syops_model.compute_average_syops_cost()
    else:
        if input_constructor:
            input = input_constructor(input_res)
            _ = syops_model(**input)
        else:
            try:
                if input_constructor is not None:
                    inputs = input_constructor(input_res)
                else:
                    # TODO 假设 input_res 是 token 序列长度（如：128）
                    param = next(syops_model.parameters())
                    seq_len = input_res if isinstance(input_res, int) else input_res[0]
                    batch_size = 1
                    vocab_size = getattr(syops_model, "config", None)
                    if hasattr(vocab_size, "vocab_size"):
                        vocab_size = vocab_size.vocab_size
                    else:
                        vocab_size = 32000  # 默认值

                    # TODO 构造 fake 的 input_ids 和 attention_mask
                    inputs = {
                        "input_ids": torch.randint(0, vocab_size, (batch_size, seq_len)).to(
                            device=param.device, dtype=param.dtype),
                        "attention_mask": torch.ones((batch_size, seq_len)).to(next(syops_model.parameters()).device),
                    }
                with torch.no_grad():
                    _ = syops_model(**inputs)
            except Exception as e:
                logger.error(f"输入构造或前向失败: {e}")
                raise e
        syops_count, params_count = syops_model.compute_average_syops_cost()

    if print_per_layer_stat:
        print_model_with_syops(
            syops_model,
            syops_count,
            params_count,
            ost=ost,
            syops_units=syops_units,
            param_units=param_units,
            precision=output_precision
        )
    syops_model.stop_syops_count()
    CUSTOM_MODULES_MAPPING = {}

    return syops_count, params_count, syops_model


# “自带__syops__，且参与累计的模块
SPECIAL_MODULES = (
    LlamaAttention, LlamaMLP,
    QuantizedLlamaAttention, QuantizedLlamaMLP,
    SpikingLlamaAttention, SpikingLlamaMLP,
)


def accumulate_syops(self) -> np.ndarray:
    """
    返回本模块的 syops 累积值：
      - 若本模块受支持且不是特殊层：直接返回自身 __syops__；
      - 否则：把子模块的 accumulate_syops() 累加，
        起点为：特殊层→自身 __syops__；非特殊层→全 0。
    """
    # 受支持且非特殊层：直接返回自身统计
    if is_supported_instance(self) and not isinstance(self, SPECIAL_MODULES):
        return np.asarray(getattr(self, "__syops__", [0.0, 0.0, 0.0, 100.0]), dtype=float)

    # 其它情况：以“起点”+子模块累加
    # 特殊层：从自身 __syops__ 开始；非特殊层：从零开始
    total = (np.array(getattr(self, "__syops__", [0.0, 0.0, 0.0, 100.0]),
                      dtype=float, copy=True)
             if isinstance(self, SPECIAL_MODULES)
             else np.zeros(4, dtype=float))

    for child in self.children():
        # 假设所有子模块都实现了 accumulate_syops；如不确定可加 hasattr 判断
        total += np.asarray(child.accumulate_syops(), dtype=float)

    return total


def check_syops_hooks(model, nameLogger: str):
    logger = logging.getLogger(nameLogger)
    logger.info("\n🔍 正在检查未加 __syops__ 属性的模块...\n")
    unhooked = []

    for name, module in model.named_modules():
        # 跳过容器模块（如 Sequential、ModuleList 等）和模型本体
        if len(list(module.children())) > 0:
            continue
        if not hasattr(module, "__syops__"):
            unhooked.append((name, type(module)))

    if not unhooked:
        logger.info("✅ 所有模块都已正确注册 __syops__ 属性！")
    else:
        logger.info(f"⚠️ 发现 {len(unhooked)} 个模块未注册 __syops__ 属性：\n")
        for name, mod_type in unhooked:
            logger.info(f" - {name:<50} ({mod_type})")


def print_model_with_syops(model,
                           total_syops,
                           total_params,
                           syops_units='GMac',
                           param_units='M',
                           precision=3,
                           ost=sys.stdout):
    for i in range(3):
        if total_syops[i] < 1:
            total_syops[i] = 1
    if total_params < 1:
        total_params = 1

    def accumulate_params(
            self):  # 如果本module在MODULES_MAPPING或CUSTOM_MODULES_MAPPING中，则直接输出记录的__params__，否则将其子module的__params__累积加和，即该函数只返回self.__params__（如果有子模块，则是累积和）
        if is_supported_instance(self):
            return self.__params__
        else:
            sum = 0
            for m in self.children():
                sum += m.accumulate_params()  # 循环递归调用，对整个网络结构进行计算
            return sum

    def syops_repr(self):
        accumulated_params_num = self.accumulate_params()  # 返回self.__params__（如果有子模块，则是累积和）
        accumulated_syops_cost = self.accumulate_syops()  # 返回self.__syops__（如果有子模块，则是累积和）
        accumulated_syops_cost[0] /= model.__batch_counter__  # 取均值
        accumulated_syops_cost[1] /= model.__batch_counter__
        accumulated_syops_cost[2] /= model.__batch_counter__
        accumulated_syops_cost[3] /= model.__times_counter__  # 取均值
        # store info for later analysis
        self.accumulated_params_num = accumulated_params_num
        self.accumulated_syops_cost = accumulated_syops_cost
        return ', '.join([self.original_extra_repr(),
                          params_to_string(accumulated_params_num,
                                           units=param_units, precision=precision),
                          '{:.3%} Params'.format(accumulated_params_num / total_params),
                          syops_to_string(accumulated_syops_cost[0],
                                          units=syops_units, precision=precision),
                          '{:.3%} oriMACs'.format(accumulated_syops_cost[0] / total_syops[0]),
                          syops_to_string(accumulated_syops_cost[1],
                                          units=syops_units, precision=precision),
                          '{:.3%} ACs'.format(accumulated_syops_cost[1] / total_syops[1]),
                          syops_to_string(accumulated_syops_cost[2],
                                          units=syops_units, precision=precision),
                          '{:.3%} MACs'.format(accumulated_syops_cost[2] / total_syops[2]),
                          '{:.3%} Spike Rate'.format(accumulated_syops_cost[3] / 100.),
                          'SpkStat: {}'.format(self.__spkhistc__)])  # print self.__spkhistc__
        # self.original_extra_repr()])
        # return ', '.join([params_to_string(accumulated_params_num,
        #                                    units=param_units, precision=precision),
        #                   '{:.3%} Params'.format(accumulated_params_num / total_params),
        #                   syops_to_string(accumulated_syops_cost[1],
        #                                   units=syops_units, precision=precision),
        #                   '{:.3%} ACs'.format(accumulated_syops_cost[1] / total_syops[1]),
        #                   syops_to_string(accumulated_syops_cost[2],
        #                                   units=syops_units, precision=precision),
        #                   '{:.3%} MACs'.format(accumulated_syops_cost[2] / total_syops[2]),
        #                   '{:.3%} Spike Rate'.format(accumulated_syops_cost[3] / 100.)])
        #                   #self.original_extra_repr()])

    def syops_repr_empty(self):
        return ''

    def add_extra_repr(m):
        m.accumulate_syops = accumulate_syops.__get__(m)  # 为module增加属性accumulate_syops
        m.accumulate_params = accumulate_params.__get__(m)
        if is_supported_instance(m):
            syops_extra_repr = syops_repr.__get__(m)
        else:
            syops_extra_repr = syops_repr_empty.__get__(m)
        if m.extra_repr != syops_extra_repr:
            m.original_extra_repr = m.extra_repr  # backup原先的extra_repr
            m.extra_repr = syops_extra_repr  # 将syops_extra_repr作为extra_repr
            assert m.extra_repr != m.original_extra_repr

    def del_extra_repr(m):
        if hasattr(m, 'original_extra_repr'):
            m.extra_repr = m.original_extra_repr
            del m.original_extra_repr
        if hasattr(m, 'accumulate_syops'):
            del m.accumulate_syops

    model.apply(add_extra_repr)
    print(repr(model), file=ost)  # 输出整个网络，且在输出过程中计算各子模块的__syops__、__params__
    model.apply(del_extra_repr)


def get_model_parameters_number(model):
    # params_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params_num = sum(p.numel() for p in model.parameters())
    return params_num


def add_syops_counting_methods(net_main_module):
    net_main_module.start_syops_count = start_syops_count.__get__(net_main_module)
    net_main_module.stop_syops_count = stop_syops_count.__get__(net_main_module)
    net_main_module.reset_syops_count = reset_syops_count.__get__(net_main_module)
    net_main_module.compute_average_syops_cost = compute_average_syops_cost.__get__(net_main_module)

    # 此处通过调用add_syops_counter_variable_or_reset(module)对各子module计算module.__params__，后面无需重新计算
    net_main_module.reset_syops_count()

    return net_main_module


def compute_average_syops_cost(self):
    """
    A method that will be available after add_syops_counting_methods() is called
    on a desired net object.

    Returns current mean syops consumption per image.
    """
    for m in self.modules():  # 递归式的为每个子模块加上属性accumulate_syops
        m.accumulate_syops = accumulate_syops.__get__(m)

    # 整个网络的操作数加和。
    # 未对网络中子模块操作：self.accumulate_syops()只返回self的__syops__
    syops_sum = self.accumulate_syops()
    syops_sum = np.array([item / self.__batch_counter__ for item in syops_sum])

    for m in self.modules():
        if hasattr(m, 'accumulate_syops'):
            del m.accumulate_syops

    params_sum = get_model_parameters_number(self)  # 整个网络的参数加和
    return syops_sum, params_sum


def _normalize_hook_signature(user_hook):
    """
    把任意风格的 forward hook 统一成：fn(module, args, kwargs, output)
    兼容：旧签名(fn(module, input, output))、新签名(fn(module, args, kwargs, output))、
         以及你已有的各种自定义 hook。
    """
    sig = inspect.signature(user_hook)
    params = list(sig.parameters)
    # 旧式：module, input, output
    if len(params) >= 3 and params[1] != 'args' and params[2] != 'kwargs':
        def _shim(module, *hook_args):
            if len(hook_args) == 2:
                args, output = hook_args
                kwargs = {}
            else:
                args, kwargs, output = hook_args
            return user_hook(module, args, output)  # 旧式原样调用

        return _shim

    # 新式：module, args, kwargs, output（或用户已经写了兼容 *rest）
    def _shim(module, *hook_args):
        if len(hook_args) == 2:
            args, output = hook_args
            kwargs = {}
        else:
            args, kwargs, output = hook_args
        return user_hook(module, args, kwargs, output)

    return _shim


def safe_register_forward_hook(module, user_hook):
    """
    优先使用 with_kwargs=True；老版本 PyTorch 自动回退。
    """
    hook_fn = _normalize_hook_signature(user_hook)
    try:
        return module.register_forward_hook(hook_fn, with_kwargs=True)  # PyTorch ≥ 2.0
    except TypeError:
        return module.register_forward_hook(hook_fn)  # 回退：老版本


def start_syops_count(self, **kwargs):
    """
    A method that will be available after add_syops_counting_methods() is called
    on a desired net object.

    Activates the computation of mean syops consumption per image.
    Call it before you run the network.

    """
    add_batch_counter_hook_function(self)  # 只给model整体，设置了 batch数，time_steps 的累加。

    seen_types = set()

    def add_syops_counter_hook_function(module, ost, verbose, ignore_list):
        if type(module) in ignore_list:
            seen_types.add(type(module))
            if is_supported_instance(module):
                module.__params__ = 0
        elif is_supported_instance(module):
            if hasattr(module, '__syops_handle__'):
                return
            if type(module) in CUSTOM_MODULES_MAPPING:
                # handle = module.register_forward_hook(CUSTOM_MODULES_MAPPING[type(module)])
                handle = safe_register_forward_hook(module, CUSTOM_MODULES_MAPPING[type(module)])
            else:
                # handle = module.register_forward_hook(MODULES_MAPPING[type(module)])
                handle = safe_register_forward_hook(module, MODULES_MAPPING[type(module)])
            module.__syops_handle__ = handle
            seen_types.add(type(module))
        else:
            if verbose and not type(module) in (nn.Sequential, nn.ModuleList) and \
                    not type(module) in seen_types:
                print('Warning: module ' + type(module).__name__ +
                      ' is treated as a zero-op.', file=ost)
            seen_types.add(type(module))

    self.apply(partial(add_syops_counter_hook_function, **kwargs))


def stop_syops_count(self):
    """
    A method that will be available after add_syops_counting_methods() is called
    on a desired net object.

    Stops computing the mean syops consumption per image.
    Call whenever you want to pause the computation.

    """
    remove_batch_counter_hook_function(self)
    self.apply(remove_syops_counter_hook_function)
    #  syops-counter 原代码这里移除了统计的操作数。以为他只关心操作数。后续能耗还要用操作数，所以这里不移除。
    # self.apply(remove_syops_counter_variables)  # keep this for later analyses


def reset_syops_count(self):
    """
    A method that will be available after add_syops_counting_methods() is called
    on a desired net object.

    Resets statistics computed so far.

    """
    add_batch_counter_variables_or_reset(self)
    self.apply(add_syops_counter_variable_or_reset)


#  region ---- Internal functions ----
def batch_counter_hook(module, input, output):
    r"""
    只对 model 整体注册，设置 batch数，time_steps 的累加。
    操作数是按 batch 归一，发放率是按 batch × Tsteps 归一。
    :param module:
    :param input:
    :param output:
    :return:
    """
    if len(input) > 0:
        # Can have multiple inputs, getting the first one
        input = input[0]
        batch_size = len(input)
    else:
        pass
        print('Warning! No positional inputs found for a module,'
              ' assuming batch size is 1.')
    module.__batch_counter__ += batch_size
    tsteps = getattr(module, "__Tsteps__", 1)
    module.__times_counter__ += tsteps


def add_batch_counter_variables_or_reset(module):
    r"""
    在 print_model_with_syops() 中定义的 syops_repr() 有这段：
    accumulated_syops_cost[3] /= model.__times_counter__ 即firing rate

    spike rate 是通过 rate × 100（在 hook 中）累加，每次 forward 加一次 Tsteps 到 __times_counter__
    然后最终输出时，除以了 __times_counter__ 来得到“平均每个时间步的发放率
    :param module:
    :return:
    """
    module.__batch_counter__ = 0
    module.__times_counter__ = 0


def add_batch_counter_hook_function(module):
    # 只对 model 整体注册，设置了 batch数，time_steps 的累加。
    if hasattr(module, '__batch_counter_handle__'):
        return
    handle = module.register_forward_hook(batch_counter_hook)
    module.__batch_counter_handle__ = handle


def remove_batch_counter_hook_function(module):
    if hasattr(module, '__batch_counter_handle__'):
        module.__batch_counter_handle__.remove()
        del module.__batch_counter_handle__


def add_syops_counter_variable_or_reset(module):
    if is_supported_instance(module):
        if hasattr(module, '__syops__') or hasattr(module, '__params__'):
            print('Warning: variables __syops__ or __params__ are already '
                  'defined for the module' + type(module).__name__ +
                  ' syops can affect your code!')
            module.__syops_backup_syops__ = module.__syops__
            module.__syops_backup_params__ = module.__params__
        module.__syops__ = np.array([0.0, 0.0, 0.0, 0.0])
        module.__params__ = get_model_parameters_number(module)
        # add __spkhistc__ for each module (by yult 2023.4.18)
        module.__spkhistc__ = None  # np.zeros(20)  # assuming there are no more than 20 spikes for one neuron


def is_supported_instance(module):
    if type(module) in MODULES_MAPPING or type(module) in CUSTOM_MODULES_MAPPING:
        return True
    return False


def remove_syops_counter_hook_function(module):
    if is_supported_instance(module):
        if hasattr(module, '__syops_handle__'):
            module.__syops_handle__.remove()
            del module.__syops_handle__


def remove_syops_counter_variables(module):
    if is_supported_instance(module):
        if hasattr(module, '__syops__'):
            del module.__syops__
            if hasattr(module, '__syops_backup_syops__'):
                module.__syops__ = module.__syops_backup_syops__
        if hasattr(module, '__params__'):
            del module.__params__
            if hasattr(module, '__syops_backup_params__'):
                module.__params__ = module.__syops_backup_params__
        # remove module.__spkhistc__ after print
        if hasattr(module, '__spkhistc__'):
            del module.__spkhistc__


# endregion


def forward_for_syops_once(model, batch):
    """
    只对这一次前向做最小处理：
      - 仅把 batch 中的 Tensor 搬到与 model 参数相同的 device（不改 dtype）
      - 若在 CUDA 上，用 autocast(dtype=参数dtype) 执行一次前向
      - 保持原有调用形式：Tensor -> model(batch); list/tuple -> model(*batch); dict -> model(**batch)
    """
    p = next(model.parameters())
    dev, p_dtype = p.device, p.dtype

    def move_to_device(x):
        if isinstance(x, torch.Tensor):
            return x.to(device=dev)  # 只搬设备，不改 dtype
        return x

    # 保持原有调用形态
    if isinstance(batch, dict):
        batch_on_dev = {k: move_to_device(v) for k, v in batch.items()}
        if dev.type == "cuda":
            with torch.cuda.amp.autocast(dtype=p_dtype):
                return model(**batch_on_dev)
        else:
            return model(**batch_on_dev)

    elif isinstance(batch, (list, tuple)):
        batch_on_dev = tuple(move_to_device(v) for v in batch)
        if dev.type == "cuda":
            with torch.cuda.amp.autocast(dtype=p_dtype):
                return model(*batch_on_dev)
        else:
            return model(*batch_on_dev)

    else:
        batch_on_dev = move_to_device(batch)
        if dev.type == "cuda":
            with torch.cuda.amp.autocast(dtype=p_dtype):
                return model(batch_on_dev)
        else:
            return model(batch_on_dev)


def remove_syops_instrumentation(model):
    """
    清理由 get_syops_pytorch / add_syops_counting_methods 等注入到模型的所有成员
    移除本脚本为统计 SyOps/能耗而动态注入到模型及其子模块上的所有方法、hook 和属性。
    可反复调用；不存在的字段会被忽略。
    移除 engine.py 为统计 SyOps/能耗而注入到模型及子模块上的所有方法、hook 和属性。
    幂等可重复调用；不存在的字段会被忽略。
    """

    # 0) 先尝试停止统计（安全移除 hooks）
    try:
        if hasattr(model, "stop_syops_count"):
            model.stop_syops_count()
    except Exception:
        pass

    # 1) 根模型层面的 batch 计数 hook / 计数器
    if hasattr(model, "__batch_counter_handle__"):
        try:
            model.__batch_counter_handle__.remove()
        except Exception:
            pass
        try:
            delattr(model, "__batch_counter_handle__")
        except Exception:
            pass
    for attr in ("__batch_counter__", "__times_counter__"):
        if hasattr(model, attr):
            try:
                delattr(model, attr)
            except Exception:
                pass

    # 2) 逐模块清理
    for m in model.modules():
        # 2.1 如果脚本里提供了“官方”还原函数，优先用（能把备份恢复回去）
        try:
            from energy_cost_Gt.engine import remove_syops_counter_variables  # 按你的包路径调整
            try:
                remove_syops_counter_variables(m)
            except Exception:
                pass
        except Exception:
            pass  # 找不到就走自定义兜底

        # 2.2 syops/params/spike 的兜底删除（以及备份字段）
        for attr in (
                "__syops__", "__params__", "__spkhistc__",
                "__syops_backup_syops__", "__syops_backup_params__",
        ):
            if hasattr(m, attr):
                try:
                    delattr(m, attr)
                except Exception:
                    pass

        # 2.3 去掉统计 hook 句柄
        if hasattr(m, "__syops_handle__"):
            try:
                m.__syops_handle__.remove()
            except Exception:
                pass
            try:
                delattr(m, "__syops_handle__")
            except Exception:
                pass

        # 2.4 还原/清理 repr 相关
        if hasattr(m, "original_extra_repr"):
            try:
                m.extra_repr = m.original_extra_repr
            except Exception:
                pass
            try:
                delattr(m, "original_extra_repr")
            except Exception:
                pass

        # 2.5 打印时写入的展示字段
        for attr in ("accumulated_params_num", "accumulated_syops_cost"):
            if hasattr(m, attr):
                try:
                    delattr(m, attr)
                except Exception:
                    pass

        # 2.6 计算均值时临时绑定的方法
        for attr in ("accumulate_syops", "accumulate_params"):
            if hasattr(m, attr):
                try:
                    delattr(m, attr)
                except Exception:
                    pass

        # 2.7 compute_average_syops_cost 可能传播到子模块的 batch 计数器（冗余）
        for attr in ("__batch_counter__", "__times_counter__"):
            if m is not model and hasattr(m, attr):
                try:
                    delattr(m, attr)
                except Exception:
                    pass

    # 3) 根模型实例上移除通过 add_syops_counting_methods() 绑定的方法
    for attr in (
            "start_syops_count",
            "stop_syops_count",
            "reset_syops_count",
            "compute_average_syops_cost",
    ):
        if hasattr(model, attr):
            try:
                delattr(model, attr)
            except Exception:
                pass

    return model
