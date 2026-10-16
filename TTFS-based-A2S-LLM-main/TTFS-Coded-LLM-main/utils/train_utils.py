import inspect
import json
import logging
import os
import re
import sys
from contextlib import contextmanager
from io import StringIO
from math import inf
from types import SimpleNamespace

import torch
from termcolor import colored
from typing_extensions import LiteralString

ANSI_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")


def ampscaler_get_grad_norm(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(),
                                                        norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.amp.GradScaler("cuda")

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True,
                 retain_graph=False):
        self._scaler.scale(loss).backward(create_graph=create_graph, retain_graph=retain_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = ampscaler_get_grad_norm(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def create_logger(dirLog: str | LiteralString | bytes,
                  t: str,
                  dist_rank: int = 0,
                  name: str = "training"):
    # create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # create formatter
    fmt = '[%(asctime)s %(name)s] (%(filename)s %(lineno)d): %(levelname)s %(message)s'
    color_fmt = colored('[%(asctime)s %(name)s]', 'green') + \
                colored('(%(filename)s %(lineno)d)', 'yellow') + \
                ': %(levelname)s %(message)s'

    # create console handlers for master process
    if dist_rank == 0:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(
            logging.Formatter(fmt=color_fmt, datefmt='%Y-%m-%d %H:%M:%S'))
        logger.addHandler(console_handler)

    # create file handlers

    file_handler = logging.FileHandler(os.path.join(dirLog, f'{name}_{t}_rank-{dist_rank}.log'),
                                       mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    return logger


@contextmanager
def capture_logs_plain(logger: logging.Logger):
    buf = StringIO()
    h = logging.StreamHandler(buf)
    h.setLevel(logging.INFO)
    h.setFormatter(logging.Formatter("%(message)s"))  # 只捕获消息体
    logger.addHandler(h)
    try:
        yield buf
    finally:
        logger.removeHandler(h)
        h.close()


@contextmanager
def capture_logs(logger: logging.Logger,
                 level: int = logging.INFO,
                 use_same_formatter: bool = True,
                 set_propagate: bool = None):
    """
    临时给 logger 加一个内存 StreamHandler，退出时移除，并返回捕获到的文本。方便最后汇总打印所有的测试结果。
    """
    stream = StringIO()
    mem_handler = logging.StreamHandler(stream)
    mem_handler.setLevel(level)

    # 复用现有 formatter（若有）
    if use_same_formatter and logger.handlers:
        fmt = logger.handlers[0].formatter
        if fmt is not None:
            mem_handler.setFormatter(fmt)

    logger.addHandler(mem_handler)
    old_propagate = logger.propagate
    if set_propagate is not None:
        logger.propagate = set_propagate

    try:
        yield stream  # 在 with 块内部可随时 read/getvalue
    finally:
        logger.removeHandler(mem_handler)
        logger.propagate = old_propagate
        mem_handler.close()


def debug_print(*args, **kwargs):
    frame = inspect.currentframe().f_back  # 调用者的栈帧
    info = inspect.getframeinfo(frame)
    print(f"[{info.filename}:{info.lineno}]", *args, **kwargs)


def cleanup_memory(verbos=True) -> None:
    """Run GC and clear GPU memory."""
    import gc
    import inspect
    caller_name = ''
    try:
        caller_name = f' (from {inspect.stack()[1].function})'
    except (ValueError, KeyError):
        pass

    def total_reserved_mem() -> int:
        return sum(torch.cuda.memory_reserved(device=i) for i in range(torch.cuda.device_count()))

    memory_before = total_reserved_mem()

    # gc.collect and empty cache are necessary to clean up GPU memory if the model was distributed
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        memory_after = total_reserved_mem()
        if verbos:
            logging.info(
                f"GPU memory{caller_name}: {memory_before / (1024 ** 3):.2f} -> {memory_after / (1024 ** 3):.2f} GB"
                f" ({(memory_after - memory_before) / (1024 ** 3):.2f} GB)"
            )


def save_dict_as_json(data, data_path):
    """
    Saves a dictionary as a JSON file to the specified path.
    
    Parameters:
    - data (dict): The dictionary to be saved.
    - data_path (str): The file path where the JSON file will be saved.
    """
    with open(data_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)


def load_json_as_dict(file_path):
    """
    Loads a JSON file from the specified path and returns its content as a dictionary.
    
    Parameters:
    - file_path (str): The file path of the JSON file to be loaded.
    
    Returns:
    - dict: The content of the JSON file as a dictionary.
    """
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
    return data


def load_json_as_namespace(file_path):
    """
    Loads a JSON file from the specified path and returns its content as a SimpleNamespace.
    
    Parameters:
    - file_path (str): The file path of the JSON file to be loaded.
    
    Returns:
    - SimpleNamespace: The content of the JSON file as a SimpleNamespace.
    """
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
    return json.loads(json.dumps(data), object_hook=lambda d: SimpleNamespace(**d))
