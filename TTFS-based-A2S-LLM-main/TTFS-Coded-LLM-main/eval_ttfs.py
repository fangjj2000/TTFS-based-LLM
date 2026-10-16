import os
import random

import numpy as np

from spiking_conversion.spike import SpikeParamDict
from spiking_conversion.spike.config import SpikeConfigTTFSFormer
from spiking_conversion.util import MinMaxMonitor
from utils.monitor_tools import attach_monitors_llama_sdpa, get_monitor_data
from utils.snn_utils import wrap_to_TTFSFormer_style

# :4096:8 与 :16:8 都能让 CuBLAS 选择确定性算法；前者用更大的 workspace，速度通常更好。
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import sys
import time
from pathlib import Path
from pprint import pformat

import torch
from ignite.utils import manual_seed
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, LlamaForCausalLM

from utils.eval_utils import evaluate
from utils.train_utils import create_logger

r"""
### `torch.backends.cudnn.benchmark = True`

* **作用**：cuDNN 会在第一次遇到某个卷积形状时跑若干实现，选最快的缓存下来（后续更快）。
* **适合开**：输入尺寸**固定/极少变化**的训练或推理，且你**不强求确定性**。
* **不要开**：输入尺寸**频繁变化**（反复重新 benchmark 反而慢）；你要**确定性**（可能选到非确定实现）。

---

### `torch.autograd.set_detect_anomaly(True)`

* **作用**：反向图上逐算子检查，遇到 `NaN/Inf` 抛出带溯源的异常（定位梯度问题非常好用）。
* **适合开**：**调试阶段**、定位 `NaN/Inf`、检查自定义 `Function`/量化算子的梯度。
* **不要开**：**正常训练/推理**（开销很大，显著变慢）。

---

### `torch.use_deterministic_algorithms(True)`

* **作用**：强制只用**确定性**实现；不行就报错（可用 `warn_only=True` 降级为警告）。
* **适合开**：需要**严苛可复现**（论文、对比实验、回归测试）。
* **不要开**：追求**最高性能**；或你用到的算子没有确定实现（会报错）。

  > 若开启且用到 CuBLAS，请**进程启动前**设置
  > `CUBLAS_WORKSPACE_CONFIG=":4096:8"`（或 `":16:8"`），否则 GEMM 会报错。

---

### `torch.backends.cudnn.deterministic = True`

* **作用**：cuDNN 只选确定性算法。
* **适合开**：你要求**确定性**。
* **不要开**：追求性能且不要求完全复现。

---

### `torch.backends.cudnn.benchmark = False`

* **作用**：禁用 benchmark；与上面“确定性”搭配常用。
* **适合开**：**确定性模式**（避免 benchmark 选到非确定实现）；输入尺寸多变（benchmark 也没用）。
* **不要开**：固定形状且追求速度（可以用 `True`）。

---

### `torch.backends.cuda.matmul.allow_tf32 = False`

### `torch.backends.cudnn.allow_tf32 = False`

* **作用**：禁用 **TF32**（Ampere+ 支持）。TF32 更快但比 FP32 稍低精度。
* **适合关（False）**：**数值稳定/可复现**优先；训练对精度敏感；做严谨对比。
* **可以开（True）**：**追求速度**的训练/推理、对轻微数值差异不敏感（很多大模型默认开启 TF32 以提速）。

---

### `os.environ['TOKENIZERS_PARALLELISM'] = 'false'`

* **作用**：关闭 HuggingFace tokenizers 的多线程并行（避免多进程 DataLoader 时的**过度并行/死锁/刷屏警告**）。
* **适合设为 'false'**：你使用 `DataLoader(num_workers>0)` 或多进程推理；或者日志里频繁警告 parallelism。
* **可以不设/设为 'true'**：单进程纯 CPU 分词、你确实需要 tokenizer 自己多线程加速且不会和外部并发打架。

---

## 推荐“预设组合”

### 1) 追求**最高速度**（固定形状、可接受非确定性）

```python
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.use_deterministic_algorithms(False)
torch.backends.cuda.matmul.allow_tf32 = True   # Ampere/TF32
torch.backends.cudnn.allow_tf32 = True
os.environ['TOKENIZERS_PARALLELISM'] = 'false' # 多进程时建议仍关
```

### 2) **完全可复现**（论文/回归测试）

```python
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # 进程启动前
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
```

### 3) **调试 NaN/梯度问题**

```python
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True, warn_only=False)
torch.autograd.set_detect_anomaly(True)        # 仅调试期开
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
```

### 4) **高吞吐推理**（形状固定、GPU新架构）

```python
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.use_deterministic_algorithms(False)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # 依负载决定
```

---

## 小提示

* 以上设置应尽量**在导入/初始化模型前**完成（尤其是 `CUBLAS_WORKSPACE_CONFIG` 必须在 `import torch` 之前或进程启动前）。
* 不同 GPU / 驱动 / PyTorch 版本的确定性支持程度略有差异；若某算子仍非确定，可换等价实现或将 `warn_only=True` 过渡。
* 若既想**可复现**又想**快一点**，可以只关闭 TF32（保持 FP32），不开严格的 deterministic（或仅在关键对比时开启）。

"""

torch.backends.cudnn.benchmark = True
# torch.autograd.set_detect_anomaly(True)

# torch.use_deterministic_algorithms(True)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# torch.backends.cuda.matmul.allow_tf32 = False
# torch.backends.cudnn.allow_tf32 = False

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
qann = True
nameLogger = "eval TTFSFormer Style SNN"
levelLogger = "INFO"


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="model path of quantized model",
                        default="./pre_quantized_models/llama-3-8b-w4a4kv4")
    parser.add_argument("--model_name", type=str, default="TTFS-Former_Reproduce",
                        help="model name, for the saving monitor data")
    parser.add_argument("--output_dir", default="./log/test", type=str, help="direction of logging file")
    parser.add_argument("--real_quant", default=False, action="store_true",
                        help="use real quantization instead of fake quantization, can reduce memory footprint")
    parser.add_argument("--use_fp32", action="store_true")
    parser.add_argument("--precision", type=int, default=256,
                        help="用于 TTFSFormer 复现的参数，相当于我们的 SNN 中的T")
    parser.add_argument("--ppl_seqlen", type=int, default=2048, help="lenth of the training sequence.")
    parser.add_argument("--seed", type=int, default=2, help="Seed for sampling the calibration data.")
    parser.add_argument("--eval_ppl", action="store_true", default=False,
                        help="evaluate perplexity on wikitext2 and c4 with 2048 context length")
    parser.add_argument("--ppl_datasets", type=str,
                        default="c4,pile,ptb,ptb-new,c4-new,wikitext2",
                        help="datasets for ppl test.")
    parser.add_argument("--eval_tasks", type=str, default="piqa",
                        help="exampe:piqa,arc_easy,arc_challenge,hellaswag,winogrande")
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--max_memory", type=str, default="70GiB", help="The maximum memory of each GPU")
    parser.add_argument("--off_load_to_disk", action="store_true", default=False,
                        help="save training dataset to disk, saving CPU memory but may reduce training speed")
    parser.add_argument("--offload_dir", default="./offload", type=str,
                        help="dir for offload model weights")
    args = parser.parse_args()
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    r"""
    random.seed(seed) —— 控制 Python 内置 random 模块
    numpy.random.seed(seed) —— 控制 NumPy
    torch.manual_seed(seed) 和 torch.cuda.manual_seed_all(seed) —— 控制 PyTorch CPU/GPU
    还会设置 torch.backends.cudnn.deterministic=True 和 torch.backends.cudnn.benchmark=False，帮助提升复现性
    """
    manual_seed(args.seed)

    dir_current_experiment = (f"{args.model_name}_"
                              f"precision-{args.precision}")
    if args.output_dir:
        args.output_dir = os.path.join(args.output_dir,
                                       dir_current_experiment)
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    timeLocal = time.strftime("%Y-%m-%d_%X", time.localtime())
    logger = create_logger(dirLog=args.output_dir,
                           t=timeLocal,
                           name=nameLogger)
    logger.setLevel(levelLogger)
    logger.info(f"used args:\n{pformat(vars(args))}")
    logger.info(f"\n日志目录为：{args.output_dir}")

    # init quantized model
    config = AutoConfig.from_pretrained(args.model_path,
                                        trust_remote_code=True)
    # 将注意力层的类型改为普通实现，方便 hooks 统计以及 TTFSFormer 的脉冲改造。
    config._attn_implementation_internal = "eager"

    tokenizer = AutoTokenizer.from_pretrained(args.model_path,
                                              use_fast=False,
                                              legacy=False,
                                              trust_remote_code=True)
    dtype = torch.float16 if not args.use_fp32 else torch.float32
    model: LlamaForCausalLM = AutoModelForCausalLM.from_pretrained(args.model_path,
                                                                   config=config,
                                                                   device_map='cpu',
                                                                   torch_dtype=dtype,
                                                                   trust_remote_code=True)
    logger.info(f"原模型:\n{pformat(model)}")
    for param in model.parameters():
        param.requires_grad = False
    logger.info("量化权重加载完毕\n")

    model.half()
    monitors, teardown = attach_monitors_llama_sdpa(model, logger_name=nameLogger, MonitorCls=MinMaxMonitor)
    logger.info(f"monitors: \n{pformat(model.monitors)}")

    logger.info(f"现在对【原】模型进行测试。。。")
    logsVanilla = evaluate(args.ppl_datasets, model, tokenizer, None, args, logger)

    monitored_data = get_monitor_data(model)
    logger.info(f"monitored data: \n{pformat(monitored_data)}")
    teardown()
    monitoredParams = SpikeParamDict(value=monitored_data)
    configSpike = SpikeConfigTTFSFormer(nameLogger=nameLogger,
                                        precision=args.precision,
                                        debug=False)

    wrap_to_TTFSFormer_style(model,
                             monitoredParams,
                             configSpike,
                             nameLogger)

    logger.info(f"转换为 TTFSFormer 风格后的脉冲模型:\n{pformat(model)}")

    logger.info(f"现在对【脉冲】模型进行测试。。。")
    logsTTFSFormer = evaluate(args.ppl_datasets, model, tokenizer, None, args, logger)

    logger.info(f"全部过程结束，现在重新打印所有测试结果。。。")

    logger.info(f"原模型：")
    for lineLog in logsVanilla.splitlines():
        logger.info(f"\t[Vanilla] {lineLog}")
    logger.info(f"\n\n\n")

    logger.info(f"TTFSFormer 风格的脉冲模型：")
    for lineLog in logsTTFSFormer.splitlines():
        logger.info(f"\t[TTFSFormer] {lineLog}")
    logger.info(f"\n\n\n")


if __name__ == "__main__":
    print(pformat(sys.argv))
    main()
