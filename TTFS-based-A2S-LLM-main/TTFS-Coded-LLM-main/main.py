import argparse
import os
import random
import sys
import time
from pathlib import Path
from pprint import pformat

import numpy as np
import torch
from accelerate import infer_auto_device_map, dispatch_model
from accelerate.hooks import remove_hook_from_module
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, LlamaForCausalLM

import utils.model_utils as model_utils
import utils.rotation_utils as rotation_utils
from quantize.block_ap import block_ap
from quantize.int_linear_real import load_quantized_model
from utils import train_utils
from utils.data_utils import get_loaders
from utils.eval_utils import evaluate, energy_calculation
from utils.quant_utils import init_weight_quantizer, init_output_act_quantizer, \
    register_online_had, \
    get_act_stat, get_quant_config, init_identity_quantizer, init_intput_act_quantizer, wrap_to_quant_model
from utils.snn_utils import wrap_to_snn_model
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
is_in_energy_test = False
nameLogger = "Ablation-Llama-2-13B"
levelLogger = "INFO"
# 彻底关闭联网
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"


def main():
    # region ==== 参数 ====
    parser = argparse.ArgumentParser()
    # -----------------model setting------------------------------------
    parser.add_argument("--model_path", type=str, help="model path")
    parser.add_argument("--model_name", type=str, default=None,
                        help="model name, for the saving of corresponding data cache")
    parser.add_argument("--cache_dir", default="./cache", type=str,
                        help="cache dir of dataset, leading to faster debug")
    parser.add_argument("--offload_dir", default="./offload", type=str,
                        help="dir for offload model weights")
    parser.add_argument("--output_dir", default="./log/",
                        type=str, help="direction of logging file")
    parser.add_argument("--save_quant_dir", default=None,
                        type=str, help="direction for saving quantization model")
    parser.add_argument("--real_quant", default=False, action="store_true",
                        help="use real quantization instead of fake quantization, can reduce memory footprint")
    parser.add_argument("--resume_quant", type=str, default=None,
                        help="model path of resumed quantized model")
    # -----------------quantization setting------------------------------------
    parser.add_argument("--wbits", type=int, default=16,
                        help="quantization bits")
    parser.add_argument("--w_group_size", type=int,
                        default=-1, help="quantization group size")
    parser.add_argument("--w_asym", dest="w_asym",
                        action="store_true", help="Set w_asym to True")
    parser.add_argument("--w_sym", dest="w_asym",
                        action="store_false", help="Set w_asym to False")
    parser.set_defaults(w_asym=False)
    parser.add_argument("--input_bits", type=int,
                        default=16, help="quantization bits")
    parser.add_argument("--input_group_size", type=int,
                        default=-1, help="quantization group size")
    parser.add_argument("--input_mode", type=str,
                        default='dynamic', help="quantization type")
    parser.add_argument("--input_asym", dest="input_asym",
                        action="store_true", help="Set input_asym to True")
    parser.add_argument("--input_sym", dest="input_asym",
                        action="store_false", help="Set input_asym to False")
    parser.add_argument("--output_bits", type=int,
                        default=16, help="out quantization bits")
    parser.add_argument("--output_group_size", type=int,
                        default=-1, help="out quantization group size")
    parser.add_argument("--output_mode", type=str,
                        default='dynamic', help="out quantization type")
    parser.add_argument("--output_asym", dest="output_asym",
                        action="store_true", help="Set output_asym to True")
    parser.set_defaults(input_asym=False)
    parser.add_argument("--k_bits", type=int, default=16, help="")
    parser.add_argument("--v_bits", type=int, default=16, help="")
    parser.add_argument("--kv_group_size", type=int,
                        default=128, help="default as head-wise")
    parser.add_argument("--k_pre_rope", action="store_true")
    parser.add_argument("--kv_mode", type=str,
                        default='dynamic', help="quantization type")
    parser.add_argument("--kv_asym", dest="kv_asym",
                        action="store_true", help="Set kv_asym to True")
    parser.add_argument("--kv_sym", dest="kv_asym",
                        action="store_false", help="Set kv_asym to False")
    parser.set_defaults(kv_asym=False)
    parser.add_argument("--mse_init", action="store_true",
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
    parser.add_argument("--down_online_had", action="store_true")
    parser.add_argument("--qk_online_had", action="store_true")
    parser.add_argument("--set_prefixed_tokens", action="store_true")
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
    parser.add_argument("--epochs", type=int, default=0)
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
    parser.add_argument("--ppl_datasets", type=str,
                        default="c4,pile,ptb,wikitext2,redpajama",
                        help="datasets for ppl test.")
    parser.add_argument("--seed", type=int, default=2,
                        help="Seed for sampling the calibration data.")
    parser.add_argument("--eval_ppl", action="store_true",
                        help="evaluate perplexity on wikitext2 and c4 with 2048 context length")
    parser.add_argument("--eval_tasks", type=str, default="",
                        help="exampe:piqa,arc_easy,arc_challenge,hellaswag,winogrande")
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--max_memory", type=str,
                        default="65GiB", help="The maximum memory of each GPU")
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
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    n_gpus = torch.cuda.device_count()
    max_mem = {i: args.max_memory for i in range(n_gpus)}
    max_mem["cpu"] = "256GiB"  # 可选
    r"""
    random.seed(seed) —— 控制 Python 内置 random 模块
    numpy.random.seed(seed) —— 控制 NumPy
    torch.manual_seed(seed) 和 torch.cuda.manual_seed_all(seed) —— 控制 PyTorch CPU/GPU
    还会设置 torch.backends.cudnn.deterministic=True 和 torch.backends.cudnn.benchmark=False，帮助提升复现性
    """
    # manual_seed(args.seed)

    # init logger
    dir_current_experiment = (f"{args.model_name}_"
                              f"W{args.wbits}I{args.input_bits}O{args.output_bits}"
                              f"_E{args.epochs}"
                              f"_lrW-{args.weight_lr}"
                              f"_lrQ-{args.quant_lr}")
    if args.output_dir:
        args.output_dir = os.path.join(args.output_dir,
                                       dir_current_experiment)
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.cache_dir:
        Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    if args.save_quant_dir:
        args.save_quant_dir = os.path.join(args.save_quant_dir,
                                           dir_current_experiment)
        Path(args.save_quant_dir).mkdir(parents=True, exist_ok=True)
    timeLocal = time.strftime("%Y-%m-%d_%X", time.localtime())
    logger = create_logger(dirLog=args.output_dir,
                           t=timeLocal,
                           name=nameLogger)
    logger.setLevel(levelLogger)
    logger.info(f"used args:{pformat(vars(args))}")
    logger.info(f"\n日志目录为：{args.output_dir}"
                f"\n量化模型保存目录为：{args.save_quant_dir}")

    if args.model_name is None:
        args.model_name = args.model_path.split('/')[-1]
        raise RuntimeError("model_name is None, setting as {args.model_name}")

    if args.resume_quant:
        # directly load quantized model for evaluation
        logger.info(f"直接测试量化过的模型：\n{args.resume_quant}")
        model, tokenizer = load_quantized_model(
            args.resume_quant, args.wbits, args.group_size)
    else:
        # load fp quantized model
        config = AutoConfig.from_pretrained(args.model_path,
                                            trust_remote_code=True)

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
        # region 能耗测试
        if is_in_energy_test:
            typeModelForEnergy = "【原模型】"
            energy_calculation(args=args,
                            nameLogger=nameLogger,
                            model=model,
                            typeModel=typeModelForEnergy,
                            nameDataset="pile",
                            tokenizer=tokenizer,
                            numBatches_for_calculate_energy=10,
                            ssa_info={'depth': 32, 'Nheads': 32, 'embSize': 4096, 'Tsteps': 1})
            logger.info(f"{typeModelForEnergy}能耗测试完毕！\n\n")
        # endregion

        logger.info(f"现在对【原】模型进行测试。。。")
        logsVanilla = evaluate(args.ppl_datasets, model, tokenizer, None, args, logger)

        if args.pre_rotate:
            rotation_utils.fuse_layer_norms(model)
            rotation_utils.rotate_model(
                model, rotate_mode=args.rotate_mode, online=args.down_online_had)
            model.half()
        logger.info(f"经过 【线性层+归一化层融合】 以及 【旋转】 之后的模型为:\n{pformat(model)}")
        for param in model.parameters():
            param.requires_grad = False
        wrap_to_quant_model(model, args, nameLogger)
        logger.info(f"量化后的模型:\n{pformat(model)}")
        # register on-line hadamard transformation
        if args.pre_rotate and args.down_online_had:
            register_online_had(model)
        # wrap rope for online_had and rope output capture
        rope_function_name = model_utils.get_rope_function_name(model)
        layers = model_utils.get_layers(model)
        for layer in layers:
            rotation_utils.add_qk_rotation_wrapper_after_function_call_in_forward(
                layer.self_attn,
                rope_function_name,
                config=model.config,
                online_had=args.qk_online_had)
        logger.info(f"量化以及后处理之后的模型为:\n{pformat(model)}")

        prefixed_tokens = None
        prefixed_key_values = None
        args.prefixed_length = 0
        activation_stat = None
        include_static = ((args.input_mode == "static" and args.input_bits < 16)
                          or (args.kv_mode == "static" and (args.k_bits < 16 or args.v_bits < 16))
                          or (args.output_mode == "static" and args.output_bits < 16))

        if args.set_prefixed_tokens or include_static:
            from utils.stat_utils import get_prefixed_tokens
            # model and data prepaer
            if model.device.type == 'cpu':
                original_device = 'cpu'
                block_class_name = model.model.layers[0].__class__.__name__
                device_map = infer_auto_device_map(model, max_memory={i: args.max_memory for i in
                                                                      range(torch.cuda.device_count())},
                                                   no_split_module_classes=[block_class_name])
                model = dispatch_model(model, device_map=device_map)
            else:
                original_device = 'cuda'

            cal_dataloader, _ = get_loaders(
                args.calib_dataset,
                nameLogger,
                tokenizer,
                train_size=64,
                val_size=0,
                seed=args.seed,
                seqlen=512,
                test_only=False
            )

            # get prefixed tokens
            if args.set_prefixed_tokens:
                tick = time.time()
                prefixed_tokens = get_prefixed_tokens(cal_dataloader,
                                                      model,
                                                      tokenizer,
                                                      args.model_name,
                                                      outlier_threshold=args.outlier_threshold,
                                                      activation_type='down_proj')
                logger.info(
                    f"获取 {len(prefixed_tokens)} 个prefixed tokens; token id:{prefixed_tokens}; text: {tokenizer.decode(prefixed_tokens)}")
                logger.info(
                    f"获取 prefixed token 耗时:{time.time() - tick:.0}s")
                model.config.prefixed_tokens = prefixed_tokens
                args.prefixed_length = len(prefixed_tokens)
                use_cache = model.config.use_cache
                model.config.use_cache = True
                if args.ablate_prefix_number is not None:
                    prefixed_tokens = prefixed_tokens[:args.ablate_prefix_number]
                    logger.info(f'ablation:set prefix as {prefixed_tokens}')
                output = model(torch.tensor(
                    [prefixed_tokens], device=model.device), return_dict=True)
                prefixed_key_values = output.past_key_values
                model.config.use_cache = use_cache

            # get activation statistic for activation quantization
            if include_static:
                # assert args.input_mode == "static" or args.kv_mode == "static","mse_init require static quantization"
                logger.info(f"开始缓存网络层中间的算子的输入、输出特征。")
                activation_stat = get_act_stat(
                    model, cal_dataloader, 'max', prefixed_tokens, args.down_online_had)
                logger.info(f"网络层中间的算子的输入、输出特征计算完毕。")
            if original_device == 'cpu':
                remove_hook_from_module(model, recurse=True)
                model = model.cpu()

        # 初始化权重的量化器
        if args.wbits < 16:
            logger.info("配置权重量化器。。。")
            init_weight_quantizer(args=args,
                                  nameLogger=nameLogger,
                                  model=model,
                                  minmax_init=True)

        # 初始化输出激活值的量化器
        if args.output_bits < 16:
            logger.info("配置输出激活值 以及 恒等映射的量化器。。。")
            init_output_act_quantizer(args=args,
                                      nameLogger=nameLogger,
                                      model=model,
                                      activation_stat=activation_stat,
                                      minmax_init=True)
            init_identity_quantizer(args=args,
                                    nameLogger=nameLogger,
                                    model=model,
                                    activation_stat=activation_stat,
                                    minmax_init=True)

        # 初始化输入激活值的量化器
        if args.input_bits < 16:
            logger.info("配置输入激活值的量化器。。。")
            init_intput_act_quantizer(args=args,
                                      nameLogger=nameLogger,
                                      model=model,
                                      activation_stat=activation_stat,
                                      minmax_init=True)

        # 已经用了 Identity 层量化旋转编码后的 K, 故不用分配 k 线性层的内部量化器。
        # init K quantizer
        # if args.k_bits < 16:
        #     logger.info('init k quantizer')
        #     init_k_quantizer(args, model, activation_stat)

        # 已经用了 Identity 层量化 V_proj的输出，故不用分配 k 线性层内的量化器。
        # init V quantizer
        # if args.v_bits < 16:
        #     logger.info('init v quantizer')
        #     init_v_quantizer(args, model, activation_stat)

        train_utils.cleanup_memory()

        logger.info(f"初始化量化器之后的模型为：\n{pformat(model)}\n\n")
        if is_in_energy_test:
            # region 能耗测试
            typeModelForEnergy = "【量化模型】"
            energy_calculation(args=args,
                            nameLogger=nameLogger,
                            model=model,
                            typeModel=typeModelForEnergy,
                            nameDataset="pile",
                            tokenizer=tokenizer,
                            numBatches_for_calculate_energy=10,
                            ssa_info={'depth': 32, 'Nheads': 32, 'embSize': 4096, 'Tsteps': 1})
            logger.info(f"{typeModelForEnergy}能耗测试完毕！\n\n")
        # endregion
        # quantization
        # block_cal = (args.epoch>0 or args.mse_init)
        if (args.epochs > 0 or args.mse_init) and not is_in_energy_test:
            assert args.wbits < 16 or args.input_bits < 16 or args.output_bits < 16
            logger.info("=== 开始量化校准 ===")
            tick = time.time()
            # load calibration dataset
            if args.epochs == 0:
                # only mse init without training
                args.train_size = args.mse_init_size
                args.val_size = 0
            cache_trainloader = f'{args.cache_dir}/dataloader_{args.model_name}_{args.calib_dataset}_{args.train_size}_{args.val_size}_{args.training_seqlen}_train.cache'
            cache_valloader = f'{args.cache_dir}/dataloader_{args.model_name}_{args.calib_dataset}_{args.train_size}_{args.val_size}_{args.training_seqlen}_val.cache'
            if os.path.exists(cache_trainloader) and os.path.exists(cache_valloader):
                trainloader = torch.load(cache_trainloader)
                logger.info(f"load trainloader from {cache_trainloader}")
                valloader = torch.load(cache_valloader)
                logger.info(f"load valloader from {cache_valloader}")
            else:
                trainloader, valloader = get_loaders(
                    args.calib_dataset,
                    nameLogger,
                    tokenizer,
                    args.train_size,
                    args.val_size,
                    seed=args.seed,
                    seqlen=args.training_seqlen,
                )
                torch.save(trainloader, cache_trainloader)
                torch.save(valloader, cache_valloader)
            block_ap(model, prefixed_key_values, args,
                     trainloader, valloader, logger)

            logger.info(f"量化校准结束，耗时 {time.time() - tick} 秒\n")

    model.half()
    torch.cuda.empty_cache()
    logger.info(f"现在对【量化】模型进行测试。。。")
    logsQuant = evaluate(args.ppl_datasets, model, tokenizer, prefixed_key_values, args, logger)

    wrap_to_snn_model(model=model,
                      convert_from="QANN",
                      args=args,
                      nameLogger=nameLogger)
    # region 能耗测试
    if is_in_energy_test:
        typeModelForEnergy = "【脉冲模型】"
        energy_calculation(args=args,
                        nameLogger=nameLogger,
                        model=model,
                        typeModel=typeModelForEnergy,
                        nameDataset="pile",
                        tokenizer=tokenizer,
                        numBatches_for_calculate_energy=10,
                        ssa_info={'depth': 32, 'Nheads': 32, 'embSize': 4096, 'Tsteps': 1})
        logger.info(f"{typeModelForEnergy}能耗测试完毕！\n\n")
    # endregion
    logger.info(f"时序脉冲转换后的模型:\n{pformat(model)}\n")
    model.half()  # to make sure same evaluation results with main

    logger.info(f"现在对【脉冲】模型进行测试。。。")
    logsSpike = evaluate(args.ppl_datasets, model, tokenizer, prefixed_key_values, args, logger)
    logger.info(f"全部过程结束，现在重新打印所有测试结果。。。")
    logger.info(f"原模型：")
    for lineLog in logsVanilla.splitlines():
        logger.info(f"\t[Vanilla] {lineLog}")
    logger.info(f"\n\n\n")

    logger.info(f"量化模型：")
    for lineLog in logsQuant.splitlines():
        logger.info(f"\t[Quant] {lineLog}")
    logger.info(f"\n\n\n")

    logger.info(f"脉冲模型：")
    for lineLog in logsSpike.splitlines():
        logger.info(f"\t[Spiking] {lineLog}")
    logger.info(f"\n\n\n")

    args.save_quant_dir = None
    if args.save_quant_dir:
        logger.info("开始保存量化、校准后的模型。")
        model.save_pretrained(args.save_quant_dir)
        tokenizer.save_pretrained(args.save_quant_dir)
        torch.save(prefixed_key_values, os.path.join(args.save_quant_dir, "prefixed_key_values.pth"))
        quant_config = get_quant_config(args)
        quant_config["prefixed_tokens"] = prefixed_tokens
        train_utils.save_dict_as_json(quant_config,
                                      os.path.join(args.save_quant_dir, "prefix_quant_config.json"))

        logger.info(f"量化、校准后的模型保存至 {args.save_quant_dir} 。\n")


if __name__ == "__main__":
    print(pformat(sys.argv))
    main()
