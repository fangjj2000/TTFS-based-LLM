import logging
import os
import random
import sys
import time
from pathlib import Path
from pprint import pformat

import numpy as np
import torch
from accelerate import infer_auto_device_map, load_checkpoint_in_model
from termcolor import colored
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

from main import evaluate
from utils.snn_utils import wrap_to_snn_model
from utils.train_utils import load_json_as_namespace

torch.backends.cudnn.benchmark = True
nameLogger = "SNN_Conversion"
levelLogger = "DEBUG"


def create_logger(dirLog: str,
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

    file_handler = logging.FileHandler(os.path.join(dirLog, f'{name}_{t}.log'),
                                       mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        fmt=fmt, datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    return logger


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--quant_model_path", type=str, help="model path of quantized model")
    parser.add_argument("--output_dir", default="./log/test_snn", type=str, help="direction of logging file")
    parser.add_argument("--real_quant", default=False, action="store_true",
                        help="use real quantization instead of fake quantization, can reduce memory footprint")
    parser.add_argument("--ppl_seqlen", type=int, default=2048, help="lenth of the training sequence.")
    parser.add_argument("--seed", type=int, default=2, help="Seed for sampling the calibration data.")
    parser.add_argument("--T", type=int, default=2
                        , help="time step")
    # parser.add_argument("--L", type=int, default=8, help="spike neuron")
    parser.add_argument("--eval_ppl", action="store_true",
                        help="evaluate perplexity on wikitext2 and c4 with 2048 context length")
    parser.add_argument("--avg_neuron", action="store_true", help="set average in spike neuron")
    parser.add_argument("--eval_tasks", type=str, default="",
                        help="exampe:piqa,arc_easy,arc_challenge,hellaswag,winogrande")
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--max_memory", type=str, default="25GiB", help="The maximum memory of each GPU")
    # parser.set_defaults(avg_neuron=True)

    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    # init logger
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    timeLocal = time.strftime("%Y-%m-%d_%X", time.localtime())

    logger = create_logger(dirLog=args.output_dir,
                           t=timeLocal,
                           name=nameLogger)
    logger.setLevel(levelLogger)
    quant_config = load_json_as_namespace(os.path.join(args.quant_model_path, 'prefixequant_config.json'))
    # if quant_config['set_prefixed_tokens']:
    if quant_config.set_prefixed_tokens:
        prefixed_key_values = torch.load(os.path.join(args.quant_model_path, 'prefixed_key_values.pth'))
    else:
        prefixed_key_values = None

    logger.info(args)

    # init quantized model
    config = AutoConfig.from_pretrained(args.quant_model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.quant_model_path, use_fast=False, legacy=False,
                                              trust_remote_code=True)
    dtype = torch.float16 if not args.use_fp32 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model_path,
                                                 config=config,
                                                 device_map='cpu',
                                                 torch_dtype=dtype,
                                                 trust_remote_code=True)

    logger.info(f"原始的模型:\n{pformat(model)}")
    wrap_to_snn_model(model, args)
    logger.info(f"时序脉冲转换后的模型:\n{pformat(model)}")

    block_class_name = model.model.layers[0].__class__.__name__
    device_map = infer_auto_device_map(model,
                                       max_memory={i: args.max_memory for i in range(torch.cuda.device_count())},
                                       no_split_module_classes=[block_class_name])

    load_checkpoint_in_model(model,
                             checkpoint=args.quant_model_path,
                             device_map=device_map,
                             dtype=torch.float16,
                             offload_folder="./hf_offload",
                             )
    for param in model.parameters():
        param.requires_grad = False
    model.half()  # to make sure same evaluation results with main
    with torch.no_grad():
        evaluate(model, tokenizer, prefixed_key_values, args, logger, )


if __name__ == "__main__":
    print(sys.argv)
    main()
