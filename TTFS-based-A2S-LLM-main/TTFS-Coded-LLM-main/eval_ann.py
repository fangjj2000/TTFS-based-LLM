import os
import sys
import time
from pathlib import Path
from pprint import pformat

import torch
from ignite.utils import manual_seed
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

from main import evaluate
from utils.train_utils import create_logger

torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

nameLogger = "eval ANN"
levelLogger = "INFO"


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="model path of model")
    parser.add_argument("--model_name", type=str, default=None,
                        help="model name, for the saving monitor data")
    parser.add_argument("--output_dir", default="./log/test", type=str, help="direction of logging file")
    parser.add_argument("--use_fp32", action="store_true", default=False)
    parser.add_argument("--ppl_seqlen", type=int, default=2048, help="lenth of the training sequence.")
    parser.add_argument("--seed", type=int, default=2, help="Seed for sampling the calibration data.")
    parser.add_argument("--eval_ppl", action="store_true", default=True,
                        help="evaluate perplexity on wikitext2 and c4 with 2048 context length")
    parser.add_argument("--ppl_datasets", type=str, default="c4,pile,ptb,ptb-new,c4-new,wikitext2",
                        help="datasets for ppl test.")
    parser.add_argument("--eval_tasks", type=str, default="",
                        help="exampe:piqa,arc_easy,arc_challenge,hellaswag,winogrande")
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--max_memory", type=str, default="70GiB", help="The maximum memory of each GPU")

    args = parser.parse_args()
    # random.seed(args.seed)
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)

    r"""
    random.seed(seed) —— 控制 Python 内置 random 模块
    numpy.random.seed(seed) —— 控制 NumPy
    torch.manual_seed(seed) 和 torch.cuda.manual_seed_all(seed) —— 控制 PyTorch CPU/GPU
    还会设置 torch.backends.cudnn.deterministic=True 和 torch.backends.cudnn.benchmark=False，帮助提升复现性
    """
    manual_seed(args.seed)

    # init logger

    dir_current_experiment = f"{args.model_name}"
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

    config = AutoConfig.from_pretrained(
        args.model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, use_fast=False, legacy=False, trust_remote_code=True)
    dtype = torch.float16 if not args.use_fp32 else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model_path,
                                                 config=config,
                                                 device_map='cpu',
                                                 torch_dtype=dtype,
                                                 trust_remote_code=True)

    for param in model.parameters():
        param.requires_grad = False

    model.half()  # to make sure same evaluation results with main
    # monitors, teardown = attach_monitors_llama_sdpa(model, logger_name=nameLogger, MonitorCls=MinMaxMonitor)
    # logger.info(f"{model.monitors}")

    logsVanilla= evaluate(args.ppl_datasets, model, tokenizer, None, args, logger)
    # monitored_data = get_monitor_data(model)
    # logger.info(f"monitored data: {pformat(monitored_data)}")
    # with open(os.path.join(args.output_dir,
    #                        f"monitor.json"), "w") as file:
    #     json.dump(monitored_data, file)


if __name__ == "__main__":
    print(pformat(sys.argv))
    main()
