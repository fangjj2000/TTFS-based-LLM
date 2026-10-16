r"""
    用来检测：加载的量化器中的 scale 与 zero_point，是否覆盖了测试阶段初始化的 scale 与 zero_point。
    用法示例：
    1) 先把模型构建出来（和你实际用的 from_pretrained 一致）
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained("你的基座", trust_remote_code=True)
    2) 指定 checkpoint 目录（就是你传给 load_checkpoint_in_model 的那个）
    diff_scales(model, "/path/to/checkpoint_dir_or_file")
"""
import gc
import logging
from collections import OrderedDict
from contextlib import contextmanager
from pprint import pformat
from typing import Optional, Dict

import torch
from accelerate import infer_auto_device_map, dispatch_model
from accelerate.hooks import remove_hook_from_module
from torch import Tensor, nn
from transformers import LlamaForCausalLM

from energy_cost_Gt import get_model_complexity_info
from energy_cost_Gt.engine import remove_syops_instrumentation
from quantize.quantizer import UniformAffineQuantizer
from utils import model_utils
from utils.data_utils import test_ppl, get_loaders
from utils.train_utils import capture_logs_plain


# 兼容 HF 分片权重目录的 scale 形状对比脚本

def _tensor_stats(t: Tensor) -> str:
    return (f"shape={tuple(t.shape)}, dtype={t.dtype}, device={t.device}, "
            f"min={t.min().item():.6g}, mean={t.mean().item():.6g}, max={t.max().item():.6g}")


def _looks_like_default(scale: Tensor, zero_point: Optional[Tensor], atol=1e-6) -> bool:
    ok_scale = torch.allclose(scale, torch.ones_like(scale), atol=atol)
    if zero_point is None:
        ok_zp = True
    else:
        ok_zp = torch.allclose(zero_point, torch.zeros_like(zero_point), atol=atol)
    return ok_scale and ok_zp


def iter_quantizers(model: nn.Module):
    """Yield (path, module) for every UniformAffineQuantizer in model."""
    for name, m in model.named_modules():
        if isinstance(m, UniformAffineQuantizer):
            yield name, m


def report_quantizer(nameLogger: str,
                     q: UniformAffineQuantizer, name: str) -> Dict[str, Tensor]:
    logger = logging.getLogger(nameLogger)
    scale = q.scale.detach().to(dtype=torch.float32, device="cpu").clone()
    zp = None
    # zero_point 可能是 Parameter，也可能是 buffer，也可能为 None
    if hasattr(q, "zero_point") and q.zero_point is not None:
        zp = q.zero_point.detach().to(dtype=torch.float32, device="cpu").clone()
    logger.debug(f"[Quantizer] {name}")
    logger.debug(
        f"  quant_type={q.quant_type}, mode={q.mode}, n_bits={q.n_bits}, group_size={q.group_size}, asym={q.asym}")
    logger.debug(f"  qmin={q.qmin}, qmax={q.qmax}, enable={getattr(q, 'enable', None)}")
    logger.debug(f"  scale:     {_tensor_stats(scale)}")
    if zp is None:
        logger.debug("  zero_point: None")
    else:
        logger.debug(f"  zero_point: {_tensor_stats(zp)}")
    logger.debug(f"  looks_like_default_init={_looks_like_default(scale, zp)}")
    return {"scale": scale,
            "zero_point": zp}


def snapshot_quantizers(nameLogger, model: nn.Module) -> Dict[str, Dict[str, Tensor]]:
    snap: Dict[str, Dict[str, Tensor]] = {}
    for name, q in iter_quantizers(model):
        snap[name] = report_quantizer(nameLogger, q, name)
    return snap


def _to_cpu_fp32(t):
    if t is None:
        return None
    if isinstance(t, Tensor):
        return t.detach().to(dtype=torch.float32, device="cpu")
    return t


def diff_snapshots(nameLogger: str,
                   before,
                   after,
                   atol: float = 1e-6,
                   rtol: float = 1e-4):
    logger = logging.getLogger(nameLogger)
    logger.debug("\n=== Diff: quantizer params changed? ===")
    names = sorted(set(before.keys()) | set(after.keys()))
    for name in names:
        b = before.get(name)
        a = after.get(name)
        if b is None:
            logger.debug(f"+ {name}: (new)")
            continue
        if a is None:
            logger.debug(f"- {name}: (removed)")
            continue

        bs = _to_cpu_fp32(b["scale"]);
        as_ = _to_cpu_fp32(a["scale"])
        bz = _to_cpu_fp32(b["zero_point"]);
        az = _to_cpu_fp32(a["zero_point"])

        def _changed(x, y):
            if (x is None) != (y is None):
                return True, None, None
            if x is None and y is None:
                return False, 0.0, 0.0
            if x.shape != y.shape:
                return True, None, None
            diff = (x - y).abs()
            max_abs = diff.max().item()
            # 避免除零：相对误差用 (|x|+|y|)/2 作为归一化
            denom = (x.abs() + y.abs()).clamp_min(1e-12) * 0.5
            max_rel = (diff / denom).max().item()
            changed = not torch.allclose(x, y, atol=atol, rtol=rtol)
            return changed, max_abs, max_rel

        c_s, s_abs, s_rel = _changed(bs, as_)
        c_z, z_abs, z_rel = _changed(bz, az)

        logger.debug(f"* {name}: "
                     f"scale_changed={c_s}"
                     + ("" if s_abs is None else f" (max_abs={s_abs:.3e}, max_rel={s_rel:.3e})")
                     + f", zero_point_changed={c_z}"
                     + ("" if z_abs is None else f" (max_abs={z_abs:.3e}, max_rel={z_rel:.3e})"))


@torch.no_grad()
def evaluate_old(datasets, model, tokenizer, prefixed_key_values, args, logger):
    block_class_name = model.model.layers[0].__class__.__name__
    device_map = infer_auto_device_map(model,
                                       max_memory={i: args.max_memory for i in range(torch.cuda.device_count())},
                                       no_split_module_classes=[block_class_name])
    # set skip_keys to avoid a bug
    model = dispatch_model(model, device_map=device_map,
                           skip_keys='past_key_values')
    prefixed_key_values = model_utils.mv_kv_cache(prefixed_key_values, model)
    results_str = ""
    if args.eval_ppl:
        logger.info(f"开始 ppl 测试。。。")
        datasets = datasets.split(",")
        # datasets = ["c4", "pile", "ptb", "ptb-new", 'c4-new', "wikitext2"]
        # datasets = ["wikitext2"]
        ppl_results = test_ppl(args,
                               logger,
                               model,
                               tokenizer,
                               prefixed_key_values,
                               datasets)
        logger.info(f"ppl 测试结束，结果如下：")
        for dataset in ppl_results:
            logger.info(f'{dataset} perplexity: {ppl_results[dataset]:.4f}')
            results_str += f"{ppl_results[dataset]:.4f} "
        logger.info(f" ")
    if args.eval_tasks != "":
        if prefixed_key_values is not None:
            model = model_utils.WrappedPrefixCausalLM(
                model, prefixed_key_values)
        import lm_eval
        from lm_eval.models.huggingface import HFLM
        from lm_eval.utils import make_table
        task_list = args.eval_tasks.split(',')
        model = HFLM(pretrained=model, batch_size=args.eval_batch_size)
        task_manager = lm_eval.tasks.TaskManager()
        results = lm_eval.simple_evaluate(
            model=model,
            tasks=task_list,
            num_fewshot=0,
            task_manager=task_manager,
        )
        logger.info(make_table(results))
        total_acc = 0
        total_acc_with_norm = 0
        for task in ['winogrande', 'hellaswag', 'arc_challenge', 'arc_easy', 'piqa']:
            if task in task_list:
                total_acc += results['results'][task]['acc,none']
                results_str += f"{results['results'][task]['acc,none'] * 100:.2f} "
                if 'acc_norm,none' in results['results'][task]:
                    results_str += f"{results['results'][task]['acc_norm,none'] * 100:.2f} "
                    total_acc_with_norm += results['results'][task]['acc_norm,none']
                else:
                    total_acc_with_norm += results['results'][task]['acc,none']
        logger.info(f'Average Acc: {total_acc / len(task_list) * 100:.2f}%')
        logger.info(
            f'Average Acc (with norm): {total_acc_with_norm / len(task_list) * 100:.2f}%')
        logger.info(f'Results string: {results_str.strip()}\n\n')

        # remove wrapper 这里是 引用操作。改变函数输入的引用类型的参数 model 指向的对象。
        if prefixed_key_values is not None:
            model = model.model


@contextmanager
def temp_model_flags(model, **flags):
    """
    临时设置 model.config 的若干属性，退出 with 后自动恢复。
    用法：
      with temp_model_flags(model, use_cache=False, output_attentions=False):
          ...  # 在这里跑只需要这些设置的逻辑
    """
    cfg = model.config
    backup = {k: getattr(cfg, k, None) for k in flags}
    for k, v in flags.items():
        setattr(cfg, k, v)
    try:
        yield
    finally:
        for k, v in backup.items():
            setattr(cfg, k, v)


def first_device_from_map(model):
    dm = getattr(model, "hf_device_map", None)
    if not dm:  # 没有分片，走默认
        return "cuda" if torch.cuda.is_available() else "cpu"

    # 优先找 embedding 所在设备
    for key in ("model.embed_tokens", "model.model.embed_tokens", "embed_tokens"):
        if key in dm and dm[key] != "disk":
            return dm[key]

    # 其次找第一个不在磁盘的子模块设备（按名字排序稳定些）
    for _, dev in sorted(dm.items()):
        if dev != "disk":
            return dev

    # 兜底
    return "cpu"


@torch.no_grad()
def evaluate(ppl_datasets, model, tokenizer, prefixed_key_values, args, logger):
    # —— 开始捕获 evaluate + test_ppl 的所有日志 ——
    with capture_logs_plain(logger) as cap:
        block_class_name = model.model.layers[0].__class__.__name__
        device_map = infer_auto_device_map(
            model,
            max_memory={i: args.max_memory for i in range(torch.cuda.device_count())},
            no_split_module_classes=[block_class_name]
        )
        model = dispatch_model(model,
                               device_map=device_map,
                               skip_keys='past_key_values',
                               offload_dir=args.offload_dir if args.off_load_to_disk else None
                               )
        hf_model = model
        prefixed_key_values = model_utils.mv_kv_cache(prefixed_key_values, model)

        # 只有在用到时才创建 hflm / wrapped；删除时也要判空
        hflm = None
        wrapped = None

        results_str = ""
        # region === ppl 测试 ===
        if args.eval_ppl:
            logger.info("开始 ppl 测试。。。")
            ppl_datasets = ppl_datasets.split(",")
            # ppl 仅需要 logits，不需要 attn/hidden、cache； 关闭这些设置可以加速
            with temp_model_flags(model,
                                  use_cache=True,
                                  output_attentions=False,
                                  output_hidden_states=False):
                ppl_results = test_ppl(args, logger, model, tokenizer, prefixed_key_values, ppl_datasets)
            logger.info("ppl 测试结束，结果如下：")
            for dataset in ppl_results:
                logger.info(f'{dataset} perplexity: {ppl_results[dataset]:.4f}')
                results_str += f"{ppl_results[dataset]:.4f} "
            logger.info(" ")
        # endregion

        # region === 准确率测试 ===
        if args.eval_tasks != "":
            task_list = args.eval_tasks.split(',')
            # 按需包一层前缀 KV 包装，不覆盖 hf_model
            if prefixed_key_values is not None:
                wrapped = model_utils.WrappedPrefixCausalLM(hf_model, prefixed_key_values)
            else:
                wrapped = hf_model

            import lm_eval
            from lm_eval.models.huggingface import HFLM
            from lm_eval.utils import make_table

            # lm-eval 里只需 forward；打开 cache，加速解码；仍然关闭其他的开关
            with temp_model_flags(hf_model,
                                  use_cache=True,
                                  output_attentions=False,
                                  output_hidden_states=False):
                # 让 HFLM 驻留在与 hf_model 第一层相同的设备
                hflm = HFLM(pretrained=wrapped,
                            batch_size=args.eval_batch_size,
                            device=first_device_from_map(hf_model))

                import lm_eval
                task_manager = lm_eval.tasks.TaskManager()
                results = lm_eval.simple_evaluate(
                    model=hflm, tasks=task_list, num_fewshot=0, task_manager=task_manager,
                )

            logger.info(make_table(results))
            total_acc = 0
            total_acc_with_norm = 0
            for task in ['winogrande', 'hellaswag', 'arc_challenge', 'arc_easy', 'piqa']:
                if task in task_list:
                    total_acc += results['results'][task]['acc,none']
                    results_str += f"{results['results'][task]['acc,none'] * 100:.2f} "
                    if 'acc_norm,none' in results['results'][task]:
                        results_str += f"{results['results'][task]['acc_norm,none'] * 100:.2f} "
                        total_acc_with_norm += results['results'][task]['acc_norm,none']
                    else:
                        total_acc_with_norm += results['results'][task]['acc,none']
            logger.info(f'Average Acc: {total_acc / len(task_list) * 100:.2f}%')
            logger.info(f'Average Acc (with norm): {total_acc_with_norm / len(task_list) * 100:.2f}%')
            logger.info(f'Results string: {results_str.strip()}\n\n')

        if prefixed_key_values is not None:
            model = wrapped.model

        # —— 清理局部对象（判空再删，避免未定义时报错）——
        if hflm is not None:
            del hflm
        if wrapped is not None:
            del wrapped

        gc.collect()
        torch.cuda.empty_cache()

        # === 关键善后：把模型权重全部移除 Accelerate 的 hooks，然后拉回 CPU  ===
        # 先移除 hooks，恢复“干净”的模块（没有 offload / meta）
        remove_hook_from_module(model, recurse=True)
        # 再放回 CPU，方便外部的操作（需要访问真实权重）。
        model.to("cpu")

    # 返回 evaluate + test_ppl 的合并日志文本
    combined_logs = cap.getvalue()
    return combined_logs


@torch.no_grad()
def energy_calculation(args,
                       nameLogger: str,
                       model: LlamaForCausalLM,
                       typeModel: str,
                       nameDataset: str,
                       tokenizer,
                       numBatches_for_calculate_energy: int,
                       ssa_info: Dict):
    model.float()
    logger = logging.getLogger(nameLogger)

    logger.info(f"为 {typeModel} 测试能耗")

    block_class_name = model.model.layers[0].__class__.__name__
    device_map: OrderedDict = infer_auto_device_map(
        model,
        max_memory={i: args.max_memory for i in range(torch.cuda.device_count())},
        no_split_module_classes=[block_class_name]
    )
    model = dispatch_model(model,
                           device_map=device_map,
                           skip_keys='past_key_values',
                           offload_dir=args.offload_dir if args.off_load_to_disk else None
                           )

    dataloader_energy, _ = get_loaders(name=nameDataset,
                                       nameLogger=nameLogger,
                                       tokenizer=tokenizer,
                                       seed=0,
                                       seqlen=args.ppl_seqlen,
                                       )

    syops_count, params_count = get_model_complexity_info(model=model,
                                                          input_res=None,
                                                          ssa_info=ssa_info,
                                                          nameLogger=nameLogger,
                                                          dataloader=dataloader_energy,
                                                          numBatches_for_calculate_energy=numBatches_for_calculate_energy,
                                                          print_per_layer_stat=False
                                                          )
    remove_syops_instrumentation(model)
    logger.info(f"\n操作数统计结果：{pformat(syops_count)}\n参数量统计结果：{pformat(params_count)}")
