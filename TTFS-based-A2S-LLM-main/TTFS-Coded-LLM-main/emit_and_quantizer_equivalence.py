import argparse

from quantize.quantizer import UniformAffineQuantizer
from spiking_conversion.model.spiking_layers import TTFS_Spiker

nameLogger = "emit_and_quantizer_test"

import torch


def compare_tensors(t1: torch.Tensor, t2: torch.Tensor, atol=1e-8, rtol=1e-5):
    """
    比较两个张量是否完全相等（形状+数值）。
    如果不相等，显示具体差异。
    """
    if t1.shape != t2.shape:
        print(f"❌ Shape 不同: {t1.shape} vs {t2.shape}")
        return False

    if torch.equal(t1, t2):
        print("✅ 两个张量完全相等。")
        return True

    # 如果不完全相等，给出差异
    diff = t1 - t2
    max_abs_diff = diff.abs().max().item()
    mean_abs_diff = diff.abs().mean().item()

    print("❌ 张量数值不完全相等：")
    print(f"  最大绝对差值: {max_abs_diff}")
    print(f"  平均绝对差值: {mean_abs_diff}")
    print(f"  不同元素数量: {(t1 != t2).sum().item()} / {t1.numel()}")

    # 如果只想看前几个不同的元素，可以额外打印
    mismatches = (t1 != t2).nonzero(as_tuple=False)
    if mismatches.numel() > 0:
        print("  前几个差异位置：")
        for idx in mismatches[:10]:
            idx_tuple = tuple(idx.tolist())
            print(f"    位置 {idx_tuple}: t1={t1[idx_tuple]}, t2={t2[idx_tuple]}")

    # 也可以用 allclose 判定近似相等
    if torch.allclose(t1, t2, atol=atol, rtol=rtol):
        print(f"⚠️ 张量在误差容忍范围内近似相等 (atol={atol}, rtol={rtol})")

    return False


if __name__ == "__main__":
    # region ==== 参数 ====
    parser = argparse.ArgumentParser()
    # -----------------model setting------------------------------------
    parser.add_argument("--model_path", type=str, help="model path")
    parser.add_argument("--model_name", type=str, default=None,
                        help="model name, for the saving of corresponding data cache")
    parser.add_argument("--cache_dir", default="./cache", type=str,
                        help="cache dir of dataset, leading to faster debug")
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
                        default=8, help="out quantization bits")
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
                        default="c4,pile,ptb,ptb-new,c4-new,wikitext2",
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

    inputTensor = torch.randn(size=(1, 8, 16)).cuda()
    args = parser.parse_args()
    quantizer = UniformAffineQuantizer(args.output_bits,
                                       (1, inputTensor.shape[-1]),
                                       True,
                                       -1,
                                       None,
                                       "activation",
                                       "static",
                                       False,
                                       ).cuda()

    optimizer = torch.optim.AdamW(params=quantizer.parameters(),
                                  lr=args.quant_lr,
                                  weight_decay=args.wd)

    for idx in range(20):
        optimizer.zero_grad()
        outputs = quantizer(inputTensor)
        loss = outputs.sum()
        loss.backward()
        optimizer.step()

    emitter = TTFS_Spiker(nameLogger,
                          quantizer,
                          False)

    outputQ = quantizer(inputTensor)
    outputS = emitter(inputTensor)

    print(f"outputQ: {outputQ}")
    print(f"outputS: {outputS}")
    compare_tensors(outputQ, outputS)
