#!/usr/bin/env bash

de=${de:-""}
gpu_no=${gpu:-1}
cmd="CUDA_VISIBLE_DEVICES=$gpu_no python"
if [[ "$de" == "de" ]]; then
  cmd="$cmd -m debugpy --listen 8764 --wait-for-client"
fi

cmd="$cmd eval_snn.py \
  --quant_model /home/user/Workspace/Models/Quantized/temp/Llama-2-7B-temp_W8I8O8_E0_lrW-5e-06_lrQ-5e-05 \
  --output_dir ./log/eval_snn/ \
  --model_name Llama-2-7B-temp \
  --max_memory 70GiB \
  --eval_ppl \
  --ppl_datasets wikitext2 \
  --eval_tasks winogrande"
echo $cmd
eval $cmd

# --eval_tasks piqa,arc_easy,arc_challenge,hellaswag,winogrande
