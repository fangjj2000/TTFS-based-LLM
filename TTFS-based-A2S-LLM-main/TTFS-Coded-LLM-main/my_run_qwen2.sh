#!/usr/bin/env bash

de=${de:-""}
gpu_no=${gpu:-1}
cmd="CUDA_VISIBLE_DEVICES=$gpu_no python"
if [[ "$de" == "de" ]]; then
  cmd="$cmd -m debugpy --listen 8762 --wait-for-client"
fi

cmd="$cmd main.py \
  --model_path /home/user/Workspace/Models/Qwen2-7B
  --model_name Qwen-2-7B \
  --output_dir ./log \
  --wbits 6 \
  --input_bits 6 \
  --input_mode static \
  --input_asym \
  --input_group_size -1 \
  --output_bits 6  \
  --output_asym \
  --output_mode static \
  --output_group_size -1 \
  --k_bits 6 \
  --v_bits 6 \
  --kv_group_size 128 \
  --kv_mode static \
  --kv_asym \
  --pre_rotate \
  --down_online_had \
  --qk_online_had \
  --set_prefixed_tokens \
  --eval_ppl \
  --ppl_datasets c4,pile,ptb,wikitext2,redpajama \
  --max_memory 70GiB \
  --epochs 10 \
  --mse_init \
  --quant_lr 5e-5 \
  --weight_lr 5e-6 \
  --eval_tasks piqa,arc_easy,arc_challenge,hellaswag,winogrande \
  --save_quant_dir /home/user/Workspace/Models/Quantized"
echo $cmd
eval $cmd

#--ppl_datasets c4,pile,ptb,wikitext2,redpajama \
#--eval_tasks piqa,arc_easy,arc_challenge,hellaswag,winogrande \
# 13 行增加一行，--off_load_to_disk， 作用是在 main.py 中使用 offload，将模型加载到硬盘。减少显存开销。
# 14 行增加一行，--offload_dir， 作用是在 main.py 中将模型加载到硬盘。减少显存开销。