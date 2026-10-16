#!/usr/bin/env bash

de=${de:-""}
gpu_no=${gpu:-1}
cmd="CUDA_VISIBLE_DEVICES=$gpu_no python"
if [[ "$de" == "de" ]]; then
  cmd="$cmd -m debugpy --listen 8765 --wait-for-client"
fi

cmd="$cmd eval_ann.py \
  --model_path /home/user/Workspace/Models/Llama-3-8B \
  --model_name Llama-3-8B \
  --output_dir ./log/eval_ann \
  --eval_ppl \
  --ppl_datasets c4,pile,ptb,wikitext2,redpajama \
  --eval_tasks  piqa,arc_easy,arc_challenge,winogrande \
  --max_memory 70GiB"
echo $cmd
eval $cmd
