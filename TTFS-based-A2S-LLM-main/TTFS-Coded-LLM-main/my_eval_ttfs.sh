#!/usr/bin/env bash

de=${de:-""}
gpu_no=${gpu:-1}
cmd="CUDA_VISIBLE_DEVICES=$gpu_no python"
if [[ "$de" == "de" ]]; then
  cmd="$cmd -m debugpy --listen 8765 --wait-for-client"
fi

cmd="$cmd eval_ttfs.py \
  --model_path /mnt/Solar/Models/Vanilla/Llama-2-13b-hf \
  --model_name Llama-2-13B-TTFSFormer \
  --precision 2048 \
  --eval_ppl \
  --output_dir ./log/eval_snn \
  --ppl_datasets c4,pile,ptb,wikitext2,redpajama \
  --eval_tasks piqa,arc_easy,arc_challenge,hellaswag,winogrande \
  --max_memory 70GiB"
echo $cmd
eval $cmd
