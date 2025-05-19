#!/bin/bash

export MODEL=Qwen3-30B-A3B
export PYTHONPATH=/lpai/volumes/lpai-yharnam-vol-ga/lt/transformers-moe/src
export CUDA_VISIBLE_DEVICES=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOP_P=0.2


# unset BLOCK_SIZE
# CUDA_VISIBLE_DEVICES=0,1 python infer_longbench.py --model /lpai/volumes/lpai-yharnam-bd-ga/lt/models/${MODEL} --result_file ${MODEL}_cot_${BLOCK_SIZE}.jsonl --cot &> ${MODEL}_cot_${BLOCK_SIZE}.log &
# export BLOCK_SIZE=2048
# CUDA_VISIBLE_DEVICES=2,3 python infer_longbench.py --model /lpai/volumes/lpai-yharnam-bd-ga/lt/models/${MODEL} --result_file ${MODEL}_cot_${BLOCK_SIZE}.jsonl --cot &> ${MODEL}_cot_${BLOCK_SIZE}.log &
# export BLOCK_SIZE=4096
# CUDA_VISIBLE_DEVICES=4,5 python infer_longbench.py --model /lpai/volumes/lpai-yharnam-bd-ga/lt/models/${MODEL} --result_file ${MODEL}_cot_${BLOCK_SIZE}.jsonl --cot &> ${MODEL}_cot_${BLOCK_SIZE}.log &
# export BLOCK_SIZE=8192
# CUDA_VISIBLE_DEVICES=6,7 python infer_longbench.py --model /lpai/volumes/lpai-yharnam-bd-ga/lt/models/${MODEL} --result_file ${MODEL}_cot_${BLOCK_SIZE}.jsonl --cot &> ${MODEL}_cot_${BLOCK_SIZE}.log &

# python3 infer_longbench.py --model /lpai/volumes/lpai-yharnam-vol-ga/lt/models/${MODEL} --result_file ${MODEL}_test.jsonl

mpirun --allow-run-as-root -n 1 python3 infer_longbench.py --model /lpai/volumes/lpai-yharnam-vol-ga/lt/models/Qwen3-30B-A3B --result_file ${MODEL}_${TOP_P}.jsonl
