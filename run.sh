#!/bin/bash

export MODEL=Meta-Llama-3.1-8B-Instruct
export PYTHONPATH=/lpai/volumes/lpai-demo-muses/lt/transformers/src

unset BLOCK_SIZE
CUDA_VISIBLE_DEVICES=0,1 python infer_longbench.py --model /lpai/volumes/lpai-demo-muses/lt/models/${MODEL} --result_file ${MODEL}_cot_${BLOCK_SIZE}.jsonl --cot &> ${MODEL}_cot_${BLOCK_SIZE}.log &
export BLOCK_SIZE=2048
CUDA_VISIBLE_DEVICES=2,3 python infer_longbench.py --model /lpai/volumes/lpai-demo-muses/lt/models/${MODEL} --result_file ${MODEL}_cot_${BLOCK_SIZE}.jsonl --cot &> ${MODEL}_cot_${BLOCK_SIZE}.log &
export BLOCK_SIZE=4096
CUDA_VISIBLE_DEVICES=4,5 python infer_longbench.py --model /lpai/volumes/lpai-demo-muses/lt/models/${MODEL} --result_file ${MODEL}_cot_${BLOCK_SIZE}.jsonl --cot &> ${MODEL}_cot_${BLOCK_SIZE}.log &
export BLOCK_SIZE=8192
CUDA_VISIBLE_DEVICES=6,7 python infer_longbench.py --model /lpai/volumes/lpai-demo-muses/lt/models/${MODEL} --result_file ${MODEL}_cot_${BLOCK_SIZE}.jsonl --cot &> ${MODEL}_cot_${BLOCK_SIZE}.log &


