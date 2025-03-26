#!/bin/bash

export MODEL=LLaVA-NeXT-Video-7B-32K-hf
export PYTHONPATH=/lpai/volumes/lpai-yharnam-bd-ga/lt/transformers/src

unset BLOCK_SIZE
CUDA_VISIBLE_DEVICES=0,1 python3 infer_vlm_longvideobench_llava.py --model /lpai/volumes/lpai-yharnam-bd-ga/lt/models/${MODEL} --result_file temp.jsonl
# CUDA_VISIBLE_DEVICES=0,1 python3 infer_vlm_longvideobench_llava.py --model /lpai/volumes/lpai-yharnam-bd-ga/lt/models/${MODEL} --result_file ${MODEL}_cot_${BLOCK_SIZE}.jsonl --cot &> ${MODEL}_cot_${BLOCK_SIZE}.log &
# export BLOCK_SIZE=2048
# CUDA_VISIBLE_DEVICES=2,3 python3 infer_longbench.py --model /lpai/volumes/lpai-yharnam-bd-ga/lt/models/${MODEL} --result_file ${MODEL}_cot_${BLOCK_SIZE}.jsonl --cot &> ${MODEL}_cot_${BLOCK_SIZE}.log &
# export BLOCK_SIZE=4096
# CUDA_VISIBLE_DEVICES=4,5 python3 infer_longbench.py --model /lpai/volumes/lpai-yharnam-bd-ga/lt/models/${MODEL} --result_file ${MODEL}_cot_${BLOCK_SIZE}.jsonl --cot &> ${MODEL}_cot_${BLOCK_SIZE}.log &
# export BLOCK_SIZE=8192
# CUDA_VISIBLE_DEVICES=6,7 python3 infer_longbench.py --model /lpai/volumes/lpai-yharnam-bd-ga/lt/models/${MODEL} --result_file ${MODEL}_cot_${BLOCK_SIZE}.jsonl --cot &> ${MODEL}_cot_${BLOCK_SIZE}.log &


