#!/bin/bash

# export MASTER_ADDR='localhost'
# export MASTER_PORT=17777

export MODEL=Qwen2.5-VL-7B-Instruct
export PYTHONPATH=/lpai/volumes/lpai-yharnam-bd-ga/lt/transformers/src
export VIDEO_MAX_PIXELS=180633600 # (256000*28*28*0.9)
export LENGTH=256k

# unset BLOCK_SIZE
# export BLOCK_SIZE=2048
# CUDA_VISIBLE_DEVICES=2,7 python3 infer_vlm_longvideobench_qwen2.5-VL.py --model /lpai/volumes/lpai-yharnam-bd-ga/lt/models/${MODEL} --result_file temp.jsonl
# unset BLOCK_SIZE
# CUDA_VISIBLE_DEVICES=0,1 python3 infer_vlm_longvideobench_qwen2.5-VL.py --model /lpai/volumes/lpai-yharnam-bd-ga/lt/models/${MODEL} --result_file ${MODEL}_${BLOCK_SIZE}_${LENGTH}.jsonl &> ${MODEL}_${BLOCK_SIZE}_${LENGTH}.log &
export BLOCK_SIZE=2048
CUDA_VISIBLE_DEVICES=2,3 python3 infer_vlm_longvideobench_qwen2.5-VL.py --model /lpai/volumes/lpai-yharnam-bd-ga/lt/models/${MODEL} --result_file ${MODEL}_${BLOCK_SIZE}_${LENGTH}.jsonl &> ${MODEL}_${BLOCK_SIZE}_${LENGTH}.log &
export BLOCK_SIZE=4096
CUDA_VISIBLE_DEVICES=4,5 python3 infer_vlm_longvideobench_qwen2.5-VL.py --model /lpai/volumes/lpai-yharnam-bd-ga/lt/models/${MODEL} --result_file ${MODEL}_${BLOCK_SIZE}_${LENGTH}.jsonl &> ${MODEL}_${BLOCK_SIZE}_${LENGTH}.log &
# export BLOCK_SIZE=8192
# CUDA_VISIBLE_DEVICES=6,7 python3 infer_vlm_longvideobench_qwen2.5-VL.py --model /lpai/volumes/lpai-yharnam-bd-ga/lt/models/${MODEL} --result_file ${MODEL}_${BLOCK_SIZE}_${LENGTH}.jsonl &> ${MODEL}_${BLOCK_SIZE}_${LENGTH}.log &


