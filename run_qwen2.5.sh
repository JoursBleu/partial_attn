#!/bin/bash

set -v

export MODEL=Qwen2.5-VL-7B-Instruct
export PYTHONPATH=/lpai/volumes/lpai-yharnam-vol-ga/lt/transformers/src
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'

# unset BLOCK_SIZE
export VIDEO_MAX_PIXELS=90316800 # (128000*28*28*0.9)
export LENGTH=128k
export BLOCK_SIZE=4096
export SINK_SIZE=256
touch ${MODEL}_${LENGTH}_${BLOCK_SIZE}_${SINK_SIZE}.jsonl
CUDA_VISIBLE_DEVICES=7 python3 infer_vlm_longvideobench_qwen2.5-VL.py --model ../models/${MODEL} --result_file ${MODEL}_${LENGTH}_${BLOCK_SIZE}_${SINK_SIZE}.jsonl --dataset /lpai/volumes/lpai-yharnam-vol-ga/lt/data/LongVideoBench

# export VIDEO_MAX_PIXELS=90316800 # (128000*28*28*0.9)
# export LENGTH=128k

# export SINK_SIZE=256
# export BLOCK_SIZE=1024
# touch ${MODEL}_${BLOCK_SIZE}_${LENGTH}_${SINK_SIZE}.jsonl
# CUDA_VISIBLE_DEVICES=0 python3 infer_vlm_longvideobench_qwen2.5-VL.py --model ../models/${MODEL} --result_file ${MODEL}_${BLOCK_SIZE}_${LENGTH}_${SINK_SIZE}.jsonl --dataset /lpai/volumes/lpai-yharnam-vol-ga/lt/data/LongVideoBench &> ${MODEL}_${BLOCK_SIZE}_${LENGTH}_${SINK_SIZE}.log &
# export BLOCK_SIZE=2048
# touch ${MODEL}_${BLOCK_SIZE}_${LENGTH}_${SINK_SIZE}.jsonl
# CUDA_VISIBLE_DEVICES=1 python3 infer_vlm_longvideobench_qwen2.5-VL.py --model ../models/${MODEL} --result_file ${MODEL}_${BLOCK_SIZE}_${LENGTH}_${SINK_SIZE}.jsonl --dataset /lpai/volumes/lpai-yharnam-vol-ga/lt/data/LongVideoBench &> ${MODEL}_${BLOCK_SIZE}_${LENGTH}_${SINK_SIZE}.log &
# export BLOCK_SIZE=4096
# touch ${MODEL}_${BLOCK_SIZE}_${LENGTH}_${SINK_SIZE}.jsonl
# CUDA_VISIBLE_DEVICES=2 python3 infer_vlm_longvideobench_qwen2.5-VL.py --model ../models/${MODEL} --result_file ${MODEL}_${BLOCK_SIZE}_${LENGTH}_${SINK_SIZE}.jsonl --dataset /lpai/volumes/lpai-yharnam-vol-ga/lt/data/LongVideoBench &> ${MODEL}_${BLOCK_SIZE}_${LENGTH}_${SINK_SIZE}.log &
# export BLOCK_SIZE=8192
# touch ${MODEL}_${BLOCK_SIZE}_${LENGTH}_${SINK_SIZE}.jsonl
# CUDA_VISIBLE_DEVICES=3 python3 infer_vlm_longvideobench_qwen2.5-VL.py --model ../models/${MODEL} --result_file ${MODEL}_${BLOCK_SIZE}_${LENGTH}_${SINK_SIZE}.jsonl --dataset /lpai/volumes/lpai-yharnam-vol-ga/lt/data/LongVideoBench &> ${MODEL}_${BLOCK_SIZE}_${LENGTH}_${SINK_SIZE}.log &

# export SINK_SIZE=512
# export BLOCK_SIZE=1024
# touch ${MODEL}_${BLOCK_SIZE}_${LENGTH}_${SINK_SIZE}.jsonl
# CUDA_VISIBLE_DEVICES=4 python3 infer_vlm_longvideobench_qwen2.5-VL.py --model ../models/${MODEL} --result_file ${MODEL}_${BLOCK_SIZE}_${LENGTH}_${SINK_SIZE}.jsonl --dataset /lpai/volumes/lpai-yharnam-vol-ga/lt/data/LongVideoBench &> ${MODEL}_${BLOCK_SIZE}_${LENGTH}_${SINK_SIZE}.log &
# export BLOCK_SIZE=2048
# touch ${MODEL}_${BLOCK_SIZE}_${LENGTH}_${SINK_SIZE}.jsonl
# CUDA_VISIBLE_DEVICES=5 python3 infer_vlm_longvideobench_qwen2.5-VL.py --model ../models/${MODEL} --result_file ${MODEL}_${BLOCK_SIZE}_${LENGTH}_${SINK_SIZE}.jsonl --dataset /lpai/volumes/lpai-yharnam-vol-ga/lt/data/LongVideoBench &> ${MODEL}_${BLOCK_SIZE}_${LENGTH}_${SINK_SIZE}.log &
# export BLOCK_SIZE=4096
# touch ${MODEL}_${BLOCK_SIZE}_${LENGTH}_${SINK_SIZE}.jsonl
# CUDA_VISIBLE_DEVICES=6 python3 infer_vlm_longvideobench_qwen2.5-VL.py --model ../models/${MODEL} --result_file ${MODEL}_${BLOCK_SIZE}_${LENGTH}_${SINK_SIZE}.jsonl --dataset /lpai/volumes/lpai-yharnam-vol-ga/lt/data/LongVideoBench &> ${MODEL}_${BLOCK_SIZE}_${LENGTH}_${SINK_SIZE}.log &
# export BLOCK_SIZE=8192
# touch ${MODEL}_${BLOCK_SIZE}_${LENGTH}_${SINK_SIZE}.jsonl
# CUDA_VISIBLE_DEVICES=7 python3 infer_vlm_longvideobench_qwen2.5-VL.py --model ../models/${MODEL} --result_file ${MODEL}_${BLOCK_SIZE}_${LENGTH}_${SINK_SIZE}.jsonl --dataset /lpai/volumes/lpai-yharnam-vol-ga/lt/data/LongVideoBench &> ${MODEL}_${BLOCK_SIZE}_${LENGTH}_${SINK_SIZE}.log &


sleep 3600000000
