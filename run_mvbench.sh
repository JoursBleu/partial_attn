#!/bin/bash

set -e
set -v


export PYTHONPATH=/lpai/volumes/lpai-yharnam-vol-ga/lt/transformers/src

# export NPROC_PER_NODE=8
# export NPROC_PER_NODE=6
export CUDA_VISIBLE_DEVICES=2
unset BLOCK_SIZE
unset SINK_SIZE
unset USE_POS
# export BLOCK_SIZE=4096
# export SINK_SIZE=64
export USE_POS=1

export VIDEO_MAX_PIXELS=90316800 # (128000*28*28*0.9)
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'


MODEL_PATH=/lpai/volumes/lpai-yharnam-vol-ga/lt/models/Qwen2.5-VL-7B-Instruct

python3 infer_vlm_mvbench.py --result_file results_frame_8192.jsonl --model /lpai/volumes/lpai-yharnam-vol-ga/lt/models/Qwen2.5-VL-7B-Instruct --dataset /lpai/volumes/lpai-yharnam-vol-ga/lt/data/MVBench

# python3 infer_vlm_mvbench.py --result_file results_${BLOCK_SIZE}.jsonl --model /lpai/volumes/lpai-yharnam-vol-ga/lt/models/Qwen2.5-VL-7B-Instruct --dataset /lpai/volumes/lpai-yharnam-vol-ga/lt/data/MVBench

mv $OUTPUT_DIR/*/*_results.json $OUTPUT_DIR/results.json || true
mv $OUTPUT_DIR/*/*_samples_*.jsonl $OUTPUT_DIR/samples.jsonl || true



