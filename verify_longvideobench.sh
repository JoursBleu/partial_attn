#!/bin/bash

set -v

export MODEL=LLaVA-Video-7B-Qwen2

export BLOCK_SIZE=
export SINK_LEN=
export USE_POS=
rm ${MODEL}_${LENGTH}_${BLOCK_SIZE}_${SINK_LEN}_1.jsonl
touch ${MODEL}_${LENGTH}_${BLOCK_SIZE}_${SINK_LEN}_1.jsonl
python3 infer_vlm_longvideobench_verify.py --result_file ${MODEL}_${LENGTH}_${BLOCK_SIZE}_${SINK_LEN}_1.jsonl --dataset /lpai/volumes/lpai-yharnam-vol-ga/lt/data/LongVideoBench --result_dir /lpai/volumes/lpai-yharnam-vol-ga/lt/lmms-eval/runs/eval/${MODEL}/lmms-longvideobench_val_v/cache_${BLOCK_SIZE}_${SINK_LEN}_${USE_POS}
