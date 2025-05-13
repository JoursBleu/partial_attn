import argparse
import gc
import json
from qwen_vl_utils import process_vision_info
import re
import time
import torch

from datasets import load_dataset

def extract_answer(response):
    response = response.replace('*', '')
    match = re.search(r'The correct answer is \(([A-Z])\)', response)
    if match:
        return match.group(1)
    else:
        match = re.search(r'The correct answer is ([A-Z])', response)
        if match:
            return match.group(1)
        else:
            return None

def get_already_done(args):
    fp = open(args.result_file,'r')
    lines = fp.readlines()
    already = set()
    for line in lines:
        json_line = json.loads(line)
        index = json_line['index']
        if index not in already:
            already.add(index)
    return already

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", "-d", type=str, default="/lpai/volumes/lpai-yharnam-vol-ga/lt/lmms-eval/runs/eval/Qwen2.5-VL-7B-Instruct/lmms-longvideobench_val_v/cache_frame_sys_1")
    parser.add_argument("--result_file", "-r", type=str, default="opt_longvideobench.jsonl")
    # parser.add_argument("--page_size", type=int, default=4096)
    parser.add_argument("--dataset", type=str, default="/lpai/volumes/lpai-yharnam-my/lt/data/LongVideoBench")
    args = parser.parse_args()

    fp = open(args.dataset+"/lvb_val.json")
    datasets = json.load(fp)

    already = get_already_done(args)

    total_time = 0.
    total_seqlen = 0
    count = 0
    for index, ele in enumerate(datasets):
        correct_choice = chr(ord("A")+ele['correct_choice'])
        with open(args.result_dir+f"/longvideobench_val_v_validation_{index}.txt", 'r') as fp:
            output_text = fp.readlines()[-1]
            output_text = output_text.split('.')[0]
        print("index:", index, "output_text:", output_text, "correct_choice:", correct_choice)
        with open(args.result_file, "a", encoding="utf-8") as f:
            json.dump({"index": index, "response": output_text, "pred": (output_text), "correct_choice": correct_choice, "judge": correct_choice==(output_text)}, f, ensure_ascii=False)
            f.write('\n')
        if correct_choice==(output_text):
            count += 1


    print("sample num:", (index))
    print("correct num:", (count))
    print("avg seqlen", total_seqlen / (index-2))
    print("avg time", total_time / (index-2))
    breakpoint()


