import argparse
import gc
import os
import json
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
import re
import time
import torch
import transformers

from datasets import load_dataset
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, DynamicCache, StaticCache

data_list = {
    "action_sequence": ("action_sequence.json", "/video/star/Charades_v1_480/", "video", True), # has start & end
    "action_prediction": ("action_prediction.json", "/video/star/Charades_v1_480/", "video", True), # has start & end
    "action_antonym": ("action_antonym.json", "/video/ssv2_video/", "video", False),
    "fine_grained_action": ("fine_grained_action.json", "/video/Moments_in_Time_Raw/videos/", "video", False),
    "unexpected_action": ("unexpected_action.json", "/video/FunQA_test/test/", "video", False),
    "object_existence": ("object_existence.json", "/video/clevrer/video_validation/", "video", False),
    "object_interaction": ("object_interaction.json", "/video/star/Charades_v1_480/", "video", True), # has start & end
    "object_shuffle": ("object_shuffle.json", "/video/perception/videos/", "video", False),
    "moving_direction": ("moving_direction.json", "/video/clevrer/video_validation/", "video", False),
    "action_localization": ("action_localization.json", "/video/sta/sta_video/", "video", True),  # has start & end
    "scene_transition": ("scene_transition.json", "/video/scene_qa/video/", "video", False),
    "action_count": ("action_count.json", "/video/perception/videos/", "video", False),
    "moving_count": ("moving_count.json", "/video/clevrer/video_validation/", "video", False),
    "moving_attribute": ("moving_attribute.json", "/video/clevrer/video_validation/", "video", False),
    "state_change": ("state_change.json", "/video/perception/videos/", "video", False),
    "character_order": ("character_order.json", "/video/perception/videos/", "video", False),
    "egocentric_navigation": ("egocentric_navigation.json", "/video/vlnqa/", "video", False),
    "counterfactual_inference": ("counterfactual_inference.json", "/video/clevrer/video_validation/", "video", False),
    # "fine_grained_pose": ("fine_grained_pose.json", "/video/nturgbd/", "video", False),
    # "episodic_reasoning": ("episodic_reasoning.json", "/video/tvqa/frames_fps3_hq/", "frame", True),  # has start & end, read frame
}

def qa_template(data):
    question = f"Question: {data['question']}\n"
    question += "Options:\n"
    answer = data['answer']
    answer_idx = -1
    for idx, c in enumerate(data['candidates']):
        question += f"({chr(ord('A') + idx)}) {c}\n"
        if c == answer:
            answer_idx = idx
    question = question.rstrip()
    answer = f"{chr(ord('A') + answer_idx)}"
    return question, answer

def extract_answer(response):
    response = response.replace('*', '')
    match = re.search(r'The correct answer is \(([A-D])\)', response)
    if match:
        return match.group(1)
    else:
        match = re.search(r'The correct answer is ([A-D])', response)
        if match:
            return match.group(1)
        else:
            return None

def check_len(result_file, length):
    count = 0
    if not os.path.exists(result_file):
        return count
    fp = open(result_file,'r')
    lines = fp.readlines()
    for line in lines:
        json_line = json.loads(line)
        video_len = json_line['video_len']
        if video_len > length:
            count += 1
    return count

# python3 infer_vlm_mvbench.py --result_file results_frame.jsonl --model /lpai/volumes/lpai-yharnam-vol-ga/lt/models/Qwen2.5-VL-7B-Instruct --dataset /lpai/volumes/lpai-yharnam-vol-ga/lt/data/MVBench

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, default="mvbench_result/mvbench_opt_sysframe_frame")
    parser.add_argument("--result_file", "-r", type=str, default="results.jsonl")
    parser.add_argument("--model", "-m", type=str, default="/lpai/volumes/lpai-yharnam-vol-ga/lt/models/Qwen2.5-VL-7B-Instruct")
    # parser.add_argument("--page_size", type=int, default=4096)
    parser.add_argument("--dataset", type=str, default="/lpai/volumes/lpai-yharnam-vol-ga/lt/data/MVBench")
    parser.add_argument("--length", type=int, default=8192)
    args = parser.parse_args()

    total_count = 0
    for data in data_list:
        base_dir = args.dataset + data_list[data][1]
        video_type = data_list[data][2]
        datasets = load_dataset(args.dataset, data, split='train')
        output_file = f'{args.result_dir}/{data}_{args.result_file}'
        count = check_len(output_file, args.length)
        total_count += count
        print(data, "count:", count)
    print("total_count:", total_count)


