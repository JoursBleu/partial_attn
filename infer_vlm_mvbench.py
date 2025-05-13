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

def get_already_done(result_file):
    if not os.path.exists(result_file):
        return set()
    fp = open(result_file,'r')
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
    parser.add_argument("--result_file", "-r", type=str, default="results.jsonl")
    parser.add_argument("--model", "-m", type=str, default="/lpai/volumes/lpai-yharnam-vol-ga/lt/models/Qwen2.5-VL-7B-Instruct")
    # parser.add_argument("--page_size", type=int, default=4096)
    parser.add_argument("--dataset", type=str, default="/lpai/volumes/lpai-yharnam-vol-ga/lt/data/MVBench")
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True,)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                args.model,
                attn_implementation="flash_attention_2",
                # attn_implementation="eager",
                torch_dtype=torch.float16,
                # load_in_8bit=True,
                device_map="cuda",
                trust_remote_code=True,
            )
    model = model.eval()
    model = torch.compile(model)

    for data in data_list:
        base_dir = args.dataset + data_list[data][1]
        video_type = data_list[data][2]
        datasets = load_dataset(args.dataset, data, split='train')
        output_file = f'mvbench_result/mvbench_opt_sysframe_frame_real/{data}_{args.result_file}'
        already = get_already_done(output_file)
        for index, ele in enumerate(tqdm(datasets)):
            if index in already:
                continue
            video_path = base_dir + ele['video']
            # print("video_path", video_path)
            question, answer = qa_template(ele)
            question += "\nFormat your response as follows: 'The correct answer is ([insert answer letter here])'"
            if video_type == "video":
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video",
                                "video": "file://"+video_path,
                            },
                            {"type": "text", "text": question},
                        ],
                    }
                ]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + 'The correct answer is ('

            image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
                **video_kwargs,
            ).to("cuda")
            generated_ids = model.generate(**inputs, max_new_tokens=1, do_sample=False)

            output_text = processor.batch_decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0] + ')'
            _, prompt_len = inputs['input_ids'].shape
            video_len, _ = inputs['pixel_values_videos'].shape
            print("index:", index, "output_text:", output_text, "correct_choice:", answer)
            with open(output_file, "a", encoding="utf-8") as f:
                json.dump({"index": index, 'prompt_len': prompt_len, 'video_len': video_len, "response": output_text, "pred": extract_answer(output_text), "correct_choice": answer, "judge": answer==extract_answer(output_text)}, f, ensure_ascii=False)
                f.write('\n')


    breakpoint()


