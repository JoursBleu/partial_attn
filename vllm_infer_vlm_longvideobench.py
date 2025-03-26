import argparse
import gc
import json
from qwen_vl_utils import process_vision_info
from vllm import LLM
import re
import time
import torch
import transformers

from datasets import load_dataset
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, DynamicCache, StaticCache

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_file", "-r", type=str, default="results.jsonl")
    parser.add_argument("--model", "-m", type=str, default="/lpai/volumes/lpai-yharnam-bd-ga/lt/models/Qwen2.5-VL-7B-Instruct")
    # parser.add_argument("--page_size", type=int, default=4096)
    parser.add_argument("--dataset", type=str, default="/lpai/volumes/lpai-yharnam-bd-ga/lt/data/LongVideoBench")
    args = parser.parse_args()

    model = LLM(
        model=args.model,
        max_num_seqs=1,
        gpu_memory_utilization=0.95,
    )

    fp = open(args.dataset+"/lvb_val.json")
    datasets = json.load(fp)

    for index, ele in enumerate(datasets):
        # if index < 788:
            # continue
        question = ["Question: " + ele["question"]]
        question += [". ".join([chr(ord("A")+i), candidate]) for i, candidate in enumerate(ele["candidates"])]
        question += ["Format your response as follows: 'The correct answer is ([insert answer letter here])'"]
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": "file://"+args.dataset+"/videos/"+ele['video_path'],
                    },
                    {"type": "text", "text": question},
                ],
            }
        ]
        generated_ids = model.chat(messages, max_new_tokens=128, do_sample=False)

        output_text = processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        correct_choice = chr(ord("A")+ele['correct_choice'])
        print("index:", index, "output_text:", output_text, "correct_choice:", correct_choice)
        with open(args.result_file, "a", encoding="utf-8") as f:
            json.dump({"index": index, "response": output_text, "pred": extract_answer(output_text), "correct_choice": correct_choice, "judge": correct_choice==extract_answer(output_text)}, f, ensure_ascii=False)
            f.write('\n')

    breakpoint()


