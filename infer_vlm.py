import argparse
import gc
import json
from qwen_vl_utils import process_vision_info
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
    parser.add_argument("--model", "-m", type=str, default="/lpai/volumes/lpai-demo-muses/lt/models/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--page_size", type=int, default=4096) # set to True if using no context (directly measuring memorization)
    parser.add_argument("--dataset", type=str, default="/lpai/volumes/lpai-demo-muses/lt/data/LongVideoBench") # set to True if using no context (directly measuring memorization)
    args = parser.parse_args()

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True,)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                args.model,
                attn_implementation="flash_attention_2",
                # attn_implementation="eager",
                torch_dtype=torch.float16,
                # load_in_8bit=True,
                device_map="auto",
                trust_remote_code=True,
            )
    model = model.eval()
    # model = torch.compile(model)

    fp = open(args.dataset+"/lvb_val.json")
    datasets = json.load(fp)

    for index, ele in enumerate(datasets):
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
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        ).to("cuda")
        generated_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)

        output_text = processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        correct_choice = chr(ord("A")+ele['correct_choice'])
        print("index:", index, "output_text:", output_text, "correct_choice:", correct_choice)
        with open(args.result_file, "a", encoding="utf-8") as f:
            json.dump({"index": index, "response": output_text, "pred": extract_answer(output_text), "judge": correct_choice==extract_answer(output_text)}, f, ensure_ascii=False)
            f.write('\n')

    breakpoint()


