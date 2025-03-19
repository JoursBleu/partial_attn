import argparse
import gc
import json
from qwen_vl_utils import process_vision_info
import re
import time
import transformers

import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

from datasets import load_dataset
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, DynamicCache, StaticCache


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values



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
    parser.add_argument("--result_file", "-r", type=str, default="opt_longvideobench.jsonl")
    parser.add_argument("--model", "-m", type=str, default="/lpai/volumes/lpai-demo-muses/lt/models/Qwen2.5-VL-7B-Instruct")
    # parser.add_argument("--page_size", type=int, default=4096)
    parser.add_argument("--dataset", type=str, default="/lpai/volumes/lpai-demo-muses/lt/data/LongVideoBench")
    args = parser.parse_args()

    model = AutoModel.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=False)

    breakpoint()

    model = model.eval()
    model = torch.compile(model)

    fp = open(args.dataset+"/lvb_val.json")
    datasets = json.load(fp)

    total_time = 0.
    total_seqlen = 0
    for index, ele in enumerate(datasets):
        # if index < 19:
            # continue
        # if index == 10:
            # break
        question = ["Question: " + ele["question"]]
        question += [". ".join([chr(ord("A")+i), candidate]) for i, candidate in enumerate(ele["candidates"])]
        question += ["Format your response as follows: 'The correct answer is ([insert answer letter here])'"]
        # question = ["Question: descript this video"]
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
        print("video:", args.dataset+"/videos/"+ele['video_path'])
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
        print("video shape:", video_inputs[0].shape)
        print("video fps:", video_kwargs)
        print("input len:", inputs.input_ids.shape)
        torch.cuda.synchronize()
        start = time.time()
        generated_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)
        torch.cuda.synchronize()
        end = time.time()

        output_text = processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        correct_choice = chr(ord("A")+ele['correct_choice'])
        if index >= 2:
            total_time += (end - start)
            total_seqlen += inputs.input_ids.shape[1]
        print("index:", index, "time", end - start, "output_text:", output_text, "correct_choice:", correct_choice)
        with open(args.result_file, "a", encoding="utf-8") as f:
            json.dump({"index": index, "response": output_text, "pred": extract_answer(output_text), "correct_choice": correct_choice, "judge": correct_choice==extract_answer(output_text)}, f, ensure_ascii=False)
            f.write('\n')

    print("sample num:", (index-2))
    print("avg seqlen", total_seqlen / (index-2))
    print("avg time", total_time / (index-2))
    breakpoint()


