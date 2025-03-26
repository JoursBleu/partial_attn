import argparse
import gc
import json
import re
import time
import torch
import transformers

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, StaticCache

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
    parser.add_argument("--model", "-m", type=str, default="/lpai/volumes/lpai-yharnam-bd-ga/lt/models/Qwen2.5-7B-Instruct")
    # parser.add_argument("--page_size", type=int, default=4096)
    parser.add_argument("--dataset", type=str, default="THUDM/LongBench-v2")
    parser.add_argument("--cot", type=bool, default=False, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True,)
    model = AutoModelForCausalLM.from_pretrained(
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

    data = load_dataset(args.dataset, split='train')
    if 'qwen' in model.config.model_type:
        system = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nPlease read the following text and answer the question below.\n\n'
        text = '<text>\n$DOC$\n</text>\n\n'
        question = 'What is the correct answer to this question: $Q$\nChoices:\n(A) $C_A$\n(B) $C_B$\n(C) $C_C$\n(D) $C_D$\n\n'
        if args.cot:
            question += 'Let’s think step by step:\n<|im_end|>\n'
            question_post_cot = '\n<|im_start|>user\nBased on the above, what is the single, most likely answer choice? Format your response as follows: "The correct answer is (insert answer here)".\n<|im_end|>\n'
        else:
            question += 'Format your response as follows: "The correct answer is (insert answer here)".\n<|im_end|>\n'
    elif 'llama' in model.config.model_type:
        system = '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nPlease read the following text and answer the question below.\n\n'
        text = '<text>\n$DOC$\n</text>\n\n'
        question = 'What is the correct answer to this question: $Q$\nChoices:\n(A) $C_A$\n(B) $C_B$\n(C) $C_C$\n(D) $C_D$\n\n'
        if args.cot:
            question += 'Let’s think step by step:\n<|eot_id|>\n'
            question_post_cot = '<|start_header_id|>user<|end_header_id|>\n\nBased on the above, what is the single, most likely answer choice? Format your response as follows: "The correct answer is (insert answer here)".\n<|eot_id|>\n'
        else:
            question += 'Format your response as follows: "The correct answer is (insert answer here)".\n<|eot_id|>\n'
    elif 'mistral' in model.config.model_type:
        system = '<s>[INST] You are a helpful assistant.\n\nPlease read the following text and answer the question below.\n\n'
        text = '<text>\n$DOC$\n</text>\n\n'
        question = 'What is the correct answer to this question: $Q$\nChoices:\n(A) $C_A$\n(B) $C_B$\n(C) $C_C$\n(D) $C_D$\n\n'
        if args.cot:
            question += 'Let’s think step by step:[/INST]'
            question_post_cot = '[INST] Based  on the above, what is the single, most likely answer choice? Format your response as follows: "The correct answer is (insert answer here)".[/INST]'
        else:
            question += 'Format your response as follows: "The correct answer is (insert answer here)".[/INST]'

    system_id = tokenizer(system, return_tensors='pt').to("cuda")
    _, system_len = system_id['input_ids'].shape
    if args.cot:
        post_cot_id = tokenizer(question_post_cot, return_tensors="pt").to("cuda")

    max_new_tokens = 128
    max_cot_tokens = 1024
    max_len = 120000

    index = 0
    for ele in data:
        index += 1
        # if index < 144:
            # continue
        # if "long" not in ele["length"]:
            # continue
        # breakpoint()
        context = ele['context']
        prompt = text.replace('$DOC$', context.strip())

        input_ids = tokenizer.encode(prompt)
        if len(input_ids) > max_len:
            input_ids = input_ids[:max_len//2] + input_ids[-max_len//2:]
            prompt = tokenizer.decode(input_ids, skip_special_tokens=True)

        choice_question = question.replace('$Q$', ele['question'].strip()).replace('$C_A$', ele['choice_A'].strip()).replace('$C_B$', ele['choice_B'].strip()).replace('$C_C$', ele['choice_C'].strip()).replace('$C_D$', ele['choice_D'].strip())
        input_ids = tokenizer(system+prompt, return_tensors='pt').to("cuda")
        question_id = tokenizer(choice_question, return_tensors="pt").to("cuda")
        _, question_len = question_id['input_ids'].shape

        torch.cuda.synchronize()
        start = time.time()
        # output_base = model.generate(**input_ids, max_new_tokens=max_new_tokens, cache_implementation="offloaded", do_sample=False)
        # prefill
        output_prefill = model.generate(**input_ids, max_new_tokens=1, return_dict_in_generate=True, do_sample=False)
        torch.cuda.synchronize()
        prefill_end = time.time()
        # decoding
        input_ids['input_ids'] = torch.cat([input_ids['input_ids'], question_id['input_ids']], dim=1)
        input_ids['attention_mask'] = torch.cat([input_ids['attention_mask'], question_id['attention_mask']], dim=1)
        _, input_len_base = input_ids["input_ids"].shape

        if args.cot:
            output_prefill = model.generate(**input_ids,
                                        max_new_tokens=max_cot_tokens,
                                        return_dict_in_generate=True,
                                        past_key_values=output_prefill["past_key_values"],
                                        do_sample=False)
            # decoding
            input_ids['input_ids'] = torch.cat([output_prefill["sequences"], post_cot_id['input_ids']], dim=1)
            input_ids['attention_mask'] = torch.ones(input_ids['input_ids'].shape).int()


        # print("input_len_base", input_len_base)
        output_base = model.generate(**input_ids,
                                    max_new_tokens=max_new_tokens,
                                    past_key_values=output_prefill["past_key_values"],
                                    do_sample=False)
        response = tokenizer.decode(output_base[0, input_len_base:], skip_special_tokens=True)
        torch.cuda.synchronize()
        end = time.time()
        print("index", index, "prefill time", prefill_end - start, "decode time", end - prefill_end, "\tanswer", ele['answer'], flush=True)
        print("response org", response, flush=True)
        print("-------------------------------------------------------", flush=True)
        with open(args.result_file, "a", encoding="utf-8") as f:
            json.dump({"index": index, "response": response, "pred": extract_answer(response), "answers": ele["answer"], "judge": ele["answer"]==extract_answer(response), "length": ele["length"]}, f, ensure_ascii=False)
            f.write('\n')

    breakpoint()


