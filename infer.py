import gc
import json
import re
import time
import torch
import transformers

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, StaticCache



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_file", "-r", type=str, default="results")
    parser.add_argument("--model", "-m", type=str, default="/lpai/volumes/lpai-demo-muses/lt/models/Qwen2.5-7B-Instruct")
    parser.add_argument("--page_size", type=int, default=4096) # set to True if using no context (directly measuring memorization)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
                args.model,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.float16,
                # load_in_8bit=True,
                device_map="cuda",
            )
    model = model.eval()
    model = torch.compile(model)

    data = load_dataset('THUDM/LongBench-v2', split='train')

    system = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nPlease read the following text and answer the question below.\n\n'
    text = '<text>\n$DOC$\n</text>\n\n'
    question = 'What is the correct answer to this question: $Q$\nChoices:\n(A) $C_A$\n(B) $C_B$\n(C) $C_C$\n(D) $C_D$\n\nFormat your response as follows: "The correct answer is (insert answer here)".\n<|im_end|>\n'

    system_id = tokenizer(system, return_tensors='pt').to("cuda")
    _, system_len = system_id['input_ids'].shape

    max_new_tokens = 128
    page_size = args.page_size
    max_len = 120000

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

        # torch.cuda.synchronize()
        # start = time.time()
        # # output_base = model.generate(**input_ids, max_new_tokens=max_new_tokens, cache_implementation="offloaded", do_sample=False)
        # # prefill
        # output_prefill = model.generate(**input_ids, max_new_tokens=1, return_dict_in_generate=True, do_sample=False)
        # torch.cuda.synchronize()
        # prefill_end = time.time()
        # # decoding
        # input_ids['input_ids'] = torch.cat([input_ids['input_ids'], question_id['input_ids']], dim=1)
        # input_ids['attention_mask'] = torch.cat([input_ids['attention_mask'], question_id['attention_mask']], dim=1)
        # _, input_len_base = input_ids["input_ids"].shape
        # # print("input_len_base", input_len_base)
        # output_base = model.generate(**input_ids,
                                    # max_new_tokens=max_new_tokens,
                                    # past_key_values=output_prefill["past_key_values"],
                                    # do_sample=False)
        # response = tokenizer.decode(output_base[0, input_len_base:], skip_special_tokens=True)
        # torch.cuda.synchronize()
        # end = time.time()
        # print("index", index, "prefill time", prefill_end - start, "decode time", end - prefill_end, "\tanswer", ele['answer'], flush=True)
        # print("response org", response, flush=True)
        # print("-------------------------------------------------------", flush=True)
        # with open("base_result.jsonl", "a", encoding="utf-8") as f:
            # json.dump({"index": index, "response": response, "pred": extract_answer(response), "answers": ele["answer"], "judge": ele["answer"]==extract_answer(response), "length": ele["length"]}, f, ensure_ascii=False)
            # f.write('\n')

        # breakpoint()

        #######################################################################################
        ## cot + batch
        #######################################################################################
        total_context = system_id.copy()
        torch.cuda.synchronize()
        start = time.time()
        input_ids_nopad = tokenizer(prompt, return_tensors='pt').to("cuda")
        _, seqlen_nopad = input_ids_nopad['input_ids'].shape
        total_context['input_ids'] = torch.cat([total_context['input_ids'], input_ids_nopad['input_ids']], dim=1)
        total_context['attention_mask'] = torch.cat([total_context['attention_mask'], input_ids_nopad['attention_mask']], dim=1)
        total_context['input_ids'] = torch.cat([total_context['input_ids'], question_id['input_ids']], dim=1)
        total_context['attention_mask'] = torch.cat([total_context['attention_mask'], question_id['attention_mask']], dim=1)
        input_ids_all = tokenizer(prompt, return_tensors='pt', padding=True, pad_to_multiple_of=page_size).to("cuda")
        _, seqlen = input_ids_all['input_ids'].shape

        current = 0
        input_ids_list = []
        count = 0
        assert(seqlen % page_size == 0)

        input_ids_all['input_ids'] = input_ids_all['input_ids'].reshape(-1,page_size)
        input_ids_all['attention_mask'] = input_ids_all['attention_mask'].reshape(-1,page_size)
        bs, _ = input_ids_all['input_ids'].shape
        input_ids_all['input_ids'] = torch.cat([system_id['input_ids'].expand(bs, -1), input_ids_all['input_ids']], dim=1)
        input_ids_all['attention_mask'] = torch.cat([system_id['attention_mask'].expand(bs, -1), input_ids_all['attention_mask']], dim=1)
        output = model.generate(**input_ids_all, max_new_tokens=1, return_dict_in_generate=True, do_sample=False)
        batch_cache = output["past_key_values"].batch_split(bs, 1)

        first = True
        for cache in batch_cache:
            if first:
                past_key_values = cache
                first = False
            else:
                cache.slice(system_len, system_len+page_size)
                past_key_values.concat(cache)

        print("seqlen:", seqlen_nopad, "\tpage_size:", page_size, "\tpage num:", seqlen_nopad // page_size + 1, flush=True)
        past_key_values.slice(0, system_len+seqlen_nopad)
        torch.cuda.synchronize()
        prefill_end = time.time()

        output = model.generate(**total_context, max_new_tokens=max_new_tokens, past_key_values=past_key_values, do_sample=False)
        _, input_len = total_context['input_ids'].shape
        response = tokenizer.decode(output[0, input_len:], skip_special_tokens=True)
        torch.cuda.synchronize()
        end = time.time()
        print("index", index, "prefill time", prefill_end - start, "decode time", end - prefill_end, "\tanswer", ele['answer'], flush=True)
        print("response", response, flush=True)
        print("-------------------------------------------------------", flush=True)
        with open(args.result_file, "a", encoding="utf-8") as f:
            json.dump({"index": index, "response": response, "pred": extract_answer(response), "answers": ele["answer"], "judge": ele["answer"]==extract_answer(response), "length": ele["length"]}, f, ensure_ascii=False)
            f.write('\n')

    breakpoint()


