# Usage example:
# To run on 4 GPUs with a batch size of 16:
# torchrun --nproc_per_node=4 utils/gen_transformers.py --model /path/to/model --sub_task metamath --output_file output/metamath_response.jsonl --batch_size 16 --num_gpus 4
#
# This will automatically split the dataset into 4 parts and process each part on a different GPU.
# Each GPU will save its results to the same output file.


import argparse
import time
import torch
import sys
import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, concatenate_datasets

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help="Model path")
parser.add_argument("--data_path", type=str, default="pissa-dataset")
parser.add_argument('--sub_task', nargs='+', help='Subtask list')
parser.add_argument('--dataset_split', type=str, default="test", help='Dataset split')
parser.add_argument('--output_file', type=str, default="model_response.jsonl", help="Output file")
parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
parser.add_argument('--temperature', type=float, default=0.0, help="Temperature")
parser.add_argument('--top_p', type=float, default=1, help="Top-p sampling")
parser.add_argument('--max_tokens', type=int, default=512, help="Max tokens")
parser.add_argument('--num_gpus', type=int, default=1, help="Number of GPUs to use")
args = parser.parse_args()

# Get current GPU ID from environment variable
local_rank = int(os.environ.get("LOCAL_RANK", 0))
if local_rank >= args.num_gpus:
    print(f"Local rank {local_rank} is greater than num_gpus {args.num_gpus}, exiting.")
    sys.exit(0)

# Load model and tokenizer
print(f"Loading model from {args.model}...")
model = AutoModelForCausalLM.from_pretrained(
    args.model,
    device_map=f"cuda:{local_rank}",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)
if model.config.model_type == "llamamoe":
    model.config.output_router_logits = False
tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, padding_side='left')
if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

# Generation config
generation_config = {
    "max_new_tokens": args.max_tokens,
    "temperature": args.temperature,
    "top_p": args.top_p,
    "do_sample": args.temperature > 0,
    "pad_token_id": tokenizer.eos_token_id
}

def batch_data(data_list, batch_size=1):
    n = len(data_list) // batch_size
    batch_data = []
    for i in range(n-1):
        start = i * batch_size
        end = (i+1)*batch_size
        batch_data.append(data_list[start:end])

    last_start = (n-1) * batch_size
    last_end = sys.maxsize
    batch_data.append(data_list[last_start:last_end])
    return batch_data

if args.sub_task is None:
    dataset = load_dataset(args.data_path, split=args.dataset_split)
else:
    all_test_dataset = []
    for task in args.sub_task:
        ds = load_dataset(args.data_path, data_dir=task, split=args.dataset_split)
        if local_rank == 0:
            print(f"{args.data_path}/{task}/{args.dataset_split}")
            for k,v in ds[0].items():
                print("-"*100)
                print(k,end=':\t')
                print(v)
            print("+"*100)
        all_test_dataset.append(ds)
        
    dataset = concatenate_datasets(all_test_dataset).select(range(64))

# Split dataset for multi-GPU processing
total_samples = len(dataset)
samples_per_gpu = total_samples // args.num_gpus
start_idx = local_rank * samples_per_gpu
end_idx = (local_rank + 1) * samples_per_gpu if local_rank < args.num_gpus - 1 else total_samples

print(f"GPU {local_rank} processing samples from {start_idx} to {end_idx-1}")
dataset = dataset.select(range(start_idx, end_idx))
    
batch_dataset_query = batch_data(dataset["instruction"], batch_size=args.batch_size)
batch_dataset_answer = batch_data(dataset["output"], batch_size=args.batch_size)
batch_dataset_task = batch_data(dataset["type"], batch_size=args.batch_size)

# modify the output file path, create independent files for each process
output_file = f"{args.output_file}.rank{local_rank}"

for idx, (batch_query, batch_answer, batch_task) in enumerate(zip(batch_dataset_query, batch_dataset_answer, batch_dataset_task)):
    
    print(f"GPU {local_rank} processing batch {idx+1}/{len(batch_dataset_query)}")
    # Encode inputs
    inputs = tokenizer(batch_query, padding=True, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            **generation_config
        )
    
    # Decode outputs
    generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    generated_texts = [o.split("### Response:")[-1].strip() for o in generated_texts]
    
    # save results to independent files
    for query, generated_text, answer, task in zip(batch_query, generated_texts, batch_answer, batch_task):
        with open(output_file, 'a', encoding='utf-8') as f:
            json.dump({
                'type': task,
                'query': query,
                'output': generated_text,
                'answer': answer
            }, f, ensure_ascii=False)
            f.write('\n')

# 创建完成标记文件
with open(f"{args.output_file}.rank{local_rank}.done", 'w') as f:
    f.write(f"GPU {local_rank} completed at {time.time()}")

# after all processes are done, rank 0 process will merge all files
if local_rank == 0:
    all_files_exist = False
    while not all_files_exist:
        all_files_exist = True
        for rank in range(args.num_gpus):
            rank_file = f"{args.output_file}.rank{rank}"
            done_file = f"{args.output_file}.rank{rank}.done"
            if not os.path.exists(rank_file) or not os.path.exists(done_file):
                all_files_exist = False
                time.sleep(1)
                break
    
    # combine all files
    with open(args.output_file, 'w', encoding='utf-8') as outfile:
        for rank in range(args.num_gpus):
            rank_file = f"{args.output_file}.rank{rank}"
            done_file = f"{args.output_file}.rank{rank}.done"
            if os.path.exists(rank_file):
                with open(rank_file, 'r', encoding='utf-8') as infile:
                    outfile.write(infile.read())
                # delete temporary files
                os.remove(rank_file)
                os.remove(done_file)