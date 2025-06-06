# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import os
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse

parser = argparse.ArgumentParser(description="Separate the principal singular value and singular vectors from base model")
parser.add_argument("--base_model_path", type=str, required=True, help="The name or path of the base model.")
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--bits", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
parser.add_argument("--init_weights", type=str, default="pissa", help="(`['pissa', 'pissa_niter_[number of iters]']`)")
parser.add_argument("--lora_r", type=int, default=128)
parser.add_argument("--lora_alpha", type=int, default=128)
parser.add_argument("--lora_dropout", type=float, default=0)
parser.add_argument('--target_modules', nargs='+', help='', required=True)
script_args = parser.parse_args()
print(script_args)

model = AutoModelForCausalLM.from_pretrained(
    script_args.base_model_path,
    torch_dtype=(
        torch.float16
        if script_args.bits == "fp16"
        else (torch.bfloat16 if script_args.bits == "bf16" else torch.float32)
    ),
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(script_args.base_model_path)
tokenizer.pad_token_id = tokenizer.eos_token_id
lora_config = LoraConfig(
    r=script_args.lora_r,
    lora_alpha=script_args.lora_alpha,
    init_lora_weights=True if script_args.init_weights=="True" else script_args.init_weights,
    lora_dropout=script_args.lora_dropout,
    target_modules=script_args.target_modules,
)
peft_model = get_peft_model(model, lora_config)

# Save PiSSA modules:
peft_model.peft_config["default"].init_lora_weights = True
peft_model.save_pretrained(os.path.join(script_args.output_dir, "pissa_init"))
# Save residual model:
peft_model = peft_model.unload()
peft_model.save_pretrained(script_args.output_dir)
# Save the tokenizer:
tokenizer.save_pretrained(script_args.output_dir)