# python>=3.10
import argparse
import json
import os
import sys

import numpy
import numpy as np
import setproctitle
import torch
from datasets import load_dataset
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoModelForCausalLM, AutoTokenizer

setproctitle.setproctitle("important_llamamoe")

parser = argparse.ArgumentParser(description="LLAMAMOE")
parser.add_argument("data", type=str, default="glue", help="input the datasets")
parser.add_argument("dataset", type=str, default="wnli", help="input the sub-dataset")
parser.add_argument(
    "subset", type=str, default="test", help="input the subset(test/val)"
)
parser.add_argument("type", type=str, default="1", help="input the connect number")
parser.add_argument(
    "--sub_one", type=str, default="question", help="input the sub-type"
)
parser.add_argument(
    "--sub_two", type=str, default="sentence", help="input the sub-type"
)
parser.add_argument(
    "--sub_three", type=str, default="sentence", help="input the sub-type"
)

args = parser.parse_args()
device = torch.device("cuda:0")
device2 = torch.device("cuda:1")
model_dir = "llama-moe/LLaMA-MoE-v1-3_5B-2_8"
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_dir, torch_dtype=torch.bfloat16, trust_remote_code=True
)
model.to(device)
model.eval()
model_dict  = model.state_dict()

class GLUEDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequences = self.sequences[idx]
        return sequences


def collate_fn(batch):
    """Define the collate function for dataloader"""
    sequences = batch
    inputs = tokenizer(sequences, padding=True, return_tensors="pt")
    return inputs


def get_index_output(module, input, output):
    expert_idx = output["topK_indices"].detach().cpu().tolist()
    layer_outputs.append(expert_idx)

def get_calculate_input(module, input, output):
    input = input[0].detach().to(device2)
    layer_inputs.append(input)


type_name = args.dataset
dataset = (
    load_dataset(args.data, type_name)
    if type_name != "none"
    else load_dataset(args.data)
)
length = int(len(dataset[args.subset]) * 0.2)
print(length)
test_data = dataset[args.subset][0:length]

if args.type == "2":
    full_sentence = []
    for q, s in zip(test_data[args.sub_one], test_data[args.sub_two]):
        prompt = q + s
        full_sentence.append(prompt)

    ax_dataset = GLUEDataset(full_sentence)
    ax_dataloader = DataLoader(ax_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)


elif args.type == "1":
    full_expert_dict = {}
    # full_sent = [i['text'] for i in test_data[args.sub_one]]
    ax_dataset = GLUEDataset(test_data[args.sub_one])
    ax_dataloader = DataLoader(ax_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    
elif args.type == "3":
    full_sentence = []

    for q, s, z in zip(
        test_data[args.sub_one], test_data[args.sub_two], test_data[args.sub_three]
    ):
        # prompt = q + " ".join(s["choices"]) + " ".join(z["choices"])
        prompt = q + s + z
        full_sentence.append(prompt)

    ax_dataset = GLUEDataset(full_sentence)
    ax_dataloader = DataLoader(ax_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

expert_dict = {i:[[] for _ in range(8)]for i in range(32)}
full_expert_dict = {}
with torch.no_grad():
    for idx, inputs in enumerate(ax_dataloader):
        print(idx)
        full_expert_dict[idx] = {}
        inputs = inputs.to(device)
        layer_outputs = []
        layer_inputs = []
        hooks = []
        hooks_input = []
        for layer in model.model.layers:
            hook = layer.mlp.gate.register_forward_hook(get_index_output)
            hook_input = layer.mlp.calculator.register_forward_hook(get_calculate_input)
            hooks.append(hook)
            hooks_input.append(hook_input)
        
        result = model.generate(**inputs, max_new_tokens=1, temperature=0.0)
        full_expert_dict[idx]['expert_index'] = layer_outputs
        full_expert_dict[idx]['inputs'] = layer_inputs

        for layer_index, X in enumerate(layer_inputs):
            C_in = X.shape[-1]
            s = 0.5
            for expert_index in range(0, 8):
                X = X.to(device2)
                # weight_gate = model_dict[f"model.layers.{layer_index}.mlp.calculator.experts.weight_gate.{expert_index}"].to(device2)
                # weight_up = model_dict[f"model.layers.{layer_index}.mlp.calculator.experts.weight_up.{expert_index}"].to(device2)
                weight_down = model_dict[f"model.layers.{layer_index}.mlp.calculator.experts.weight_down.{expert_index}"].to(device2)

                # gate = weight_gate.abs() * X.norm(p=2, dim=0)
                # up = weight_up.abs() * X.norm(p=2, dim=0)
                # gather = gate * up
                # metric = gather @ weight_down
                metric = weight_down.T.abs() * X.norm(p=2, dim=0)

                _, sorted_idx = torch.sort(metric, dim=1, descending=True)  
                pruned_idx = sorted_idx[:, :int(C_in * s)] 

                pruned_matrix = torch.gather(metric, 1, pruned_idx)
                mean_value = float(pruned_matrix.mean().to(device2))

                expert_dict[layer_index][expert_index].append(mean_value)

        # expert_expand = np.array(list(expert_dict.values()))
        # break
        torch.cuda.set_device("cuda:0")
        torch.cuda.set_device(device2)
        torch.cuda.empty_cache()
        # del weight_gate
        # del weight_up
        del weight_down
        del mean_value

        for hook in hooks:
            hook.remove()
        for hook in hooks_input:
            hook.remove()


        


output_name = type_name if type_name != "none" else args.data
output_name = output_name if "/" not in output_name else output_name.split("/")[-1]

with open(
    f"./results/mmlu/{output_name}.json", "w"
) as fw:
    json.dump(expert_dict, fw)

"""
CUDA_VISIBLE_DEVICES=0,2 nohup python cal_mt.py piqa none test 3 --sub_one goal --sub_two sol1 --sub_three sol2 > piqa.lb 2>&1 &
CUDA_VISIBLE_DEVICES=0,2 nohup python cal_mt.py nyu-mll/glue mrpc test 2 --sub_one sentence1 --sub_two sentence2 > glue.ax 2>&1 &
CUDA_VISIBLE_DEVICES=0,2 nohup python cal_mt.py Rowan/hellaswag none test 3 --sub_one ctx_a --sub_two ctx_b --sub_three activity_label > hellaswag.lb 2>&1 &
"""
