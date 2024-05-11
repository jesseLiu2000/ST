import torch
import json
import argparse
import numpy as np

from datasets import load_dataset
from deepspeed.pipe import PipelineModule
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from torch.utils.data import DataLoader, Dataset, random_split

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


model_name = "deepseek-ai/deepseek-moe-16b-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
model.generation_config = GenerationConfig.from_pretrained(model_name)
model.generation_config.pad_token_id = model.generation_config.eos_token_id
model_state_dict = model.state_dict()
# model.to(device)
# model.eval()


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
    inputs = tokenizer(sequences, return_tensors="pt")
    return inputs


def get_layer_output(module, input, output):
    inputs = input[0].squeeze().detach()
    layer_outputs.append(inputs)


type_name = args.dataset 
dataset = load_dataset(args.data, type_name) if type_name != 'none' else load_dataset(args.data)
test_data = dataset[args.subset]

output_name = type_name if type_name != "none" else args.data
output_name = output_name if "/" not in output_name else output_name.split("/")[-1]

if args.type == "2":
    full_sentence = []
    for q, s in zip(test_data[args.sub_one], test_data[args.sub_two]):
        prompt = q + s
        full_sentence.append(prompt)

elif args.type == "1":
    full_sentence = test_data[args.sub_one]
    
elif args.type == "3":
    full_sentence = []
    for q, s, z in zip(
        test_data[args.sub_one], test_data[args.sub_two], test_data[args.sub_three]
    ):
        if output_name == "multiple_choice":
            prompt = q + " ".join(s['choices']) + " ".join(z['choices'])
        else:
            prompt = q + s + z
        full_sentence.append(prompt)


ax_dataset = GLUEDataset(full_sentence)
ax_dataloader = DataLoader(
    ax_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn
)

expert_dict = {i:[[] for _ in range(64)]for i in range(27)}
full_expert_dict = {}
with torch.no_grad():
    for idx, inputs in enumerate(ax_dataloader):
        inputs = inputs.to(device)
        layer_outputs = []
        hooks = []
        for decoder_layer in model.model.layers[1:]:
            hook = decoder_layer.mlp.gate.register_forward_hook(
                get_layer_output
            )
            hooks.append(hook)
        
        result = model.generate(**inputs.to(model.device), max_new_tokens=1)

        for layer_index, X in enumerate(layer_outputs):
            C_in = X.shape[-1]
            s = 0.5
            for expert_index in range(0, 64):
                weight_down = model_state_dict[f"model.layers.{layer_index+1}.mlp.experts.{expert_index}.down_proj.weight"].to(device2)
                X = X.to(device2)
                metric = weight_down.T.abs() * X.norm(p=2, dim=0)

                _, sorted_idx = torch.sort(metric, dim=1, descending=True)
                pruned_idx = sorted_idx[:, :int(C_in * s)]

                pruned_matrix = torch.gather(metric, 1, pruned_idx)
                mean_value = float(pruned_matrix.mean().to(device2))

                expert_dict[layer_index][expert_index].append(mean_value)
        

        for hook in hooks:
            hook.remove()



output_name = type_name if type_name != "none" else args.data
output_name = output_name if "/" not in output_name else output_name.split("/")[-1]

with open(
    f"/mnt/deepseek/cal_important/resutls/mmlu_new/{output_name}.json", "w"
) as fw:
    json.dump(expert_dict, fw, indent=4)



"""
CUDA_VISIBLE_DEVICES=6,5 nohup python eval_mt_model.py winogrande winogrande_debiased test 3 --sub_one sentence --sub_two option1 --sub_three option2 > ./log/raw/winogrande_debiased.lb 2>&1 &
CUDA_VISIBLE_DEVICES=5,6 nohup python eval_mt_model.py Rowan/hellaswag none test 3 --sub_one ctx_a --sub_two ctx_b --sub_three activity_label > ./log/raw/hellaswag.lb 2>&1 &
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python eval_mt_model.py truthful_qa generation validation 2 --sub_one question --sub_two best_answer > ./log/raw/generation.lb 2>&1 &
CUDA_VISIBLE_DEVICES=6,5 nohup python eval_raw_model.py truthful_qa multiple_choice validation 3 --sub_one question --sub_two mc1_targets --sub_three mc2_targets > ./log/raw/multiple_choice.lb 2>&1 &
CUDA_VISIBLE_DEVICES=4,3 nohup python eval_mt_model.py gsm8k main test 2 --sub_one question --sub_two answer > ./log/raw/main.lb 2>&1 &
CUDA_VISIBLE_DEVICES=4,3 nohup python eval_mt_model.py gsm8k socratic test 2 --sub_one question --sub_two answer > ./log/raw/socratic.lb 2>&1 &


"""
