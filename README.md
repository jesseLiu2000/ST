需要create一个./results/mmlu的文件夹，他需要写入：


with open(
    f"./results/mmlu/{output_name}.json", "w"
) as fw:
    json.dump(expert_dict, fw)

然后还需要create一个./log


