需要create一个results/mmlu的文件夹，他需要写入：
with open(
    f"./results/mmlu/{output_name}.json", "w"
) as fw:
    json.dump(expert_dict, fw)

