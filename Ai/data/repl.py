import json

# 输入和输出文件路径
input_file = "../data/valid.jsonl"  # 替换为你的输入文件路径
output_file = "../data/valid_re.jsonl"  # 替换为你的输出文件路径


def transform_line(line):
    """
    将单行 JSON 转换为目标格式。
    """
    # 解析原始 JSON 行
    data = json.loads(line)
    text = data.get("text", "")

    # 分割原始内容
    parts = text.split("<|im_end|>")
    messages = []

    for part in parts:
        part = part.strip()
        if not part:
            continue
        if part.startswith("<|im_start|>"):
            # 获取角色和内容
            role, content = part[12:].split("\n", 1)
            role = role.strip()
            content = content.strip()
            # 映射角色名
            if role == "system":
                role = "system"
            elif role == "user":
                role = "user"
            elif role == "assistant":
                role = "assistant"
            messages.append({"role": role, "content": content})

    # 构造目标 JSON 行
    transformed_data = {
        "type": "chatml",
        "messages": messages,
        "source": "self-made"
    }
    return transformed_data


def process_file(input_file, output_file):
    """
    读取输入文件并转换为目标格式，写入输出文件。
    """
    with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
        for line in infile:
            transformed_line = transform_line(line)
            outfile.write(json.dumps(transformed_line, ensure_ascii=False) + "\n")


# 执行转换
process_file(input_file, output_file)
print(f"转换完成，结果已保存到 {output_file}")
