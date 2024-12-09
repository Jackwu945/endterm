# import json
# import time
#
# from mlx_lm import load, generate
#
# # with open('../models/Qwen1.5-32B-Chat/tokenizer_config.json', 'r') as file:
# #     tokenizer_config = json.load(file)
#
# with open('../models/Qwen1.5-32B-Chat-FT-4Bit/tokenizer_config.json', 'r') as file:
#     tokenizer_config = json.load(file)
#
# # model, tokenizer = load(
# #     "mlx_model/Qwen1.5-32B-Chat/",
# #     tokenizer_config=tokenizer_config
# # )
#
# model, tokenizer = load(
#     "../models/Qwen1.5-32B-Chat-FT-4Bit/",
#     tokenizer_config=tokenizer_config
# )
#
# sys_msg = 'You are a helpful assistant'
#
# # with open('../text/chat_template.txt', 'r') as template_file:
# #     template = template_file.read()
#
# with open('../text/chat_template.txt', 'r') as template_file:
#     template = template_file.read()
#
# while True:
#     usr_msg = input("用户: ")  # Get user message from terminal
#     if usr_msg.lower() == 'quit()':  # Allows the user to exit the loop
#         break
#
#     prompt = template.replace("{usr_msg}", usr_msg)
#
#     time_ckpt = time.time()
#     response = generate(
#         model,
#         tokenizer,
#         prompt=prompt,
#         temp=0.3,
#         max_tokens=500,
#         verbose=False
#     )
#
#     print("%s: %s (Time %d ms)\n" % ("回答", response, (time.time() - time_ckpt) * 1000))

import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# 加载 tokenizer 配置文件
with open('../models/merged/tokenizer_config.json', 'r',encoding='utf-8') as file:
    tokenizer_config = json.load(file)

# 加载模型和 tokenizer
# model_path = "../models/Qwen1.5-4B-Chat/"
# config = AutoConfig.from_pretrained(model_path)  # 手动加载 con
# tokenizer = AutoTokenizer.from_pretrained(model_path, config=config)
# model = AutoModelForCausalLM.from_pretrained(model_path, config=config)

model_path = "../models/merged/"
token_path = "../models/merged/"
tokenizer = AutoTokenizer.from_pretrained(token_path)
model = AutoModelForCausalLM.from_pretrained(model_path,device_map="auto",torch_dtype=torch.float16)

# 定义系统消息和模板
sys_msg = 'You are a helpful assistant'
with open('../text/chat_template.txt', 'r',encoding='utf-8') as template_file:
    template = template_file.read()

print("输入 'quit()' 退出程序。\n")

# 开始对话循环
while True:
    usr_msg = input("用户: ")  # 获取用户输入
    if usr_msg.lower() == 'quit()':  # 用户输入 quit() 退出循环
        print("退出对话。")
        break

    # 替换模板中的用户消息
    prompt = template.replace("{usr_msg}", usr_msg)

    # 测量生成时间
    time_ckpt = time.time()
    # 加载模型
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs["input_ids"],
        max_length=300,
        temperature=0.3,
        do_sample=True
    )

    # 解码并显示响应
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("%s: %s (Time %d ms)\n" % ("回答", response, (time.time() - time_ckpt) * 1000))
