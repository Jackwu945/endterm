from transformers import AutoModelForCausalLM, AutoTokenizer
import os
os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"

model_name = "Qwen/Qwen1.5-1.8B-Chat"  # 替换为模型名称
while True:
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name,device_map="cpu")
        break
    except:
        continue
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("saving!")
model.save_pretrained("../models/Qwen1.5-1.8B-Chat/")
tokenizer.save_pretrained("../models/Qwen1.5-1.8B-Chat-token/")

# https://pypi.tuna.tsinghua.edu.cn/simple/ is the best mirror for me


