# Only for reference, these codes is not part of the current task.
# Please refer pre_process.py and repl.py for the complete code.

import json
import os
from typing import Dict

import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq
from transformers import get_scheduler
from transformers.trainer_pt_utils import LabelSmoother

tokenizer = AutoTokenizer.from_pretrained("../models/Qwen1.5-4B-Chat-token/", trust_remote_code=True, padding_side="left")
TEMPLATE = "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content']}}{% if loop.last %}{{ '<|im_end|>'}}{% else %}{{ '<|im_end|>\n' }}{% endif %}{% endfor %}"
IGNORE_TOKEN_ID = LabelSmoother.ignore_index


def preprocess(
        messages,
        tokenizer: transformers.PreTrainedTokenizer,
        max_len: int,
) -> Dict:
    """Preprocesses the data for supervised fine-tuning."""

    texts = []
    for i, msg in enumerate(messages):
        texts.append(
            tokenizer.apply_chat_template(
                msg,
                chat_template=TEMPLATE,
                tokenize=True,
                add_generation_prompt=False,
                padding="max_length",
                max_length=max_len,
                truncation=True,
            )
        )
    input_ids = torch.tensor(texts, dtype=torch.int)
    target_ids = input_ids.clone()
    target_ids[target_ids == tokenizer.pad_token_id] = IGNORE_TOKEN_ID
    attention_mask = input_ids.ne(tokenizer.pad_token_id)

    return dict(
        input_ids=input_ids, target_ids=target_ids, attention_mask=attention_mask
    )


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
            self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len: int
    ):
        super(SupervisedDataset, self).__init__()

        messages = [example["messages"] for example in raw_data]
        data_dict = preprocess(messages, tokenizer, max_len)

        self.input_ids = data_dict["input_ids"]
        self.target_ids = data_dict["target_ids"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.target_ids[i],
            attention_mask=self.attention_mask[i],
        )


train_data = []
with open("../data/rawtrain.jsonl", "r", encoding='utf-8') as f:
    for line in f:
        train_data.append(json.loads(line))

dataset = SupervisedDataset(train_data, tokenizer=tokenizer, max_len=512)

# 配置
MODEL_NAME = "../models/Qwen1.5-4B-Chat"  # 替换为模型路径
DATA_PATH = "/root/data/train.jsonl"  # 替换为数据路径
OUTPUT_DIR = "../models/qwen-4B-lora-ft2/"  # 微调模型保存路径
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 4
LR = 1e-4
EPOCHS = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained("../models/Qwen1.5-4B-Chat-token/", trust_remote_code=True, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)

# 配置 LoRA
lora_config = LoraConfig(
    r=16,  # Bottleneck大小
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # 应用LoRA的模块
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)


# 修改 preprocess_function
def preprocess_function(examples):
    # 提取文本
    text = examples["text"]
    # 使用 tokenizer 进行分词，并确保输出为 PyTorch 张量格式
    tokens = tokenizer(
        text,
        max_length=512,  # 最大长度
        truncation=True,  # 截断
        padding="max_length",  # 填充到最大长度
        return_tensors="pt"  # 返回 PyTorch 格式张量
    )

    # 确保返回的是 "input_ids" 和 "labels"（用来做语言建模的标签）
    tokens["labels"] = tokens["input_ids"].clone()  # 用同样的 input_ids 作为标签
    print(tokens)
    return tokens


dataset = load_dataset("json", data_files=DATA_PATH)
tokenized_dataset = dataset["train"].map(preprocess_function, batched=True)

# 定义数据集和批处理器
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,  # 用于正确处理 padding 和对齐
    padding=True,  # 自动填充
)

# 数据加载器
dataloader = DataLoader(
    tokenized_dataset,
    collate_fn=data_collator,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# 优化器 & 学习率调度器
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0,
                          num_training_steps=len(dataloader) * EPOCHS)

# 训练
model.to(DEVICE)
model.train()

for epoch in range(EPOCHS):
    for step, batch in enumerate(dataloader):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        if (step + 1) % GRAD_ACCUM_STEPS == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if step % 10 == 0:
            print(f"Epoch {epoch + 1}/{EPOCHS}, Step {step}, Loss: {loss.item()}")

# 保存微调模型（LoRA 权重）
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("LoRA 微调模型保存完成！")