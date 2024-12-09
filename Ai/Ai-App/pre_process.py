import json
import os
import transformers
from typing import Dict, List, Optional
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq, get_scheduler
from transformers.trainer_pt_utils import LabelSmoother
import torch



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

train_dataset = SupervisedDataset(train_data, tokenizer=tokenizer, max_len=512)

# LoRA 配置
lora_config = LoraConfig(
    r=16,  # Bottleneck大小
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # 应用 LoRA 的模块
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

# 加载预训练模型
model = AutoModelForCausalLM.from_pretrained(
    "../models/Qwen1.5-4B-Chat", trust_remote_code=True
)
model = get_peft_model(model, lora_config)

# 数据加载器
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer, model=model, padding=True, max_length=512, truncation=True
)

train_dataloader = DataLoader(
    train_dataset, batch_size=2, shuffle=True, collate_fn=data_collator
)

# 优化器和学习率调度器
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
num_training_steps = len(train_dataloader) * 3  # 假设 3 个 epoch
scheduler = get_scheduler(
    "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

# 设备设置
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# 训练循环
model.train()
for epoch in range(3):
    for step, batch in enumerate(train_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        if (step + 1) % 4 == 0:  # 梯度累积，每 4 步更新一次参数
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if step % 10 == 0:
            print(f"Epoch {epoch + 1}, Step {step}, Loss: {loss.item()}")

# 保存模型
output_dir = "../models/qwen-4B-lora-ft"
os.makedirs(output_dir, exist_ok=True)
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print("LoRA 微调完成并保存！")