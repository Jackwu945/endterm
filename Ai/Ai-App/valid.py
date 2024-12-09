import json
from typing import Dict

import torch
import transformers
from peft import get_peft_model, LoraConfig
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, AutoModelForCausalLM
from transformers.trainer_pt_utils import LabelSmoother

tokenizer = AutoTokenizer.from_pretrained("../models/Qwen1.5-4B-Chat-token/", trust_remote_code=True, padding_side="left")
TEMPLATE = "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content']}}{% if loop.last %}{{ '<|im_end|>'}}{% else %}{{ '<|im_end|>\n' }}{% endif %}{% endfor %}"
IGNORE_TOKEN_ID = LabelSmoother.ignore_index
# 设备设置
device = "cuda" if torch.cuda.is_available() else "cpu"

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
model.to(device)

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


val_data = []
with open("../data/valid_re.jsonl", "r", encoding='utf-8') as f:
    for line in f:
        val_data.append(json.loads(line))

# 数据加载器
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer, model=model, padding=True, max_length=512
)

# 假设你已经准备了验证集数据并使用与训练集相同的预处理方法
# 这里使用一个简单的验证集加载器
val_dataset = SupervisedDataset(val_data, tokenizer=tokenizer, max_len=512)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=data_collator)

# 设置模型为评估模式
model.eval()

# 存储总损失和生成的文本
total_loss = 0
generated_texts = []

# 验证集上的测试
with torch.no_grad():  # 禁用梯度计算
    for batch in val_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}

        # 计算损失
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.item()

        # 生成文本
        generated_ids = model.generate(batch['input_ids'], max_length=512, num_beams=5)
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        generated_texts.append(generated_text)

# 计算验证集的平均损失
avg_loss = total_loss / len(val_dataloader)

print(f"Validation Loss: {avg_loss}")

# 打印部分生成的文本以手动评估
for text in generated_texts[:5]:  # 打印前 5 个生成的文本
    print(text)