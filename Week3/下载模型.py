from transformers import AutoTokenizer, AutoModel
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
model = AutoModel.from_pretrained("bert-base-chinese")

# 输入一句话
text = "我爱深度学习"
inputs = tokenizer(text, return_tensors="pt")

# 前向传播
outputs = model(**inputs)
print(outputs.last_hidden_state.shape)