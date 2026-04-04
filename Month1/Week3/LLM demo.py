import os

# 【关键】国内镜像，满速下载，不用梯子
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 设置环境变量，将 Hugging Face 的默认下载地址从 https://huggingface.co 切换到国内镜像站 hf-mirror.com。
# Hugging Face 的服务器在国外，国内访问速度慢且可能不稳定。使用镜像站可以大幅提升下载速度，无需 VPN 或梯子。

from transformers import AutoTokenizer, AutoModelForCausalLM
# AutoTokenizer：自动加载与模型匹配的分词器。分词器负责把人类语言（如"今天天气很好"）转换成模型能理解的数字（token IDs）。
# AutoModelForCausalLM：自动加载与模型匹配的因果语言模型。"Causal"（因果）意味着这是一个自回归模型，它根据前面已生成的文本，预测下一个词是什么。
# GPT 系列、QwQ 系列都属于这种类型。

# 极小模型，300MB，秒下
model_name = "distilgpt2"

# force_download=True 强制重新下载，不读坏缓存
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 输入句子
prompt = "today is a good day, because"


# 生成
inputs = tokenizer(prompt, return_tensors="pt")
# tokenizer()：将输入文本转换成模型需要的格式。它会把文本分词，并将每个词转换成对应的 token ID。
# return_tensors="pt"：指定返回 PyTorch 张量格式的数据，适用于 PyTorch 模型。如果使用 TensorFlow，则可以设置为 "tf"。

output = model.generate(**inputs, max_length=50)
# model.generate() 是生成文本的核心方法。它接受输入文本的 token IDs，并根据模型的语言理解能力，预测后续的文本内容。
# max_length=50 参数限制了生成文本的最大长度为 50 个 token，

result = tokenizer.decode(output[0], skip_special_tokens=True)
# output[0]：generate() 返回的是一个批次的生成结果，即使只有一个输入也是一个批次，所以取第一个（也是唯一一个）结果。
# tokenizer.decode()：将生成的一串 token IDs（数字）反向转换回人类可读的文本。
# skip_special_tokens=True：跳过那些用于模型内部控制的特殊 token（如 [CLS]、[SEP]、[PAD] 等），只输出真正的文本内容。

print("生成结果：", result)