# config.py

# 模型配置
MODEL_NAME = "Qwen/Qwen1.5-1.8B-Chat"

# 检索配置
SIMILARITY_THRESHOLD = 1.0      # 原来可能是1.2，改为1.0更严格
TOP_K = 1                        # 返回最相似的一个块

# 生成配置
DEFAULT_MAX_LENGTH = 200
DEFAULT_TEMPERATURE = 0.3

# 默认知识库（当docs文件夹为空时使用）
KNOWLEDGE_BASE = [
    "RAG（检索增强生成）是一种技术，它让大模型先从知识库检索相关资料，再基于资料生成答案，可以有效防止模型瞎编（幻觉）。",
    "CNN（卷积神经网络）是一种深度学习架构，专门用于处理网格状数据，如图像。",
    "Transformer是当前大语言模型的基础架构，它通过自注意力机制捕获长距离依赖关系。",
    "LoRA（低秩适配）是一种高效微调大模型的方法。",
]

# 意图识别配置
GREETINGS = ["你好", "您好", "hi", "hello", "嗨"]
THANKS = ["谢谢", "感谢", "多谢", "thx", "thanks"]
GOODBYE = ["再见", "拜拜", "bye", "goodbye"]
FUZZY_QUESTIONS = ["那个", "这个", "那个是什么", "这个是什么"]

# 多轮对话配置
FOLLOWUP_WORDS = ["它", "这个", "那个", "其"]
OMIT_STARTS = ["怎么", "如何", "为什么", "有什么", "哪些"]