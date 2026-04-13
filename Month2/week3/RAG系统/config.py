# config.py

# 模型配置
MODEL_NAME = "Qwen/Qwen1.5-1.8B-Chat"

# 检索配置
SIMILARITY_THRESHOLD = 0.5          # 相似度阈值（低于此值认为不相关）
HYBRID_SIMILARITY_THRESHOLD = 0.3   # 混合检索相似度阈值（得分≥此值才采纳）
TOP_K = 10                           # 检索返回的最相似块数量

# 生成配置
USE_GENERATION =  True           # True: 让模型基于检索内容生成答案；False: 直接返回原文
DEFAULT_MAX_LENGTH = 100         # 生成的最大 token 数
MAX_PROMPT_LENGTH = 2048         # 输入 prompt 的最大长度（超出截断）
DEFAULT_TEMPERATURE = 0.1        # 生成温度（越低越保守）

# 文档分块配置
CHUNK_SIZE = 500        # 每个块字符数
CHUNK_OVERLAP = 50      # 块间重叠字符数

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
FOLLOWUP_WORDS = ["它", "这个", "那个", "其","这", "那"]
OMIT_STARTS = ["它的", "这个的", "那个的", "其", "怎么", "如何", "为什么", "有什么", "哪些"]

LOG_LEVEL = "INFO"
LOG_FILE = "rag_system.log"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

#使用语义分块配置
USE_SEMANTIC_CHUNKING = False   # 默认关闭，需要时改为 True

USE_HYBRID_SEARCH = True   # 是否启用混合检索
HYBRID_ALPHA = 0.5         # 向量相似度权重，BM25 权重为 1-alpha

USE_RERANK = True   # 是否启用重排序
RERANK_MODEL = "BAAI/bge-reranker-base"

# 分层检索配置
USE_HIERARCHICAL = True          # 是否启用分层检索（适用于长文档）
CHAPTER_TOP_K = 2                # 第一层检索返回的相关章节数