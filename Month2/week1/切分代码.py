from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. 一段长文本
text = """
RAG是检索增强生成。
它先把文档切成小段，转成向量存在向量库。
用户提问时，系统先检索相关段落，再送给大模型回答。
这样可以解决大模型幻觉，让回答更准确。
"""

# 2. 定义切分规则
splitter = RecursiveCharacterTextSplitter(
    chunk_size=30,    # 每段最多30字符
    chunk_overlap=20,  # 重叠20字符
    length_function=len
)

# 3. 开始切分
chunks = splitter.split_text(text)

# 4. 看结果
for i, c in enumerate(chunks):
    print(f"第{i+1}块：", c)
