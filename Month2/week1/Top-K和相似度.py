import asyncio
asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 知识库
text = """
RAG是检索增强生成。
RAG可以解决大模型幻觉问题。
大模型幻觉就是指AI凭空编造答案。
切分文本可以让检索更精准。
向量库用来存储文本对应的Embedding向量。
"""

# 切分
splitter = RecursiveCharacterTextSplitter(chunk_size=30, chunk_overlap=20)
chunks = splitter.split_text(text)

# 向量模型
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 建库
db = Chroma.from_texts(chunks, embedding)

# 提问
query = "RAG 能解决什么问题？"

# ======================
# 这里改 k：1、2、3、4
# ======================
k = 3
results = db.similarity_search_with_relevance_scores(query, k=k)

print("用户问题：", query)
print(f"取 top-{k} 条结果\n")

for i, (doc, score) in enumerate(results):
    print(f"第{i+1}条")
    print("内容：", doc.page_content)
    print("相似度分数：", round(score, 3))
    print("---")