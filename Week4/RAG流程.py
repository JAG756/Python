# ==========================
# Day24：最简 RAG 流程（100%可运行）
# 1. 文档切分
# 2. 向量化
# 3. 存入向量库
# 4. 语义检索
# ==========================

# 正确新版导入！
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# ======================
# 1. 准备文档
# ======================
doc_content = """
RAG是检索增强生成，它让大模型先从知识库检索资料，再生成答案，不会瞎编。
向量数据库用来存储文本的语义向量，可以快速找到意思相似的内容。
Chroma是最简单的轻量级向量数据库，适合学习和小型项目。
FAISS是Facebook开源的向量搜索库，速度快，但只存在内存。
企业用RAG解决大模型幻觉、知识过时、数据安全问题。

CNN是卷积神经网络，常用于图像处理、图像分类、目标检测。
它通过卷积层提取图像特征，池化层降维，最后全连接层输出分类结果。
CNN在CV领域非常常用，是深度学习的基础模型。
"""

documents = [Document(page_content=doc_content)]

# ======================
# 2. 文档切分（核心步骤）
# ======================
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,    # 每块大小
    chunk_overlap=20   # 块之间重叠
)
splits = text_splitter.split_documents(documents)
print("✅ 切分完成，总块数：", len(splits))

# ======================
# 3. 向量化（生成语义向量）
# ======================
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ======================
# 4. 存入向量数据库 Chroma
# ======================
vector_db = Chroma.from_documents(
    documents=splits,
    embedding=embedding
)
retriever = vector_db.as_retriever(search_kwargs={"k": 2})

# ======================
# 5. 用户提问 → 检索最相关内容
# ======================
print("\n===== 检索结果 =====")
question = "什么是CNN？"
result = retriever.invoke(question)

print("问题：", question)
print("\n找到的相关内容：")
for i, doc in enumerate(result):
    print(f"{i+1}. {doc.page_content}")