from langchain_huggingface import HuggingFaceEmbeddings
from scipy.spatial.distance import cosine

# 加载向量模型
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 三句话
q1 = "RAG 怎么解决幻觉"
q2 = "检索增强生成为什么更准确"
q3 = "今天吃什么"

# 转向量
v1 = embedding.embed_query(q1)
v2 = embedding.embed_query(q2)
v3 = embedding.embed_query(q3)

# 计算距离（越小越相似）
sim12 = cosine(v1, v2)
sim13 = cosine(v1, v3)

print("q1 与 q2 距离：", sim12)
print("q1 与 q3 距离：", sim13)