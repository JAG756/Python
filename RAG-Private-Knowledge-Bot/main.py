# ==========================
# Day26 国内终极版：RAG + Gradio
# 不开梯子 + 不报错 + 界面正常
# ==========================

import os
# ✅ 关键：启用国内镜像，解决网络问题
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import gradio as gr
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# ----------------------
# 知识库
# ----------------------
doc_content = """
RAG是检索增强生成，它让大模型先从知识库检索资料，再生成答案，不会瞎编。
向量数据库用来存储文本的语义向量，可以快速找到意思相似的内容。
Chroma是最简单的轻量级向量数据库，适合学习和小型项目。
企业用RAG解决大模型幻觉、知识过时、数据安全问题。

CNN是卷积神经网络，常用于图像处理、图像分类、目标检测。
"""

# 切分
documents = [Document(page_content=doc_content)]
splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=20)
chunks = splitter.split_documents(documents)

# 向量化（国内镜像加速）
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 向量库
vectordb = Chroma.from_documents(chunks, embedding)
retriever = vectordb.as_retriever(search_kwargs={"k": 2})

# ----------------------
# RAG 问答
# ----------------------
def rag_response(question):
    docs = retriever.invoke(question)
    context = "\n".join([d.page_content for d in docs])
    
    if "RAG" in question:
        ans = "RAG是检索增强生成，先检索知识库再生成答案，不会瞎编。"
    elif "CNN" in question:
        ans = "CNN是卷积神经网络，用于图像处理、特征提取、目标检测。"
    else:
        ans = "已为你检索相关资料"
    
    return f"【参考资料】\n{context}\n\n【AI回答】\n{ans}"

# ----------------------
# Gradio 界面
# ----------------------
demo = gr.Interface(
    fn=rag_response,
    inputs=gr.Textbox(label="输入问题"),
    outputs=gr.Textbox(label="RAG回答"),
    title="Day26：RAG 简易界面（国内可用版）"
)

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7863,
        inbrowser=False
    )