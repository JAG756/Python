# ==========================
# Day25：完整 RAG 问答系统
# 步骤：文档切分 → 向量化 → 检索 → 大模型生成答案
# ==========================

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# ======================
# 1. 准备私有文档（知识库）
# ======================
doc_content = """
RAG是检索增强生成，它让大模型先从知识库检索资料，再生成答案，不会瞎编。
向量数据库用来存储文本的语义向量，可以快速找到意思相似的内容。
Chroma是最简单的轻量级向量数据库，适合学习和小型项目。
企业用RAG解决大模型幻觉、知识过时、数据安全问题。

CNN是卷积神经网络，常用于图像处理、图像分类、目标检测。
它通过卷积层提取图像特征，池化层降维，最后输出分类结果。
"""

# ======================
# 2. 文档切分
# ======================
documents = [Document(page_content=doc_content)]
text_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=20)
splits = text_splitter.split_documents(documents)

# ======================
# 3. 向量化 + 向量库
# ======================
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db = Chroma.from_documents(documents=splits, embedding=embedding)
retriever = vector_db.as_retriever(search_kwargs={"k": 2})

# ======================
# 4. 加载本地大模型（RTX4060专用）
# ======================
model_name = "Qwen/Qwen-1_8B-Chat"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto"
)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=300)

# ======================
# 5. RAG 提示词模板（核心！）
# ======================
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个严谨的问答助手，必须只根据资料回答，不许瞎编。\n资料：{context}"),
    ("user", "{question}")
])

# ======================
# 6. 组合 RAG 链
# ======================
def format_docs(docs):
    return "\n".join([doc.page_content for doc in docs])

print("\n===== 🚀 RAG 问答系统启动（输入 exit 退出）=====")
while True:
    question = input("你：")
    if question.lower() == "exit":
        print("AI：再见！")
        break

    # 检索相关文档
    docs = retriever.invoke(question)
    context = format_docs(docs)

    # 构造提示词
    prompt_text = prompt.format(context=context, question=question)

    # 大模型生成回答
    response = pipe(prompt_text)[0]["generated_text"].split("user")[-1].split("assistant")[-1].strip()

    # 输出
    print("\n📚 检索到的资料：")
    for d in docs:
        print("-", d.page_content)
    
    print("\n🤖 AI 最终回答：")
    print(response)
    print("-" * 60)