import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import gradio as gr
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime

# ====================== 初始化 ======================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ 使用设备：{device}")

# ====================== 加载模型 ======================
model_name = "Qwen/Qwen1.5-1.8B-Chat"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None,
    trust_remote_code=True,
).eval()

if device == "cpu":
    model = model.to(device)

# ====================== 知识库 ======================
knowledge_base = [
    "RAG（检索增强生成）是一种技术，它让大模型先从知识库检索相关资料，再基于资料生成答案，可以有效防止模型瞎编（幻觉）。企业应用RAG可以解决大模型的幻觉问题、知识过时问题以及数据安全合规问题。",
    "向量数据库（如Chroma、Milvus）用来存储文本的语义向量，通过向量相似度搜索，可以快速找到意思最相似的内容。企业级向量数据库需要支持高并发、高可用和水平扩展。",
    "CNN（卷积神经网络）是一种深度学习架构，专门用于处理网格状数据，如图像。它在图像分类、目标检测、图像分割等任务中表现优异。",
    "Transformer是当前大语言模型的基础架构，它通过自注意力机制捕获长距离依赖关系。企业级应用包括智能客服、代码助手、文档分析等。",
    "LoRA（低秩适配）是一种高效微调大模型的方法，通过训练少量参数就能让模型适应特定任务，大幅降低训练成本。",
]

documents = [Document(page_content=text) for text in knowledge_base]
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

embedding = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': device}
)
vectordb = Chroma.from_documents(chunks, embedding)
print(f"✅ 知识库已加载，共 {len(knowledge_base)} 条知识")

# ====================== 对话管理（核心）======================
class ChatSession:
    def __init__(self):
        self.history = []  # 对话历史
        self.last_context = None  # 上一轮的检索结果
        self.last_topic = None  # 上一轮的话题
    
    def add(self, question, answer, context):
        self.history.append({"q": question, "a": answer})
        self.last_context = context
        # 提取话题
        for kw in ["RAG", "CNN", "Transformer", "LoRA", "向量数据库"]:
            if kw in question or (context and kw in context):
                self.last_topic = kw
                break
    
    def get_contextual_question(self, question):
        """将追问转换为完整问题"""
        # 检测指代词
        if self.last_topic and any(w in question for w in ["它", "这个", "那个", "其", "它有什么", "它怎么"]):
            return f"{self.last_topic} {question}"
        return question
    
    def clear(self):
        self.history = []
        self.last_context = None
        self.last_topic = None

sessions = {}

def get_session(sid):
    if sid not in sessions:
        sessions[sid] = ChatSession()
    return sessions[sid]

# ====================== 核心回答函数 ======================
def answer(question, session_id, max_len=200, temp=0.3):
    if not question.strip():
        return "请输入问题", session_id
    
    if not session_id:
        session_id = str(datetime.now().timestamp())
    
    session = get_session(session_id)
    
    # 1. 处理指令
    if question in ["清空", "重置"]:
        session.clear()
        return "✅ 对话已清空", session_id
    
    # 2. 处理问候
    if question in ["你好", "您好", "hi"]:
        return "你好！我可以回答RAG、CNN、Transformer、LoRA等技术问题。", session_id

    #  处理感谢（新增）
    if question in ["谢谢", "感谢", "多谢", "thx"]:
        return "不客气！如果还有其他问题，随时问我。", session_id

         # 处理感谢（新增）
    if question in ["谢谢", "感谢", "多谢", "thx"]:
        return "不客气！如果还有其他问题，随时问我。", session_id

    # 在检索之前添加
    fuzzy_questions = ["那个", "这个", "那个是什么", "这个是什么"]
    if any(fq in question for fq in fuzzy_questions):
        return "抱歉，请具体描述你想了解的技术问题。", session_id
    
    # 3. 处理追问：将指代词替换为上一轮的话题
    full_question = session.get_contextual_question(question)
    
    # 4. 检索：如果有上一轮的上下文，优先使用（追问场景）
    if session.last_context and "它" in question:
        context = session.last_context
    else:
        docs = vectordb.similarity_search_with_score(full_question, k=1)
        if not docs or docs[0][1] > 1.2:
            return f"未找到关于「{question}」的信息。\n\n💡 可以问：什么是RAG？CNN是什么？", session_id
        context = docs[0][0].page_content
    
    # 5. 生成回答
    prompt = f"""资料：{context}
问题：{full_question}
回答："""
    
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_len,
            temperature=temp,
            do_sample=temp > 0,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    answer_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    if not answer_text:
        answer_text = context
    
    # 6. 记录历史
    session.add(question, answer_text, context)
    
    return answer_text, session_id

# ====================== 简洁界面 ======================
with gr.Blocks(title="RAG问答", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚀 RAG 智能问答\n\n**支持多轮对话 | 自动理解追问**")
    
    session_id = gr.State()
    
    with gr.Row():
        with gr.Column(scale=3):
            question = gr.Textbox(
                label="问题",
                placeholder="例如：什么是RAG？\n或者追问：它有什么优点？",
                lines=2
            )
            with gr.Row():
                submit = gr.Button("提交", variant="primary")
                clear = gr.Button("清空对话")
        
        with gr.Column(scale=4):
            output = gr.Textbox(
                label="回答",
                lines=12,
                interactive=False
            )
    
    # 参数（折叠起来，保持简洁）
    with gr.Accordion("⚙️ 高级参数", open=False):
        max_len = gr.Slider(50, 500, value=200, label="最大长度")
        temp = gr.Slider(0.0, 1.0, value=0.3, label="温度")
    
    # 示例
    gr.Markdown("### 💡 试试这些")
    examples = [
        ["什么是RAG？"],
        ["它有什么优点？"],
        ["CNN是什么？"],
        ["它怎么工作？"],
        ["你好"],
        ["清空"],
    ]
    gr.Examples(examples, inputs=question, label="点击测试")
    
    # 事件绑定
    submit.click(answer, [question, session_id, max_len, temp], [output, session_id])
    question.submit(answer, [question, session_id, max_len, temp], [output, session_id])
    clear.click(lambda sid: ("对话已清空", None), [session_id], [output, session_id])

# ====================== 启动 ======================
if __name__ == "__main__":
    print("\n🚀 启动 http://localhost:7860\n")
    demo.launch(server_name="0.0.0.0", server_port=7860)