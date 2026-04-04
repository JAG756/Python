import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import gradio as gr
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from transformers import AutoTokenizer, AutoModelForCausalLM
import socket
import webbrowser
from functools import lru_cache

# ====================== 初始化 ======================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ 使用设备：{device}")

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"

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
    "RAG（检索增强生成）是一种技术，它让大模型先从知识库检索相关资料，再基于资料生成答案，可以有效防止模型瞎编（幻觉）。企业应用RAG可以解决大模型的幻觉问题、知识过时问题以及数据安全合规问题。RAG的核心流程包括：检索阶段从知识库中找到相关内容，增强阶段将检索结果与问题组合，生成阶段基于资料生成答案。",
    
    "向量数据库（如Chroma、Milvus、Pinecone）用来存储文本的语义向量，通过向量相似度搜索，可以快速找到意思最相似的内容。企业级向量数据库需要支持高并发、高可用和水平扩展。向量数据库的核心技术包括：向量索引（如HNSW、IVF）、相似度计算（余弦相似度、欧氏距离）、分布式架构等。",
    
    "CNN（卷积神经网络）是一种深度学习架构，专门用于处理网格状数据，如图像。它在图像分类、目标检测、图像分割等任务中表现优异。企业应用包括工业质检、医疗影像分析、自动驾驶等。CNN的核心组件包括：卷积层（提取特征）、池化层（降维）、全连接层（分类）。",
    
    "Transformer是当前大语言模型的基础架构，它通过自注意力机制捕获长距离依赖关系。企业级应用包括智能客服、代码助手、文档分析等场景。Transformer的核心创新包括：自注意力机制（Self-Attention）、位置编码（Positional Encoding）、多头注意力（Multi-Head Attention）、前馈神经网络（FFN）。",
    
    "LoRA（低秩适配）是一种高效微调大模型的方法，通过训练少量参数就能让模型适应特定任务。企业可以用LoRA低成本定制专属模型，大幅降低训练成本。LoRA的核心思想是：在预训练模型的基础上，添加低秩矩阵进行微调，只训练新增参数，冻结原始模型权重。",
]

documents = [Document(page_content=text) for text in knowledge_base]

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

embedding = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': device},
    encode_kwargs={'normalize_embeddings': True}
)
vectordb = Chroma.from_documents(chunks, embedding)
print(f"✅ 知识库已加载，共 {len(knowledge_base)} 条知识")

# ====================== 缓存 ======================
@lru_cache(maxsize=128)
def cached_retrieval(question):
    docs = vectordb.similarity_search_with_score(question, k=1)
    if docs and docs[0][1] <= 1.2:
        return docs[0][0].page_content
    return None

# ====================== 带参数的生成函数 ======================
def rag_with_params(question, mode, max_length, temperature):
    """
    支持前端参数控制的 RAG 生成
    """
    if not question or not question.strip():
        return "⚠️ 请输入问题"
    
    # 检索
    cached = cached_retrieval(question)
    if cached:
        context = cached
    else:
        docs = vectordb.similarity_search_with_score(question, k=1)
        if not docs or docs[0][1] > 1.2:
            return "❌ 知识库中暂无相关信息。"
        context = docs[0][0].page_content.strip()
    
    # 严格模式：直接返回资料
    if mode == "strict":
        return f"""📄 **严格模式** (直接返回资料)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{context}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚙️ 参数设置: 模式=严格 | 长度={max_length} | 温度={temperature}"""
    
    # 扩展模式：根据参数生成
    if mode == "expanded":
        # 安全限制：生成长度不超过资料的1.5倍
        safe_length = min(max_length, int(len(context) * 1.5))
        
        prompt = f"""请基于以下资料回答问题。可以适当整理和总结，但不要添加资料外的内容。

【资料】
{context}

【问题】
{question}

【回答】"""
        
        inputs = tokenizer(prompt, return_tensors="pt", max_length=1024).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=safe_length,
                temperature=temperature,
                do_sample=(temperature > 0),
                top_p=0.9 if temperature > 0 else 1.0,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        
        answer = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        if not answer.strip():
            answer = context
        
        # 添加参数信息
        return f"""✨ **扩展模式** (模型生成)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{answer}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚙️ 参数设置: 模式=扩展 | 长度={safe_length} | 温度={temperature}
📊 资料长度: {len(context)} 字符 | 生成长度: {len(answer)} 字符"""

# ====================== Gradio 界面 ======================
with gr.Blocks(title="RAG智能问答系统", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🚀 RAG 智能问答系统 - 可调参数版
    
    **实时调整生成长度和温度，控制模型输出**
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            question = gr.Textbox(
                label="💬 输入问题",
                placeholder="例如：什么是RAG？",
                lines=3
            )
            
            # 模式选择
            mode = gr.Radio(
                choices=["strict", "expanded"],
                label="🔧 回答模式",
                value="strict",
                info="严格模式：直接返回资料 | 扩展模式：模型生成回答"
            )
            
            # 参数控制滑块
            with gr.Row():
                with gr.Column():
                    max_length = gr.Slider(
                        minimum=50,
                        maximum=500,
                        value=200,
                        step=10,
                        label="📏 最大生成长度 (字符数)",
                        info="值越大回答越详细，但可能编造内容"
                    )
                
                with gr.Column():
                    temperature = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.3,
                        step=0.05,
                        label="🌡️ 温度参数",
                        info="值越高越随机，越低越确定"
                    )
            
            # 快速预设按钮
            gr.Markdown("### 🎯 快速预设")
            with gr.Row():
                safe_btn = gr.Button("🛡️ 安全预设", size="sm", variant="secondary")
                creative_btn = gr.Button("🎨 创意预设", size="sm", variant="secondary")
                detailed_btn = gr.Button("📖 详细预设", size="sm", variant="secondary")
            
            with gr.Row():
                submit_btn = gr.Button("提交", variant="primary", size="lg")
                clear_btn = gr.Button("清空", variant="secondary", size="lg")
            
        with gr.Column(scale=2):
            output = gr.Textbox(
                label="📝 回答",
                lines=20,
                interactive=False
            )
    
    # 预设函数
    def set_safe_preset():
        return gr.update(value="strict"), gr.update(value=100), gr.update(value=0.1)
    
    def set_creative_preset():
        return gr.update(value="expanded"), gr.update(value=200), gr.update(value=0.7)
    
    def set_detailed_preset():
        return gr.update(value="expanded"), gr.update(value=400), gr.update(value=0.5)
    
    # 绑定预设按钮
    safe_btn.click(
        set_safe_preset,
        outputs=[mode, max_length, temperature]
    )
    
    creative_btn.click(
        set_creative_preset,
        outputs=[mode, max_length, temperature]
    )
    
    detailed_btn.click(
        set_detailed_preset,
        outputs=[mode, max_length, temperature]
    )
    
    # 绑定提交按钮
    submit_btn.click(
        rag_with_params,
        inputs=[question, mode, max_length, temperature],
        outputs=output
    )
    
    # 回车提交
    question.submit(
        rag_with_params,
        inputs=[question, mode, max_length, temperature],
        outputs=output
    )
    
    clear_btn.click(
        lambda: ("", "", "strict", 200, 0.3),
        inputs=[],
        outputs=[question, output, mode, max_length, temperature]
    )
    
    # 示例
    examples = [
        ["什么是RAG？", "strict", 100, 0.1],
        ["CNN的核心组件有哪些？", "expanded", 200, 0.3],
        ["Transformer的核心创新是什么？", "expanded", 300, 0.5],
    ]
    
    gr.Examples(
        examples=examples,
        inputs=[question, mode, max_length, temperature],
        outputs=output,
        fn=rag_with_params,
        label="点击快速测试"
    )

# ====================== 启动 ======================
if __name__ == "__main__":
    local_ip = get_local_ip()
    
    print("\n" + "="*60)
    print("🚀 RAG 智能问答系统已启动（可调参数版）")
    print("="*60)
    print(f"📱 本地访问: http://localhost:7860")
    
    if local_ip != "127.0.0.1":
        print(f"🌐 局域网访问: http://{local_ip}:7860")
    
    print("\n⚙️ 可调参数:")
    print("   • 回答模式: 严格模式 / 扩展模式")
    print("   • 生成长度: 50-500 字符")
    print("   • 温度参数: 0.0-1.0")
    print("\n💡 快速预设:")
    print("   • 安全预设: 严格模式 + 短长度 + 低温度")
    print("   • 创意预设: 扩展模式 + 中长度 + 高温度")
    print("   • 详细预设: 扩展模式 + 长长度 + 中温度")
    print("="*60 + "\n")
    
    try:
        webbrowser.open("http://localhost:7860")
    except:
        pass
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )