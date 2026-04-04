# ui/interface.py
"""
极简版企业级界面 - 带溯源显示
"""

import gradio as gr

EXAMPLE_QUESTIONS = [
    "什么是RAG？",
    "CNN的核心组件有哪些？",
    "Transformer的核心创新是什么？",
    "LoRA有什么用？",
]

def create_interface(rag_engine):
    """创建极简版 Gradio 界面"""
    
    with gr.Blocks(
        title="RAG 智能问答系统", 
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1000px !important;
            margin: auto !important;
        }
        footer {
            visibility: hidden;
        }
        """
    ) as demo:
        
        gr.Markdown("""
        # 🚀 企业知识库问答
        
        <div style="text-align: center; margin-bottom: 30px;">
        基于企业知识库，智能回答技术问题 | 每个答案都有据可查
        </div>
        """)
        
        session_id = gr.State()
        
        chatbot = gr.Chatbot(
            height=500,
            show_label=False
        )
        
        with gr.Row():
            msg = gr.Textbox(
                placeholder="输入你的问题...\n例如：什么是整数？",
                lines=2,
                scale=9,
                show_label=False
            )
            send_btn = gr.Button("发送", variant="primary", scale=1)
        
        with gr.Row():
            clear_btn = gr.Button("🗑️ 清空对话", variant="secondary", size="sm")
        
        with gr.Accordion("📋 示例问题", open=False):
            gr.Markdown("点击下方问题快速体验：")
            for i in range(0, len(EXAMPLE_QUESTIONS), 2):
                with gr.Row():
                    for q in EXAMPLE_QUESTIONS[i:i+2]:
                        btn = gr.Button(q, size="sm", variant="secondary")
                        btn.click(fn=lambda q=q: q, outputs=msg)
        
        gr.Markdown(
            '<div style="text-align: center; color: #888; font-size: 12px; margin-top: 20px;">'
            '基于检索增强生成 | 知识库问答系统 | 答案可溯源'
            '</div>'
        )
        
        # ========== 事件绑定 ==========
        
        def send_message(message, history, sid):
            if not message or not message.strip():
                return history, "", sid
            
            if message.strip() in ["清空", "重置"]:
                return [], "", None
            
            answer, new_sid = rag_engine.answer(
                message, sid, 
                max_len=50,
                temp=0.1
            )
            
            history = history or []
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": answer})
            
            return history, "", new_sid
        
        def do_clear(sid):
            if sid:
                rag_engine.answer("清空", sid)
            return [], "", None
        
        send_btn.click(
            send_message,
            [msg, chatbot, session_id],
            [chatbot, msg, session_id]
        )
        
        msg.submit(
            send_message,
            [msg, chatbot, session_id],
            [chatbot, msg, session_id]
        )
        
        clear_btn.click(do_clear, [session_id], [chatbot, msg, session_id])
    
    return demo