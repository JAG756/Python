# ui/interface.py
"""
极简版企业级界面 - 带溯源显示
"""
import os
import gradio as gr
from config import DEFAULT_MAX_LENGTH, DEFAULT_TEMPERATURE
from utils.logger import logger   # 确保可以导入

EXAMPLE_QUESTIONS = [
    "什么是进程？",
    "什么是死锁？",
    "死锁的四个必要条件是什么？",
    "TCP三次握手的过程是什么？",
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
                placeholder="输入你的问题...\n例如：什么是死锁？",
                lines=2,
                scale=9,
                show_label=False
            )
            send_btn = gr.Button("发送", variant="primary", scale=1)
        
        with gr.Row():
            clear_btn = gr.Button("🗑️ 清空对话", variant="secondary")
            update_btn = gr.Button("🔄 更新知识库", variant="secondary")
            update_status = gr.Textbox(label="更新状态", interactive=False, visible=True, scale=2)
        
        with gr.Row():
            export_btn = gr.Button("📄 导出对话", variant="secondary")
            export_output = gr.Textbox(label="对话记录", interactive=False, visible=True, lines=10)

        with gr.Row():
            file_upload = gr.File(
                label="上传知识库文档 (PDF/TXT/MD)",
                file_count="multiple",
                file_types=[".pdf", ".txt", ".md"]
            )
            upload_btn = gr.Button("📤 上传并更新", variant="primary")
            upload_status = gr.Textbox(label="上传状态", interactive=False, visible=True, scale=2)

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
        
        def update_kb():
            try:
                # 获取当前文件（interface.py）所在目录，然后向上到项目根目录
                current_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.dirname(current_dir)
                docs_dir = os.path.join(project_root, "docs")
                rag_engine.kb.incremental_update(docs_dir)
                rag_engine._cache.clear()
                return "✅ 知识库已更新"
            except Exception as e:
                return f"❌ 失败: {e}"

        update_btn.click(update_kb, outputs=update_status)


                # 文件上传处理函数
        def handle_upload(files):
            if not files:
                return "未选择文件"
            try:
                import shutil
                # 获取 docs 目录绝对路径
                current_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.dirname(current_dir)
                docs_dir = os.path.join(project_root, "docs")
                os.makedirs(docs_dir, exist_ok=True)

                saved_files = []
                for file in files:
                    # 获取原始文件名
                    if hasattr(file, 'name'):
                        src_path = file.name
                        original_name = os.path.basename(src_path)
                    else:
                        src_path = file
                        original_name = os.path.basename(file)
                    dest_path = os.path.join(docs_dir, original_name)
                    # 如果文件已存在，可先删除（或覆盖）
                    if os.path.exists(dest_path):
                        os.remove(dest_path)
                    shutil.copy(src_path, dest_path)
                    saved_files.append(dest_path)

                # 调用增量更新（会自动处理新增、修改、删除）
                rag_engine.kb.incremental_update(docs_dir)
                rag_engine._cache.clear()
                return f"✅ 成功上传 {len(saved_files)} 个文件，知识库已更新"
            except Exception as e:
                return f"❌ 失败: {e}"
                
        # 导出对话函数
        def export_history(history):
            if not history:
                return "暂无对话记录"
            lines = []
            for turn in history:
                role = "用户" if turn["role"] == "user" else "助手"
                lines.append(f"**{role}**: {turn['content']}")
            return "\n\n---\n\n".join(lines)


        # ========== 事件绑定 ==========
        
        def send_message(message, history, sid):
            if not message or not message.strip():
                return history, "", sid
            
            if message.strip() in ["清空", "重置"]:
                if sid:
                    rag_engine.answer("清空", sid)   # 通知后端重置
                return [], "", None
            
            answer, new_sid = rag_engine.answer(
                message, sid, 
                max_len=DEFAULT_MAX_LENGTH,
                temp=DEFAULT_TEMPERATURE
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

        upload_btn.click(handle_upload, inputs=file_upload, outputs=upload_status)
        
        export_btn.click(export_history, inputs=chatbot, outputs=export_output)
    
    return demo