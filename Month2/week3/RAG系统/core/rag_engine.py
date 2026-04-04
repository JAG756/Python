# core/rag_engine.py
"""
RAG核心引擎 - 带检索溯源
"""

from config import DEFAULT_MAX_LENGTH, DEFAULT_TEMPERATURE
from utils.chat_session import get_session, clear_session
from core.knowledge_base import KnowledgeBase

class RAGEngine:
    """RAG引擎（带溯源）"""
    
    def __init__(self, model_loader, knowledge_base):
        self.model = model_loader
        self.kb = knowledge_base
        self.tokenizer = model_loader.tokenizer
    
    def detect_intent(self, question):
        """意图识别"""
        from config import GREETINGS, THANKS, GOODBYE, FUZZY_QUESTIONS
        
        if question in GREETINGS:
            return "greeting", "你好！我可以回答RAG、CNN、Transformer、LoRA等技术问题。"
        
        if question in THANKS:
            return "thanks", "不客气！有问题随时问我。"
        
        if question in GOODBYE:
            return "goodbye", "再见！欢迎下次咨询。"
        
        if any(fq in question for fq in FUZZY_QUESTIONS):
            return "fuzzy", "抱歉，我不太明白你的问题。请具体描述你想了解的技术，比如：什么是RAG？"
        
        if question in ["清空", "重置", "clear"]:
            return "clear", None
        
        return "normal", None
    
    def answer(self, question, session_id=None, max_len=50, temp=0.1):
        """核心回答函数 - 带溯源"""
        
        # 输入验证
        if not question or not question.strip():
            return "⚠️ 请输入问题", session_id
        
        # 创建会话ID
        if not session_id:
            import time
            session_id = str(time.time())
        
        # 意图识别
        intent, response = self.detect_intent(question)
        
        if intent == "clear":
            clear_session(session_id)
            return "✅ 对话已清空", session_id
        
        if intent != "normal":
            return response, session_id
        
        # 获取会话
        session = get_session(session_id)
        
        # 处理追问
        full_question = session.get_contextual_question(question)
        
        # 检索（带来源）
        #if session.should_use_last_context(question):
        #    context = session.last_context
        #    source_info = None
        #else:
        #    result = self.kb.search_with_source(full_question)

        # 强制每次都重新检索
        result = self.kb.search_with_source(full_question)
            
        if result is None:
            return f"未找到关于「{question}」的信息。\n\n💡 可以问：什么是RAG？CNN是什么？", session_id
        context = result["content"]
        source_info = result
        
        # 构建回答（带溯源）
        answer_text = context
        
        # 如果有来源信息，添加到回答中
        if source_info:
            source_display = f"""
---
📚 **来源信息**：
• 文档：{source_info['source']}
• 页码：第{source_info['page']}页
• 相关度：{source_info['score']:.2f}
"""
            answer_text = context + source_display
        else:
            # 没有来源信息时，可能是内置知识库
            answer_text = context + """
---
📚 **来源**：内置知识库"""
        
        # 记录历史
        session.add(question, answer_text, context)
        
        return answer_text, session_id