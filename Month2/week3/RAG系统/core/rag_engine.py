# core/rag_engine.py
"""
RAG核心引擎 - 带检索溯源
"""

from config import DEFAULT_MAX_LENGTH, DEFAULT_TEMPERATURE
from utils.chat_session import get_session, clear_session
from core.knowledge_base import KnowledgeBase
from utils.logger import logger

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
        # 输入校验
        if not question or not isinstance(question, str):
            return "请输入有效的问题。", session_id
        if len(question) > 500:
            question = question[:500]
            logger.warning(f"问题过长，已截断至500字符")
        question = question.strip()
        
        try:
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
            
            session = get_session(session_id)
            full_question = session.get_contextual_question(question)
            
            result = self.kb.search_with_source(full_question)
            if result is None:
                logger.info(f"未找到关于「{question}」的信息")
                return f"未找到关于「{question}」的信息。\n\n💡 可以问：什么是RAG？CNN是什么？", session_id
            
            context = result["content"]
            source_info = result
            
            answer_text = context
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
                answer_text = context + "\n---\n📚 **来源**：内置知识库"
            
            session.add(question, answer_text, context)
            logger.info(f"问题: {question} -> 检索到来源: {source_info.get('source')} 第{source_info.get('page')}页 相似度:{source_info.get('score'):.2f}")
            return answer_text, session_id
            
        except Exception as e:
            logger.error(f"回答生成失败: {e}", exc_info=True)
            return "系统内部错误，请稍后重试。", session_id