# utils/chat_session.py
"""
对话管理模块
"""

from config import FOLLOWUP_WORDS, OMIT_STARTS

class ChatSession:
    """对话会话管理"""
    
    def __init__(self):
        self.history = []
        self.last_context = None
        self.last_topic = None
    
    def add(self, question, answer, context):
        """添加一轮对话"""
        self.history.append({"q": question, "a": answer})
        self.last_context = context
        
        # 提取话题
        for kw in ["RAG", "CNN", "Transformer", "LoRA", "向量数据库"]:
            if kw in question or (context and kw in context):
                self.last_topic = kw
                break
    
    def get_contextual_question(self, question):
        """将追问转换为完整问题"""
        # 处理指代词
        if self.last_topic and any(w in question for w in FOLLOWUP_WORDS):
            return f"{self.last_topic} {question}"
        
        # 处理省略主语的情况
        if self.last_topic and any(question.startswith(w) for w in OMIT_STARTS):
            return f"{self.last_topic} {question}"
        
        return question
    
    def should_use_last_context(self, question):
        """判断是否应该复用上一轮的检索结果"""
        return self.last_context and any(w in question for w in FOLLOWUP_WORDS)
    
    def clear(self):
        """清空会话"""
        self.history = []
        self.last_context = None
        self.last_topic = None
    
    def get_history_count(self):
        """获取历史轮数"""
        return len(self.history)

# 会话存储
sessions = {}

def get_session(session_id):
    """获取或创建会话"""
    if session_id not in sessions:
        sessions[session_id] = ChatSession()
    return sessions[session_id]

def clear_session(session_id):
    """清除会话"""
    if session_id in sessions:
        del sessions[session_id]