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
        """添加一轮对话，并提取话题（支持通用问题）"""
        self.history.append({"q": question, "a": answer})
        self.last_context = context

        # 1. 优先从用户问题中提取核心名词（去掉疑问词前缀）
        topic = None
        # 常见疑问词前缀
        prefixes = ["什么是", "解释一下", "请说明", "介绍一下", "何为", "说说"]
        cleaned = question
        for prefix in prefixes:
            if question.startswith(prefix):
                cleaned = question[len(prefix):]
                break
        # 取第一个逗号、句号或空格前的内容作为话题
        for sep in ["，", "。", "？", " ", "的"]:
            if sep in cleaned:
                topic = cleaned.split(sep)[0].strip()
                break
        if not topic:
            topic = cleaned.strip()
        
        # 只保留长度适中的话题（避免整段文字）
        if topic and 2 <= len(topic) <= 30:
            self.last_topic = topic
            return
        
        # 2. 后备：预设关键词匹配
        for kw in ["RAG", "CNN", "Transformer", "LoRA", "向量数据库", "死锁", "进程", "线程"]:
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