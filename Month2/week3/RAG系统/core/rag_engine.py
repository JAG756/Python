# core/rag_engine.py
"""
RAG核心引擎 - 带检索溯源（支持生成式/检索式切换）
"""
import time
from config import DEFAULT_MAX_LENGTH, DEFAULT_TEMPERATURE, USE_GENERATION
from config import GREETINGS, THANKS, GOODBYE, FUZZY_QUESTIONS
from utils.chat_session import get_session, clear_session
from core.knowledge_base import KnowledgeBase
from utils.logger import logger

class RAGEngine:
    """RAG引擎（带溯源，可切换生成/直接返回）"""
    
    def __init__(self, model_loader, knowledge_base):
        self.model = model_loader
        self.kb = knowledge_base
        self.tokenizer = model_loader.tokenizer
        self._cache = {}   # {question: (answer_text, session_id)}
    
    def detect_intent(self, question):
        """意图识别"""
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
    
    def answer(self, question, session_id=None, max_len=DEFAULT_MAX_LENGTH, temp=DEFAULT_TEMPERATURE):
        # 输入校验
        if not question or not isinstance(question, str):
            #如果问题不是字符串或者为空，直接返回提示信息，并保持会话ID不变（如果有的话）
            return "请输入有效的问题。", session_id
        if len(question) > 500:
            question = question[:500]
            logger.warning(f"问题过长，已截断至500字符")
            # 注意：这里我们不直接返回错误，而是截断问题并继续处理，避免用户体验过差
        question = question.strip()
        #去掉问题首尾的空白字符，避免因为多余的空格导致检索失败或意图识别错误
        
        try:
            # 创建会话ID
            # 捕捉异常
            if not session_id:
                session_id = str(time.time())
            # 如果用户输入了一个新的问题但没有提供session_id，我们会创建一个新的session_id。这样可以确保每个独立的对话都有一个唯一的标识符，
            # 方便后续的会话管理和上下文维护。

            # 意图识别
            intent, response = self.detect_intent(question)
            
            if intent == "clear":
                clear_session(session_id)
                return "✅ 对话已清空", session_id
            if intent != "normal":
                return response, session_id

            # ========== 新增：缓存检查 ==========
            cache_key = f"{question}_{max_len}_{temp}"   # 可根据需要加入 session_id 或 context
            if cache_key in self._cache:
                logger.info(f"✅ 命中缓存：{question}")
                cached_answer = self._cache[cache_key]
                return cached_answer, session_id
            # ==================================

            session = get_session(session_id)
            full_question = session.get_contextual_question(question)
            # 获取该会话的对话历史对象，然后根据历史将当前问题补全为完整问题

            t0 = time.time()

            # 检索
            result = self.kb.search_with_source(full_question)
            #调用知识库的检索方法，传入补全后的问题，获取检索结果（包含内容和来源信息）

            t1 = time.time()

            if result is None:
                logger.info(f"未找到关于「{question}」的信息")
                return f"未找到关于「{question}」的信息。\n\n💡 可以问：什么是RAG？CNN是什么？", session_id
            
            context = result["content"]
            source_info = result
            # 这里我们保留了完整的检索结果对象（包含content、source、page、score等），以便后续生成答案时可以附加来源信息。
            
            # ---------- 核心改动：根据开关决定是否生成 ----------
            if USE_GENERATION:
                prompt = f"""你是一个专业的技术助手。请严格基于下面提供的资料回答用户的问题。
            如果资料中没有答案，请直接说“资料中未找到相关信息”，不要自己编造。

            【资料】：
            {context}

            【用户问题】：
            {full_question}

            【要求】：
            - 直接给出答案，不要添加“答案是：”、“回答：”等任何前缀。
            - 答案必须是自然语言文本，不要使用代码块（不要出现```python或任何编程语言代码）。
            - 答案应简洁、完整，结尾要有句号。

            【答案】："""
                t_gen_start = time.time()
                generated_answer = self.model.generate(
                    prompt, 
                    max_new_tokens=max_len, 
                    temperature=temp
                )
                t_gen_end = time.time()
                generated_answer = generated_answer.strip()
                
                # 空答案处理
                if not generated_answer:
                    generated_answer = f"（模型未能生成答案，以下为检索到的原始资料）\n{context}"
                    logger.warning(f"模型生成空答案，使用原文片段。问题：{question}")
                else:
                    # 截断提示
                    if generated_answer[-1] not in "。！？":
                        generated_answer += "…（回答已截断）"
                
                answer_text = generated_answer
                logger.info(f"生成耗时: {t_gen_end - t_gen_start:.2f}s")
            else:
                answer_text = context

            # 记录检索耗时
            logger.info(f"检索耗时: {t1 - t0:.2f}s")

            # 添加来源信息（无论哪种模式都保留）
            if source_info:
                source_display = f"""
---
📚 **来源信息**：
• 文档：{source_info['source']}
• 页码：第{source_info['page']}页
• 相关度：{source_info['score']:.2f}
"""
                answer_text = answer_text + source_display
            else:
                answer_text = answer_text + "\n---\n📚 **来源**：内置知识库"
            
            # ========== 新增：存入缓存 ==========
            self._cache[cache_key] = answer_text
            # ==================================

            # 记录会话（只记录原始上下文，不记录生成的答案，避免冗余）
            session.add(question, answer_text, context)
            logger.info(f"问题: {question} -> 检索到来源: {source_info.get('source')} 第{source_info.get('page')}页 相似度:{source_info.get('score'):.2f}")
            return answer_text, session_id
            
        except Exception as e:
            logger.error(f"回答生成失败: {e}", exc_info=True)
            return "系统内部错误，请稍后重试。", session_id