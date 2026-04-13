# core/rag_engine.py
"""
RAG核心引擎 - 带检索溯源（支持生成式/检索式切换）
"""
import os
os.environ["FLASH_ATTENTION_DISABLE"] = "1"
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import jieba
from config import DEFAULT_MAX_LENGTH, DEFAULT_TEMPERATURE, USE_GENERATION
from config import GREETINGS, THANKS, GOODBYE, FUZZY_QUESTIONS
from config import USE_HYBRID_SEARCH, HYBRID_ALPHA, USE_RERANK, RERANK_MODEL
from config import HYBRID_SIMILARITY_THRESHOLD
from utils.chat_session import get_session, clear_session
from core.knowledge_base import KnowledgeBase
from core.reranker import BGEReranker
from utils.logger import logger


class RAGEngine:
    """RAG引擎（带溯源，可切换生成/直接返回）"""
    
    def __init__(self, model_loader, knowledge_base):
        self.model = model_loader
        self.kb = knowledge_base
        self.tokenizer = model_loader.tokenizer
        self._cache = {}   # {question: (answer_text, session_id)}
        if USE_RERANK:
            self.reranker = BGEReranker(model_name=RERANK_MODEL)
        else:
            self.reranker = None
    
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

    def hybrid_search(self, query: str, top_k: int = 3, alpha: float = 0.5):
        """
        混合检索：向量检索 + BM25，加权融合得分
        alpha: 向量相似度权重，BM25 权重为 1-alpha
        """
        # 1. 向量检索（取更多候选，比如 top_k * 2）
        vector_results = self.kb.vectordb.similarity_search_with_score(query, k=top_k * 2)
        # 将得分（距离）转换为相似度（距离越小相似度越高）
        max_dist = max([score for _, score in vector_results]) if vector_results else 1.0
        vector_scores = {}
        for doc, score in vector_results:
            sim = 1 - (score / max_dist)   # 归一化到 [0,1]
            vector_scores[doc.page_content] = sim   # 用内容作为临时 key（实际应该用 id，但这里简化）
        
        # 2. BM25 检索
        if self.kb.bm25_index is not None:
            tokenized_query = list(jieba.cut(query))
            bm25_scores = self.kb.bm25_index.get_scores(tokenized_query)
            # 归一化 BM25 分数
            if bm25_scores is not None and len(bm25_scores) > 0:
                max_bm25 = max(bm25_scores)
                bm25_norm = [s / max_bm25 for s in bm25_scores]
            else:
                bm25_norm = []

            
            # 建立 BM25 得分映射（注意顺序与 self.kb.bm25_chunks 一致）
            bm25_dict = {}
            for idx, chunk in enumerate(self.kb.bm25_chunks):
                bm25_dict[chunk.page_content] = bm25_norm[idx]
        else:
            bm25_dict = {}
        
        # 3. 融合得分（取并集）
        all_contents = set(vector_scores.keys()) | set(bm25_dict.keys())
        fused = []
        for content in all_contents:
            v_score = vector_scores.get(content, 0.0)
            b_score = bm25_dict.get(content, 0.0)
            final = alpha * v_score + (1 - alpha) * b_score
            fused.append((content, final))
        
        # 4. 排序取 top_k
        fused.sort(key=lambda x: x[1], reverse=True)
        top_contents = fused[:top_k]
        
        # 5. 返回对应的 Document 对象和得分（需要从原 chunks 中找回）
        result_docs = []
        for content, score in top_contents:
            # 从 bm25_chunks 或 vector_results 中找回原始 doc
            for chunk in self.kb.bm25_chunks:
                if chunk.page_content == content:
                    result_docs.append((chunk, score))
                    break
        return result_docs
    
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

            # 检索：先召回 Top-K * 2 个候选（例如 TOP_K=3，则召回 6 个）
            candidate_k = self.kb.top_k * 2
            if USE_HYBRID_SEARCH and self.kb.bm25_index is not None:
                docs = self.hybrid_search(full_question, top_k=candidate_k, alpha=HYBRID_ALPHA)
                # 注意：hybrid_search 返回格式为 [(doc, score), ...]
            else:
                docs = self.kb.vectordb.similarity_search_with_score(full_question, k=candidate_k)


            # ========== 新增：Rerank 重排序 ==========
            if USE_RERANK and hasattr(self, 'reranker') and self.reranker and len(docs) > 0:
                logger.info("🔄 执行 Rerank 重排序...")
                query = full_question
                candidates = [doc.page_content for doc, _ in docs]
                reranked = self.reranker.rerank(query, candidates, top_k=self.kb.top_k)
                # 根据重排序结果构建新的 docs 列表
                reranked_docs = []
                for idx, score in reranked:
                    original_doc, original_score = docs[idx]   # original_score 是原始检索得分
                    # 使用 Rerank 模型给出的新分数（替换原得分）
                    reranked_docs.append((original_doc, score))
                docs = reranked_docs
            else:
                # 如果没有 Rerank，直接截取前 top_k 个
                docs = docs[:self.kb.top_k]
            # =======================================    

            # 收集有效片段（根据检索类型使用不同的阈值判断）
            if USE_HYBRID_SEARCH and self.kb.bm25_index is not None:
                # 混合检索：得分越高越相似，阈值设为 0.3（可调整）
                valid_docs = [(doc, score) for doc, score in docs if score >= HYBRID_SIMILARITY_THRESHOLD]
            else:
                # 纯向量检索：得分（距离）越小越相似，阈值使用 config 中的 SIMILARITY_THRESHOLD
                valid_docs = [(doc, score) for doc, score in docs if score <= self.kb.threshold]

            if not valid_docs and docs:
                valid_docs = [docs[0]]  # 保底取最相关的一个

            # 合并多个片段作为上下文（用 --- 分隔）
            context = "\n\n---\n\n".join([doc.page_content for doc, _ in valid_docs])
            # 取第一个片段用于来源显示（转换为字典）
            first_doc, first_score = valid_docs[0]
            source_info = {
                "source": first_doc.metadata.get("source", "未知来源"),
                "page": first_doc.metadata.get("page", "未知"),
                "score": first_score,
                "content": first_doc.page_content,   # 可选，保留
}

            t1 = time.time()
            
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