import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ===================== 【强制离线模式】全局生效 =====================
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# ==================================================================

from config import SIMILARITY_THRESHOLD, TOP_K, USE_HYBRID_SEARCH, HYBRID_ALPHA, USE_RERANK
from core.model_loader import ModelLoader
from core.knowledge_base import KnowledgeBase
from core.reranker import BGEReranker
from utils.logger import logger

def test_retrieval(question, mode="vector", kb=None, reranker=None, top_k=3):
    full_question = question

    if mode == "vector":
        docs = kb.vectordb.similarity_search_with_score(full_question, k=top_k)
        results = []
        for doc, dist in docs:
            sim = 1 / (1 + dist)
            results.append((doc, sim))
        return results

    elif mode == "hybrid":
        if kb.bm25_index is None:
            return []

        vector_results = kb.vectordb.similarity_search_with_score(full_question, k=top_k * 2)
        max_dist = max([s for _, s in vector_results]) if vector_results else 1.0
        vector_scores = {}
        for doc, score in vector_results:
            sim = 1 - (score / max_dist)
            vector_scores[doc.page_content] = sim

        import jieba
        tokenized_query = list(jieba.cut(full_question))
        bm25_scores = kb.bm25_index.get_scores(tokenized_query)
        max_bm25 = max(bm25_scores) if len(bm25_scores) > 0 else 1.0
        bm25_norm = [s / max_bm25 for s in bm25_scores]
        bm25_dict = {chunk.page_content: bm25_norm[idx] for idx, chunk in enumerate(kb.bm25_chunks)}

        all_contents = set(vector_scores.keys()) | set(bm25_dict.keys())
        fused = []
        for content in all_contents:
            v_score = vector_scores.get(content, 0.0)
            b_score = bm25_dict.get(content, 0.0)
            final = HYBRID_ALPHA * v_score + (1 - HYBRID_ALPHA) * b_score
            fused.append((content, final))

        fused.sort(key=lambda x: x[1], reverse=True)
        top_contents = fused[:top_k]
        result_docs = []
        for content, score in top_contents:
            for chunk in kb.bm25_chunks:
                if chunk.page_content == content:
                    result_docs.append((chunk, score))
                    break
        return result_docs

    elif mode == "hybrid_rerank":
        hybrid_results = test_retrieval(question, "hybrid", kb=kb, top_k=top_k * 2)
        if not hybrid_results:
            return []
        candidates = [doc.page_content for doc, _ in hybrid_results]
        reranked = reranker.rerank(full_question, candidates, top_k=top_k)
        reranked_docs = []
        for idx, score in reranked:
            original_doc, _ = hybrid_results[idx]
            reranked_docs.append((original_doc, score))
        return reranked_docs
    else:
        return []

def print_results(question, results, mode_name):
    print(f"\n{'='*60}")
    print(f"模式: {mode_name}")
    print(f"问题: {question}")
    print(f"{'='*60}")
    if not results:
        print("无检索结果")
        return
    for i, (doc, score) in enumerate(results):
        source = doc.metadata.get("source", "未知")
        page = doc.metadata.get("page", "?")
        preview = doc.page_content[:80].replace('\n', ' ')
        print(f"{i+1}. [{source} 第{page}页] 得分: {score:.4f}")
        print(f"   内容: {preview}...")

def main():
    print("加载模型中...")

    # ===================== 【正常加载模型】离线自动生效 =====================
    try:
        model_loader = ModelLoader()
        model = model_loader.load()
    except Exception as e:
        print(f"❌ 模型加载失败：{str(e)}")
        return

    kb = KnowledgeBase(model.device, vector_db_path="./vector_db")
    kb.load_persisted()

    reranker = None
    if USE_RERANK:
        print("加载 Rerank 模型...")
        reranker = BGEReranker()

    questions = [
        "银行家算法",
        "死锁的四个必要条件",
        "什么是进程？",
        "HTTP 状态码 404"
    ]

    for q in questions:
        print(f"\n\n>>> 测试问题: {q}")
        vec_results = test_retrieval(q, "vector", kb=kb, top_k=3)
        print_results(q, vec_results, "纯向量检索")

        hybrid_results = test_retrieval(q, "hybrid", kb=kb, top_k=3)
        print_results(q, hybrid_results, "混合检索")

        if reranker:
            rerank_results = test_retrieval(q, "hybrid_rerank", kb=kb, reranker=reranker, top_k=3)
            print_results(q, rerank_results, "混合检索+Rerank")

        input("按 Enter 继续下一个问题...")

if __name__ == "__main__":
    main()