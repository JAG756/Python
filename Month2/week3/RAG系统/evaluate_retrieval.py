# evaluate_retrieval.py
import json
import sys
import os

# 获取当前脚本所在目录（绝对路径）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# ===================== 强制离线模式（与 test_retrieval 保持一致） =====================
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# ====================================================================================

from config import USE_RERANK
from core.knowledge_base import KnowledgeBase
from core.reranker import BGEReranker
from test_retrieval import test_retrieval
import torch


def ensure_test_queries_file():
    """如果测试文件不存在，则在项目根目录创建示例文件"""
    json_path = os.path.join(BASE_DIR, "eval_queries.json")
    if not os.path.exists(json_path):
        print(f"⚠️ 未找到 {json_path}，正在创建示例文件...")
        sample = [
            {"query": "银行家算法", "relevant_docs": ["死锁.txt"]},
            {"query": "死锁的四个必要条件", "relevant_docs": ["死锁.txt"]},
            {"query": "什么是进程", "relevant_docs": ["进程与线程.txt"]},
            {"query": "HTTP 状态码 404", "relevant_docs": ["HTTP 状态码.txt"]}
        ]
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(sample, f, indent=2, ensure_ascii=False)
        print(f"✅ 已创建示例文件 {json_path}")
        print("⚠️ 请根据您实际的文档名称修改 relevant_docs 字段。")
    else:
        print(f"✅ 找到测试文件 {json_path}")
    return json_path


def load_test_queries(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def evaluate_recall_mrr_hitrate(retriever_func, queries, k_values=[1, 3, 5]):
    """
    评估检索性能
    retriever_func: 接受 (query, top_k) 返回 [(doc, score), ...]
    """
    # 初始化结果字典
    results = {k: {"recall": 0.0, "hit": 0, "mrr_sum": 0.0} for k in k_values}
    total_queries = len(queries)

    for q in queries:
        query = q["query"]
        relevant_set = set(q["relevant_docs"])
        if not relevant_set:
            continue  # 跳过没有标注的查询

        retrieved = retriever_func(query, top_k=max(k_values))

        # 找出第一个相关文档的排名 (1-indexed)
        first_rank = None
        # 记录每个K下是否命中（用于Hit Rate）
        hit_for_k = {k: False for k in k_values}
        # 记录每个K下命中相关文档的数量（用于Recall）
        hits_count_for_k = {k: 0 for k in k_values}

        for rank, (doc, _) in enumerate(retrieved, start=1):
            doc_name = doc.metadata.get("source", "")
            if doc_name in relevant_set:
                if first_rank is None:
                    first_rank = rank
                # 对于所有k >= rank，标记为命中
                for k in k_values:
                    if rank <= k:
                        hit_for_k[k] = True
                        hits_count_for_k[k] += 1

        # 累加 Recall
        for k in k_values:
            results[k]["recall"] += hits_count_for_k[k] / len(relevant_set)
            if hit_for_k[k]:
                results[k]["hit"] += 1

        # 累加 MRR
        if first_rank is not None:
            rr = 1.0 / first_rank
            for k in k_values:
                results[k]["mrr_sum"] += rr

    # 计算平均值
    for k in k_values:
        results[k]["recall"] /= total_queries
        results[k]["hit_rate"] = results[k]["hit"] / total_queries
        results[k]["mrr"] = results[k]["mrr_sum"] / total_queries
        # 删除临时字段
        del results[k]["hit"]
        del results[k]["mrr_sum"]

    return results


def print_results(results, k_values):
    for k in k_values:
        print(f"K={k}:")
        print(f"  Recall@{k}: {results[k]['recall']:.4f}")
        print(f"  Hit Rate@{k}: {results[k]['hit_rate']:.4f}")
        print(f"  MRR@{k}: {results[k]['mrr']:.4f}")


def main():
    # 确保测试文件存在
    json_path = ensure_test_queries_file()
    queries = load_test_queries(json_path)

    # 加载知识库
    device = "cuda" if torch.cuda.is_available() else "cpu"
    kb = KnowledgeBase(device, vector_db_path="./vector_db")
    kb.load_persisted()

    # 初始化 Reranker（如果配置启用）
    reranker = None
    if USE_RERANK:
        print("加载 Rerank 模型...")
        reranker = BGEReranker()

    # 定义三种检索模式（复用 test_retrieval 函数）
    def vector_retriever(query, top_k):
        return test_retrieval(query, mode="vector", kb=kb, top_k=top_k)

    def hybrid_retriever(query, top_k):
        return test_retrieval(query, mode="hybrid", kb=kb, top_k=top_k)

    def hybrid_rerank_retriever(query, top_k):
        return test_retrieval(query, mode="hybrid_rerank", kb=kb, reranker=reranker, top_k=top_k)

    k_values = [1, 3, 5]

    # 评估纯向量检索
    print("\n" + "="*60)
    print("评估模式：纯向量检索")
    print("="*60)
    vec_results = evaluate_recall_mrr_hitrate(vector_retriever, queries, k_values)
    print_results(vec_results, k_values)

    # 评估混合检索
    print("\n" + "="*60)
    print("评估模式：混合检索 (BM25 + 向量)")
    print("="*60)
    hyb_results = evaluate_recall_mrr_hitrate(hybrid_retriever, queries, k_values)
    print_results(hyb_results, k_values)

    # 评估混合检索 + Rerank
    if reranker:
        print("\n" + "="*60)
        print("评估模式：混合检索 + Rerank")
        print("="*60)
        rerank_results = evaluate_recall_mrr_hitrate(hybrid_rerank_retriever, queries, k_values)
        print_results(rerank_results, k_values)
    else:
        print("\n⚠️ Rerank 未启用，跳过该模式评估。")


if __name__ == "__main__":
    main()