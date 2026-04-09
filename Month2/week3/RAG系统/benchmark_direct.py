import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"   # 可选，与 main.py 一致

import time
import concurrent.futures
import statistics
from core.model_loader import ModelLoader
from core.knowledge_base import KnowledgeBase
from core.rag_engine import RAGEngine

def main():
    print("正在加载模型和知识库...")
    model = ModelLoader().load()
    kb = KnowledgeBase(model.device, vector_db_path="./vector_db")
    kb.load_persisted()
    rag = RAGEngine(model, kb)

    # 测试问题集（可重复多次）
    questions = [
        "什么是RAG？",
        "CNN的核心组件有哪些？",
        "Transformer的核心创新是什么？",
        "LoRA有什么用？",
        "死锁定义",
        "什么是线程",
        "DNS"
    ] * 3   # 每个问题重复3次，共21个请求

    def bench(question):
        start = time.time()
        answer, _ = rag.answer(question, max_len=100, temp=0.3)   # 与配置一致
        elapsed = time.time() - start
        return elapsed, len(answer)

    print(f"开始压测，共 {len(questions)} 个请求，并发线程数: 5")
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(bench, questions))

    times = [t for t, _ in results]
    print("\n========== 压测结果 ==========")
    print(f"总请求数: {len(times)}")
    print(f"平均耗时: {statistics.mean(times):.2f} 秒")
    print(f"中位数: {statistics.median(times):.2f} 秒")
    print(f"P95耗时: {sorted(times)[int(len(times)*0.95)]:.2f} 秒")
    print(f"最大耗时: {max(times):.2f} 秒")
    print(f"最小耗时: {min(times):.2f} 秒")

if __name__ == "__main__":
    main()