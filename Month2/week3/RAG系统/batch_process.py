# batch_process.py
"""
批量处理脚本 - 处理大量文档
"""

import os
import sys
sys.path.append(".")

from core.document_loader import DocumentLoader
from core.batch_vectorizer import BatchVectorizer
from core.knowledge_base import KnowledgeBase
from core.model_loader import ModelLoader

def process_documents():
    """处理文档示例"""
    
    print("=" * 60)
    print("开始处理文档...")
    print("=" * 60)
    
    # 1. 加载模型
    model = ModelLoader().load()
    device = model.device
    
    # 2. 创建知识库（使用和 main.py 相同的路径）
    kb = KnowledgeBase(device, vector_db_path="./vector_db")
    
    # 3. 从目录批量构建
    kb.build_from_directory("./docs/")
    
    # 4. 查看统计
    stats = kb.get_stats()
    print(f"\n📊 知识库统计: {stats}")
    
    # 5. 测试检索
    print("\n" + "=" * 60)
    print("测试检索效果")
    print("=" * 60)
    
    test_questions = [
        "什么是整数",
        "什么是张量",
        "什么是PyTorch",
    ]
    
    for q in test_questions:
        result = kb.search_with_source(q)
        if result:
            print(f"\n问题: {q}")
            print(f"  匹配: {result['source']} 第{result['page']}页")
            print(f"  相似度: {result['score']:.3f}")
            print(f"  内容: {result['content'][:80]}...")
        else:
            print(f"\n问题: {q} -> 未找到")
    
    return kb

def clear_vector_db():
    """清空向量库"""
    import shutil
    paths = ["./vector_db", "./my_vector_db"]
    for path in paths:
        if os.path.exists(path):
            shutil.rmtree(path)
            print(f"✅ 已清空: {path}")

if __name__ == "__main__":
    print("=" * 60)
    print("📚 批量文档处理工具")
    print("=" * 60)
    print("1. 处理文档（构建向量库）")
    print("2. 清空向量库")
    
    choice = input("\n请选择 (1/2): ").strip()
    
    if choice == "1":
        process_documents()
    elif choice == "2":
        clear_vector_db()
    else:
        print("无效选项")