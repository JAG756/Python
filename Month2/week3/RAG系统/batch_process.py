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
from utils.logger import logger

def process_documents():
    """处理文档示例"""
    
    logger.info("=" * 60)
    logger.info("开始处理文档...")
    logger.info("=" * 60)
    
    # 1. 加载模型
    model = ModelLoader().load()
    device = model.device
    
    # 2. 创建知识库（使用和 main.py 相同的路径）
    kb = KnowledgeBase(device, vector_db_path="./vector_db")
    
    # 3. 从目录批量构建
    kb.build_from_directory("./docs/")
    
    # 4. 查看统计
    stats = kb.get_stats()
    logger.info(f"\n📊 知识库统计: {stats}")
    
    # 5. 测试检索
    logger.info("\n" + "=" * 60)
    logger.info("测试检索效果")
    logger.info("=" * 60)
    
    test_questions = [
        "什么是整数",
        "什么是张量",
        "什么是PyTorch",
    ]
    
    for q in test_questions:
        result = kb.search_with_source(q)
        if result:
            logger.info(f"\n问题: {q}")
            logger.info(f"  匹配: {result['source']} 第{result['page']}页")
            logger.info(f"  相似度: {result['score']:.3f}")
            logger.info(f"  内容: {result['content'][:80]}...")
        else:
            logger.info(f"\n问题: {q} -> 未找到")
    
    return kb

def clear_vector_db():
    """清空向量库及相关状态文件"""
    import shutil
    
    # 要删除的路径列表
    paths_to_delete = [
        "./vector_db",          # 向量库目录
        "./file_state.json",    # 文件状态记录
        "./bm25_index.pkl"      # BM25 索引文件
    ]
    
    for path in paths_to_delete:
        if os.path.exists(path):
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                    logger.info(f"✅ 已删除目录: {path}")
                else:
                    os.remove(path)
                    logger.info(f"✅ 已删除文件: {path}")
            except Exception as e:
                logger.error(f"删除失败 {path}: {e}")
        else:
            logger.info(f"⚠️ 路径不存在，跳过: {path}")
    
    logger.info("✅ 知识库及相关文件已清空，下次启动将自动重建")

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("📚 批量文档处理工具")
    logger.info("=" * 60)
    logger.info("1. 处理文档（构建向量库）")
    logger.info("2. 清空向量库")
    
    choice = input("\n请选择 (1/2): ").strip()
    
    if choice == "1":
        process_documents()
    elif choice == "2":
        clear_vector_db()
    else:
        logger.info("无效选项")