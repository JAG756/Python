# main.py
import os
import threading
import time
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from core.model_loader import ModelLoader
from core.knowledge_base import KnowledgeBase
from core.rag_engine import RAGEngine
from ui.interface import create_interface
from utils.logger import logger



def auto_update_loop(kb, docs_dir, interval=60):
    """
    后台线程函数：每隔 interval 秒调用一次 kb.incremental_update
    """
    while True:
        time.sleep(interval)
        try:
            logger.info("🔄 定时触发知识库增量更新...")
            kb.incremental_update(docs_dir)
        except Exception as e:
            logger.error(f"自动更新失败: {e}")


def main():
    logger.info("加载模型中...")
    model = ModelLoader().load()

    # 创建知识库对象
    kb = KnowledgeBase(model.device, vector_db_path="./vector_db")

    # ========== 新增：获取 docs 目录的绝对路径 ==========
    docs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")
    logger.info(f"docs 目录绝对路径: {docs_dir}")
    # =================================================

    # 尝试加载已有向量库，若失败则重建
    try:
        kb.load_persisted()
        logger.info("✅ 从已有向量库加载成功")
    except Exception as e:
        logger.warning(f"⚠️ 加载已有向量库失败: {e}，开始构建...")
        if os.path.exists(docs_dir):
            kb.build_from_directory(docs_dir)
        else:
            from config import KNOWLEDGE_BASE
            kb.build_from_strings(KNOWLEDGE_BASE)

    # 启动后台自动更新线程（每隔 60 秒检查 docs 目录变化）
    updater_thread = threading.Thread(
        target=auto_update_loop,
        args=(kb, docs_dir, 60),
        daemon=True
    )
    updater_thread.start()
    logger.info("已启动知识库自动更新线程（间隔60秒）")

    # 创建 RAG 引擎
    logger.info("创建 RAG 引擎...")
    rag = RAGEngine(model, kb)

    # 启动 Gradio 界面
    logger.info("启动界面...")
    demo = create_interface(rag)
    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()