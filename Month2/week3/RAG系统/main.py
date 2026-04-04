# main.py
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from core.model_loader import ModelLoader
from core.knowledge_base import KnowledgeBase
from core.rag_engine import RAGEngine
from ui.interface import create_interface

def main():
    print("加载模型中...")
    model = ModelLoader().load()
    
    # 方式1：从已有向量库加载
    kb = KnowledgeBase(model.device, vector_db_path="./vector_db")
    
    # 尝试加载已有向量库
    try:
        kb.load_persisted()
        print("✅ 从已有向量库加载成功")
    except:
        print("⚠️ 未找到已有向量库，开始构建...")
        # 方式2：从目录构建（如果有 docs 文件夹）
        import os
        if os.path.exists("./docs"):
            kb.build_from_directory("./docs")
        else:
            # 方式3：从默认字符串构建
            from config import KNOWLEDGE_BASE
            kb.build_from_strings(KNOWLEDGE_BASE)
    
    print("创建 RAG 引擎...")
    rag = RAGEngine(model, kb)
    
    print("启动界面...")
    demo = create_interface(rag)
    demo.launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    main()