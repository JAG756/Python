# core/batch_vectorizer.py
"""
批量向量化模块 - 处理大规模文档
"""

import os
import json
import pickle
from typing import List, Optional
from datetime import datetime
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

class BatchVectorizer:
    """批量向量化处理器"""
    
    def __init__(self, vector_db_path: str = "./vector_db"):
        self.vector_db_path = vector_db_path
        self.vectordb = None
        self.embedding = None
        
    def init_embedding(self, device: str):
        """初始化 Embedding 模型"""
        self.embedding = HuggingFaceEmbeddings(
            model_name="moka-ai/m3e-base",
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True}
        )
    
    def batch_vectorize(self, chunks: List[Document], batch_size: int = 100):
        """批量向量化"""
        total = len(chunks)
        print(f"🚀 开始批量向量化，共 {total} 个块，批次大小 {batch_size}")
        
        for i in range(0, total, batch_size):
            batch = chunks[i:i+batch_size]
            print(f"  处理批次 {i//batch_size + 1}/{(total-1)//batch_size + 1}")
            
            # 这里可以根据需要批量处理
            # 目前 Chroma 内部已经支持批量
        
        print(f"✅ 批量向量化完成")
    
    def create_vector_store(self, chunks: List[Document], persist: bool = True):
        """创建向量数据库"""
        if self.embedding is None:
            raise ValueError("请先调用 init_embedding()")
        
        # 创建向量库
        self.vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=self.embedding,
            persist_directory=self.vector_db_path if persist else None
        )
        
        if persist:
            print(f"💾 向量库已保存到: {self.vector_db_path}")
        
        return self.vectordb
    
    def load_vector_store(self):
        """加载已存在的向量库"""
        if self.embedding is None:
            raise ValueError("请先调用 init_embedding()")
        
        if os.path.exists(self.vector_db_path):
            self.vectordb = Chroma(
                persist_directory=self.vector_db_path,
                embedding_function=self.embedding
            )
            print(f"📂 加载已有向量库: {self.vector_db_path}")
        else:
            print(f"⚠️ 向量库不存在: {self.vector_db_path}")
        
        return self.vectordb
    
    def add_documents(self, documents: List[Document]):
        """增量添加文档"""
        if self.vectordb is None:
            raise ValueError("请先创建或加载向量库")
        
        self.vectordb.add_documents(documents)
        print(f"➕ 增量添加 {len(documents)} 个文档")
    
    def delete_collection(self):
        """删除向量库"""
        import shutil
        if os.path.exists(self.vector_db_path):
            shutil.rmtree(self.vector_db_path)
            print(f"🗑️ 已删除向量库: {self.vector_db_path}")
    
    def get_stats(self) -> dict:
        """获取向量库统计信息"""
        if self.vectordb is None:
            return {"status": "未初始化"}
        
        try:
            # 获取集合中的文档数量
            collection = self.vectordb._collection
            count = collection.count()
            return {
                "status": "正常",
                "document_count": count,
                "vector_db_path": self.vector_db_path
            }
        except Exception as e:
            return {"status": "错误", "error": str(e)}