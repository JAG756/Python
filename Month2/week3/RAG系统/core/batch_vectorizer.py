# core/batch_vectorizer.py
"""
批量向量化模块 - 处理大规模文档
"""

import os
from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from utils.logger import logger

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
    
    def create_vector_store(self, chunks: List[Document], persist: bool = True):
        """创建向量数据库"""
        if self.embedding is None:
            raise ValueError("请先调用 init_embedding()")
        
        # 创建向量库
        self.vectordb = Chroma.from_documents(
            documents=chunks,                     #文本块列表
            embedding=self.embedding,             #向量化模型
            persist_directory=self.vector_db_path if persist else None
            # 如果 persist=True 则保存到磁盘，否则仅在内存中使用
        )
        
        if persist:
            logger.info(f"💾 向量库已保存到: {self.vector_db_path}")
        
        return self.vectordb
    
    def create_vector_store_with_ids(self, chunks: List[Document], ids: List[str], persist: bool = True):
        if self.embedding is None:
            raise ValueError("请先调用 init_embedding()")
        
        # 1. 创建空向量库
        self.vectordb = Chroma(
            embedding_function=self.embedding,
            persist_directory=self.vector_db_path if persist else None
        )
        
        # 2. 批量生成 embedding（关键优化点）
        texts = [chunk.page_content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        # HuggingFaceEmbeddings 内部已支持 batch，但为了确保批量，直接调用 embed_documents
        embeddings = self.embedding.embed_documents(texts)   # 一次性生成所有向量
        
        # 3. 批量插入底层 collection（绕过 add_documents 的逐条循环）
        self.vectordb._collection.add(
            embeddings=embeddings,
            metadatas=metadatas,
            documents=texts,
            ids=ids
        )
        
        if persist:
            self.vectordb.persist()
            logger.info(f"💾 向量库已保存到: {self.vector_db_path}，共 {len(chunks)} 个块（批量插入）")
        
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
            logger.info(f"📂 加载已有向量库: {self.vector_db_path}")
        else:
            logger.warning(f"⚠️ 向量库不存在: {self.vector_db_path}")
        
        return self.vectordb
    
    def add_documents(self, documents: List[Document], ids: List[str] = None):
        if self.vectordb is None:
            raise ValueError("请先创建或加载向量库")
        if ids:
            self.vectordb.add_documents(documents, ids=ids)
        else:
            self.vectordb.add_documents(documents)
        self.vectordb.persist()  # 新增：强制持久化
        logger.info(f"➕ 增量添加 {len(documents)} 个文档")

    def add_documents_batch(self, documents: List[Document], ids: List[str], batch_size=32):
        """批量添加文档（用于增量更新，比 add_documents 快）"""
        if self.vectordb is None:
            raise ValueError("请先创建或加载向量库")
        
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        # 分批生成 embedding（避免 OOM）
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_emb = self.embedding.embed_documents(batch_texts)
            all_embeddings.extend(batch_emb)
        
        self.vectordb._collection.add(
            embeddings=all_embeddings,
            metadatas=metadatas,
            documents=texts,
            ids=ids
        )
        self.vectordb.persist()
        logger.info(f"➕ 批量添加 {len(documents)} 个文档")        
    
    def delete_collection(self):
        """删除向量库"""
        import shutil
        if os.path.exists(self.vector_db_path):
            shutil.rmtree(self.vector_db_path)
            logger.info(f"🗑️ 已删除向量库: {self.vector_db_path}")

    def delete_by_ids(self, ids: List[str]):
        if self.vectordb is None:
            raise ValueError("向量库未初始化")
        # 新增：过滤不存在的ID
        existing_ids = set(self.vectordb.get()['ids'])
        valid_ids = [i for i in ids if i in existing_ids]
        if valid_ids:
            self.vectordb.delete(valid_ids)
            self.vectordb.persist()  # 新增：持久化删除操作
            logger.info(f"🗑️ 已删除 {len(valid_ids)} 个文档块")
        else:
            logger.warning("无有效ID可删除")
    
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