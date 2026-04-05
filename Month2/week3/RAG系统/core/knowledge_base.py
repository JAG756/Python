# core/knowledge_base.py
"""
知识库模块 - 支持多源加载，带检索溯源
"""

import os
from typing import List, Dict, Optional, Tuple
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from config import KNOWLEDGE_BASE, SIMILARITY_THRESHOLD, TOP_K
from core.document_loader import DocumentLoader
from core.batch_vectorizer import BatchVectorizer
from utils.logger import logger

class KnowledgeBase:
    """知识库管理器（带溯源）"""
    
    def __init__(self, device: str, vector_db_path: str = "./vector_db"):
        self.device = device
        self.vectordb = None
        self.threshold = SIMILARITY_THRESHOLD
        self.top_k = TOP_K
        self.vector_db_path = vector_db_path
        self.doc_loader = DocumentLoader()
        self.batch_vectorizer = BatchVectorizer(vector_db_path)
    
    def build_from_strings(self, texts: List[str]):
        """从字符串列表构建知识库"""
        logger.info("从字符串构建知识库...")

        documents = []
        for i, text in enumerate(texts):
            doc = Document(
                page_content=text,
                metadata={
                    "source": f"内置知识库_{i+1}",
                    "page": 1,
                    "type": "builtin"
                }
            )
            documents.append(doc)
        
        chunks = self.doc_loader.split_documents(documents)
        
        self.batch_vectorizer.init_embedding(self.device)
        self.vectordb = self.batch_vectorizer.create_vector_store(chunks)
        
        logger.info(f"知识库构建完成，共 {len(texts)} 条知识")
        return self
    
    def build_from_directory(self, directory_path: str):
        """从目录批量构建知识库"""
        logger.info(f"从目录构建知识库: {directory_path}")
        
        if not os.path.exists(directory_path):
            logger.warning(f"目录不存在: {directory_path}")
            logger.info("使用默认知识库")
            return self.build_from_strings(KNOWLEDGE_BASE)
        
        files = os.listdir(directory_path)
        pdf_files = [f for f in files if f.endswith('.pdf')]
        txt_files = [f for f in files if f.endswith('.txt')]
        
        if not pdf_files and not txt_files:
            logger.warning(f"目录中没有 PDF 或 TXT 文件: {directory_path}")
            logger.info("使用默认知识库")
            return self.build_from_strings(KNOWLEDGE_BASE)
        
        logger.info(f"找到 {len(pdf_files)} 个 PDF 文件，{len(txt_files)} 个 TXT 文件")
        
        documents = self.doc_loader.load_directory(directory_path)
        
        if not documents:
            logger.warning("没有加载到任何文档，使用默认知识库")
            return self.build_from_strings(KNOWLEDGE_BASE)
        
        chunks = self.doc_loader.split_documents(documents)
        
        self.batch_vectorizer.init_embedding(self.device)
        self.vectordb = self.batch_vectorizer.create_vector_store(chunks)
        
        logger.info(f"知识库构建完成，共 {len(chunks)} 个文档块")
        return self
    
    def search_with_source(self, question: str) -> Optional[Dict]:
        """
        检索并返回内容和来源信息
        返回: {"content": 原文, "source": 文件名, "page": 页码, "score": 相似度}
        """
        if self.vectordb is None:
            return None

        # 恢复为只检索 Top 1
        docs = self.vectordb.similarity_search_with_score(question, k=self.top_k)

        if not docs or docs[0][1] > self.threshold:
            return None

        doc = docs[0][0]
        score = docs[0][1]

        # 获取元数据
        metadata = doc.metadata if hasattr(doc, 'metadata') else {}

        return {
            "content": doc.page_content,
            "source": metadata.get("source", "未知来源"),
            "source_path": metadata.get("source_path", ""),
            "page": metadata.get("page", "未知"),
            "doc_type": metadata.get("type", "unknown"),
            "score": score
        }
    
    def search(self, question: str) -> Tuple[Optional[str], Optional[float]]:
        """简化版检索，只返回内容和分数（兼容旧代码）"""
        result = self.search_with_source(question)
        if result:
            return result["content"], result["score"]
        return None, None
    
    def load_persisted(self):
        """加载已存在的向量库"""
        logger.info("📂 加载已有向量库...")
        
        self.batch_vectorizer.init_embedding(self.device)
        self.vectordb = self.batch_vectorizer.load_vector_store()
        
        if self.vectordb:
            logger.info("✅ 向量库加载成功")
        else:
            logger.warning("⚠️ 向量库不存在，正在从 docs 目录构建...")
            import os
            current_file = os.path.abspath(__file__)
            project_root = os.path.dirname(os.path.dirname(current_file))
            docs_path = os.path.join(project_root, "docs")
            logger.info(f"📁 使用文档目录: {docs_path}")
            self.build_from_directory(docs_path)
        
        return self
    
    def get_stats(self):
        """获取知识库统计"""
        return self.batch_vectorizer.get_stats()