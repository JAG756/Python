# core/knowledge_base.py
"""
知识库模块 - 支持多源加载，带检索溯源
"""

import os
import json
import hashlib
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
    # 初始化知识库，支持从字符串列表或目录构建，并提供检索功能
    def __init__(self, device: str, vector_db_path: str = "./vector_db"):
        self.device = device
        self.vectordb = None
        self.threshold = SIMILARITY_THRESHOLD
        self.top_k = TOP_K
        self.vector_db_path = vector_db_path
        self.doc_loader = DocumentLoader()
        self.batch_vectorizer = BatchVectorizer(vector_db_path)
    #初始化
    
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
        # 创建向量库
        self.vectordb = self.batch_vectorizer.create_vector_store(chunks)
        # 将字符串列表转换为 Document 对象，添加元数据（来源、页码、类型等），然后切分成更小的块。接着初始化向量化模型，并用 Chroma 创建向量数据库。
        
        logger.info(f"知识库构建完成，共 {len(texts)} 条知识")
        return self
    
    def build_from_directory(self, directory_path: str):
        """从目录批量构建知识库（递归支持子文件夹）"""
        logger.info(f"从目录构建知识库: {directory_path}")
        
        if not os.path.exists(directory_path):
            logger.warning(f"目录不存在: {directory_path}")
            logger.info("使用默认知识库")
            return self.build_from_strings(KNOWLEDGE_BASE)
        
        # 加载所有文档
        documents = self.doc_loader.load_directory(directory_path)
        if not documents:
            logger.warning("没有加载到任何文档，使用默认知识库")
            return self.build_from_strings(KNOWLEDGE_BASE)
        
        # 分块
        chunks = self.doc_loader.split_documents(documents)
        
        # ========== 新增：为每个 chunk 生成自定义 ID ==========
        # 需要知道每个 chunk 来自哪个文件，以便后续生成 ID
        # 由于 chunks 已经保留了 metadata（包含 source_path 和 page），我们可以遍历生成
        chunk_ids = []
        for chunk in chunks:
            file_path = chunk.metadata.get("source_path", "")
            page = chunk.metadata.get("page", 0)
            # 使用 _get_all_document_ids 的逻辑，但这里需要为单个 chunk 生成 ID
            # 注意：_get_all_document_ids 是为整个文件的 chunks 批量生成的，这里我们模拟一个简单规则
            file_hash = hashlib.md5(file_path.encode()).hexdigest()[:8]
            # 由于分块后每个 chunk 没有块内索引，我们用它在列表中的位置作为临时索引（不完美但唯一）
            # 更好的做法：在 split_documents 时保留一个 chunk_index，但为了简化，我们基于内容生成
            # 这里采用更可靠的方式：使用文件路径+页码+内容前50字符的哈希
            content_hash = hashlib.md5(chunk.page_content[:50].encode()).hexdigest()[:4]
            chunk_id = f"{file_hash}_{page}_{content_hash}"
            chunk_ids.append(chunk_id)
        # ==================================================
        
        self.batch_vectorizer.init_embedding(self.device)
        # 使用带 ID 的方法创建向量库
        self.vectordb = self.batch_vectorizer.create_vector_store_with_ids(chunks, chunk_ids)
        
        # ========== 新增：保存文件状态 ==========
        # 构建文件状态字典：需要知道每个文件对应的所有 chunk_ids
        file_state = {}
        for chunk, cid in zip(chunks, chunk_ids):
            file_path = chunk.metadata.get("source_path", "")
            rel_path = os.path.relpath(file_path, directory_path) if file_path else ""
            if not rel_path:
                continue
            if rel_path not in file_state:
                # 记录文件的修改时间和 chunk_ids 列表
                mtime = os.path.getmtime(file_path) if os.path.exists(file_path) else 0
                file_state[rel_path] = {
                    "hash": self._get_file_hash(file_path),
                    "chunk_ids": []
                }
            file_state[rel_path]["chunk_ids"].append(cid)
        self._save_file_state(file_state)
        # ======================================
        
        logger.info(f"知识库构建完成，共 {len(chunks)} 个文档块")
        return self
    
    def search_with_source(self, question: str) -> Optional[Dict]:
        if self.vectordb is None:
            logger.warning("向量库未初始化，无法检索")
            return None
        
        try:
            docs = self.vectordb.similarity_search_with_score(question, k=self.top_k)
        except Exception as e:
            logger.error(f"检索失败: {e}")
            return None
        
        # 遍历所有结果，找到第一个相似度低于阈值的
        for doc, score in docs:
            if score <= self.threshold:   # 注意：score 越小越相关
                metadata = doc.metadata if hasattr(doc, 'metadata') else {}
                return {
                    "content": doc.page_content,
                    "source": metadata.get("source", "未知来源"),
                    "source_path": metadata.get("source_path", ""),
                    "page": metadata.get("page", "未知"),
                    "doc_type": metadata.get("type", "unknown"),
                    "score": score,
                    "is_relevant": True
                }
        return None   # 所有结果都不满足阈值

    
    def search(self, question: str) -> Tuple[Optional[str], Optional[float]]:
        """简化版检索，只返回内容和分数（兼容旧代码）"""
        result = self.search_with_source(question)
        if result:
            return result["content"], result["score"]
        return None, None
    
    def load_persisted(self):
        """加载已存在的向量库，失败则自动重建"""
        logger.info("加载已有向量库...")
        
        try:
            self.batch_vectorizer.init_embedding(self.device)
            self.vectordb = self.batch_vectorizer.load_vector_store()
            
            if self.vectordb:
                logger.info("向量库加载成功")
                return self
            else:
                raise FileNotFoundError("向量库目录存在但无法加载")
        except Exception as e:
            logger.warning(f"向量库加载失败: {e}，将尝试重建")
            
            # 删除可能损坏的向量库文件
            import shutil
            if os.path.exists(self.vector_db_path):
                shutil.rmtree(self.vector_db_path)
                logger.info(f"已删除损坏的向量库: {self.vector_db_path}")
            
            # 从 docs 目录重建（递归检查是否有文档）
            current_file = os.path.abspath(__file__)
            project_root = os.path.dirname(os.path.dirname(current_file))
            docs_path = os.path.join(project_root, "docs")
            
            # 递归检查 docs 目录下是否有任何 PDF 或 TXT 文件
            has_docs = False
            if os.path.exists(docs_path):
                for root, dirs, files in os.walk(docs_path):
                    if any(f.endswith(('.pdf', '.txt', '.md')) for f in files):
                        has_docs = True
                        break
            
            if has_docs:
                logger.info(f"从文档目录重建: {docs_path}")
                self.build_from_directory(docs_path)
            else:
                logger.info("无可用文档，使用内置知识库")
                self.build_from_strings(KNOWLEDGE_BASE)
            
            return self
    
    def _get_file_state_path(self):
        """状态文件路径，放在向量库目录旁边"""
        return os.path.join(os.path.dirname(self.vector_db_path), "file_state.json")

    def _load_file_state(self):
        """加载已入库文件的状态"""
        path = self._get_file_state_path()
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _save_file_state(self, state):
        """保存文件状态"""
        with open(self._get_file_state_path(), 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

    def _get_file_hash(self, file_path):
        """计算文件的MD5，用于检测内容变化（比修改时间更可靠）"""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _get_all_document_ids(self, file_path, chunks):
        """为文件的所有块生成唯一ID（用于删除）"""
        # ID格式：文件路径的MD5 + 页码 + 块内索引
        file_hash = hashlib.md5(file_path.encode()).hexdigest()[:8]
        ids = []
        for i, chunk in enumerate(chunks):
            # 从chunk.metadata中获取页码
            page = chunk.metadata.get("page", 0)
        # 统一为：file_hash + page + 索引i（与build_from_directory对齐）
            ids.append(f"{file_hash}_{page}_{i}")
        return ids

    def incremental_update(self, docs_dir: str):
        """
        增量更新知识库：扫描docs_dir，对比状态文件，增删改对应的向量
        """
        logger.info("🔍 开始增量更新知识库...")

        logger.info(f"扫描目录: {os.path.abspath(docs_dir)}")
        logger.info(f"目录是否存在: {os.path.exists(docs_dir)}")
        if os.path.exists(docs_dir):
            file_list = []
            for root, dirs, files in os.walk(docs_dir):
                for f in files:
                    file_list.append(os.path.join(root, f))
            logger.info(f"找到文件: {file_list}")

        if self.vectordb is None:
            raise ValueError("请先初始化向量库（调用 load_persisted 或 build_from_directory）")

        
        old_state = self._load_file_state()
        new_state = {}

        # 递归遍历当前docs目录
        current_files = {}
        for root, dirs, files in os.walk(docs_dir):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext not in ['.pdf', '.txt', '.md']:
                    continue
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, docs_dir)
                current_files[rel_path] = full_path

        # 识别删除的文件
        to_delete_files = [rel for rel in old_state if rel not in current_files]

        # 识别需要添加/更新的文件（新文件或修改时间变化）
        to_process_files = []



        for rel_path, full_path in current_files.items():
        # 使用文件 MD5 判断内容是否变化
            current_hash = self._get_file_hash(full_path)
            if rel_path not in old_state:
                to_process_files.append((rel_path, full_path))
            else:
                old_hash = old_state[rel_path].get("hash")  # 注意：状态文件中存储的 key 改为 'hash'
                if current_hash != old_hash:
                    to_process_files.append((rel_path, full_path))

        # 处理删除：从向量库中移除这些文件的所有块
        if to_delete_files:
            ids_to_delete = []
            for rel_path in to_delete_files:
                if "chunk_ids" in old_state.get(rel_path, {}):
                    ids_to_delete.extend(old_state[rel_path]["chunk_ids"])
                else:
                    # 兜底：如果旧状态没有 chunk_ids，尝试通过 metadata 删除（需要 Chroma 支持 where）
                    # 为了简单，这里记录警告并跳过（或者可以全量重建，但为了代码简洁，先警告）
                    logger.warning(f"无法删除文件 {rel_path} 的块，缺少ID记录，建议重建向量库")
            if ids_to_delete:
                self.batch_vectorizer.delete_by_ids(ids_to_delete)
                logger.info(f"已删除 {len(ids_to_delete)} 个块，来自 {len(to_delete_files)} 个移除的文件")

        # 处理新增/更新
        added_chunk_count = 0
        for rel_path, full_path in to_process_files:
            logger.info(f"处理文件: {rel_path}")
            # 加载文档
            ext = os.path.splitext(full_path)[1].lower()
            if ext == '.pdf':
                docs = self.doc_loader.load_pdf(full_path)
            elif ext == '.txt':
                docs = self.doc_loader.load_text(full_path)
            elif ext == '.md':
                docs = self.doc_loader.load_md(full_path)
            else:
                continue
            if not docs:
                continue
            # 分块
            chunks = self.doc_loader.split_documents(docs)
            if not chunks:
                continue
            
            # ========== 生成自定义 ID ==========
            chunk_ids = self._get_all_document_ids(full_path, chunks)
            # =================================
            
            # 如果是更新，先删除该文件原有的块
            if rel_path in old_state and "chunk_ids" in old_state[rel_path]:
                old_ids = old_state[rel_path]["chunk_ids"]
                self.batch_vectorizer.delete_by_ids(old_ids)
                logger.info(f"删除旧块 {len(old_ids)} 个")
            
            # 添加新块，并传入自定义 ID
            self.batch_vectorizer.add_documents(chunks, ids=chunk_ids)   # 注意这里传入了 ids
            added_chunk_count += len(chunks)
            
            # 记录新状态
            new_state[rel_path] = {
                "hash": self._get_file_hash(full_path),
                "chunk_ids": chunk_ids
            }

        # 保留未变化的文件状态
        for rel_path, info in old_state.items():
            if rel_path not in to_delete_files and rel_path not in [p for p, _ in to_process_files]:
                new_state[rel_path] = info

        # 保存新状态
        self._save_file_state(new_state)
        logger.info(f"✅ 增量更新完成：新增/更新 {len(to_process_files)} 个文件，添加 {added_chunk_count} 个块；删除 {len(to_delete_files)} 个文件")
        
    def get_stats(self):
        """获取知识库统计"""
        return self.batch_vectorizer.get_stats()