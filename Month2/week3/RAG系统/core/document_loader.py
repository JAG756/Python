# core/document_loader.py
"""
文档加载模块 - 支持 PDF、TXT，带元数据
"""

import os
from typing import List
from config import CHUNK_SIZE, CHUNK_OVERLAP
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from utils.logger import logger

class DocumentLoader:
    """文档加载器（带元数据）"""
    
    def __init__(self, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
        """
        chunk_size: 每个块的大小（字符数），建议 150-200
        chunk_overlap: 块之间的重叠字符数
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "，", " ", ""]
        )
    
    def load_pdf(self, pdf_path: str) -> List[Document]:
        """加载 PDF 文件，添加元数据"""
        try:
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
            # 为每一页添加元数据
            for i, doc in enumerate(documents):
                doc.metadata = {
                    "source": os.path.basename(pdf_path),
                    "source_path": pdf_path,
                    "page": i + 1,
                    "type": "pdf"
                }
            
            logger.info(f"📄 加载 PDF: {os.path.basename(pdf_path)}，共 {len(documents)} 页")
            return documents
        except Exception as e:
            logger.error(f"❌ 加载 PDF 失败: {e}")
            return []
    
    def load_text(self, text_path: str) -> List[Document]:
        """加载 TXT 文件，添加元数据"""
        try:
            loader = TextLoader(text_path, encoding='utf-8')
            documents = loader.load()
            for doc in documents:
                doc.metadata = {
                    "source": os.path.basename(text_path),
                    "source_path": text_path,
                    "page": 1,
                    "type": "text"
                }
            
            logger.info(f"📄 加载文本: {os.path.basename(text_path)}")
            return documents
        except Exception as e:
            logger.error(f"❌ 加载文本失败: {e}")
            return []

    def load_md(self, md_path: str) -> List[Document]:
        try:
            loader = TextLoader(md_path, encoding='utf-8')
            documents = loader.load()
            for doc in documents:
                doc.metadata = {
                    "source": os.path.basename(md_path),
                    "source_path": md_path,
                    "page": 1,
                    "type": "markdown"
                }
            logger.info(f"📄 加载 Markdown: {os.path.basename(md_path)}")
            return documents
        except Exception as e:
            logger.error(f"❌ 加载 Markdown 失败: {e}")
            return []
    
    def load_from_string(self, content: str, source: str = "内置知识库") -> List[Document]:
        """从字符串加载"""
        doc = Document(
            page_content=content, 
            metadata={
                "source": source,
                "page": 1,
                "type": "builtin"
            }
        )
        return [doc]
    
    def load_directory(self, directory_path: str, extensions: List[str] = [".pdf", ".txt", ".md"]) -> List[Document]:
        all_documents = []
        if not os.path.exists(directory_path):
            logger.error(f"目录不存在: {directory_path}")
            return []

        pdf_count = txt_count = 0
        # 使用 os.walk 递归遍历所有子文件夹
        for root, dirs, files in os.walk(directory_path):
            for filename in files:
                file_path = os.path.join(root, filename)
                ext = os.path.splitext(filename)[1].lower()
                try:
                    if ext == ".pdf" and ".pdf" in extensions:
                        docs = self.load_pdf(file_path)
                        all_documents.extend(docs)
                        pdf_count += 1
                    elif ext == ".txt" and ".txt" in extensions:
                        docs = self.load_text(file_path)
                        all_documents.extend(docs)
                        txt_count += 1
                    elif ext == ".md" and ".md" in extensions:
                        docs = self.load_md(file_path)
                        all_documents.extend(docs)
                        txt_count += 1
                except Exception as e:
                    logger.error(f"跳过无法加载的文件 {filename}: {e}")
                    continue

        logger.info(f"批量加载完成: {pdf_count} 个 PDF, {txt_count} 个 TXT, 共 {len(all_documents)} 页")
        return all_documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """分块处理，保留元数据"""
        chunks = self.splitter.split_documents(documents)
        logger.info(f"✂️ 文档分块: {len(documents)} 页 → {len(chunks)} 个块")
        logger.info(f"   块大小: {self.chunk_size} 字符, 重叠: {self.chunk_overlap} 字符")
        return chunks

    def split_by_chapters(self, document: Document) -> List[Document]:
        """
        将 Markdown 文档按二级标题（##）切分成章节，每个章节作为一个 Document。
        保留原文档的元数据，并添加 chapter_title 字段。
        """
        import re
        content = document.page_content
        # 匹配二级标题及其后的内容（直到下一个二级标题或文档结束）
        pattern = r'(##\s+[^\n]+)\n(.*?)(?=\n##\s+|$)'
        matches = re.findall(pattern, content, re.DOTALL)
        
        if not matches:
            # 如果没有二级标题，返回整个文档作为一个章节
            doc = Document(
                page_content=content,
                metadata={**document.metadata, "chapter_title": "全文", "is_chapter": True}
            )
            return [doc]
        
        chapters = []
        for title, body in matches:
            # 生成章节的摘要（取前 300 字符作为摘要）
            summary = body.strip()[:300]
            chapter_content = f"{title}\n{body.strip()}"
            metadata = {
                **document.metadata,
                "chapter_title": title.strip(),
                "summary": summary,
                "is_chapter": True
            }
            chapters.append(Document(page_content=chapter_content, metadata=metadata))
        return chapters    

    def _semantic_split_text(self, text: str, threshold: float = 0.5) -> List[str]:
        """语义分块：基于句子向量相似度切分"""
        from sentence_transformers import SentenceTransformer
        import numpy as np
        
        # 加载模型（只加载一次，缓存到实例）
        if not hasattr(self, '_semantic_model'):
            self._semantic_model = SentenceTransformer("moka-ai/m3e-base")
        
        # 分句（简单按句号、感叹号、问号分割）
        sentences = []
        for sep in ['。', '！', '？', '\n\n', '\n']:
            if sep in text:
                parts = text.split(sep)
                for p in parts:
                    if p.strip():
                        sentences.append(p.strip() + sep)
                break
        else:
            sentences = [text]
        
        if len(sentences) <= 1:
            return sentences
        
        # 计算句子向量
        embeds = self._semantic_model.encode(sentences)
        chunks = []
        current_chunk = [sentences[0]]
        
        for i in range(1, len(sentences)):
            # 计算当前句子与前一句的余弦相似度
            sim = np.dot(embeds[i-1], embeds[i]) / (np.linalg.norm(embeds[i-1]) * np.linalg.norm(embeds[i]))
            if sim < threshold:
                # 语义转折，切分
                chunks.append(''.join(current_chunk))
                current_chunk = [sentences[i]]
            else:
                current_chunk.append(sentences[i])
        
        if current_chunk:
            chunks.append(''.join(current_chunk))
        
        return chunks

    def semantic_split_documents(self, documents: List[Document], threshold: float = 0.5) -> List[Document]:
        """对文档列表进行语义分块（保留元数据）"""
        from langchain_core.documents import Document
        all_chunks = []
        for doc in documents:
            text = doc.page_content
            chunks_text = self._semantic_split_text(text, threshold)
            for i, chunk_text in enumerate(chunks_text):
                # 复制原文档的元数据，并添加块索引
                new_metadata = doc.metadata.copy()
                new_metadata["chunk_index"] = i
                new_doc = Document(page_content=chunk_text, metadata=new_metadata)
                all_chunks.append(new_doc)
        logger.info(f"✂️ 语义分块: {len(documents)} 篇文档 → {len(all_chunks)} 个块")
        return all_chunks