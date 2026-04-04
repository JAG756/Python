# core/document_loader.py
"""
文档加载模块 - 支持 PDF、TXT，带元数据
"""

import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class DocumentLoader:
    """文档加载器（带元数据）"""
    
    def __init__(self, chunk_size=200, chunk_overlap=30):
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
            
            print(f"📄 加载 PDF: {os.path.basename(pdf_path)}，共 {len(documents)} 页")
            return documents
        except Exception as e:
            print(f"❌ 加载 PDF 失败: {e}")
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
            
            print(f"📄 加载文本: {os.path.basename(text_path)}")
            return documents
        except Exception as e:
            print(f"❌ 加载文本失败: {e}")
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
    
    def load_directory(self, directory_path: str, extensions: List[str] = [".pdf", ".txt"]) -> List[Document]:
        """批量加载目录下的所有文档"""
        all_documents = []
        
        if not os.path.exists(directory_path):
            print(f"❌ 目录不存在: {directory_path}")
            return []
        
        pdf_count = 0
        txt_count = 0
        
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            ext = os.path.splitext(filename)[1].lower()
            
            if ext == ".pdf" and ".pdf" in extensions:
                docs = self.load_pdf(file_path)
                all_documents.extend(docs)
                pdf_count += 1
            elif ext == ".txt" and ".txt" in extensions:
                docs = self.load_text(file_path)
                all_documents.extend(docs)
                txt_count += 1
        
        print(f"✅ 批量加载完成: {pdf_count} 个 PDF, {txt_count} 个 TXT, 共 {len(all_documents)} 页")
        return all_documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """分块处理，保留元数据"""
        chunks = self.splitter.split_documents(documents)
        print(f"✂️ 文档分块: {len(documents)} 页 → {len(chunks)} 个块")
        print(f"   块大小: {self.chunk_size} 字符, 重叠: {self.chunk_overlap} 字符")
        return chunks