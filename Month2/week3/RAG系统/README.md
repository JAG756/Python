markdown
# RAG 智能问答系统

基于检索增强生成（Retrieval-Augmented Generation）的企业级知识库问答系统。支持多种文档格式（PDF、TXT、Markdown），提供检索溯源、多轮对话、增量更新等功能。

## 功能特点

- 📚 **多源知识库**：支持 PDF、TXT、Markdown 文档，自动递归加载子目录
- 🔎 **高级检索**：支持混合检索（BM25+向量）、重排序（Rerank）和分层检索（针对长文档），显著提升召回率和答案相关性
- 🤖 **生成式问答**：可选启用大模型生成答案（支持 Qwen 系列模型）
- 🔗 **答案溯源**：每条回答附来源文档、页码、相关度分数
- 💬 **多轮对话**：自动处理指代（“它”、“这个”）和省略主语
- 🔄 **增量更新**：自动检测文档变化，仅更新变更的文件，无需重建全量索引
- ⚡ **性能优化**：支持 GPU 加速、答案缓存、生成长度控制
- 🌐 **Web 界面**：基于 Gradio 的简洁交互界面，开箱即用
- 📤 **多文件上传**：支持通过 Web 界面上传 PDF/TXT/Markdown 文件，自动增量更新知识库，无需重启
- 💾 **对话记忆**：保留完整对话历史，支持导出记录，可手动清空会话
- 🧠 **智能查询改写**：自动识别定义类问题，强化关键词，提高答案准确度

## 系统架构
用户输入 → 意图识别 → 多轮对话补全 → 向量检索 → （可选）LLM生成 → 溯源拼接 → 输出

## 核心流程

1. **文档加载**：递归扫描 `docs/` 目录，支持 PDF、TXT、Markdown 格式，自动提取元数据（文件名、页码）。
2. **智能分块**：可配置递归分块或语义分块（基于句子向量相似度），默认块大小 200 字符，重叠 20 字符。
3. **向量索引**：使用 `moka-ai/m3e-base` 生成 768 维向量，存储于 Chroma 数据库，同时构建 BM25 关键词索引。
4. **分层检索（可选）**：对长文档自动提取章节（支持 `#` 和 `##` 标题），先检索相关章节，再在章节内精细检索。
5. **混合检索**：融合向量相似度和 BM25 得分（权重可调），兼顾语义理解与关键词匹配。
6. **重排序（Rerank）**：利用 `bge-reranker-base` 交叉编码器对候选片段重新打分，提升 Top-1 准确率。
7. **答案生成**：将检索到的相关片段拼接后送入 `Qwen1.5-1.8B-Chat` 生成最终答案，并附来源信息。
8. **增量更新**：对比文件 MD5 哈希，仅处理新增、修改、删除的文件，自动重建 BM25 索引，无需全量重建。

## 环境要求

- Python 3.8+
- 推荐 8GB+ RAM（CPU 模式）或 4GB+ 显存（GPU 模式）
- 可选：NVIDIA GPU（CUDA 支持）

## 快速开始

### 1. 克隆项目

```bash
git clone <your-repo-url>
cd RAG系统
2. 安装依赖
bash
pip install -r requirements.txt
主要依赖：

torch (>=1.13)

transformers (>=4.35)

langchain (>=0.1)

chromadb (>=0.4)

gradio (>=4.0)

sentence-transformers (>=2.2)

pypdf (PDF 解析)

3. 准备知识库
将您的文档放入 docs/ 目录（支持子文件夹），系统会递归加载所有 .pdf、.txt、.md 文件。

默认内置知识库（config.py 中的 KNOWLEDGE_BASE）会在 docs/ 为空时使用。

4. 下载模型（离线）
本项目默认使用 Qwen/Qwen1.5-1.8B-Chat。请提前下载到本地缓存目录（例如 ~/.cache/huggingface/hub/）。

设置环境变量强制离线：

bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
5. 启动系统
bash
python main.py
访问 http://localhost:7860 开始使用。


配置说明
所有配置集中在 config.py 中：

参数	说明	默认值
MODEL_NAME	生成模型名称	Qwen/Qwen1.5-1.8B-Chat
SIMILARITY_THRESHOLD	相似度阈值（越小越严格）	0.5
TOP_K	检索返回的最相似块数量	3
USE_GENERATION	是否启用 LLM 生成	True
DEFAULT_MAX_LENGTH	生成最大 token 数	200
DEFAULT_TEMPERATURE	生成温度（0~1）	0.3
CHUNK_SIZE	文档分块字符数	200
CHUNK_OVERLAP	块间重叠字符数	30

补充

| USE_HYBRID_SEARCH | 是否启用混合检索（BM25+向量） | True |
| HYBRID_ALPHA | 混合检索中向量权重（0~1） | 0.5 |
| USE_RERANK | 是否启用重排序 | True |
| RERANK_MODEL | 重排序模型名称 | BAAI/bge-reranker-base |
| USE_HIERARCHICAL | 是否启用分层检索（针对长文档） | True |
| CHAPTER_TOP_K | 分层检索第一层返回的章节数 | 2 |
| USE_SEMANTIC_CHUNKING | 是否启用语义分块 | False |



项目结构
text
.
├── main.py                 # 入口文件
├── config.py               # 配置文件
├── core/
│   ├── model_loader.py     # 模型加载（支持 GPU/CPU）
│   ├── knowledge_base.py   # 知识库管理（向量库、增量更新）
│   ├── rag_engine.py       # RAG 核心逻辑
│   ├── document_loader.py  # 文档加载与分块
│   └── batch_vectorizer.py # 批量向量化
├── ui/
│   └── interface.py        # Gradio 界面
├── utils/
│   ├── chat_session.py     # 多轮对话管理
│   └── logger.py           # 日志配置
├── docs/                   # 放置知识库文档
├── vector_db/              # 向量库持久化目录（自动生成）
└── rag_system.log          # 运行日志
使用示例
Web 界面
在输入框输入问题，例如 “什么是死锁？”

系统返回答案及来源信息（文档名、页码、相关度）

支持多轮追问：“它的四个必要条件是什么？”

命令行快速测试
python
from core.model_loader import ModelLoader
from core.knowledge_base import KnowledgeBase
from core.rag_engine import RAGEngine

model = ModelLoader().load()
kb = KnowledgeBase(model.device).load_persisted()
rag = RAGEngine(model, kb)

answer, session_id = rag.answer("什么是RAG？")
print(answer)
性能优化建议
场景	推荐配置
CPU 环境，追求速度	USE_GENERATION=False（直接返回检索原文）
GPU 环境，追求质量	USE_GENERATION=True，DEFAULT_MAX_LENGTH=150
重复问题多	启用答案缓存（参考 rag_engine.py 中的 _cache）
显存不足	降低 TOP_K，或使用 4-bit 量化（需 bitsandbytes）
详细优化指南见 Day24 压测与优化（内部文档）。

常见问题
Q：启动时报错 “Tokenizer 加载失败”？
A：模型未下载或缓存路径不正确。请设置 HF_HUB_OFFLINE=0 允许联网下载一次，或手动下载模型到缓存目录。

Q：检索结果不相关？
A：调低 SIMILARITY_THRESHOLD（如 0.3），或增加 CHUNK_SIZE 使每个块包含更完整语义。

Q：如何更新知识库？
A：将新文档放入 docs/，点击 Web 界面中的“更新知识库”按钮，或重启程序（自动增量更新）。

Q：支持哪些文档格式？
A：PDF、TXT、Markdown。其他格式可自行扩展 document_loader.py。

技术栈
向量数据库：Chroma（本地持久化）

Embedding 模型：moka-ai/m3e-base（中文优化）

LLM：Qwen1.5-1.8B-Chat（可替换）

前端：Gradio 4.x

框架：LangChain（文档处理）、Transformers（模型推理）

许可证
MIT License

贡献指南
欢迎提交 Issue 和 Pull Request。如需添加新的文档格式或自定义生成逻辑，请参考代码注释。

Happy RAGging! 🚀

text

---

## 使用方法

1. 在项目根目录（与 `main.py` 同级）新建文件 `README.md`。
2. 将上述内容复制粘贴进去。
3. 根据需要修改部分信息（如项目名称、Git 仓库地址等）。

如果您需要添加更详细的 API 文档、部署指南或故障排查章节，可以在此基础上扩展。