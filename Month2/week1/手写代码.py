# 1,加载文档
text = "你的知识库内容"
# 2，切分文档
from langchain_text_splitters import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=30,    # 每段最多30字符  
    chunk_overlap=20,  # 重叠20字符
    length_function=len)
chunks = splitter.split_text(text)
# 3, 向量化
from langchain_huggingface import HuggingFaceEmbeddings
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectors = embedding.embed_documents(chunks)
# 4，存向量库
from langchain_vectorstores import FAISS
vectorstore = FAISS.from_embeddings(vectors, chunks)
# 5，根据用户问题检索相似度最大的
query = "用户问题"
query_vector = embedding.embed_query(query)
docs = vectorstore.similarity_search_by_vector(query_vector, k=1)
print("检索到的相关文档：", docs[0].page_content)
# 6，打包发送给大模型，大模型根据文档回答问题
from langchain_chat_models import ChatOpenAI    
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
response = llm(f"根据以下文档回答问题：{docs[0].page_content}\n问题：{query}")
print("大模型回答：", response.content)
