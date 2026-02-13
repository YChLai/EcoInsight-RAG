import os
import json
import uuid
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from zhipuai import ZhipuAI
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

MD_DIR = "../preprocess/output_merged"
DATA_DIR = "../processed_data"
MILVUS_HOST = "localhost"
MILVUS_PORT = 19530
MILVUS_COLLECTION = "env_rag"
EMBEDDING_MODEL = "BAAI/bge-m3"

client = ZhipuAI(api_key=os.getenv("ZHIPUAI_API_KEY"))

def summarize(content):
    if not client or len(content) < 500:
        return content
    try:
        resp = client.chat.completions.create(
            model="glm-4-flash",
            messages=[{"role": "user", "content": f"为以下内容生成150字以内的摘要：\n{content}"}],
            max_tokens=300, temperature=0.0
        )
        return resp.choices[0].message.content
    except:
        return "摘要失败"

def load_docs():
    print("--- 加载文档 ---")
    if not os.path.exists(MD_DIR):
        print(f"目录不存在: {MD_DIR}")
        return []
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=250, separators=["\n\n", "\n", " ", ""])
    docs = []
    
    for md in [f for f in os.listdir(MD_DIR) if f.endswith(".md")]:
        with open(os.path.join(MD_DIR, md), 'r', encoding='utf-8') as f:
            content = f.read()
        
        for i, chunk in enumerate(splitter.split_text(content)):
            docs.append(Document(page_content=chunk, metadata={"source": md.replace(".md", ""), "page": i + 1, "doc_id": str(uuid.uuid4())}))
    
    print(f"创建 {len(docs)} 个文档")
    return docs

def split_docs(parent_docs):
    print("\n--- 分割文档 ---")
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100, separators=["\n\n", "\n", " ", ""])
    children, summaries = [], []
    
    for i, doc in enumerate(parent_docs):
        print(f"处理 {i+1}/{len(parent_docs)}: {doc.metadata['source']}")
        
        summaries.append(Document(page_content=summarize(doc.page_content), metadata={**doc.metadata, "is_summary": True}))
        
        for j, chunk in enumerate(splitter.split_text(doc.page_content)):
            children.append(Document(page_content=chunk, metadata={**doc.metadata, "is_child": True, "chunk_index": j}))
    
    print(f"创建 {len(children)} 个子块, {len(summaries)} 个摘要")
    return children, summaries

def build_index(docs, embeddings):
    print("\n--- 构建索引 ---")
    connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
    
    if utility.has_collection(MILVUS_COLLECTION):
        utility.drop_collection(MILVUS_COLLECTION)
    
    fields = [
        FieldSchema("id", DataType.VARCHAR, max_length=40, is_primary=True),
        FieldSchema("content", DataType.VARCHAR, max_length=65535),
        FieldSchema("source", DataType.VARCHAR, max_length=255),
        FieldSchema("page", DataType.INT32),
        FieldSchema("doc_id", DataType.VARCHAR, max_length=36),
        FieldSchema("is_child", DataType.BOOL, default_value=False),
        FieldSchema("is_summary", DataType.BOOL, default_value=False),
        FieldSchema("chunk_index", DataType.INT64, default_value=-1),
        FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=1024)
    ]
    
    collection = Collection(MILVUS_COLLECTION, CollectionSchema(fields))
    
    data = []
    for doc in docs:
        doc_id = doc.metadata.get("doc_id", str(uuid.uuid4()))
        chunk_idx = doc.metadata.get("chunk_index", -1)
        uid = f"{doc_id.replace('-', '')[:32]}{chunk_idx:04d}"[:36] if chunk_idx != -1 else doc_id
        
        data.append([uid, doc.page_content, doc.metadata.get("source", ""), doc.metadata.get("page", 0),
                     doc_id, doc.metadata.get("is_child", False), doc.metadata.get("is_summary", False),
                     chunk_idx, embeddings.embed_query(doc.page_content)])
    
    collection.insert([[d[i] for d in data] for i in range(9)])
    collection.create_index("embedding", {"index_type": "HNSW", "metric_type": "L2", "params": {"M": 8, "efConstruction": 64}})
    collection.load()
    print(f"插入 {len(data)} 条数据")

def verify(embeddings):
    print("\n--- 验证 ---")
    collection = Collection(MILVUS_COLLECTION)
    collection.load()
    
    query = "我国核电步入建设投运双高峰的第一年是哪一年？"
    results = collection.search(
        data=[embeddings.embed_query(query)], anns_field="embedding",
        param={"metric_type": "L2", "params": {"ef": 64}}, limit=3, expr="is_child == True",
        output_fields=["content", "source", "page"]
    )
    
    for hit in results[0]:
        print(f"内容: {hit.entity.get('content', '')[:100]}...")
        print(f"来源: {hit.entity.get('source')}, 页码: {hit.entity.get('page')}")

if __name__ == "__main__":
    docs = load_docs()
    if docs:
        children, summaries = split_docs(docs)
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={"device": "cuda"}, encode_kwargs={"normalize_embeddings": True})
        build_index(children + summaries, embeddings)
        verify(embeddings)
        print("\n完成")
