import os
import json
import shutil
import uuid
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from zhipuai import ZhipuAI
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

# --- 配置 ---
# 使用绝对路径
PROCESSED_DATA_FOLDER = "d:\\LLM\\RAG\\Nvidia-Finance-Rag\\processed_data"
# Milvus配置
MILVUS_HOST = "localhost"
MILVUS_PORT = 19530
MILVUS_COLLECTION = "environmental_data_rag"
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"

# --- 初始化 LLM 用于生成摘要 ---
try:
    client = ZhipuAI(api_key=os.getenv("ZHIPUAI_API_KEY"))
    print("智谱AI客户端初始化成功，将用于生成摘要。")
except Exception as e:
    client = None
    print(f"智谱AI客户端初始化失败: {e}。摘要生成功能将不可用。")


def generate_summary_with_llm(content):
    """使用LLM为文本块生成摘要。"""
    if not client:
        return "摘要功能未启用。"
    if len(content) < 500:
        return content
    prompt = f"""你是一个专业的环境数据分析师。请为以下环境报告内容生成一个简洁、精确、信息密集的摘要，不超过150个字。摘要需要捕捉最关键的实体、数据和结论。\n\n内容如下：\n---\n{content}\n---\n\n摘要："""
    try:
        response = client.chat.completions.create(
            model="glm-4-flash",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300, temperature=0.0
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"  - 调用LLM生成摘要时出错: {e}")
        return "摘要生成失败。"


def load_and_prepare_docs():
    """
    加载所有处理好的JSON和TXT文件，创建父文档。
    """
    print("--- 步骤一: 加载并准备父文档 (已修正) ---")

    image_summaries_path = os.path.join(PROCESSED_DATA_FOLDER, "image_summaries.json")
    if os.path.exists(image_summaries_path):
        with open(image_summaries_path, 'r', encoding='utf-8') as f:
            image_summaries = json.load(f)
        print(f"  - 已加载 {len(image_summaries)} 条图片摘要。")
    else:
        image_summaries = {}
        print("  - 未找到图片摘要文件，将继续处理。")

    parent_docs = []
    for filename in sorted(os.listdir(PROCESSED_DATA_FOLDER)):
        file_path = os.path.join(PROCESSED_DATA_FOLDER, filename)

        # --- 新增逻辑：同时处理 .txt 文件 ---
        if filename.endswith("_transcribed.txt"):
            print(f"  - 正在处理音频转录稿: {filename}")
            with open(file_path, 'r', encoding='utf-8') as f:
                full_text = f.read()

            # 将整个转录稿视为一个大的父文档
            doc_id = str(uuid.uuid4())
            doc = Document(
                page_content=full_text,
                metadata={
                    "source": filename.replace("_transcribed.txt", ""),
                    "page": 1,  # 对于转录稿，我们将其视为单页
                    "doc_id": doc_id
                }
            )
            parent_docs.append(doc)

        # --- 原有逻辑：处理 .json 文件 ---
        elif filename.endswith(".json") and filename != "image_summaries.json":
            print(f"  - 正在处理PDF解析文件: {filename}")
            with open(file_path, 'r', encoding='utf-8') as f:
                blocks = json.load(f)

            pages = {}
            for block in blocks:
                page_num = block["metadata"]["page"]
                if page_num not in pages:
                    pages[page_num] = []

                content = block["content"]
                if block["type"] == "image_placeholder":
                    img_name = content.replace("[IMAGE: ", "").replace("]", "")
                    summary = image_summaries.get(img_name, "")
                    content = f"--- [参考图片: {img_name}] ---\n{summary}\n--- [图片描述结束] ---"

                pages[page_num].append(content)

            for page_num, contents in sorted(pages.items()):
                full_page_content = "\n\n".join(contents)
                doc_id = str(uuid.uuid4())
                doc = Document(
                    page_content=full_page_content,
                    metadata={
                        "source": filename.replace("_processed.json", ""),
                        "page": page_num,
                        "doc_id": doc_id
                    }
                )
                parent_docs.append(doc)

    print(f"  - 总共创建了 {len(parent_docs)} 个父文档 (来自PDF和TXT)。")
    return parent_docs


def create_child_chunks_and_summaries(parent_docs):
    """
    为父文档创建子块和摘要。
    """
    print("\n--- 步骤二: 创建子块与摘要 ---")

    child_chunks = []
    summary_docs = []

    child_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

    for i, parent_doc in enumerate(parent_docs):
        doc_id = parent_doc.metadata["doc_id"]
        source_info = f"{parent_doc.metadata['source']} (页码 {parent_doc.metadata['page']})"
        print(f"  - 正在处理父文档 ({i + 1}/{len(parent_docs)}): {source_info}")

        chunks = child_splitter.split_text(parent_doc.page_content)
        for j, chunk_text in enumerate(chunks):
            chunk_doc = Document(
                page_content=chunk_text,
                metadata={**parent_doc.metadata, "is_child": True, "chunk_index": j}
            )
            child_chunks.append(chunk_doc)

        summary_text = generate_summary_with_llm(parent_doc.page_content)
        summary_doc = Document(
            page_content=summary_text,
            metadata={**parent_doc.metadata, "is_summary": True}
        )
        summary_docs.append(summary_doc)

    print(f"  - 已创建 {len(child_chunks)} 个子块。")
    print(f"  - 已创建 {len(summary_docs)} 个摘要。")

    return child_chunks, summary_docs


def create_and_persist_vector_store(docs_to_index, embeddings):
    """创建并持久化向量库。"""
    print("\n--- 步骤三: 嵌入数据并构建向量库 ---")
    
    # 连接到Milvus
    print(f"  - 连接到Milvus: {MILVUS_HOST}:{MILVUS_PORT}")
    connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
    
    # 检查集合是否存在，如果存在则删除
    if utility.has_collection(MILVUS_COLLECTION):
        print(f"  - 发现旧的集合，正在删除: {MILVUS_COLLECTION}")
        utility.drop_collection(MILVUS_COLLECTION)
    
    # 创建字段
    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=40, is_primary=True),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=255),
        FieldSchema(name="page", dtype=DataType.INT32),
        FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=36),
        FieldSchema(name="is_child", dtype=DataType.BOOL, default_value=False),
        FieldSchema(name="is_summary", dtype=DataType.BOOL, default_value=False),
        FieldSchema(name="chunk_index", dtype=DataType.INT64, default_value=-1),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024)  # BAAI/bge-m3 模型的维度
    ]
    
    # 创建集合
    schema = CollectionSchema(fields, description="环境数据RAG向量库")
    collection = Collection(MILVUS_COLLECTION, schema)
    print(f"  - 集合创建成功: {MILVUS_COLLECTION}")
    
    # 准备数据
    print("  - 正在处理文档并生成嵌入向量...")
    data = []
    for doc in docs_to_index:
        # 生成唯一ID
        doc_id = doc.metadata.get("doc_id", str(uuid.uuid4()))
        chunk_index = doc.metadata.get("chunk_index", -1)
        if chunk_index != -1:
            # 使用前32个字符的doc_id加上4位chunk_index
            short_doc_id = doc_id.replace("-", "")[:32]
            unique_id = f"{short_doc_id}{chunk_index:04d}"
            # 确保长度不超过36个字符
            unique_id = unique_id[:36]
        else:
            unique_id = doc_id
        
        # 生成嵌入向量
        embedding = embeddings.embed_query(doc.page_content)
        
        # 提取元数据
        source = doc.metadata.get("source", "")
        page = doc.metadata.get("page", 0)
        is_child = doc.metadata.get("is_child", False)
        is_summary = doc.metadata.get("is_summary", False)
        
        # 添加到数据列表
        data.append({
            "id": unique_id,
            "content": doc.page_content,
            "source": source,
            "page": page,
            "doc_id": doc_id,
            "is_child": is_child,
            "is_summary": is_summary,
            "chunk_index": chunk_index,
            "embedding": embedding
        })
    
    # 批量插入数据
    print(f"  - 正在插入 {len(data)} 条数据...")
    collection.insert([
        [item["id"] for item in data],
        [item["content"] for item in data],
        [item["source"] for item in data],
        [item["page"] for item in data],
        [item["doc_id"] for item in data],
        [item["is_child"] for item in data],
        [item["is_summary"] for item in data],
        [item["chunk_index"] for item in data],
        [item["embedding"] for item in data]
    ])
    
    # 创建索引
    print("  - 正在创建索引...")
    index_params = {
        "index_type": "HNSW",
        "metric_type": "L2",
        "params": {
            "M": 8,
            "efConstruction": 64
        }
    }
    collection.create_index("embedding", index_params)
    
    # 加载集合到内存
    collection.load()
    print("  - 向量数据库创建完毕并已成功加载到内存！")
    
    return collection


def verify_retrieval(embeddings):
    """验证检索效果。"""
    print("\n--- 步骤四: 验证检索效果 ---")
    
    # 连接到Milvus
    try:
        connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
    except Exception as e:
        print(f"连接Milvus失败: {e}")
        return
    
    # 检查集合是否存在
    if not utility.has_collection(MILVUS_COLLECTION):
        print("集合不存在，无法验证。")
        return
    
    # 加载集合
    collection = Collection(MILVUS_COLLECTION)
    collection.load()
    
    # 验证一个来自音频转录稿的查询
    query = "are there other that have yet to be announced of the same kind of scale and magnitude?"
    print(f"  - 模拟音频内容查询: '{query}'")
    
    # 生成查询向量
    query_embedding = embeddings.embed_query(query)
    
    # 执行相似性搜索
    search_params = {
        "metric_type": "L2",
        "params": {
            "ef": 64
        }
    }
    
    # 构建过滤条件
    expr = "is_child == True"
    
    # 执行搜索
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=3,
        expr=expr,
        output_fields=["id", "content", "source", "page", "doc_id", "is_child", "chunk_index"]
    )
    
    print("\n  - 检索到的前3个精确块为：\n")
    if not results or not results[0]:
        print("    未能检索到任何相关内容。")
    else:
        for i, hit in enumerate(results[0]):
            content = hit.entity.get("content", "")
            source = hit.entity.get("source", "")
            page = hit.entity.get("page", 0)
            doc_id = hit.entity.get("doc_id", "")
            is_child = hit.entity.get("is_child", False)
            chunk_index = hit.entity.get("chunk_index", -1)
            
            print(f"    内容片段: {content[:150].replace('\n', ' ')}...")
            print(f"      元数据: {{'source': '{source}', 'page': {page}, 'doc_id': '{doc_id}', 'is_child': {is_child}, 'chunk_index': {chunk_index}}}")
            print(f"      相似度: {hit.distance:.4f}")
            print("      " + "-" * 20)


if __name__ == "__main__":
    # 1. 加载并准备父文档
    parent_documents = load_and_prepare_docs()

    if parent_documents:
        # 2. 创建子块和摘要
        child_chunks, summary_docs = create_child_chunks_and_summaries(parent_documents)
        all_docs_to_index = child_chunks + summary_docs

        # 3. 嵌入与存储
        print("\n正在初始化嵌入模型...")
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cuda"},
            encode_kwargs={"normalize_embeddings": True}
        )
        create_and_persist_vector_store(all_docs_to_index, embeddings)

        # 4. 验证
        verify_retrieval(embeddings)
        print(f"\n🎉 恭喜！数据处理与索引流程已全部修复并完成！")
    else:
        print("未能加载任何文档，流程终止。")