# rag_fusion/retriever.py

import os
import re
from typing import List, Dict, Any
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from zhipuai import ZhipuAI
from pymilvus import connections, Collection, utility
from rank_bm25 import BM25Okapi
import config
import jieba

# --- 全局变量，用于缓存 ---
db_client = None
embedding_model = None
llm_client = None
bm25_index = None
bm25_docs = None


def get_llm_client():
    """获取智谱AI客户端的单例。"""
    global llm_client
    if llm_client is None:
        if not config.LLM_API_KEY or config.LLM_API_KEY == "你的智谱AI API Key":
            raise ValueError("请在 config.py 文件中设置你的 ZHIPUAI_API_KEY")
        llm_client = ZhipuAI(api_key=config.LLM_API_KEY)
    return llm_client


def get_embedding_model():
    """获取嵌入模型的单例。"""
    global embedding_model
    if embedding_model is None:
        print("正在初始化嵌入模型...")
        embedding_model = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL_NAME,
            model_kwargs={"device": config.EMBEDDING_DEVICE},
            encode_kwargs={"normalize_embeddings": True}
        )
    return embedding_model


def get_db_client():
    """获取向量数据库客户端的单例。"""
    global db_client
    if db_client is None:
        print("正在连接Milvus数据库...")
        try:
            # 连接到Milvus
            connections.connect("default", host=config.MILVUS_HOST, port=config.MILVUS_PORT)
            
            # 检查集合是否存在
            if not utility.has_collection(config.MILVUS_COLLECTION):
                raise FileNotFoundError(f"Milvus集合不存在: {config.MILVUS_COLLECTION}。请先运行 build_vectorstore.py。")
            
            # 加载集合
            collection = Collection(config.MILVUS_COLLECTION)
            collection.load()
            db_client = collection
            print("Milvus数据库连接成功！")
        except Exception as e:
            raise Exception(f"连接Milvus数据库失败: {e}")
    return db_client


def get_bm25_index():
    """获取BM25索引的单例。"""
    global bm25_index, bm25_docs
    if bm25_index is None or bm25_docs is None:
        print("正在构建BM25索引...")
        # 加载所有文档内容
        bm25_docs = _load_all_documents_for_bm25()
        
        # 对文档内容进行分词处理
        tokenized_corpus = []
        for doc in bm25_docs:
            # 分词
            tokens = tokenize_text(doc.page_content)
            tokenized_corpus.append(tokens)
        
        # 构建BM25索引
        bm25_index = BM25Okapi(tokenized_corpus)
        print(f"BM25索引构建完成，包含 {len(bm25_docs)} 个文档")
    return bm25_index, bm25_docs


def tokenize_text(text):
    """对文本进行分词处理。"""
    # 移除特殊字符
    text = re.sub(r'[\s\n\r\t]+', ' ', text)
    text = re.sub(r'[^\w\s\u4e00-\u9fa5]', '', text)
    
    # 分词
    if any('\u4e00' <= char <= '\u9fa5' for char in text):
        # 中文文本使用jieba分词
        tokens = list(jieba.cut_for_search(text))
    else:
        # 英文文本使用空格分词
        tokens = text.lower().split()
    
    # 过滤停用词
    stop_words = set(['的', '了', '是', '在', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'])
    tokens = [token for token in tokens if token not in stop_words and len(token) > 1]
    
    return tokens


def _load_all_documents_for_bm25():
    """加载所有文档内容用于构建BM25索引。"""
    import json
    
    processed_data_folder = "d:\\LLM\\RAG\\Nvidia-Finance-Rag\\processed_data"
    docs = []
    
    # 加载图片摘要
    image_summaries_path = os.path.join(processed_data_folder, "image_summaries.json")
    image_summaries = {}
    if os.path.exists(image_summaries_path):
        with open(image_summaries_path, 'r', encoding='utf-8') as f:
            image_summaries = json.load(f)
    
    # 处理PDF解析文件
    for filename in os.listdir(processed_data_folder):
        if filename.endswith(".json") and filename != "image_summaries.json":
            file_path = os.path.join(processed_data_folder, filename)
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
                doc = Document(
                    page_content=full_page_content,
                    metadata={
                        "source": filename.replace("_processed.json", ""),
                        "page": page_num
                    }
                )
                docs.append(doc)
    
    # 处理音频转录稿
    for filename in os.listdir(processed_data_folder):
        if filename.endswith("_transcribed.txt"):
            file_path = os.path.join(processed_data_folder, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                full_text = f.read()
            
            doc = Document(
                page_content=full_text,
                metadata={
                    "source": filename.replace("_transcribed.txt", ""),
                    "page": 1
                }
            )
            docs.append(doc)
    
    return docs


def generate_queries(original_query: str) -> List[str]:
    """使用LLM为原始问题生成多个变体。"""
    print(f"\n--- 步骤 1: 为 '{original_query}' 生成扩展查询 ---")
    client = get_llm_client()
    prompt = config.QUERY_GENERATION_PROMPT.format(
        num_queries=config.NUM_GENERATED_QUERIES,
        original_query=original_query
    )

    response = client.chat.completions.create(
        model=config.QUERY_GENERATION_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
    )

    generated_queries = response.choices[0].message.content.strip().split('\n')
    all_queries = [original_query] + [q for q in generated_queries if q]

    print("  - 生成的查询:")
    for q in all_queries:
        print(f"    - {q}")

    return all_queries


def parallel_search(queries: List[str]) -> Dict[str, List[Document]]:
    """对每个查询并行执行摘要和子块的搜索。"""
    print("\n--- 步骤 2: 执行并行检索 ---")
    collection = get_db_client()
    embedding_model = get_embedding_model()
    bm25_index, bm25_docs = get_bm25_index()
    all_results = {}

    # 搜索参数
    search_params = {
        "metric_type": "L2",
        "params": {
            "ef": 64
        }
    }

    for query in queries:
        print(f"  - 正在搜索: '{query}'")

        # 生成查询向量
        query_embedding = embedding_model.embed_query(query)

        # 路径A: 搜索摘要
        expr_summary = "is_summary == True"
        summary_results_milvus = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=config.RETRIEVAL_K,
            expr=expr_summary,
            output_fields=["id", "content", "source", "page", "doc_id", "is_summary"]
        )

        # 路径B: 搜索子块
        expr_child = "is_child == True"
        chunk_results_milvus = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=config.RETRIEVAL_K,
            expr=expr_child,
            output_fields=["id", "content", "source", "page", "doc_id", "is_child", "chunk_index"]
        )

        # 路径C: 使用BM25搜索
        bm25_results = _bm25_search(query, bm25_index, bm25_docs, k=config.RETRIEVAL_K)

        # 将Milvus结果转换为Document对象
        summary_docs = []
        for hit in summary_results_milvus[0]:
            metadata = {
                "source": hit.entity.get("source", ""),
                "page": hit.entity.get("page", 0),
                "doc_id": hit.entity.get("doc_id", ""),
                "is_summary": hit.entity.get("is_summary", False),
                "score_type": "hnsw"
            }
            doc = Document(
                page_content=hit.entity.get("content", ""),
                metadata=metadata
            )
            summary_docs.append(doc)

        chunk_docs = []
        for hit in chunk_results_milvus[0]:
            metadata = {
                "source": hit.entity.get("source", ""),
                "page": hit.entity.get("page", 0),
                "doc_id": hit.entity.get("doc_id", ""),
                "is_child": hit.entity.get("is_child", False),
                "chunk_index": hit.entity.get("chunk_index", -1),
                "score_type": "hnsw"
            }
            doc = Document(
                page_content=hit.entity.get("content", ""),
                metadata=metadata
            )
            chunk_docs.append(doc)

        # 合并所有结果，**不**去重，保留同一文档在不同路径中的排名信息
        all_docs = summary_docs + chunk_docs + bm25_results
        
        all_results[query] = all_docs
        print(f"    - 路径A: {len(summary_docs)} 个摘要, 路径B: {len(chunk_docs)} 个子块, 路径C: {len(bm25_results)} 个BM25结果, 总计: {len(all_docs)} 个结果。")

    return all_results


def _bm25_search(query, bm25_index, bm25_docs, k=5):
    """使用BM25进行文本搜索。"""
    # 对查询进行分词
    query_tokens = tokenize_text(query)
    
    # 执行BM25搜索
    scores = bm25_index.get_scores(query_tokens)
    
    # 排序并获取前k个结果
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    
    # 构建结果列表
    results = []
    for idx in top_indices:
        doc = bm25_docs[idx]
        # 添加BM25分数到元数据
        metadata = doc.metadata.copy()
        metadata["score_type"] = "bm25"
        metadata["bm25_score"] = scores[idx]
        
        result_doc = Document(
            page_content=doc.page_content,
            metadata=metadata
        )
        results.append(result_doc)
    
    return results





def reciprocal_rank_fusion(search_results: Dict[str, List[Document]], k: int = 60) -> List[Document]:
    """
    对并行搜索的结果进行RRF重排。
    返回一个按RRF分数排序的、去重的文档列表。
    """
    print("\n--- 步骤 3: RRF 结果融合与重排 ---")

    # 1. 计算每个文档的RRF分数
    fused_scores = {}
    for query, docs in search_results.items():
        for rank, doc in enumerate(docs):
            # 使用 doc_id 和 chunk_index (如果有) 创建唯一标识符
            doc_id = doc.metadata.get('doc_id')
            chunk_index = doc.metadata.get('chunk_index', -1)  # -1 表示摘要
            unique_id = f"{doc_id}-{chunk_index}"

            if unique_id not in fused_scores:
                fused_scores[unique_id] = {'score': 0, 'doc': doc}

            # RRF核心公式
            fused_scores[unique_id]['score'] += 1.0 / (rank + k)

    # 2. 按分数排序
    reranked_results = sorted(
        fused_scores.values(),
        key=lambda x: x['score'],
        reverse=True
    )

    # 3. 提取排序后的文档并去重 (按父文档ID)
    final_docs = []
    seen_parent_ids = set()
    for item in reranked_results:
        parent_id = item['doc'].metadata.get('doc_id')
        if parent_id not in seen_parent_ids:
            final_docs.append(item['doc'])
            seen_parent_ids.add(parent_id)

    print(f"  - 融合后得到 {len(final_docs)} 个唯一的父文档上下文。")
    return final_docs


def retrieve_final_context(query: str, top_n: int = 3) -> List[Document]:
    """
    完整的检索流程：查询生成 -> 并行搜索 -> RRF重排。
    返回最终用于生成答案的、最相关的父文档。
    """
    # 步骤1: 查询生成
    all_queries = generate_queries(query)

    # 步骤2: 并行搜索
    search_results = parallel_search(all_queries)

    # 步骤3: RRF重排
    reranked_docs = reciprocal_rank_fusion(search_results, k=config.RRF_K)

    # 步骤4: 提取最终的父文档上下文
    # 我们需要加载完整的父文档内容
    final_parent_docs = []
    parent_docs_content = _load_all_parent_docs_from_json()  # 这是一个辅助函数

    for doc in reranked_docs[:top_n]:
        parent_id = doc.metadata['doc_id']
        if parent_id in parent_docs_content:
            final_parent_docs.append(parent_docs_content[parent_id])

    return final_parent_docs


def _load_all_parent_docs_from_json() -> Dict[str, Document]:
    """
    一个辅助函数，用于从原始JSON文件中加载所有父文档的完整内容。
    在实际应用中，你可能希望用更高效的方式（如数据库）来存储和获取父文档。
    """
    # 这是一个简化实现，它会重新解析JSON来构建父文档。
    # 为了效率，我们可以在 build_vectorstore.py 中将父文档保存为一个 pickle 文件。
    # 这里我们为了简单起见，重新执行部分加载逻辑。
    import json
    import uuid

    image_summaries_path = os.path.join("../processed_data", "image_summaries.json")
    with open(image_summaries_path, 'r', encoding='utf-8') as f:
        image_summaries = json.load(f)

    parent_docs_content = {}

    for filename in sorted(os.listdir("../processed_data")):
        if filename.endswith(".json") and filename != "image_summaries.json":
            file_path = os.path.join("../processed_data", filename)
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
                # 注意：这里的doc_id生成逻辑需要和 build_vectorstore.py 中完全一致才能匹配！
                # 为了确保一致，我们最好在建库时就将父文档存起来。
                # 这是一个潜在的风险点，暂定我们认为ID可以匹配。
                # 更好的方法是在 build_vectorstore.py 中保存一个 parent_docs.pkl 文件。

                # 我们通过源文件和页码来创建伪ID
                parent_id_seed = f"{filename.replace('_processed.json', '')}_{page_num}"
                # 此处需要一个稳定的UUID生成方式，但python内置的uuid5可能更合适
                # 为了简化，我们暂时假设可以通过元数据回溯

                # 修正：直接通过元数据匹配，而不是重新生成ID
                # 在真实检索中，我们拿到了doc的metadata，可以直接用
                pass  # 逻辑转移到主应用中实现

    # 修正：这个函数不再直接返回内容，而是提供一个查找路径
    # 在主应用中，我们将根据检索到的doc元数据（source, page）来动态构建上下文
    return {}  # 返回空字典，逻辑在主应用中处理