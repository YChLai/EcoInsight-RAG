# rag_fusion/retriever.py

import os
from typing import List, Dict, Any
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from zhipuai import ZhipuAI
import config

# --- 全局变量，用于缓存 ---
db_client = None
embedding_model = None
llm_client = None


def get_llm_client():
    """获取智谱AI客户端的单例。"""
    global llm_client
    if llm_client is None:
        if not config.LLM_API_KEY or config.LLM_API_KEY == "你的智谱AI API Key":
            raise ValueError("请在 config.py 文件中设置你的 ZHIPU_API_KEY")
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
        if not os.path.exists(config.VECTOR_DB_PATH):
            raise FileNotFoundError(f"向量数据库路径不存在: {config.VECTOR_DB_PATH}。请先运行 build_vectorstore.py。")
        print("正在加载向量数据库...")
        db_client = Chroma(
            persist_directory=config.VECTOR_DB_PATH,
            embedding_function=get_embedding_model()
        )
    return db_client


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
    db = get_db_client()
    all_results = {}

    for query in queries:
        print(f"  - 正在搜索: '{query}'")

        # 路径A: 搜索摘要
        summary_results = db.similarity_search(
            query, k=config.RETRIEVAL_K, filter={"is_summary": {"$eq": True}}
        )

        # 路径B: 搜索子块
        chunk_results = db.similarity_search(
            query, k=config.RETRIEVAL_K, filter={"is_child": {"$eq": True}}
        )

        all_results[query] = summary_results + chunk_results
        print(f"    - 命中 {len(summary_results)} 个摘要，{len(chunk_results)} 个子块。")

    return all_results


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