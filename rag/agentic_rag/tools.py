import os
import re
import json
import sys
import torch
import jieba
from typing import List, Dict, Union

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_community.embeddings import HuggingFaceEmbeddings
from zhipuai import ZhipuAI
from pymilvus import connections, Collection, utility
from rank_bm25 import BM25Okapi
from modelscope import AutoModelForCausalLM, AutoTokenizer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import config
DATA_DIR = os.path.join(BASE_DIR, "../processed_data")

llm = None
embeddings = None
db = None
bm25 = None
bm25_docs = None
reranker = None
tokenizer = None

RERANKER_PATH = r"C:\Users\LYC\.cache\modelscope\hub\models\qwen\Qwen3-Reranker-0.6B"

def get_llm():
    global llm
    if llm is None:
        llm = ZhipuAI(api_key=config.LLM_API_KEY)
    return llm

def get_embeddings():
    global embeddings
    if embeddings is None:
        embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={"device": config.DEVICE},
            encode_kwargs={"normalize_embeddings": True}
        )
    return embeddings

def get_db():
    global db
    if db is None:
        connections.connect("default", host=config.MILVUS_HOST, port=config.MILVUS_PORT)
        if not utility.has_collection(config.MILVUS_COLLECTION):
            raise FileNotFoundError(f"集合不存在: {config.MILVUS_COLLECTION}")
        db = Collection(config.MILVUS_COLLECTION)
        db.load()
    return db

def get_reranker():
    global reranker, tokenizer
    if reranker is None:
        tokenizer = AutoTokenizer.from_pretrained(RERANKER_PATH, trust_remote_code=True)
        reranker = AutoModelForCausalLM.from_pretrained(RERANKER_PATH, trust_remote_code=True)
        reranker.eval()
    return reranker, tokenizer

def get_bm25():
    global bm25, bm25_docs
    if bm25 is None:
        bm25_docs = _load_bm25_docs()
        tokenized = [tokenize(d.page_content) for d in bm25_docs]
        bm25 = BM25Okapi(tokenized)
    return bm25, bm25_docs

def tokenize(text):
    text = re.sub(r'[\s\n\r\t]+', ' ', text)
    text = re.sub(r'[^\w\s\u4e00-\u9fa5]', '', text)
    tokens = list(jieba.cut_for_search(text)) if any('\u4e00' <= c <= '\u9fa5' for c in text) else text.lower().split()
    stop = set(['的', '了', '是', '在', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'])
    return [t for t in tokens if t not in stop and len(t) > 1]

def _load_bm25_docs():
    docs = []
    img_file = os.path.join(DATA_DIR, "images.json")
    imgs = json.load(open(img_file, 'r', encoding='utf-8')) if os.path.exists(img_file) else {}

    for f in sorted(os.listdir(DATA_DIR)):
        path = os.path.join(DATA_DIR, f)
        if f.endswith(".json") and f != "images.json":
            blocks = json.load(open(path, 'r', encoding='utf-8'))
            pages = {}
            for b in blocks:
                p = b["metadata"]["page"]
                pages.setdefault(p, [])
                c = b["content"]
                if b["type"] == "image":
                    name = c.replace("[IMAGE: ", "").replace("]", "")
                    c = f"--- [图片: {name}] ---\n{imgs.get(name, '')}\n---"
                pages[p].append(c)
            for p, contents in sorted(pages.items()):
                docs.append(Document(page_content="\n\n".join(contents), metadata={"source": f.replace(".json", ""), "page": p}))
        elif f.endswith(".txt"):
            docs.append(Document(page_content=open(path, 'r', encoding='utf-8').read(), metadata={"source": f.replace(".txt", ""), "page": 1}))
    return docs

def _load_raw_data():
    data = {}
    for f in os.listdir(DATA_DIR):
        path = os.path.join(DATA_DIR, f)
        if f.endswith(".json"):
            data[f.replace(".json", "")] = {'type': 'json', 'content': json.load(open(path, 'r', encoding='utf-8'))}
        elif f.endswith(".txt"):
            data[f.replace(".txt", "")] = {'type': 'txt', 'content': open(path, 'r', encoding='utf-8').read()}
    return data

def _find_content(data, source, page):
    src = source.replace(".pdf", "").replace(".json", "")
    if src not in data:
        return ""
    d = data[src]
    if d['type'] == 'txt':
        return d['content']
    return "\n\n".join([b['content'] for b in d['content'] if b['metadata'].get('page') == page])

def _expand_query(query: str) -> List[str]:
    client = get_llm()
    prompt = config.EXPAND_PROMPT.format(n=3, q=query)
    resp = client.chat.completions.create(model=config.LLM_MODEL, messages=[{"role": "user", "content": prompt}])
    generated = resp.choices[0].message.content.strip().split('\n')
    queries = [query] + [q for q in generated if q]
    return queries

def _vector_search(query: str, top_k: int = 5) -> List[Document]:
    collection = get_db()
    emb = get_embeddings()
    q_emb = emb.embed_query(query)

    summary_hits = collection.search(
        data=[q_emb],
        anns_field="embedding",
        param={"metric_type": "L2", "params": {"ef": 64}},
        limit=top_k,
        expr="is_summary == True",
        output_fields=["content", "source", "page", "doc_id"]
    )[0]

    chunk_hits = collection.search(
        data=[q_emb],
        anns_field="embedding",
        param={"metric_type": "L2", "params": {"ef": 64}},
        limit=top_k,
        expr="is_child == True",
        output_fields=["content", "source", "page", "doc_id", "chunk_index"]
    )[0]

    summary_docs = [
        Document(
            page_content=h.entity.get("content", ""),
            metadata={"source": h.entity.get("source", ""), "page": h.entity.get("page", 0), "doc_id": h.entity.get("doc_id", "")}
        ) for h in summary_hits
    ]
    chunk_docs = [
        Document(
            page_content=h.entity.get("content", ""),
            metadata={"source": h.entity.get("source", ""), "page": h.entity.get("page", 0), "doc_id": h.entity.get("doc_id", ""), "chunk_index": h.entity.get("chunk_index", -1)}
        ) for h in chunk_hits
    ]

    return summary_docs + chunk_docs

def _bm25_search(query: str, top_k: int = 5) -> List[Document]:
    bm25_idx, bm25_d = get_bm25()
    scores = bm25_idx.get_scores(tokenize(query))
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [Document(page_content=bm25_d[i].page_content, metadata={**bm25_d[i].metadata, "bm25_score": scores[i]}) for i in top]

def _rerank_documents(query: str, docs: List[Document]) -> List[Document]:
    if not docs:
        return []

    model, tok = get_reranker()
    scores = {}

    for d in docs:
        try:
            inputs = tok([query], [d.page_content[:1000]], return_tensors="pt", padding=True)
            with torch.no_grad():
                out = model(**inputs)
                score = out.logits.mean().item() if hasattr(out, 'logits') else 0.0
        except:
            score = 0.0
        uid = f"{d.metadata.get('doc_id')}-{d.metadata.get('chunk_index', -1)}"
        scores.setdefault(uid, {'score': 0, 'doc': d})['score'] += score

    ranked = sorted(scores.values(), key=lambda x: x['score'], reverse=True)
    result, seen = [], set()
    for item in ranked:
        pid = item['doc'].metadata.get('doc_id')
        if pid not in seen:
            result.append(item['doc'])
            seen.add(pid)
    return result

@tool
def vector_search_tool(query: str) -> str:
    """
    使用向量检索从知识库中搜索相关文档。
    适用于：需要查找与问题语义相关的文档内容。

    参数:
        query: 用户的问题或查询关键词

    返回:
        格式化的检索结果，包含来源和页码信息
    """
    print(f"\n--- 向量检索 ---")
    print(f"  查询: {query}")

    try:
        docs = _vector_search(query, top_k=config.TOP_K)
        data = _load_raw_data()

        results = []
        for i, doc in enumerate(docs):
            src = doc.metadata.get('source', '未知')
            page = doc.metadata.get('page', '未知')
            content = _find_content(data, src, page)
            results.append({
                "index": i + 1,
                "source": src,
                "page": page,
                "content": content[:1000] if content else doc.page_content[:1000]
            })

        if not results:
            return "未找到相关文档"

        output = "向量检索结果:\n"
        for r in results:
            output += f"\n[{r['index']}] 来源: {r['source']}, 第 {r['page']} 页\n"
            output += f"内容: {r['content'][:500]}...\n"

        return output
    except Exception as e:
        return f"检索出错: {str(e)}"

@tool
def bm25_search_tool(query: str) -> str:
    """
    使用BM25算法从知识库中搜索相关文档。
    适用于：需要精确关键词匹配或术语匹配的场景。

    参数:
        query: 用户的问题或查询关键词

    返回:
        格式化的检索结果，包含来源和页码信息
    """
    print(f"\n--- BM25检索 ---")
    print(f"  查询: {query}")

    try:
        docs = _bm25_search(query, top_k=config.TOP_K)
        data = _load_raw_data()

        results = []
        for i, doc in enumerate(docs):
            src = doc.metadata.get('source', '未知')
            page = doc.metadata.get('page', '未知')
            content = _find_content(data, src, page)
            results.append({
                "index": i + 1,
                "source": src,
                "page": page,
                "content": content[:1000] if content else doc.page_content[:1000]
            })

        if not results:
            return "未找到相关文档"

        output = "BM25检索结果:\n"
        for r in results:
            output += f"\n[{r['index']}] 来源: {r['source']}, 第 {r['page']} 页\n"
            output += f"内容: {r['content'][:500]}...\n"

        return output
    except Exception as e:
        return f"检索出错: {str(e)}"

@tool
def expand_query_tool(query: str) -> str:
    """
    使用LLM将用户问题扩展为多个不同角度的查询。
    适用于：问题比较简短或需要多角度检索时。

    参数:
        query: 原始用户问题

    返回:
        扩展后的查询列表
    """
    print(f"\n--- 查询扩展 ---")
    print(f"  原始: {query}")

    try:
        expanded = _expand_query(query)
        output = "扩展后的查询:\n"
        for i, q in enumerate(expanded):
            output += f"{i+1}. {q}\n"
        return output
    except Exception as e:
        return f"查询扩展出错: {str(e)}"

@tool
def hybrid_search_tool(query: str) -> str:
    """
    使用混合检索策略（向量检索 + BM25）并重排结果。
    适用于：需要综合多种检索方法获得最佳结果时。

    参数:
        query: 用户的问题或查询关键词

    返回:
        格式化的检索结果，包含来源和页码信息
    """
    print(f"\n--- 混合检索 + 重排 ---")
    print(f"  查询: {query}")

    try:
        expanded_queries = _expand_query(query)
        all_docs = []

        for q in expanded_queries:
            vec_docs = _vector_search(q, top_k=config.TOP_K)
            bm25_docs = _bm25_search(q, top_k=config.TOP_K)
            all_docs.extend(vec_docs + bm25_docs)

        unique_docs = {}
        for d in all_docs:
            uid = f"{d.metadata.get('doc_id')}-{d.metadata.get('chunk_index', -1)}"
            if uid not in unique_docs:
                unique_docs[uid] = d

        docs_list = list(unique_docs.values())
        reranked_docs = _rerank_documents(query, docs_list)

        data = _load_raw_data()
        results = []
        for i, doc in enumerate(reranked_docs[:5]):
            src = doc.metadata.get('source', '未知')
            page = doc.metadata.get('page', '未知')
            content = _find_content(data, src, page)
            results.append({
                "index": i + 1,
                "source": src,
                "page": page,
                "content": content[:1000] if content else doc.page_content[:1000]
            })

        if not results:
            return "未找到相关文档"

        output = "混合检索结果（重排后）:\n"
        for r in results:
            output += f"\n[{r['index']}] 来源: {r['source']}, 第 {r['page']} 页\n"
            output += f"内容: {r['content'][:500]}...\n"

        return output
    except Exception as e:
        return f"混合检索出错: {str(e)}"

@tool
def get_context_for_answer(query: str) -> str:
    """
    获取用于回答问题的完整上下文信息。
    适用于：准备回答问题时需要获取完整的上下文。

    参数:
        query: 用户的问题

    返回:
        格式化的上下文信息，包含所有检索到的内容
    """
    print(f"\n--- 获取回答上下文 ---")
    print(f"  查询: {query}")

    try:
        expanded_queries = _expand_query(query)
        all_docs = []

        for q in expanded_queries:
            vec_docs = _vector_search(q, top_k=config.TOP_K)
            bm25_docs = _bm25_search(q, top_k=config.TOP_K)
            all_docs.extend(vec_docs + bm25_docs)

        unique_docs = {}
        for d in all_docs:
            uid = f"{d.metadata.get('doc_id')}-{d.metadata.get('chunk_index', -1)}"
            if uid not in unique_docs:
                unique_docs[uid] = d

        docs_list = list(unique_docs.values())
        reranked_docs = _rerank_documents(query, docs_list)

        data = _load_raw_data()
        context_parts = []

        for doc in reranked_docs[:3]:
            src = doc.metadata.get('source', '未知')
            page = doc.metadata.get('page', '未知')
            content = _find_content(data, src, page)
            if content:
                context_parts.append(f"来源: {src}, 第 {page} 页\n{content}")

        if not context_parts:
            return "无法获取上下文"

        return "\n\n---\n\n".join(context_parts)
    except Exception as e:
        return f"获取上下文出错: {str(e)}"

@tool
def generate_answer(query: str, context: str) -> str:
    """
    基于提供的上下文和问题，使用LLM生成回答。

    参数:
        query: 用户的问题
        context: 检索到的上下文信息

    返回:
        生成的回答
    """
    print(f"\n--- 生成回答 ---")

    if not context or "无法获取上下文" in context:
        return "根据提供的资料，我无法回答这个问题。"

    if "未找到相关文档" in context:
        return "根据提供的资料，我无法回答这个问题。"

    try:
        client = get_llm()
        prompt = config.ANSWER_PROMPT.format(ctx=context, q=query)
        resp = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        answer = resp.choices[0].message.content.strip()
        return answer
    except Exception as e:
        return f"生成回答出错: {str(e)}"

def get_all_tools():
    """返回所有可用的工具"""
    return [
        vector_search_tool,
        bm25_search_tool,
        expand_query_tool,
        hybrid_search_tool,
        get_context_for_answer,
        generate_answer
    ]
