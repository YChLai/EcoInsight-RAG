import os
import re
import json
import torch
import jieba
from typing import List, Dict
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from zhipuai import ZhipuAI
from pymilvus import connections, Collection, utility
from rank_bm25 import BM25Okapi
from modelscope import AutoModelForCausalLM, AutoTokenizer
import config

DATA_DIR = "../processed_data"

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
        embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL, model_kwargs={"device": config.DEVICE}, encode_kwargs={"normalize_embeddings": True})
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

def expand_query(query: str) -> List[str]:
    print(f"\n--- 扩展查询 ---")
    client = get_llm()
    prompt = config.EXPAND_PROMPT.format(n=3, q=query)
    resp = client.chat.completions.create(model=config.LLM_MODEL, messages=[{"role": "user", "content": prompt}])
    generated = resp.choices[0].message.content.strip().split('\n')
    queries = [query] + [q for q in generated if q]
    print(f"  查询: {queries}")
    return queries

def search(queries: List[str]) -> Dict[str, List[Document]]:
    print("\n--- 检索 ---")
    collection = get_db()
    emb = get_embeddings()
    bm25_idx, bm25_d = get_bm25()
    results = {}
    
    for q in queries:
        print(f"  搜索: '{q}'")
        q_emb = emb.embed_query(q)
        
        summary_hits = collection.search(data=[q_emb], anns_field="embedding", param={"metric_type": "L2", "params": {"ef": 64}}, limit=config.TOP_K, expr="is_summary == True", output_fields=["content", "source", "page", "doc_id"])[0]
        chunk_hits = collection.search(data=[q_emb], anns_field="embedding", param={"metric_type": "L2", "params": {"ef": 64}}, limit=config.TOP_K, expr="is_child == True", output_fields=["content", "source", "page", "doc_id", "chunk_index"])[0]
        
        summary_docs = [Document(page_content=h.entity.get("content", ""), metadata={"source": h.entity.get("source", ""), "page": h.entity.get("page", 0), "doc_id": h.entity.get("doc_id", "")}) for h in summary_hits]
        chunk_docs = [Document(page_content=h.entity.get("content", ""), metadata={"source": h.entity.get("source", ""), "page": h.entity.get("page", 0), "doc_id": h.entity.get("doc_id", ""), "chunk_index": h.entity.get("chunk_index", -1)}) for h in chunk_hits]
        
        bm25_results = _bm25(q, bm25_idx, bm25_d, config.TOP_K)
        results[q] = summary_docs + chunk_docs + bm25_results
        print(f"    HNSW: {len(summary_docs)+len(chunk_docs)}, BM25: {len(bm25_results)}")
    return results

def _bm25(query, idx, docs, k):
    scores = idx.get_scores(tokenize(query))
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [Document(page_content=docs[i].page_content, metadata={**docs[i].metadata, "bm25_score": scores[i]}) for i in top]

def rerank(search_results: Dict[str, List[Document]]) -> List[Document]:
    print("\n--- 重排 ---")
    model, tok = get_reranker()
    
    unique = {}
    for q, docs in search_results.items():
        for d in docs:
            uid = f"{d.metadata.get('doc_id')}-{d.metadata.get('chunk_index', -1)}"
            if uid not in unique:
                unique[uid] = d
    
    docs_list = list(unique.values())
    if not docs_list:
        return []
    
    print(f"  {len(docs_list)} 个文档")
    scores = {}
    
    for q in search_results.keys():
        for d in docs_list:
            try:
                inputs = tok([q], [d.page_content[:1000]], return_tensors="pt", padding=True)
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
    print(f"  融合后 {len(result)} 个文档")
    return result
