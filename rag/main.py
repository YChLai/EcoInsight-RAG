import os
import json
from typing import List, Tuple
from langchain_core.documents import Document
import retriever
import config

DATA_DIR = "../processed_data"

def rephrase(question: str, history: List[Tuple[str, str]]) -> str:
    if not history:
        return question
    print("\n--- 重构问题 ---")
    formatted = "\n".join([f"用户: {u}\n助手: {a}" for u, a in history])
    client = retriever.get_llm()
    prompt = config.REPHRASE_PROMPT.format(history=formatted, q=question)
    resp = client.chat.completions.create(model=config.LLM_MODEL, messages=[{"role": "user", "content": prompt}])
    result = resp.choices[0].message.content.strip()
    print(f"  {question} -> {result}")
    return result

def build_context(docs: List[Document]) -> str:
    print("\n--- 构建上下文 ---")
    data = _load_data()
    context = ""
    for doc in docs:
        src = doc.metadata.get('source', '未知')
        page = doc.metadata.get('page', '未知')
        content = _find_content(data, src, page) or doc.page_content
        context += f"--- 来源: {src}, 第 {page} 页 ---\n{content}\n\n"
    return context.strip()

def _load_data():
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

def answer(query: str, context: str):
    print("--- 生成答案 ---")
    if not context:
        return "根据提供的资料，我无法回答这个问题。"
    client = retriever.get_llm()
    prompt = config.ANSWER_PROMPT.format(ctx=context, q=query)
    resp = client.chat.completions.create(model=config.LLM_MODEL, messages=[{"role": "user", "content": prompt}], stream=True)
    print("\n--- 答案 ---\n")
    result = ""
    for chunk in resp:
        c = chunk.choices[0].delta.content or ""
        print(c, end="", flush=True)
        result += c
    print("\n" + "=" * 20)
    return result

def ask(query: str, history: List[Tuple[str, str]]):
    q = rephrase(query, history)
    docs = retriever.rerank(retriever.search(retriever.expand_query(q)))
    if not docs:
        return "未能检索到相关信息。"
    return answer(q, build_context(docs[:3]))

if __name__ == "__main__":
    print("--- 环境数据智能分析助手 (输入 exit 退出) ---")
    history = []
    while True:
        q = input("\n问题: ")
        if q.lower() == 'exit':
            break
        ans = ask(q, history)
        history.append((q, ans))
