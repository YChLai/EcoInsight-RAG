import os
import json
import streamlit as st
from typing import List, Tuple
from langchain_core.documents import Document
import retriever
import config

DATA_DIR = "../processed_data"

@st.cache_resource
def init():
    retriever.get_llm()
    retriever.get_embeddings()
    retriever.get_db()

@st.cache_data
def load_data():
    data = {}
    for f in os.listdir(DATA_DIR):
        path = os.path.join(DATA_DIR, f)
        if f.endswith(".json"):
            data[f.replace(".json", "")] = {'type': 'json', 'content': json.load(open(path, 'r', encoding='utf-8'))}
        elif f.endswith(".txt"):
            data[f.replace(".txt", "")] = {'type': 'txt', 'content': open(path, 'r', encoding='utf-8').read()}
    return data

def rephrase(question, history):
    if not history:
        return question
    formatted = "\n".join([f"用户: {u}\n助手: {a}" for u, a in history])
    client = retriever.get_llm()
    prompt = config.REPHRASE_PROMPT.format(history=formatted, q=question)
    resp = client.chat.completions.create(model=config.LLM_MODEL, messages=[{"role": "user", "content": prompt}])
    result = resp.choices[0].message.content.strip()
    st.info(f"重构后: {result}")
    return result

def find_content(data, source, page):
    src = source.replace(".pdf", "").replace(".json", "")
    if src not in data:
        return ""
    d = data[src]
    if d['type'] == 'txt':
        return d['content']
    return "\n\n".join([b['content'] for b in d['content'] if b['metadata'].get('page') == page])

def build_context(docs):
    context, sources = "", []
    data = load_data()
    for doc in docs:
        src, page = doc.metadata.get('source', '未知'), doc.metadata.get('page', '未知')
        content = find_content(data, src, page)
        if content:
            context += f"--- 来源: {src}, 第 {page} 页 ---\n{content}\n\n"
            sources.append({"source": src, "page": page})
    return context.strip(), sources

def stream_answer(query, context):
    if not context:
        yield "根据提供的资料，我无法回答这个问题。"
        return
    client = retriever.get_llm()
    prompt = config.ANSWER_PROMPT.format(ctx=context, q=query)
    resp = client.chat.completions.create(model=config.LLM_MODEL, messages=[{"role": "user", "content": prompt}], stream=True)
    for chunk in resp:
        yield chunk.choices[0].delta.content or ""

st.set_page_config(page_title="环境数据智能分析助手", page_icon="🌱")
st.title("🌱 环境数据智能分析助手")

init()

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.history = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("请输入问题..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        q = rephrase(prompt, st.session_state.history)
        with st.spinner("检索中..."):
            docs = retriever.rerank(retriever.search(retriever.expand_query(q)))
        
        if not docs:
            resp = "未能检索到相关信息。"
            st.warning(resp)
        else:
            context, sources = build_context(docs[:3])
            with st.sidebar:
                st.subheader("引用来源")
                for s in sources:
                    st.markdown(f"- {s['source']}, 第 {s['page']} 页")
            resp = st.write_stream(stream_answer(q, context))
    
    st.session_state.messages.append({"role": "assistant", "content": resp})
    st.session_state.history.append((prompt, resp))

with st.sidebar:
    st.header("说明")
    st.markdown("基于RAG的环境数据问答系统，支持多轮对话。")
