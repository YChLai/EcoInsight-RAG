import streamlit as st
import os
import json
from typing import List, Tuple, Dict  # <--- BUG修复在这里！
from langchain_core.documents import Document
import retriever
import config


# --- 缓存加载函数 ---
@st.cache_resource
def get_llm_client_cached():
    """获取并缓存智谱AI客户端。"""
    return retriever.get_llm_client()


@st.cache_resource
def get_embedding_model_cached():
    """获取并缓存嵌入模型。"""
    return retriever.get_embedding_model()


@st.cache_resource
def get_db_client_cached():
    """获取并缓存向量数据库客户端。"""
    retriever.embedding_model = get_embedding_model_cached()
    return retriever.get_db_client()


# --- 核心逻辑函数 ---

def rephrase_question_with_history(question: str, chat_history: List[Tuple[str, str]]) -> str:
    """根据对话历史重构问题。"""
    if not chat_history:
        return question

    formatted_history = ""
    for user_msg, bot_msg in chat_history:
        formatted_history += f"用户: {user_msg}\n助手: {bot_msg}\n"

    client = get_llm_client_cached()
    prompt = config.HISTORY_REPHRASE_PROMPT.format(chat_history=formatted_history, question=question)

    with st.spinner("正在理解上下文..."):
        response = client.chat.completions.create(
            model=config.QUERY_GENERATION_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
        )
        rephrased_question = response.choices[0].message.content.strip()

    st.info(f"重构后问题: *{rephrased_question}*")
    return rephrased_question


@st.cache_data(show_spinner=False)
def _load_all_processed_data() -> dict:
    """加载所有处理好的json和txt文件内容到内存。"""
    data = {}
    folder = "D:\\LLM\\RAG\\Nvidia-Finance-Rag\\processed_data"
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if filename.endswith(".json"):
            with open(file_path, 'r', encoding='utf-8') as f:
                key = filename.replace("_processed.json", "")
                data[key] = {'type': 'json', 'content': json.load(f)}
        elif filename.endswith("_transcribed.txt"):
            with open(file_path, 'r', encoding='utf-8') as f:
                key = filename.replace("_transcribed.txt", "")
                data[key] = {'type': 'txt', 'content': f.read()}
    return data


def _find_page_content(all_data: dict, source: str, page: int) -> str:
    """从加载的数据中找到特定源文件的特定页面的内容。"""
    clean_source = source.replace(".pdf", "").replace(".json", "")
    source_data = all_data.get(clean_source)
    if not source_data: return ""

    if source_data['type'] == 'txt':
        return source_data['content']
    elif source_data['type'] == 'json':
        json_content = source_data['content']
        page_blocks = [block['content'] for block in json_content if block['metadata'].get('page') == page]
        return "\n\n".join(page_blocks)
    return ""


def format_context_for_llm(docs: List[Document]) -> Tuple[str, List[Dict]]:
    """将检索到的文档格式化为LLM上下文，并提取引用来源。"""
    context_str = ""
    sources = []
    all_processed_data = _load_all_processed_data()

    for doc in docs:
        source = doc.metadata.get('source', '未知来源')
        page = doc.metadata.get('page', '未知页码')

        page_content = _find_page_content(all_processed_data, source, page)

        if page_content:
            context_str += f"--- 相关资料来源: {source}, 第 {page} 页 ---\n"
            context_str += page_content
            context_str += "\n\n"
            sources.append({"source": source, "page": page})

    return context_str.strip(), sources


def generate_final_answer_stream(query: str, context: str):
    """以流式方式调用LLM生成最终答案。"""
    if not context or not context.strip():
        yield "根据提供的资料，我无法回答这个问题，因为未能加载到相关的上下文信息。"
        return

    client = get_llm_client_cached()
    prompt = config.FINAL_ANSWER_PROMPT.format(context=context, question=query)

    response_stream = client.chat.completions.create(
        model=config.GENERATION_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )

    for chunk in response_stream:
        content = chunk.choices[0].delta.content or ""
        yield content


# --- Streamlit UI ---

st.set_page_config(page_title="环境数据智能分析助手", page_icon="🌱")
st.title("🌱 环境数据智能分析助手")
st.caption("由 RAG-Fusion 和多模态数据处理驱动")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("请输入你关于环境数据的问题..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        standalone_query = rephrase_question_with_history(prompt, st.session_state.chat_history)

        with st.spinner("正在执行RAG-Fusion检索..."):
            final_docs_for_context = retriever.reranker_based_fusion(
                retriever.parallel_search(
                    retriever.generate_queries(standalone_query)
                )
            )

        if not final_docs_for_context:
            st.warning("抱歉，未能检索到任何相关信息。")
            full_response = "抱歉，未能检索到任何相关信息。"
        else:
            context, sources = format_context_for_llm(final_docs_for_context[:3])

            with st.sidebar:
                st.subheader("本次回答引用的资料来源")
                st.empty()  # 清空旧的来源
                for src in sources:
                    st.markdown(f"- **来源**: `{src['source']}`\n- **页码**: `{src['page']}`")

            response_generator = generate_final_answer_stream(standalone_query, context)
            full_response = st.write_stream(response_generator)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
    st.session_state.chat_history.append((prompt, full_response))

with st.sidebar:
    st.header("应用说明")
    st.markdown("""
    这是一个基于多模态环境数据（PDFs和音频转录）构建的高级RAG问答应用。

    **主要技术栈:**
    - **Streamlit**: 用于构建交互式Web界面。
    - **RAG-Fusion**: 通过生成多个查询并融合结果，提升检索准确性。
    - **父子块分块**: 优化检索单元和上下文单元。
    - **对话记忆**: 支持多轮对话，理解上下文。

    **如何使用:**
    1. 在下方的输入框中提问。
    2. 应用会自动理解上下文并检索相关信息。
    3. 答案会实时显示，引用的来源会展示在这里。
    """)