# rag_fusion/main_app.py (完整替换)

import json
import os
from typing import List, Tuple
from langchain_core.documents import Document
import retriever
import config


def rephrase_question_with_history(question: str, chat_history: List[Tuple[str, str]]) -> str:
    """根据对话历史重构问题。"""
    if not chat_history:
        print("--- 对话历史为空，直接使用原始问题 ---")
        return question

    print("\n--- 检测到对话历史，正在重构问题以获取上下文 ---")

    formatted_history = ""
    for user_msg, bot_msg in chat_history:
        formatted_history += f"用户: {user_msg}\n助手: {bot_msg}\n"

    client = retriever.get_llm_client()
    prompt = config.HISTORY_REPHRASE_PROMPT.format(chat_history=formatted_history, question=question)

    response = client.chat.completions.create(
        model=config.QUERY_GENERATION_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
    )

    rephrased_question = response.choices[0].message.content.strip()
    print(f"  - 原始问题: '{question}'")
    print(f"  - 重构后问题: '{rephrased_question}'")

    return rephrased_question


def format_context_for_llm(docs: List[Document]) -> str:
    """将检索到的父文档格式化为LLM的上下文。"""
    print("\n--- 步骤 4: 构建最终上下文 ---")

    context_str = ""
    all_processed_data = _load_all_processed_data()

    for doc in docs:
        source = doc.metadata.get('source', '未知来源')
        page = doc.metadata.get('page', '未知页码')

        page_content = _find_page_content(all_processed_data, source, page)

        if page_content:
            context_str += f"--- 相关资料来源: {source}, 第 {page} 页 ---\n"
            context_str += page_content
            context_str += "\n\n"
        else:
            # 这是一个关键的调试信息
            print(f"  - 警告: 未能找到 {source} 第 {page} 页的完整内容。")

    return context_str.strip()


def _load_all_processed_data() -> dict:
    """
    辅助函数：加载所有processed_data下的json和txt文件内容到内存。
    (已修正)
    """
    data = {}
    folder = "../processed_data"
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)

        # --- 修正逻辑：同时处理 .json 和 .txt 文件 ---
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
    """
    辅助函数：从加载的数据中找到特定源文件的特定页面的所有内容块并拼接。
    (已修正)
    """
    clean_source = source.replace(".pdf", "").replace(".json", "")

    source_data = all_data.get(clean_source)
    if not source_data:
        return ""

    # --- 修正逻辑：根据文件类型分别处理 ---
    if source_data['type'] == 'txt':
        # 对于 TXT 文件，我们返回其全部内容，因为我们视其为单页
        return source_data['content']
    elif source_data['type'] == 'json':
        # 对于 JSON 文件，按页码查找
        json_content = source_data['content']
        page_blocks = [block['content'] for block in json_content if block['metadata'].get('page') == page]
        return "\n\n".join(page_blocks)

    return ""


def generate_final_answer(query: str, context: str):
    """调用LLM生成最终答案。"""
    print("--- 步骤 5: 生成最终答案 ---")
    if not context or not context.strip():
        print("\n\n--- 最终答案 ---\n")
        final_answer = "根据提供的资料，我无法回答这个问题，因为未能加载到相关的上下文信息。"
        print(final_answer)
        print("\n" + "=" * 20)
        return final_answer

    client = retriever.get_llm_client()
    prompt = config.FINAL_ANSWER_PROMPT.format(context=context, question=query)

    response = client.chat.completions.create(
        model=config.GENERATION_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )

    print("\n\n--- 最终答案 ---\n")
    full_response = ""
    for chunk in response:
        content = chunk.choices[0].delta.content or ""
        print(content, end="", flush=True)
        full_response += content
    print("\n" + "=" * 20)

    return full_response


def ask_question(query: str, chat_history: List[Tuple[str, str]]):
    """执行完整的RAG问答流程（包含对话记忆）。"""
    standalone_query = rephrase_question_with_history(query, chat_history)
    final_docs_for_context = retriever.reciprocal_rank_fusion(
        retriever.parallel_search(
            retriever.generate_queries(standalone_query)
        )
    )

    if not final_docs_for_context:
        final_answer = "抱歉，未能检索到任何相关信息。"
        print(final_answer)
        chat_history.append((query, final_answer))
        return

    context = format_context_for_llm(final_docs_for_context[:3])
    final_answer = generate_final_answer(standalone_query, context)
    chat_history.append((query, final_answer))


if __name__ == "__main__":
    print("--- Nvidia 财报智能分析问答工具 (RAG-Fusion版 + 对话记忆) ---")
    print("--- 输入 'exit' 退出程序 ---")

    chat_history = []
    while True:
        user_query = input("\n请输入你的问题: ")
        if user_query.lower() == 'exit':
            break
        ask_question(user_query, chat_history)