import json
import os
from typing import List, Dict

# 假设您的 retriever 和 main 都在 rag_fusion 目录下
from rag_fusion import config,retriever, main as rag_main
from langchain_core.documents import Document

# 1. 初始化所有客户端 (复用您 retriever.py 中的逻辑)
retriever.get_llm_client()
retriever.get_embedding_model()
retriever.get_db_client()


def get_retrieved_contexts_and_answer(query: str) -> Dict[str, any]:
    """
    封装您的 RAG 流程，以返回 ragas 所需的格式。
    """
    # 步骤 1 & 2 & 3: RAG-Fusion 检索
    # (复用 retriever.py 和 main.py 的逻辑)
    standalone_query = query  # 假设没有聊天记录

    final_docs_for_context: List[Document] = retriever.reciprocal_rank_fusion(
        retriever.parallel_search(
            retriever.generate_queries(standalone_query)
        )
    )

    # 限制到前3个，如您 main.py 中所示
    top_docs = final_docs_for_context[:3]

    # --- 关键修改 ---
    # Ragas 需要字符串列表，而不是一个合并的字符串
    contexts_list: List[str] = []

    # 复用您 main.py 中的 _load_all_processed_data
    all_processed_data = rag_main._load_all_processed_data()

    for doc in top_docs:
        source = doc.metadata.get('source', '未知来源')
        page = doc.metadata.get('page', '未知页码')

        # 复用您 main.py 中的 _find_page_content
        page_content = rag_main._find_page_content(all_processed_data, source, page)

        if page_content:
            # Ragas 需要的上下文列表
            contexts_list.append(page_content)
        else:
            # 如果找不到完整父文档，退而求其次使用子块内容
            # （在您的结构中，doc 已经是父文档的引用了）
            print(f"警告: 未能找到 {source} 第 {page} 页的完整内容。")

    # 步骤 4: 构建用于LLM的完整上下文
    # (这部分不变，用于生成答案)
    final_context_str = "\n\n".join(contexts_list)

    # 步骤 5: 生成最终答案 (复用 main.py 逻辑, 但需修改为非流式)
    client = retriever.get_llm_client()
    prompt = config.FINAL_ANSWER_PROMPT.format(context=final_context_str, question=standalone_query)

    response = client.chat.completions.create(
        model=config.GENERATION_MODEL_NAME,  #
        messages=[{"role": "user", "content": prompt}],
        stream=False  # 评估时使用非流式
    )
    final_answer = response.choices[0].message.content.strip()

    return {
        "answer": final_answer,
        "contexts": contexts_list  # 返回字符串列表
    }


# --- 步骤 3: 运行 RAG 流程并收集结果 ---

print("正在运行RAG流程，收集评估数据...")
# 加载黄金标准数据集
with open('../evaluation_dataset.json', 'r', encoding='utf-8') as f:
    golden_dataset = json.load(f)

results_data = []
for item in golden_dataset:
    question = item['question']
    print(f"正在处理问题: {question}")

    rag_output = get_retrieved_contexts_and_answer(question)

    results_data.append({
        "question": question,
        "ground_truth": item['ground_truth'],
        "answer": rag_output['answer'],
        "contexts": rag_output['contexts']
    })

# 将结果保存，以便 ragas 使用
with open('rag_evaluation_results.json', 'w', encoding='utf-8') as f:
    json.dump(results_data, f, indent=4, ensure_ascii=False)

print("RAG 流程数据收集完毕！")