# rag_fusion/config.py
import os

# --- 向量数据库与嵌入模型配置 ---
# Chroma配置（已废弃，使用Milvus）
VECTOR_DB_PATH = "../chroma_db_optimized"
# Milvus配置
MILVUS_HOST = "localhost"
MILVUS_PORT = 19530
MILVUS_COLLECTION = "environmental_data_rag"
MILVUS_INDEX_PARAM = {
    "index_type": "HNSW",
    "params": {
        "M": 8,
        "efConstruction": 64
    }
}
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
EMBEDDING_DEVICE = "cuda"  # 如果没有GPU，可以改为 "cpu"

# --- LLM 模型配置 ---
# 用于最终答案生成的强大模型
GENERATION_MODEL_NAME = "glm-4-flash"
# 用于查询生成的经济型模型 (RAG-Fusion步骤一)
QUERY_GENERATION_MODEL_NAME = "glm-4-flash"
LLM_API_KEY = os.getenv("ZHIPUAI_API_KEY")  # 请替换为你的 API Key

# --- RAG Fusion 流程配置 ---
# 在并行检索中，从摘要和子块中各取回多少个结果
RETRIEVAL_K = 5
# RRF融合算法中的一个常数，通常设置为60
RRF_K = 60
# 生成多少个扩展查询
NUM_GENERATED_QUERIES = 3

# --- Prompt 模板 ---
# RAG-Fusion: 查询生成模板
QUERY_GENERATION_PROMPT = """
你是一个精通环境数据和报告的AI分析师。
你的任务是根据用户的原始问题，生成 {num_queries} 个不同角度、但语义相似的查询。
这些查询应该更具体、更适合在向量数据库中进行搜索。

请使用以下格式，每个查询占一行，不要有任何额外的解释或编号。

原始问题: {original_query}

生成的查询:
"""

# 最终答案生成模板
FINAL_ANSWER_PROMPT = """
你是一个专业的环境数据AI分析师。
你的任务是基于下面提供的上下文信息，用中文清晰、准确、全面地回答用户的问题。

**规则:**
1.  **忠于原文**: 你的回答必须完全基于所提供的上下文，禁止使用任何外部知识。
2.  **引用来源**: 在回答的每一句话或每一个关键信息点后面，必须用 `[来源: <source>, 第 <page> 页]` 的格式注明信息来源。例如：该地区2025年第一季度的PM2.5浓度为50μg/m³ [来源: 2025_env_report, 第 5 页]。
3.  **内容全面**: 如果上下文信息足以回答问题，请综合所有相关信息，提供一个完整的答案。
4.  **无法回答**: 如果上下文中没有足够的信息来回答问题，请直接回复：“根据提供的资料，我无法回答这个问题。”

**上下文信息:**
---
{context}
---

**用户问题:** {question}

**你的回答:**
"""

HISTORY_REPHRASE_PROMPT = """
你的任务是根据一段聊天记录和一个后续问题，将这个后续问题改写成一个独立的、无需上下文就能理解的新问题。

规则:
1. 如果后续问题已经是一个完整的、独立的问题，则无需改写，直接返回原问题。
2. 如果后续问题依赖于聊天记录（例如 "为什么", "具体点说", "他呢"），请结合聊天记录，生成一个完整的问题。
3. 返回的新问题应该是清晰、具体的，适合用于向量数据库的检索。
4. 只返回改写后的问题，不要添加任何额外的解释。

--- 聊天记录 ---
{chat_history}
---

后续问题: {question}

改写后的独立问题:
"""