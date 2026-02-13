import os

MILVUS_HOST = "localhost"
MILVUS_PORT = 19530
MILVUS_COLLECTION = "env_rag"
EMBEDDING_MODEL = "BAAI/bge-m3"
DEVICE = "cuda"

LLM_MODEL = "glm-4-flash"
LLM_API_KEY = os.getenv("ZHIPUAI_API_KEY")

TOP_K = 3

EXPAND_PROMPT = """根据用户问题生成 {n} 个不同角度的查询，每行一个，不要编号。

原始问题: {q}

生成的查询:"""

ANSWER_PROMPT = """基于上下文回答问题。规则：
1. 忠于原文，禁止使用外部知识
2. 每个关键信息后注明来源 [来源: <source>, 第 <page> 页]
3. 信息不足时回复"根据提供的资料，我无法回答这个问题。"

上下文:
---
{ctx}
---

问题: {q}

回答:"""

REPHRASE_PROMPT = """根据聊天记录将后续问题改写为独立问题。只返回改写后的问题。

聊天记录:
{history}

后续问题: {q}

改写后的问题:"""
