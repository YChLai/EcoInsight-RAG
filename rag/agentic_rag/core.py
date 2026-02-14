"""
Agentic RAG - Agentic检索增强生成
---------------------------------
基于LangGraph的Agentic RAG实现，支持：
1. 自动选择合适的检索工具
2. 多轮检索和推理
3. 自主决定是否需要再次检索
"""

import os
import asyncio
from typing import List, Dict, Any
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from langchain_community.chat_models import ChatZhipuAI
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage

from .tools import get_all_tools

load_dotenv()

AGENT_PROMPT = """你是一个专业的环境数据智能分析助手。你的任务是根据用户的问题，通过检索知识库来获取相关信息并回答。

## 可用工具
1. `vector_search_tool` - 向量语义检索，适合语义相似的内容
2. `bm25_search_tool` - 关键词精确匹配检索，适合专业术语
3. `expand_query_tool` - 使用LLM扩展查询，获取更多角度
4. `hybrid_search_tool` - 混合检索+重排，综合多种方法
5. `get_context_for_answer` - 获取完整上下文用于回答
6. `generate_answer` - 基于上下文生成最终回答

## 工作流程
1. 首先理解用户问题
2. 选择合适的工具进行检索（可以先用expand_query_tool扩展问题）
3. 查看检索结果，如果结果不相关，尝试其他检索方法
4. 使用get_context_for_answer获取完整上下文
5. 使用generate_answer生成最终回答

## 重要规则
- 必须先检索才能回答，不能仅凭常识回答
- 每个关键信息需要注明来源
- 如果检索不到相关信息，明确告知用户
- 回答必须基于检索到的内容，不要编造信息

请根据用户问题自主决定使用哪些工具，完成检索和回答。"""

class AgenticRAG:
    def __init__(self, model_name: str = None, thread_id: str = "default"):
        self.model_name = model_name or os.getenv("MODEL", "glm-4-flash")
        self.thread_id = thread_id
        self.agent = None
        self.checkpointer = InMemorySaver()

    def _init_agent(self):
        if self.agent is None:
            api_key = os.getenv("ZHIPUAI_API_KEY")
            if not api_key:
                raise ValueError("未找到 API Key，请在 .env 中配置 ZHIPUAI_API_KEY")

            model = ChatZhipuAI(
                model=self.model_name,
                temperature=0.7,
                streaming=False,
                api_key=api_key
            )

            tools = get_all_tools()

            self.agent = create_react_agent(
                model=model,
                tools=tools,
                prompt=AGENT_PROMPT,
                checkpointer=self.checkpointer
            )

        return self.agent

    async def chat(self, message: str) -> str:
        agent = self._init_agent()

        config = {"configurable": {"thread_id": self.thread_id}}

        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=message)]},
            config
        )

        if "messages" in result and len(result["messages"]) > 0:
            return result["messages"][-1].content
        return "未获取到回复"

    def chat_sync(self, message: str) -> str:
        return asyncio.run(self.chat(message))

    async def chat_with_history(self, message: str, thread_id: str = None) -> Dict[str, Any]:
        if thread_id:
            self.thread_id = thread_id

        agent = self._init_agent()
        config = {"configurable": {"thread_id": self.thread_id}}

        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=message)]},
            config
        )

        messages = []
        for msg in result.get("messages", []):
            if hasattr(msg, 'content'):
                content = msg.content
            elif isinstance(msg, dict):
                content = msg.get('content', '')
            else:
                content = str(msg)

            role = getattr(msg, 'type', 'assistant') if not isinstance(msg, dict) else msg.get('type', 'assistant')
            messages.append({
                "role": role,
                "content": content
            })

        reply_content = ""
        for msg in reversed(messages):
            if msg["content"]:
                reply_content = msg["content"]
                break

        return {
            "reply": reply_content,
            "messages": messages,
            "thread_id": self.thread_id
        }


def create_agentic_rag(model_name: str = None, thread_id: str = "default") -> AgenticRAG:
    """创建Agentic RAG实例的工厂函数"""
    return AgenticRAG(model_name=model_name, thread_id=thread_id)
