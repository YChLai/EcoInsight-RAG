"""
Agentic RAG CLI 客户端
---------------------------------
提供命令行界面与Agentic RAG交互
"""

import asyncio
import os
import sys

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(FILE_DIR))
sys.path.insert(0, BASE_DIR)

from dotenv import load_dotenv
from rag.agentic_rag import AgenticRAG

load_dotenv()

async def run_chat_loop():
    print("\n" + "=" * 50)
    print("🌱 Agentic RAG 智能问答系统")
    print("=" * 50)
    print("输入 'quit' 退出，输入 'clear' 清除对话历史\n")

    agent = AgenticRAG(thread_id="cli-session")

    while True:
        try:
            user_input = input("\n你: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["quit", "exit", "q"]:
                print("\n👋 再见!")
                break

            if user_input.lower() == "clear":
                agent = AgenticRAG(thread_id="cli-session")
                print("✅ 对话历史已清除\n")
                continue

            print("\n🤖 Agent思考中...")

            result = await agent.chat_with_history(user_input)

            print(f"\nAgent: {result['reply']}")

        except KeyboardInterrupt:
            print("\n\n👋 再见!")
            break
        except Exception as e:
            print(f"\n⚠️  出错: {e}")

def main():
    asyncio.run(run_chat_loop())

if __name__ == "__main__":
    main()
