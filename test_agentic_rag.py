import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'rag'))

from agentic_rag import AgenticRAG, get_all_tools

print("✅ Agentic RAG 导入成功!")
print(f"✅ 可用工具数量: {len(get_all_tools())}")
for t in get_all_tools():
    print(f"   - {t.name}")
