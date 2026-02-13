import json
import os
from typing import Dict
from rag import config, retriever, main as rag

def get_result(query: str) -> Dict:
    docs = retriever.rerank(retriever.search(retriever.expand_query(query)))[:3]
    
    contexts = []
    data = rag._load_data()
    for doc in docs:
        src, page = doc.metadata.get('source', ''), doc.metadata.get('page', 0)
        content = rag._find_content(data, src, page)
        if content:
            contexts.append(content)
    
    context_str = "\n\n".join(contexts)
    client = retriever.get_llm()
    prompt = config.ANSWER_PROMPT.format(ctx=context_str, q=query)
    resp = client.chat.completions.create(model=config.LLM_MODEL, messages=[{"role": "user", "content": prompt}])
    
    return {"answer": resp.choices[0].message.content.strip(), "contexts": contexts}

if __name__ == "__main__":
    with open('../eval_data.json', 'r', encoding='utf-8') as f:
        golden = json.load(f)
    
    results = []
    for item in golden:
        print(f"处理: {item['question']}")
        r = get_result(item['question'])
        results.append({"question": item['question'], "ground_truth": item['ground_truth'], "answer": r['answer'], "contexts": r['contexts']})
    
    with open('results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("完成")
