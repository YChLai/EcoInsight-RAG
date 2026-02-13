import json
from collections import defaultdict
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import mine_hard_negatives

INPUT_FILE = "qa.json"
OUTPUT_FILE = "train.json"
MODEL_PATH = "BAAI/bge-m3"

SYSTEM = "Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
INSTRUCT = "Given a search query, retrieve relevant passages that answer the query"

def main():
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        pairs = json.load(f)
    print(f"加载 {len(pairs)} 条QA")
    
    dataset = Dataset.from_dict({
        "prompt": [p["query"] for p in pairs],
        "response": [p["response"] for p in pairs]
    })
    
    print("加载模型...")
    model = SentenceTransformer(MODEL_PATH)
    
    print("挖掘负样本...")
    mined = mine_hard_negatives(
        dataset, model,
        anchor_column_name="prompt",
        positive_column_name="response",
        num_negatives=5, range_min=20, range_max=50,
        max_score=0.8, absolute_margin=0.1, use_faiss=False
    )
    
    query_map = {p['query']: p['response'] for p in pairs}
    grouped = defaultdict(lambda: {"query": "", "positive": "", "negatives": []})
    
    for item in mined:
        q = item['prompt']
        if not grouped[q]["query"]:
            grouped[q]["query"] = q
            grouped[q]["positive"] = query_map.get(q, "")
        if 'negative' in item and item['negative'] not in grouped[q]["negatives"]:
            grouped[q]["negatives"].append(item['negative'])
    
    result = []
    for item in grouped.values():
        if item["positive"]:
            result.append({"system": SYSTEM, "input": f"<Instruct>: {INSTRUCT}\n<Query>: {item['query']}\n<Document>: {item['positive']}", "output": "yes"})
        for neg in item["negatives"]:
            result.append({"system": SYSTEM, "input": f"<Instruct>: {INSTRUCT}\n<Query>: {item['query']}\n<Document>: {neg}", "output": "no"})
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"完成，共 {len(result)} 条")

if __name__ == "__main__":
    main()
