import json
import os
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from langchain_community.chat_models.zhipuai import ChatZhipuAI
from langchain_huggingface import HuggingFaceEmbeddings

api_key = os.getenv("ZHIPUAI_API_KEY")
if not api_key:
    print("错误：未找到 ZHIPUAI_API_KEY")
    exit()

llm = ChatZhipuAI(model="glm-4-flash", api_key=api_key)
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3", model_kwargs={"device": "cuda"}, encode_kwargs={"normalize_embeddings": True})

def load_data(path):
    data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    with open(path, 'r', encoding='utf-8') as f:
        items = json.load(f)
    for item in items:
        data["question"].append(item.get("question"))
        data["answer"].append(item.get("answer"))
        data["contexts"].append(item.get("contexts"))
        data["ground_truth"].append(item.get("ground_truth"))
    return data

if __name__ == "__main__":
    data = load_data("results.json")
    dataset = Dataset.from_dict(data)
    
    print("开始评估...")
    result = evaluate(dataset, metrics=[context_precision, context_recall, faithfulness, answer_relevancy], llm=llm, embeddings=embeddings)
    
    df = result.to_pandas()
    print(df)
    df.to_csv("report.csv", index=False, encoding="utf-8-sig")
    
    print("\n平均分:")
    print(df[['context_precision', 'context_recall', 'faithfulness', 'answer_relevancy']].mean())
