import json
import os
from datasets import Dataset
from ragas import evaluate  # 移除了 RagasSettings
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)

# 导入 Zhipu AI 和 HuggingFace 的 LangChain 包装器
try:
    from langchain_community.chat_models.zhipuai import ChatZhipuAI
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    print("错误：请运行 'pip install langchain-community langchain-huggingface sentence-transformers' 来安装所需的库。")
    exit()

# --- RAGAS 模型配置 (使用 Zhipu AI 和 HuggingFace) ---

# 1. 检查 ZHIPU_API_KEY
api_key = os.getenv("ZHIPUAI_API_KEY")
if not api_key:
    print("错误：未找到 ZHIPU_API_KEY 环境变量。")
    print("请确保在运行此脚本前已正确设置了该变量。")
    exit()
else:
    print("成功加载 ZHIPU_API_KEY。")

# 2. 显式初始化 LLM (使用您指定的 glm-4-flash)
llm = ChatZhipuAI(
    model="glm-4-flash",
    api_key=api_key
)

# 3. 显式初始化嵌入模型
embeddings_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True}
)
print("已成功初始化 ChatZhipuAI (glm-4-flash) 和 HuggingFaceEmbeddings (BAAI/bge-m3)。")


# --- 模型配置结束 ---


def convert_json_to_dataset_dict(json_file_path):
    """
    加载一个面向记录的 JSON 文件，并将其转换为
    一个面向列的字典（列表的字典）。
    """
    data_dict = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        for item in json_data:
            data_dict["question"].append(item.get("question"))
            data_dict["answer"].append(item.get("answer"))
            data_dict["contexts"].append(item.get("contexts"))
            gt = item.get("ground_truth")
            data_dict["ground_truth"].append(gt)  # 附加字符串

        print(f"成功从 '{json_file_path}' 加载并转换了 {len(json_data)} 条记录。")
        return data_dict

    except FileNotFoundError:
        print(f"错误：文件 '{json_file_path}' 未找到。")
        return None
    except json.JSONDecodeError:
        print(f"错误：文件 '{json_file_path}' 格式不正确，无法解析 JSON。")
        return None
    except Exception as e:
        print(f"发生了一个未预料的错误： {e}")
        return None


# --- 使用 ---
file_name = "rag_evaluation_results.json"
converted_data = convert_json_to_dataset_dict(file_name)

if converted_data:
    print("\n--- 转换后第一条数据的示例 ---")
    print("Question:", converted_data["question"][0])
    print("Answer:", converted_data["answer"][0])
    print("Contexts (count):", len(converted_data["contexts"][0]))
    print("Ground Truth:", converted_data["ground_truth"][0])

    dataset = Dataset.from_dict(converted_data)
    print("\nDataset 结构:")
    print(dataset)

    print("\n正在开始 RAGAS 评估 (使用 Zhipu glm-4-flash)...")

    result = evaluate(
        dataset=dataset,
        metrics=[
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy,
        ],
        llm=llm,  # 指定 LLM
        embeddings=embeddings_model  # 指定 Embeddings
    )

    # --- [已更改] 保存与打印结果 ---

    # 1. 转换为 Pandas DataFrame
    df = result.to_pandas()

    # 2. 打印完整的 DataFrame 到控制台
    print("\n评估结果 (Pandas DataFrame):")
    # 设置 pandas 打印选项，以显示所有列
    import pandas as pd

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(df)

    # 3. 保存 DataFrame 到 CSV 文件
    output_csv_file = "rag_evaluation_report.csv"
    df.to_csv(output_csv_file, index=False, encoding="utf-8-sig")
    print(f"\n--- 评估结果已成功保存到: {output_csv_file} ---")

    # 4. [已更改] 直接从 DataFrame 计算平均分
    print("\n--- 评估指标平均分 ---")
    try:
        # .mean() 会自动计算所有数值列的平均值
        # 我们可以只选择我们关心的指标列
        metrics_to_average = ['context_precision', 'context_recall', 'faithfulness', 'answer_relevancy']
        # 过滤掉数据集中可能存在的非指标列
        available_metrics = [m for m in metrics_to_average if m in df.columns]

        if available_metrics:
            averages = df[available_metrics].mean()
            print(averages)
        else:
            print("在 DataFrame 中未找到任何 RAGAS 指标列。")

    except Exception as e:
        print(f"计算平均分时出错: {e}")



"""
context_precision    0.988889
context_recall       0.966667
faithfulness         0.760556
answer_relevancy     0.810647
"""