import os
import json
import shutil
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

PROCESSED_DATA_FOLDER = "../processed_data"
VECTOR_DB_PATH = "../chroma_db"
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"

def load_docs_from_json():
    """
    职责一：加载所有处理好的JSON文件，整合图片摘要，
    并返回一个 Document 对象的列表。
    """
    print("--- 步骤一: 从JSON加载并整合所有文档块 ---")

    # 加载图片摘要
    image_summaries_path = os.path.join(PROCESSED_DATA_FOLDER, "image_summaries.json")
    with open(image_summaries_path, 'r', encoding='utf-8') as f:
        image_summaries = json.load(f)
    print(f"  - 已加载 {len(image_summaries)} 条图片摘要。")

    all_docs = []
    for filename in sorted(os.listdir(PROCESSED_DATA_FOLDER)):
        if filename.endswith(".json") and filename != "image_summaries.json":
            file_path = os.path.join(PROCESSED_DATA_FOLDER, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                blocks = json.load(f)

            print(f"  - 正在处理: {filename}")
            for block in blocks:
                content = block["content"]
                metadata = block["metadata"]

                if block["type"] == "image_placeholder":
                    img_name = content.replace("[IMAGE: ", "").replace("]", "")
                    summary = image_summaries.get(img_name, "")  # 使用 .get() 避免KeyError
                    content = f"--- [参考图片: {img_name} 来自第 {metadata['page']} 页] ---\n{summary}\n--- [图片描述结束] ---"

                # 为每个块创建一个Document对象
                doc = Document(page_content=content, metadata=metadata)
                # 将块的类型也加入元数据，方便下一步处理
                doc.metadata["type"] = block["type"]
                all_docs.append(doc)

    print(f"  - 总共加载了 {len(all_docs)} 个原始文档块。")
    return all_docs

def chunk_documents(docs):
    """
    职责二：对加载好的Document列表进行差异化分割。
    """
    print("\n--- 步骤二: 对文档块进行差异化分割 ---")

    final_chunks = []
    # 初始化一个仅用于处理长篇文本的分割器
    prose_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    for doc in docs:
        # 根据块的类型决定处理策略
        doc_type = doc.metadata.get("type", "prose")

        if doc_type in ["table", "table_summary", "image_placeholder"]:
            # 对于这些类型，我们保留其完整性，不进行分割
            final_chunks.append(doc)
        else:  # 默认情况，包括 "prose" 类型
            # 对普通文本块进行分割
            prose_chunks = prose_splitter.split_documents([doc])
            final_chunks.extend(prose_chunks)

    print(f"  - 所有块被成功处理成 {len(final_chunks)} 个最终Chunk。")
    return final_chunks

def create_and_persist_vector_store(chunks, embeddings):
    print("\n--- 步骤三: 嵌入数据并构建向量库 ---")
    if os.path.exists(VECTOR_DB_PATH):
        print(f"  - 发现旧的数据库，正在删除: {VECTOR_DB_PATH}")
        shutil.rmtree(VECTOR_DB_PATH)
    Chroma.from_documents(chunks, embeddings, persist_directory=VECTOR_DB_PATH)
    print("  - 向量数据库创建完毕并已成功持久化！")

def verify_metadata(embeddings):
    print("\n--- 步骤四: 验证元数据 ---")
    if not os.path.exists(VECTOR_DB_PATH):
        print("数据库不存在，无法验证。")
        return
    db = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embeddings)
    retrieved_docs = db.similarity_search("revenue", k=3)
    print("  - 对'revenue'进行相似度搜索，检索到的前3个块为：\n")
    for doc in retrieved_docs:
        print("内容片段: {}...".format(doc.page_content[:120].replace("\n", " ")))
        print(f"    元数据: {doc.metadata}")
        print("    " + "-" * 20)

if __name__ == "__main__":
    # --- 经过重构，主流程更加清晰 ---
    # 1. 加载
    all_documents = load_docs_from_json()

    if all_documents:
        # 2. 分割
        final_chunks = chunk_documents(all_documents)

        # 3. 嵌入与存储
        print("\n正在初始化嵌入模型...")
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cuda"},
            encode_kwargs={"normalize_embeddings": True}
        )
        create_and_persist_vector_store(final_chunks, embeddings)

        # 4. 验证
        verify_metadata(embeddings)
        print(f"\n🎉 恭喜！数据处理与索引流程已全部优化完成！")
    else:
        print("未能加载任何文档，流程终止。")