import os
import json
import shutil
import uuid
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from zhipuai import ZhipuAI

# --- 配置 ---
PROCESSED_DATA_FOLDER = "../processed_data"
VECTOR_DB_PATH = "../chroma_db_optimized"
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"

# --- 初始化 LLM 用于生成摘要 ---
try:
    client = ZhipuAI(api_key=os.getenv("ZHIPU_API_KEY"))
    print("智谱AI客户端初始化成功，将用于生成摘要。")
except Exception as e:
    client = None
    print(f"智谱AI客户端初始化失败: {e}。摘要生成功能将不可用。")


def generate_summary_with_llm(content):
    """使用LLM为文本块生成摘要。"""
    if not client:
        return "摘要功能未启用。"
    if len(content) < 500:
        return content
    prompt = f"""你是一个专业的金融文档分析师。请为以下财报内容生成一个简洁、精确、信息密集的摘要，不超过150个字。摘要需要捕捉最关键的实体、数据和结论。\n\n内容如下：\n---\n{content}\n---\n\n摘要："""
    try:
        response = client.chat.completions.create(
            model="glm-4-flash",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300, temperature=0.0
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"  - 调用LLM生成摘要时出错: {e}")
        return "摘要生成失败。"


def load_and_prepare_docs():
    """
    加载所有处理好的JSON和TXT文件，创建父文档。
    """
    print("--- 步骤一: 加载并准备父文档 (已修正) ---")

    image_summaries_path = os.path.join(PROCESSED_DATA_FOLDER, "image_summaries.json")
    if os.path.exists(image_summaries_path):
        with open(image_summaries_path, 'r', encoding='utf-8') as f:
            image_summaries = json.load(f)
        print(f"  - 已加载 {len(image_summaries)} 条图片摘要。")
    else:
        image_summaries = {}
        print("  - 未找到图片摘要文件，将继续处理。")

    parent_docs = []
    for filename in sorted(os.listdir(PROCESSED_DATA_FOLDER)):
        file_path = os.path.join(PROCESSED_DATA_FOLDER, filename)

        # --- 新增逻辑：同时处理 .txt 文件 ---
        if filename.endswith("_transcribed.txt"):
            print(f"  - 正在处理音频转录稿: {filename}")
            with open(file_path, 'r', encoding='utf-8') as f:
                full_text = f.read()

            # 将整个转录稿视为一个大的父文档
            doc_id = str(uuid.uuid4())
            doc = Document(
                page_content=full_text,
                metadata={
                    "source": filename.replace("_transcribed.txt", ""),
                    "page": 1,  # 对于转录稿，我们将其视为单页
                    "doc_id": doc_id
                }
            )
            parent_docs.append(doc)

        # --- 原有逻辑：处理 .json 文件 ---
        elif filename.endswith(".json") and filename != "image_summaries.json":
            print(f"  - 正在处理PDF解析文件: {filename}")
            with open(file_path, 'r', encoding='utf-8') as f:
                blocks = json.load(f)

            pages = {}
            for block in blocks:
                page_num = block["metadata"]["page"]
                if page_num not in pages:
                    pages[page_num] = []

                content = block["content"]
                if block["type"] == "image_placeholder":
                    img_name = content.replace("[IMAGE: ", "").replace("]", "")
                    summary = image_summaries.get(img_name, "")
                    content = f"--- [参考图片: {img_name}] ---\n{summary}\n--- [图片描述结束] ---"

                pages[page_num].append(content)

            for page_num, contents in sorted(pages.items()):
                full_page_content = "\n\n".join(contents)
                doc_id = str(uuid.uuid4())
                doc = Document(
                    page_content=full_page_content,
                    metadata={
                        "source": filename.replace("_processed.json", ""),
                        "page": page_num,
                        "doc_id": doc_id
                    }
                )
                parent_docs.append(doc)

    print(f"  - 总共创建了 {len(parent_docs)} 个父文档 (来自PDF和TXT)。")
    return parent_docs


def create_child_chunks_and_summaries(parent_docs):
    """
    为父文档创建子块和摘要。
    """
    print("\n--- 步骤二: 创建子块与摘要 ---")

    child_chunks = []
    summary_docs = []

    child_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

    for i, parent_doc in enumerate(parent_docs):
        doc_id = parent_doc.metadata["doc_id"]
        source_info = f"{parent_doc.metadata['source']} (页码 {parent_doc.metadata['page']})"
        print(f"  - 正在处理父文档 ({i + 1}/{len(parent_docs)}): {source_info}")

        chunks = child_splitter.split_text(parent_doc.page_content)
        for j, chunk_text in enumerate(chunks):
            chunk_doc = Document(
                page_content=chunk_text,
                metadata={**parent_doc.metadata, "is_child": True, "chunk_index": j}
            )
            child_chunks.append(chunk_doc)

        summary_text = generate_summary_with_llm(parent_doc.page_content)
        summary_doc = Document(
            page_content=summary_text,
            metadata={**parent_doc.metadata, "is_summary": True}
        )
        summary_docs.append(summary_doc)

    print(f"  - 已创建 {len(child_chunks)} 个子块。")
    print(f"  - 已创建 {len(summary_docs)} 个摘要。")

    return child_chunks, summary_docs


def create_and_persist_vector_store(docs_to_index, embeddings):
    """创建并持久化向量库。"""
    print("\n--- 步骤三: 嵌入数据并构建向量库 ---")
    if os.path.exists(VECTOR_DB_PATH):
        print(f"  - 发现旧的数据库，正在删除: {VECTOR_DB_PATH}")
        shutil.rmtree(VECTOR_DB_PATH)

    Chroma.from_documents(docs_to_index, embeddings, persist_directory=VECTOR_DB_PATH)
    print("  - 向量数据库创建完毕并已成功持久化！")


def verify_retrieval(embeddings):
    """验证检索效果。"""
    print("\n--- 步骤四: 验证检索效果 ---")
    if not os.path.exists(VECTOR_DB_PATH):
        print("数据库不存在，无法验证。")
        return

    db = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embeddings)

    # 验证一个来自音频转录稿的查询
    query = "are there other that have yet to be announced of the same kind of scale and magnitude?"
    print(f"  - 模拟音频内容查询: '{query}'")

    retrieved_chunks = db.similarity_search(
        query,
        k=3,
        filter={"is_child": True}
    )

    print("\n  - 检索到的前3个精确块为：\n")
    if not retrieved_chunks:
        print("    未能检索到任何相关内容。")
    else:
        for doc in retrieved_chunks:
            print("    内容片段: {}...".format(doc.page_content[:150].replace("\n", " ")))
            print(f"      元数据: {doc.metadata}")
            print("      " + "-" * 20)


if __name__ == "__main__":
    # 1. 加载并准备父文档
    parent_documents = load_and_prepare_docs()

    if parent_documents:
        # 2. 创建子块和摘要
        child_chunks, summary_docs = create_child_chunks_and_summaries(parent_documents)
        all_docs_to_index = child_chunks + summary_docs

        # 3. 嵌入与存储
        print("\n正在初始化嵌入模型...")
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cuda"},
            encode_kwargs={"normalize_embeddings": True}
        )
        create_and_persist_vector_store(all_docs_to_index, embeddings)

        # 4. 验证
        verify_retrieval(embeddings)
        print(f"\n🎉 恭喜！数据处理与索引流程已全部修复并完成！")
    else:
        print("未能加载任何文档，流程终止。")