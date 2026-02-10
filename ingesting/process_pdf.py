import json
import os
from collections import Counter
from zhipuai import ZhipuAI
import fitz  # PyMuPDF
from image_filter import is_image_valuable

# --- 配置 ---
PDF_FOLDER = "../data"
OUTPUT_FOLDER = "../processed_data"
IMAGE_FOLDER = os.path.join(OUTPUT_FOLDER, "images")

# --- 健壮性提升：在初始化时检查API Key ---
API_KEY = os.getenv("ZHIPUAI_API_KEY")
ZHIPU_CLIENT = None
if not API_KEY:
    print("警告: 环境变量 ZHIPUAI_API_KEY 未设置，表格摘要功能将不可用。")
else:
    try:
        ZHIPU_CLIENT = ZhipuAI(api_key=API_KEY)
        print("智谱AI客户端初始化成功。")
    except Exception as e:
        print(f"智谱AI客户端初始化失败: {e}")

# --- 初始化文件目录 ---
if not os.path.exists(OUTPUT_FOLDER): os.makedirs(OUTPUT_FOLDER)
if not os.path.exists(IMAGE_FOLDER): os.makedirs(IMAGE_FOLDER)


# (识别页眉页脚 和 to_markdown_table 函数非常清晰，无需改动)
def identify_headers_footers(doc, sample_pages=10, top_margin=0.1, bottom_margin=0.9):
    # ... 代码和之前完全一样 ...
    header_footer_candidates = []
    page_height = doc[0].rect.height if len(doc) > 0 else 0
    num_pages = len(doc)
    if num_pages <= sample_pages:
        pages_to_sample = range(num_pages)
    else:
        step = num_pages // sample_pages; pages_to_sample = [i * step for i in range(sample_pages)]
    for page_num in pages_to_sample:
        page = doc.load_page(page_num)
        blocks = page.get_text("blocks")
        for block in blocks:
            y0, y1, block_text = block[1], block[3], block[4].strip()
            if y0 < page_height * top_margin or y1 > page_height * bottom_margin:
                if block_text: header_footer_candidates.append(block_text)
    counts = Counter(header_footer_candidates)
    return {text for text, count in counts.items() if count > 1}


def to_markdown_table(table_data):
    # ... 代码和之前完全一样 ...
    table_data = [['' if item is None else str(item).replace('\n', ' ') for item in row] for row in table_data]
    header = table_data[0];
    body = table_data[1:]
    markdown = "| " + " | ".join(header) + " |\n"
    markdown += "| " + " | ".join(["---"] * len(header)) + " |\n"
    for row in body: markdown += "| " + " | ".join(row) + " |\n"
    return markdown


def summarize_table_with_llm(markdown_table):
    if not ZHIPU_CLIENT: return None  # 如果客户端未初始化，则跳过
    prompt = f"""你是一个专业的金融数据分析师。请用一段话（不超过150字）总结以下Markdown表格的核心内容和主要趋势。你的总结应该是对表格的高度概括。\n表格如下：\n{markdown_table}"""
    try:
        response = ZHIPU_CLIENT.chat.completions.create(
            model="glm-4", messages=[{"role": "user", "content": prompt}], max_tokens=300, temperature=0.1
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"  - 调用LLM生成表格摘要时出错: {e}")
        return None


# --- 核心重构：将页面处理逻辑提炼为独立函数 ---
def extract_content_from_page(page, pdf_name, noise_strings):
    """处理单个页面，提取所有内容块。"""
    page_blocks = []
    page_num = page.number + 1
    page_area = page.rect.width * page.rect.height
    base_metadata = {"source": pdf_name, "page": page_num}

    # 1. 图片处理
    valuable_image_bboxes = []
    for img_index, img in enumerate(page.get_images(full=True)):
        try:
            img_bbox = page.get_image_bbox(img)
            pix = page.get_pixmap(clip=img_bbox, alpha=False)
            image_name = f"{os.path.splitext(pdf_name)[0]}_page{page_num}_img{img_index}.png"
            image_path = os.path.join(IMAGE_FOLDER, image_name)
            pix.save(image_path)
            is_valuable, _ = is_image_valuable(image_path, page_area)
            if is_valuable:
                valuable_image_bboxes.append(img_bbox)
                page_blocks.append(
                    {"type": "image_placeholder", "content": f"[IMAGE: {image_name}]", "metadata": base_metadata})
            else:
                os.remove(image_path)
        except Exception:
            continue

    # 2. 表格处理 (逻辑内聚)
    tables = page.find_tables()
    table_bboxes = [fitz.Rect(t.bbox) for t in tables]
    for table in tables:
        table_data = table.extract()
        if table_data:
            markdown_table = to_markdown_table(table_data)
            page_blocks.append(
                {"type": "table", "content": f"[TABLE]\n{markdown_table}[/TABLE]", "metadata": base_metadata})
            summary = summarize_table_with_llm(markdown_table)
            if summary:
                page_blocks.append({"type": "table_summary", "content": summary, "metadata": base_metadata})

    # 3. 文本处理
    text_blocks = page.get_text("blocks")
    for block in text_blocks:
        block_text = block[4].strip()
        if block_text and block_text not in noise_strings:
            block_rect = fitz.Rect(block[0:4])
            is_in_image = any(block_rect.intersects(bbox) for bbox in valuable_image_bboxes)
            is_in_table = any(block_rect.intersects(bbox) for bbox in table_bboxes)
            if not is_in_image and not is_in_table:
                page_blocks.append({"type": "prose", "content": block_text, "metadata": base_metadata})

    return page_blocks


def process_pdf_to_json(pdf_path, output_json_filename, pdf_name):
    """主协调函数：负责文档级操作，调用页面处理函数。"""
    print(f"正在处理 PDF: {pdf_path}")
    doc = fitz.open(pdf_path)
    noise_strings = identify_headers_footers(doc)
    print(f"  -> 已识别并过滤页眉/脚噪声: {noise_strings or '无'}")

    all_content_blocks = []
    for page in doc:
        page_content = extract_content_from_page(page, pdf_name, noise_strings)
        all_content_blocks.extend(page_content)

    with open(output_json_filename, "w", encoding="utf-8") as f:
        json.dump(all_content_blocks, f, indent=4, ensure_ascii=False)
    print(f"  -> 成功生成纯净版的JSON: {output_json_filename}")


def run_process_pdf():
    """脚本主入口"""
    print("--- 开始数据预处理流程 (重构版) ---")
    for filename in os.listdir(PDF_FOLDER):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(PDF_FOLDER, filename)
            output_filename = f"{os.path.splitext(filename)[0]}_processed.json"
            output_path = os.path.join(OUTPUT_FOLDER, output_filename)
            process_pdf_to_json(pdf_path, output_path, filename)
    print("\n--- 所有PDF已处理成结构化的JSON文件！ ---")


if __name__ == "__main__":
    run_process_pdf()