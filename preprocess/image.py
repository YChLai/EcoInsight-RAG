import os
import json
import base64
import time
from zhipuai import ZhipuAI

OUTPUT_DIR = "./output"
OUTPUT_FILE = "../processed_data/images.json"

PROMPT = """描述这张图片。如果是图表，解读标题、坐标轴、趋势、关键数据；如果是表格，总结核心内容。
如果图片无信息（服务器、Logo、装饰图等），只返回: [SKIP]"""

SKIP_WORDS = ["[SKIP]", "抱歉", "无法", "模糊"]

client = ZhipuAI(api_key=os.getenv("ZHIPUAI_API_KEY"))

def is_valid(text):
    return text and not any(k in text for k in SKIP_WORDS)

def process_image():
    summaries = {}
    if os.path.exists(OUTPUT_FILE):
        summaries = json.load(open(OUTPUT_FILE, 'r', encoding='utf-8'))
    
    images = []
    for pdf_dir in os.listdir(OUTPUT_DIR):
        pdf_path = os.path.join(OUTPUT_DIR, pdf_dir)
        if os.path.isdir(pdf_path):
            for root, dirs, _ in os.walk(pdf_path):
                if 'imgs' in dirs:
                    for img in os.listdir(os.path.join(root, 'imgs')):
                        if img.lower().endswith(('.png', '.jpg', '.jpeg')) and 'header' not in img.lower() and 'footer' not in img.lower():
                            images.append(os.path.join(root, 'imgs', img))
    
    to_process = [i for i in images if os.path.basename(i) not in summaries]
    print(f"共 {len(images)} 张图片，待处理 {len(to_process)} 张")
    
    for img_path in to_process:
        name = os.path.basename(img_path)
        try:
            with open(img_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            
            resp = client.chat.completions.create(
                model="GLM-4.6V-Flash",
                messages=[{"role": "user", "content": [{"type": "text", "text": PROMPT}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]}]
            )
            summary = resp.choices[0].message.content.strip()
            
            if is_valid(summary):
                summaries[name] = summary
                with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                    json.dump(summaries, f, indent=2, ensure_ascii=False)
            time.sleep(1)
        except Exception as e:
            print(f"错误: {name} - {e}")
    
    print(f"图片处理完成，共 {len(summaries)} 条摘要")

if __name__ == "__main__":
    process_image()
