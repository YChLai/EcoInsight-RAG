import os
from zhipuai import ZhipuAI
import base64
import json
import time

API_KEY = os.getenv("ZHIPU_API_KEY")

IMAGE_FOLDER = os.path.join("../processed_data", "images")
OUTPUT_FILE = os.path.join("../processed_data", "image_summaries.json")

PROMPT = """你是一位顶尖的金融分析师。请用中文详细、客观地描述这张图片。
如果它是一个图表（如折线图、柱状图），请解读其标题、X轴和Y轴的含义、数据的总体趋势、关键数据点、峰值和谷值。
如果它是一个表格，请总结其核心内容。
你的描述需要专业、信息密集，以便于后续的问答系统使用。

**重要规则**：如果图片是一张没有信息含量的照片（例如：服务器机架、电线、纯粹的公司Logo、演讲者照片）、装饰性背景图、或者图像模糊到无法辨认内容，**请不要进行任何描述，只返回一个词：[NO_DATA]**
"""

NEGATIVE_KEYWORDS = [
    "[NO_DATA]",
    "很抱歉",
    "抱歉",
    "无法",
    "模糊不清",
    "服务器",
    "机柜",
    "设备",
    "电线",
]

client = ZhipuAI(api_key=API_KEY)


def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


# --- 新增的辅助函数：判断摘要是否有用 ---
def is_summary_valuable(summary_text):
    """
    根据负面关键词列表，判断摘要是否为有效信息。
    """
    # 检查摘要是否为空
    if not summary_text or not summary_text.strip():
        return False
    # 检查是否包含任何一个负面关键词
    for keyword in NEGATIVE_KEYWORDS:
        if keyword in summary_text:
            return False
    # 如果通过所有检查，则认为是有价值的
    return True


# --- 主函数 ---
def summarize_and_clean():
    valuable_summaries = {}

    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            valuable_summaries = json.load(f)
        print(f"已加载 {len(valuable_summaries)} 条已存在的有效摘要。")

    all_files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    images_to_process = [f for f in all_files if f not in valuable_summaries]

    print(f"共发现 {len(all_files)} 张图片，还需处理 {len(images_to_process)} 张。")

    for index, filename in enumerate(images_to_process):
        image_path = os.path.join(IMAGE_FOLDER, filename)
        try:
            base64_image = encode_image_to_base64(image_path)
            print(f"正在处理图片 ({index + 1}/{len(images_to_process)}): {filename}...")

            response = client.chat.completions.create(
                model="glm-4v",
                messages=[{"role": "user", "content": [{"type": "text", "text": PROMPT}, {"type": "image_url",
                                                                                          "image_url": {
                                                                                              "url": f"data:image/png;base64,{base64_image}"}}]}]
            )
            summary = response.choices[0].message.content.strip()

            # --- 使用新的、更强大的判断逻辑 ---
            if is_summary_valuable(summary):
                valuable_summaries[filename] = summary
                print(f"    -> [采纳] 摘要有效！")
                # 实时保存，确保断点续传的是干净数据
                with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                    json.dump(valuable_summaries, f, indent=4, ensure_ascii=False)
            else:
                print(f"    -> [忽略] 摘要包含负面关键词或为空。内容: '{summary[:50]}...'")

            time.sleep(1)
        except Exception as e:
            print(f"处理图片 '{filename}' 时发生严重错误: {e}")

    print("\n--- 摘要生成与清洗流程全部完成！ ---")
    print(f"最终获得 {len(valuable_summaries)} 条有价值的图片摘要。")
    print(f"结果已保存在: {OUTPUT_FILE}")


if __name__ == "__main__":
    summarize_and_clean()