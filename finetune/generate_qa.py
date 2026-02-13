import os
import json
import time
from zhipuai import ZhipuAI

MD_DIR = "../preprocess/output_merged"
OUTPUT_FILE = "qa.json"

client = ZhipuAI(api_key=os.getenv("ZHIPUAI_API_KEY"))

def generate_pairs(content):
    prompt = f"""基于以下文本生成3对问答，格式如下：
问题1: xxx
回答1: xxx
问题2: xxx
回答2: xxx
问题3: xxx
回答3: xxx

文本：{content[:1000]}"""
    
    for _ in range(3):
        try:
            resp = client.chat.completions.create(
                model="glm-4-flash",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800, temperature=0.0
            )
            text = resp.choices[0].message.content.strip()
            
            pairs, current = [], {}
            for line in text.split('\n'):
                line = line.strip()
                if line.startswith('问题'):
                    if 'query' in current and 'response' in current:
                        pairs.append(current)
                    current = {'query': line.split(':', 1)[1].strip()}
                elif line.startswith('回答') and current:
                    current['response'] = line.split(':', 1)[1].strip()
            if 'query' in current and 'response' in current:
                pairs.append(current)
            
            time.sleep(1)
            return pairs[:3]
        except Exception as e:
            print(f"重试: {e}")
            time.sleep(2)
    return []

def main():
    pairs = []
    for md in sorted(os.listdir(MD_DIR)):
        if not md.endswith('.md'):
            continue
        with open(os.path.join(MD_DIR, md), 'r', encoding='utf-8') as f:
            content = f.read()
        for i in range(0, len(content), 3000):
            pairs.extend(generate_pairs(content[i:i+3000]))
        print(f"处理: {md}, 累计: {len(pairs)}")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)
    print(f"完成，共 {len(pairs)} 条")

if __name__ == "__main__":
    main()
