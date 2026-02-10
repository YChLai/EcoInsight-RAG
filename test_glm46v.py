import os
from zhipuai import ZhipuAI
import base64

# 检查API密钥
API_KEY = os.getenv("ZHIPUAI_API_KEY")
if not API_KEY:
    print("错误: 环境变量 ZHIPUAI_API_KEY 未设置，无法测试GLM-4.6V-Flash。")
    exit(1)

# 初始化客户端
client = ZhipuAI(api_key=API_KEY)
print("智谱AI客户端初始化成功。")

# 测试提示词
PROMPT = "你是一位顶尖的金融分析师。请用中文详细、客观地描述这张图片。如果它是一个图表，请解读其标题、坐标轴含义、数据趋势等。"

# 测试函数
def test_glm46v_flash():
    print("\n测试GLM-4.6V-Flash模型...")
    
    # 创建一个简单的测试请求
    try:
        # 使用一个示例图片URL（这里使用一个公开的测试图片）
        # 注意：在实际使用中，你需要将图片转换为base64编码
        # 这里我们只测试模型是否能被正确调用
        response = client.chat.completions.create(
            model="GLM-4.6V-Flash",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3b/Line_chart_example.svg/1200px-Line_chart_example.svg.png"
                        }
                    }
                ]
            }]
        )
        
        # 打印响应
        if response and response.choices:
            content = response.choices[0].message.content
            print("\n模型响应:")
            print(content[:500] + "..." if len(content) > 500 else content)
            print("\n✓ GLM-4.6V-Flash模型调用成功！")
        else:
            print("\n✗ 模型响应异常。")
            
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")

if __name__ == "__main__":
    test_glm46v_flash()
