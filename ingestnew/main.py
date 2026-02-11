# 尝试导入处理模块

# 导入音频处理模块
from process_audio import run_process_audio

# 导入PDF处理模块
from process_pdf import run_process_pdf

# 导入图片摘要模块
from summarize_images import summarize_and_clean

if __name__ == "__main__":
    print("=== 开始完整处理流程 ===")
    
    # 处理音频文件
    print("\n1. 处理音频文件...")
    run_process_audio()
    
    # 处理PDF文件
    print("\n2. 处理PDF文件...")
    run_process_pdf()
    
    # 处理图片摘要
    print("\n3. 生成图片摘要...")
    summarize_and_clean()
    
    print("\n=== 完整处理流程完成 ===")
