import os
import json
import re
from pathlib import Path
from bs4 import BeautifulSoup

class MDToJSONConverter:
    def __init__(self):
        pass
    
    def clean_markdown(self, md_content):
        """清洗Markdown内容，去除HTML标签等
        
        Args:
            md_content: Markdown内容
            
        Returns:
            str: 清洗后的内容
        """
        # 使用BeautifulSoup去除HTML标签
        soup = BeautifulSoup(md_content, 'html.parser')
        cleaned_content = soup.get_text()
        
        # 去除多余的空行
        cleaned_content = re.sub(r'\n+', '\n', cleaned_content)
        
        # 去除首尾空白
        cleaned_content = cleaned_content.strip()
        
        return cleaned_content
    
    def extract_images(self, md_content):
        """从Markdown内容中提取图片路径
        
        Args:
            md_content: Markdown内容
            
        Returns:
            list: 图片路径列表
        """
        # 匹配Markdown图片格式: ![alt](path)
        img_pattern = r'!\[.*?\]\((.*?)\)'
        img_paths = re.findall(img_pattern, md_content)
        
        return img_paths
    
    def process_file(self, md_path, pdf_filename):
        """处理单个MD文件并转换为JSON
        
        Args:
            md_path: MD文件路径
            pdf_filename: PDF文件名
            
        Returns:
            list: JSON对象列表
        """
        # 读取MD文件
        with open(md_path, 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        # 提取页码
        page_num = 1
        page_match = re.search(r'page_(\d+)_result', str(md_path))
        if page_match:
            page_num = int(page_match.group(1))
        
        # 构建metadata
        metadata = {
            'source': pdf_filename,
            'page': page_num
        }
        
        # 处理内容
        json_objects = []
        
        # 提取并添加图片占位符
        img_paths = self.extract_images(md_content)
        for img_path in img_paths:
            # 提取图片文件名
            img_name = os.path.basename(img_path)
            json_objects.append({
                'type': 'image_placeholder',
                'content': f"[IMAGE: {img_name}]",
                'metadata': metadata.copy()
            })
        
        # 清洗Markdown内容并添加为prose类型
        cleaned_content = self.clean_markdown(md_content)
        if cleaned_content:
            json_objects.append({
                'type': 'prose',
                'content': cleaned_content,
                'metadata': metadata.copy()
            })
        
        return json_objects
    
    def convert_all(self, input_dir, output_json_path):
        """转换所有MD文件为单个JSON文件
        
        Args:
            input_dir: 输入目录，包含MD文件
            output_json_path: 输出JSON文件路径
        """
        print(f"=== 开始转换MD文件为JSON ===")
        
        # 收集所有MD文件
        md_files = list(Path(input_dir).glob('*.md'))
        print(f"找到 {len(md_files)} 个MD文件")
        
        # 处理所有MD文件
        all_json_objects = []
        for md_file in md_files:
            print(f"处理文件: {md_file.name}")
            
            # 提取PDF文件名
            pdf_filename = Path(input_dir).name
            
            # 处理单个文件
            json_objects = self.process_file(md_file, pdf_filename)
            all_json_objects.extend(json_objects)
        
        # 保存为JSON
        output_dir = Path(output_json_path).parent
        output_dir.mkdir(exist_ok=True)
        
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(all_json_objects, f, indent=4, ensure_ascii=False)
        
        print(f"转换完成，保存到: {output_json_path}")
        print(f"共生成 {len(all_json_objects)} 个JSON对象")

def main():
    """主函数"""
    # 配置路径
    input_dir = Path("./output")
    output_base_dir = Path("../processed_data")
    
    # 确保输出目录存在
    output_base_dir.mkdir(exist_ok=True)
    
    # 获取所有PDF子目录
    pdf_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
    
    # 创建转换器
    converter = MDToJSONConverter()
    
    # 处理每个PDF目录
    for pdf_dir in pdf_dirs:
        pdf_filename = pdf_dir.name
        output_json_path = output_base_dir / f"{pdf_filename}_processed.json"
        
        print(f"\n处理PDF: {pdf_filename}")
        converter.convert_all(pdf_dir, str(output_json_path))

if __name__ == "__main__":
    main()
