from paddleocr import PaddleOCRVL
import os
from pathlib import Path

class PaddleOCRWrapper:
    def __init__(self):
        self.pipeline = None
        self.initialize()
    
    def initialize(self):
        """初始化PaddleOCR-VL"""
        print("正在初始化PaddleOCR-VL...")
        try:
            self.pipeline = PaddleOCRVL()
            print("PaddleOCR-VL初始化成功！")
        except Exception as e:
            print(f"PaddleOCR-VL初始化失败: {e}")
            self.pipeline = None
    
    def process_pdf(self, pdf_path, output_dir):
        """处理PDF文件并返回结构化结果
        
        Args:
            pdf_path: PDF文件路径
            output_dir: 输出目录
            
        Returns:
            list: 处理结果列表
        """
        if not self.pipeline:
            print("PaddleOCR-VL未初始化，无法处理PDF")
            return None
        
        try:
            print(f"正在处理PDF文件: {pdf_path}")
            
            # 创建输出目录
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)
            
            # 处理PDF
            output = self.pipeline.predict(input=pdf_path)
            pages_res = list(output)
            
            # 重组页面
            # 尝试使用合并跨页表格参数
            try:
                restructured_output = self.pipeline.restructure_pages(pages_res, merge_table=True)
                print("成功使用跨页表格合并功能")
            except TypeError:
                # 如果参数不支持，使用默认方式
                restructured_output = self.pipeline.restructure_pages(pages_res)
                print("跨页表格合并参数不支持，使用默认方式")
            
            # 保存结果
            results = []
            for i, res in enumerate(restructured_output):
                page_num = i + 1
                print(f"处理页面 {page_num}...")
                
                # 获取PDF文件名
                pdf_filename = Path(pdf_path).stem
                
                # 保存JSON结果
                json_path = output_dir / f"{pdf_filename}_page_{page_num}_result.json"
                res.save_to_json(save_path=str(json_path))
                
                # 保存Markdown结果
                md_path = output_dir / f"{pdf_filename}_page_{page_num}_result.md"
                # 先保存Markdown，图片会默认保存在md文件同目录的imgs文件夹
                res.save_to_markdown(save_path=str(md_path))
                
                # 读取Markdown内容
                with open(md_path, 'r', encoding='utf-8') as f:
                    markdown_content = f.read()
                
                results.append({
                    'page_num': page_num,
                    'markdown_content': markdown_content,
                    'json_path': str(json_path),
                    'md_path': str(md_path)
                })
            
            print(f"PDF处理完成，共处理 {len(results)} 页")
            return results
            
        except Exception as e:
            print(f"处理PDF时出错: {e}")
            import traceback
            traceback.print_exc()
            return None

# 测试代码
if __name__ == "__main__":
    ocr_wrapper = PaddleOCRWrapper()
    
    # 测试PDF处理
    test_pdf = "../data/2.pdf"
    output_dir = "./output_test"
    
    if os.path.exists(test_pdf):
        # 测试PDF处理
        print("测试PDF处理...")
        results = ocr_wrapper.process_pdf(test_pdf, output_dir)
        if results:
            print(f"测试成功！共处理 {len(results)} 页")
        else:
            print("测试失败！")
    else:
        print(f"测试文件不存在: {test_pdf}")
