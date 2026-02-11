from paddleocr_wrapper import PaddleOCRWrapper
from clean_and_convert import MDToJSONConverter
import os
from pathlib import Path

class PDFProcessingPipeline:
    def __init__(self):
        """初始化处理管道"""
        self.ocr_wrapper = PaddleOCRWrapper()
        self.converter = MDToJSONConverter()
    
    def process_all_pdfs(self, data_dir, output_base_dir):
        """处理所有PDF文件
        
        Args:
            data_dir: 数据目录，包含PDF文件
            output_base_dir: 输出基础目录
        """
        print("=== PaddleOCR-VL PDF处理流程 ===")
        
        # 创建输出目录
        output_base_dir = Path(output_base_dir)
        output_base_dir.mkdir(exist_ok=True)
        
        # 处理所有PDF文件
        pdf_files = list(data_dir.glob("*.pdf"))
        print(f"找到 {len(pdf_files)} 个PDF文件")
        
        for pdf_file in pdf_files:
            print(f"\n=== 处理文件: {pdf_file.name} ===")
            
            # 创建输出目录
            output_dir = output_base_dir / pdf_file.stem
            output_dir.mkdir(exist_ok=True)
            
            # 处理PDF
            results = self.ocr_wrapper.process_pdf(str(pdf_file), str(output_dir))
            
            if results:
                print(f"文件处理完成，共处理 {len(results)} 页")
            else:
                print("文件处理失败")
        
        print("\n=== 所有PDF文件处理完成 ===")
    
    def convert_to_json(self, input_dir, output_base_dir):
        """将处理结果转换为JSON
        
        Args:
            input_dir: 输入目录，包含MD文件
            output_base_dir: 输出基础目录
        """
        print("\n=== 开始转换MD文件为JSON ===")
        
        # 获取所有PDF子目录
        pdf_dirs = [d for d in Path(input_dir).iterdir() if d.is_dir()]
        
        for pdf_dir in pdf_dirs:
            pdf_filename = pdf_dir.name
            output_json_path = Path(output_base_dir) / f"{pdf_filename}_processed.json"
            
            print(f"\n处理PDF: {pdf_filename}")
            self.converter.convert_all(pdf_dir, str(output_json_path))

def run_process_pdf():
    """运行PDF处理流程"""
    # 配置路径
    data_dir = Path("../data")
    output_base_dir = Path("./output")
    processed_data_dir = Path("../processed_data")
    
    # 确保处理数据目录存在
    processed_data_dir.mkdir(exist_ok=True)
    
    # 创建处理管道
    pipeline = PDFProcessingPipeline()
    
    # 处理所有PDF
    pipeline.process_all_pdfs(data_dir, output_base_dir)
    
    # 转换为JSON
    pipeline.convert_to_json(output_base_dir, processed_data_dir)

if __name__ == "__main__":
    run_process_pdf()
