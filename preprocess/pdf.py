import os
import json
import re
from pathlib import Path
from bs4 import BeautifulSoup
from paddleocr import PaddleOCRVL

DATA_DIR = "../data"
OUTPUT_DIR = "./output"
PROCESSED_DIR = "../processed_data"

class PDFProcessor:
    def __init__(self):
        self.pipeline = PaddleOCRVL()
    
    def process(self, pdf_path, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        output = self.pipeline.predict(input=pdf_path)
        pages = list(output)
        
        try:
            results = self.pipeline.restructure_pages(pages, merge_table=True)
        except:
            results = self.pipeline.restructure_pages(pages)
        
        for i, res in enumerate(results):
            name = Path(pdf_path).stem
            res.save_to_json(str(output_dir / f"{name}_page_{i+1}.json"))
            res.save_to_markdown(str(output_dir / f"{name}_page_{i+1}.md"))
        return len(results)

class MDConverter:
    def clean(self, content):
        return re.sub(r'\n+', '\n', BeautifulSoup(content, 'html.parser').get_text()).strip()
    
    def extract_images(self, content):
        return re.findall(r'!\[.*?\]\((.*?)\)', content)
    
    def convert(self, md_path, pdf_name):
        with open(md_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        page_match = re.search(r'page_(\d+)', str(md_path))
        page = int(page_match.group(1)) if page_match else 1
        
        metadata = {'source': pdf_name, 'page': page}
        objects = []
        
        for img in self.extract_images(content):
            objects.append({'type': 'image', 'content': f"[IMAGE: {os.path.basename(img)}]", 'metadata': metadata.copy()})
        
        cleaned = self.clean(content)
        if cleaned:
            objects.append({'type': 'text', 'content': cleaned, 'metadata': metadata.copy()})
        return objects

def process_pdf():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    processor = PDFProcessor()
    converter = MDConverter()
    
    for pdf in Path(DATA_DIR).glob("*.pdf"):
        print(f"处理: {pdf.name}")
        out_dir = Path(OUTPUT_DIR) / pdf.stem
        out_dir.mkdir(exist_ok=True)
        processor.process(str(pdf), str(out_dir))
        
        objects = []
        for md in out_dir.glob("*.md"):
            objects.extend(converter.convert(md, pdf.stem))
        
        with open(f"{PROCESSED_DIR}/{pdf.stem}.json", 'w', encoding='utf-8') as f:
            json.dump(objects, f, indent=2, ensure_ascii=False)
    
    print("PDF处理完成")

if __name__ == "__main__":
    process_pdf()
