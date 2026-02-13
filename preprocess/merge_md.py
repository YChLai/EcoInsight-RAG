import os
import glob

OUTPUT_DIR = "./output"
MERGED_DIR = "./output_merged"

def merge_pages():
    os.makedirs(MERGED_DIR, exist_ok=True)
    
    for pdf_folder in os.listdir(OUTPUT_DIR):
        path = os.path.join(OUTPUT_DIR, pdf_folder)
        if not os.path.isdir(path):
            continue
        
        md_files = glob.glob(os.path.join(path, f"{pdf_folder}_page_*.md"))
        md_files.sort(key=lambda f: int(f.split('page_')[1].split('.')[0]) if 'page_' in f else 0)
        
        if not md_files:
            continue
        
        content = []
        for md in md_files:
            with open(md, 'r', encoding='utf-8') as f:
                content.append(f.read())
        
        with open(os.path.join(MERGED_DIR, f"{pdf_folder}.md"), 'w', encoding='utf-8') as f:
            f.write(("\n\n" + "="*80 + "\n\n").join(content))
        
        print(f"合并: {pdf_folder} ({len(md_files)} 页)")
    
    print("合并完成")

if __name__ == "__main__":
    merge_pages()
