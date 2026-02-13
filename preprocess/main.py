from audio import process_audio
from pdf import process_pdf
from image import process_image

if __name__ == "__main__":
    print("=== 开始处理 ===")
    process_audio()
    process_pdf()
    process_image()
    print("=== 完成 ===")
