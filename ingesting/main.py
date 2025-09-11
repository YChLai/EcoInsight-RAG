from process_audio import run_process_audio
from process_pdf import run_process_pdf
from summarize_images import summarize_and_clean

if __name__ == "__main__":
    run_process_audio()
    run_process_pdf()
    summarize_and_clean()