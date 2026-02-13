import os
from faster_whisper import WhisperModel

DATA_DIR = "../data"
OUTPUT_DIR = "../processed_data"

model = WhisperModel("medium", device="cuda", compute_type="float16")

def transcribe(input_path, output_path):
    print(f"转录: {input_path}")
    segments, _ = model.transcribe(input_path, beam_size=5)
    with open(output_path, "w", encoding="utf-8") as f:
        for s in segments:
            f.write(f"[{int(s.start)}s->{int(s.end)}s]{s.text.strip()}\n")

def process_audio():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for f in os.listdir(DATA_DIR):
        if f.lower().endswith(('.mp3', '.mp4', '.m4a', '.wav')):
            transcribe(os.path.join(DATA_DIR, f), os.path.join(OUTPUT_DIR, f"{os.path.splitext(f)[0]}.txt"))
    print("音频处理完成")

if __name__ == "__main__":
    process_audio()
