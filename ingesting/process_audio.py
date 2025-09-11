import os
from faster_whisper import WhisperModel
import time
AUDIO_FOLDER="../data"
OUTPUT_FOLDER="../processed_data"
MODEL_SIZE="medium"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

model=WhisperModel(MODEL_SIZE,device="cuda",compute_type="float16")
print("模型加载完毕。")

def transcribe_audio(audio_path,output_filename):
    """
    使用faster whisper模型将指定音频文件转录为文本
    """
    print(f"开始转录文件:{audio_path}...")
    start_time=time.time()
    segments,info=model.transcribe(audio_path,beam_size=5)
    print(f"检测到语言{info.language}，概率为{info.language_probability:.2f}")

    with open(output_filename,"w",encoding="utf-8") as f:
        for segment in segments:
            line=f"[{int(segment.start)}s->{int(segment.end)}s]{segment.text.strip()}\n"
            f.write(line)

    end_time=time.time()
    duration=end_time-start_time
    print(f"文件转录完成: {output_filename}")


def run_process_audio():
    # 遍历 data 文件夹中的所有文件
    for filename in os.listdir(AUDIO_FOLDER):
        # 检查文件是否为常见的音频格式
        if filename.lower().endswith(('.mp3', '.mp4', '.m4a', '.wav')):
            audio_file_path = os.path.join(AUDIO_FOLDER, filename)
            # 构建输出文件名，例如 a.mp3 -> a_transcribed.txt
            output_file_name = f"{os.path.splitext(filename)[0]}_transcribed.txt"
            output_file_path = os.path.join(OUTPUT_FOLDER, output_file_name)

            transcribe_audio(audio_file_path, output_file_path)

    print("\n所有音频文件处理完毕！")

if __name__ == "__main__":
    run_process_audio()