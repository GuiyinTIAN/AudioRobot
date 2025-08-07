import os
import time
from datetime import datetime
import torch
from faster_whisper import WhisperModel


class ASRTranscriber:
    """
    ASR transcription service using Whisper models.
    
    Handles model initialization and provides methods for transcribing audio files.
    """
    
    def __init__(self, model_name="large-v2", device="auto", compute_type="int8"):
        """
        Initialize the ASR transcriber with specified model settings.
        
        Args:
            model_name (str): Name of the Whisper model to use
            device (str): Device to run the model on ('cpu', 'cuda', or 'auto')
            compute_type (str): Computation type for model inference
        """
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
        print(f"🔧 Initializing Whisper model: {model_name}")
        print(f"💻 Using device: {device}")
        print(f"🧮 Compute type: {compute_type}")

        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type

        self.model = WhisperModel(
            model_name, 
            device=device, 
            compute_type=compute_type, 
            cpu_threads=16, 
            num_workers=1
        )

    def transcribe(self, audio_path, language="zh"):
        """
        Transcribe an audio file.
        
        Args:
            audio_path (str): Path to the audio file
            language (str): Language code for transcription (default: 'zh')
            
        Returns:
            tuple: (transcribed_text, detected_language)
        """
        segments, info = self.model.transcribe(
            audio_path, 
            beam_size=1,  # Reduced from 5 for 5x speed improvement
            language=language,
            temperature=0.0,  # Fixed temperature
            vad_filter=True,  # Filter silent parts
            vad_parameters=dict(min_silence_duration_ms=300),
            condition_on_previous_text=False
        )
        
        text = " ".join([seg.text for seg in segments]).strip()
        return text, info.language
            
if __name__ == "__main__":
    import time
    import os
    from datetime import datetime
    segments_dir = "outputs/session_20250731_195445/segments"  # 替换为实际的音频文件目录
    
    if os.path.exists(segments_dir):
        audio_files = sorted([f for f in os.listdir(segments_dir) if f.endswith('.wav')])
        
        if audio_files:

            model_load_start = time.time()

            asr = ASRTranscriber(model_name="large-v2", device="auto")

            model_load_time = time.time() - model_load_start
            print(f"⚡ 模型加载完成，耗时: {model_load_time:.2f}秒")
            
            # 记录处理开始时间
            processing_start_time = time.time()
            results = []
            
            for audio_file in audio_files:
                audio_path = os.path.join(segments_dir, audio_file)
                print(f"\n🎵 正在处理: {audio_file}")
                
                start_time = time.time()
                text, language = asr.transcribe(audio_path)
                elapsed_time = time.time() - start_time
                
                print(f"📝 文本: {text}")
                print(f"🌍 语言: {language}")
                print(f"⏱️  耗时: {elapsed_time:.2f}秒")

                results.append({
                    'file': audio_file,
                    'text': text,
                    'language': language,
                    'duration': elapsed_time
                })
            
            processing_time = time.time() - processing_start_time
            total_time = time.time() - model_load_start
            
            print(f"\n🎉 所有处理完成！")
            print(f"📊 总文件数: {len(audio_files)}")
            print(f"⚡ 模型加载时间: {model_load_time:.2f}秒")
            print(f"🔄 纯处理时间: {processing_time:.2f}秒")
            print(f"⏱️  总时间: {total_time:.2f}秒")
            print(f"📈 平均处理时间: {processing_time/len(audio_files):.2f}秒/文件")
            
            # 生成输出文件名（带时间戳）
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = asr.model_name
            output_file = f"{segments_dir}/{model_name}_transcription_results_{timestamp}.txt"
            
            # 将结果写入文件
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"语音识别结果报告\n")
                f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"处理文件数: {len(audio_files)}\n")
                f.write(f"模型加载时间: {model_load_time:.2f}秒\n")
                f.write(f"纯处理时间: {processing_time:.2f}秒\n")
                f.write(f"总时间: {total_time:.2f}秒\n")
                f.write(f"平均处理时间: {processing_time/len(audio_files):.2f}秒/文件\n")
                f.write("="*50 + "\n\n")
                
                for i, result in enumerate(results, 1):
                    f.write(f"[{i:03d}] {result['file']}\n")
                    f.write(f"语言: {result['language']}\n")
                    f.write(f"耗时: {result['duration']:.2f}秒\n")
                    f.write(f"内容: {result['text']}\n")
                    f.write("-" * 30 + "\n\n")
            
            print(f"📄 转录结果已保存到: {output_file}")
            
        else:
            print("❌ 没有找到 .wav 文件")
    else:
        print(f"❌ 目录不存在: {segments_dir}")



