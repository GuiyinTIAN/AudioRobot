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
            

# =============================
# For testing purposes
# =============================
if __name__ == "__main__":
    import time
    import os
    from datetime import datetime

    segments_dir = "outputs/session_20250731_195445/segments" # Replace with the actual audio segments directory

    if os.path.exists(segments_dir):
        audio_files = sorted([f for f in os.listdir(segments_dir) if f.endswith('.wav')])

        if audio_files:

            model_load_start = time.time()
            asr = ASRTranscriber(model_name="large-v2", device="auto")
            model_load_time = time.time() - model_load_start
            print(f"⚡ Model loaded in: {model_load_time:.2f}s")

            # Track processing time
            processing_start_time = time.time()
            results = []

            for audio_file in audio_files:
                audio_path = os.path.join(segments_dir, audio_file)
                print(f"\n🎵 Processing: {audio_file}")

                start_time = time.time()
                text, language = asr.transcribe(audio_path)
                elapsed_time = time.time() - start_time

                print(f"📝 Text: {text}")
                print(f"🌍 Language: {language}")
                print(f"⏱️  Elapsed: {elapsed_time:.2f}s")

                results.append({
                    'file': audio_file,
                    'text': text,
                    'language': language,
                    'duration': elapsed_time
                })

            processing_time = time.time() - processing_start_time
            total_time = time.time() - model_load_start

            print(f"\n🎉 All processing done!")
            print(f"📊 Total files: {len(audio_files)}")
            print(f"⚡ Model load time: {model_load_time:.2f}s")
            print(f"🔄 Pure processing time: {processing_time:.2f}s")
            print(f"⏱️  Total time: {total_time:.2f}s")
            print(f"📈 Avg processing time: {processing_time/len(audio_files):.2f}s/file")

            # Generate output filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = asr.model_name
            output_file = f"{segments_dir}/{model_name}_transcription_results_{timestamp}.txt"

            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("ASR Transcription Report\n")
                f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Files processed: {len(audio_files)}\n")
                f.write(f"Model load time: {model_load_time:.2f}s\n")
                f.write(f"Pure processing time: {processing_time:.2f}s\n")
                f.write(f"Total time: {total_time:.2f}s\n")
                f.write(f"Average processing time: {processing_time/len(audio_files):.2f}s/file\n")
                f.write("=" * 50 + "\n\n")

                for i, result in enumerate(results, 1):
                    f.write(f"[{i:03d}] {result['file']}\n")
                    f.write(f"Language: {result['language']}\n")
                    f.write(f"Elapsed: {result['duration']:.2f}s\n")
                    f.write(f"Content: {result['text']}\n")
                    f.write("-" * 30 + "\n\n")

            print(f"📄 Transcription results saved to: {output_file}")

        else:
            print("❌ No .wav files found")
    else:
        print(f"❌ Directory does not exist: {segments_dir}")



