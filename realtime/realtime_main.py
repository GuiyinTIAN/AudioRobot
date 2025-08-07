"""
Real-time Audio Transcription and Summarization System

This module provides the main entry point for the real-time audio transcription
system, handling initialization, recording, and summarization processes.
"""

import os
import sys
import time
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import configuration and components
from config import WHISPER_MODEL_NAME, WHISPER_DEVICE, OUTPUT_BASE_DIR, HUGGINGFACE_TOKEN
from realtime.realtime_transcriber import RealTimeTranscriber
from summarizer import choose_summarizer_type, OpenAISummarizer, DeepSeekSummarizer, OfflineSummarizer

# Constants
DEFAULT_CHUNK_DURATION = 3.5
DEFAULT_OVERLAP_DURATION = 0.5
MAX_WAIT_TIME = 10

def main_realtime():
    """
    Main function for the real-time transcription system.
    
    Handles the complete workflow from initialization to summarization.
    """
    total_start_time = time.time()
    
    print("\n" + "="*50)
    print("🚀 AudioRobot Real-time Transcription System")
    print("="*50)
    print(f"🧠 Model: {WHISPER_MODEL_NAME}")
    print(f"💻 Device: {WHISPER_DEVICE}")
    print(f"👥 Speaker recognition: Pyannote diarization")
    
    # Get HuggingFace token for speaker diarization
    print("\n🔧 Speaker recognition configuration:")
    hf_token = os.getenv('HF_TOKEN') or HUGGINGFACE_TOKEN
    
    if not hf_token:
        print("\nPlease enter your HuggingFace Token:")
        print("(Get one at: https://huggingface.co/settings/tokens)")
        hf_token = input("Token: ").strip()
        
        if not hf_token:
            print("❌ No HuggingFace Token provided. Cannot continue.")
            return
    
    # Step 1: Initialize the transcriber
    print("\n🔧 Step 1: Initializing transcription system...")
    init_start_time = time.time()
    
    try:
        transcriber = RealTimeTranscriber(
            model_name=WHISPER_MODEL_NAME,
            device=WHISPER_DEVICE,
            chunk_duration=DEFAULT_CHUNK_DURATION,
            overlap_duration=DEFAULT_OVERLAP_DURATION,
            hf_token=hf_token
        )
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        print("Suggestions:")
        print("  - Check if your HuggingFace Token is valid")
        print("  - Check your network connection")
        print("  - Ensure pyannote.audio is installed")
        return
    
    init_time = time.time() - init_start_time
    print(f"⏱️ Initialization completed in {init_time:.2f}s")
    
    # Step 2: Start real-time recording and transcription
    print(f"\n🎤 Step 2: Starting real-time recording...")
    
    recording_start_time = time.time()
    
    try:
        # Start real-time transcription (with built-in Enter interaction)
        transcriber.start_recording()
    except KeyboardInterrupt:
        print("\n🛑 Recording stopped by user")
    except Exception as e:
        print(f"\n❌ Error during recording: {e}")
    
    recording_time = time.time() - recording_start_time
    
    # Step 3: Save transcription results
    print(f"\n💾 Step 3: Saving results...")
    save_start_time = time.time()
    
    if not transcriber.conversation_history:
        print("❌ No valid conversation detected")
        print("Suggestions:")
        print("  - Check if your microphone is working")
        print("  - Ensure speaking volume is sufficient")
        return
    
    text_file, structured_file, ai_file = transcriber.save_results()
    save_time = time.time() - save_start_time
    
    print(f"⏱️ Results saved in {save_time:.2f}s")
    
    # Display transcription statistics
    print(f"\n📊 Transcription statistics:")
    print(f"  - Conversation turns: {len(transcriber.conversation_history)}")
    print(f"  - Audio segments: {len(transcriber.segments_data)}")
    print(f"  - Recording duration: {recording_time:.1f}s")
    
    for speaker, count in transcriber.speaker_activity.items():
        print(f"  - {speaker}: {count} utterances")
    
    # Step 4: Generate conversation summary
    print(f"\n📝 Step 4: Generating summary...")
    
    try:
        # Let user choose summary method
        summarizer_type = choose_summarizer_type()
        
        # Create summarizer based on selection
        if summarizer_type == "openai":
            summarizer = OpenAISummarizer()
            print("Selected: OpenAI API")
        elif summarizer_type == "deepseek":
            summarizer = DeepSeekSummarizer()
            print("Selected: DeepSeek API")
        else:
            summarizer = OfflineSummarizer()
            print("Selected: Offline model")
        
        print("🔄 Generating summary, please wait...")
        
        summary_start_time = time.time()
        
        summary_file = summarizer.summarize_to_file(
            conversation_data=ai_file,
            output_dir=transcriber.output_dir,
            timestamp=transcriber.timestamp
        )
        
        summary_time = time.time() - summary_start_time
        
        print(f"✅ Summary saved to: {summary_file}")
        print(f"⏱️ Summary generation completed in {summary_time:.2f}s")
        
        # Display summary preview
        try:
            with open(summary_file, 'r', encoding='utf-8') as f:
                summary_content = f.read()
                summary_parts = summary_content.split("=" * 50)
                if len(summary_parts) >= 3:
                    summary_body = summary_parts[2].strip()
                    preview = summary_body[:300]
                    if len(summary_body) > 300:
                        preview += "..."
                    print(f"\n📋 Summary preview:")
                    print("-" * 30)
                    print(preview)
                    print("-" * 30)
        except Exception as e:
            print(f"Cannot read summary preview: {e}")
            
    except Exception as e:
        print(f"⚠️ Failed to generate summary: {e}")
        print("Transcription results have been saved, you can generate a summary manually later")
        summary_time = 0
    
    # Calculate total time and generate final report
    total_time = init_time + recording_time + save_time + summary_time
    
    print(f"\n⌛ Processing time statistics:")
    print(f"{'='*30}")
    print(f"🕒 System initialization: {init_time:.2f}s")
    print(f"🕒 Recording & transcription: {recording_time:.2f}s")
    print(f"🕒 Saving results: {save_time:.2f}s")
    print(f"🕒 Summary generation: {summary_time:.2f}s")
    print(f"🕒 Total processing time: {total_time:.2f}s")
    
    # Write process summary to file
    process_summary_file = os.path.join(transcriber.output_dir, "process_summary.txt")
    with open(process_summary_file, 'a', encoding='utf-8') as f:
        f.write("\nTime statistics:\n")
        f.write(f"{'-'*30}\n")
        f.write(f"System initialization: {init_time:.2f}s\n")
        f.write(f"Recording & transcription: {recording_time:.2f}s\n")
        f.write(f"Saving results: {save_time:.2f}s\n")
        f.write(f"Summary generation: {summary_time:.2f}s\n")
        f.write(f"Total processing time: {total_time:.2f}s\n")
    
    print(f"\n🎉 All files saved to: {transcriber.output_dir}")
 


if __name__ == "__main__":
    
    try:
        main_realtime()
    except KeyboardInterrupt:
        print("\n\n🛑 Program interrupted by user")
    except Exception as e:
        print(f"\n❌ Program error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n👋 Thank you for using AudioRobot!")