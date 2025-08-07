"""
AudioRobot - Main Application

This is the main entry point for the AudioRobot application, which handles the
complete workflow of audio recording, speaker diarization, transcription,
and summarization.
"""

# Standard library imports
import os
import json
import time
from datetime import datetime

# Local imports
from config import (
    AUDIO_FILE, RTTM_FILE, SEGMENTS_DIR, HUGGINGFACE_TOKEN, WHISPER_MODEL_NAME,
    WHISPER_DEVICE, OUTPUT_DIR, create_output_dirs, TIMESTAMP
)
from audio_recorder import record_with_pyaudio
from diarization import DiarizationPipeline
from segmenter import split_audio_by_diarization
from asr_transcriber import ASRTranscriber
from summarizer import choose_summarizer_type, OpenAISummarizer, DeepSeekSummarizer, OfflineSummarizer


def diarize(audio_path, rttm_path, hf_token):
    """
    Run speaker diarization on audio file and save RTTM file.
    
    Args:
        audio_path (str): Path to the audio file
        rttm_path (str): Output RTTM file path
        hf_token (str): HuggingFace API token
        
    Returns:
        pyannote.core.Annotation: Diarization result object
    """
    print("🚀 Step 1: Starting speaker diarization...")
    diarizer = DiarizationPipeline(hf_token=hf_token)
    diarization_result = diarizer.run(audio_path, output_rttm_path=rttm_path)
    return diarization_result


def save_transcription_results(segments, asr, output_dir, timestamp):
    """
    Save transcription results to files in appropriate formats.
    
    Args:
        segments (list): Audio segments list [(file_path, speaker_id),...]
        asr (ASRTranscriber): ASR transcriber instance
        output_dir (str): Output directory path
        timestamp (str): Timestamp string for file naming
        
    Returns:
        tuple: Paths to output files and results data
    """
    results = []
    
    print("\n=== 📝 Transcription Results ===\n")
    
    for segment_file, speaker in segments:
        text, lang = asr.transcribe(segment_file)
        if text:
            # Console output
            print(f"{speaker} [{lang.upper()}]: {text}")
            
            # Collect results for file output
            results.append({
                'speaker': speaker,
                'language': lang.upper(),
                'text': text,
                'file': segment_file
            })
    
    # Save text format transcription
    text_output_file = os.path.join(output_dir, f"transcription_{timestamp}.txt")
    
    with open(text_output_file, 'w', encoding='utf-8') as f:
        # Write header
        f.write("=" * 50 + "\n")
        f.write(f"Conversation Transcript\n")
        f.write(f"Recorded: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 50 + "\n\n")
        
        # Write conversation
        for i, result in enumerate(results, 1):
            f.write(f"[{i:03d}] {result['speaker']} [{result['language']}]:\n")
            f.write(f"{result['text']}\n\n")
        
        # Write statistics
        f.write("\n" + "=" * 50 + "\n")
        f.write("Statistics:\n")
        f.write(f"- Total turns: {len(results)}\n")
        
        # Count speaking turns per speaker
        speaker_counts = {}
        for result in results:
            speaker = result['speaker']
            speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
        
        for speaker, count in speaker_counts.items():
            f.write(f"- {speaker}: {count} turns\n")
    
    # Save structured data
    json_output_file = os.path.join(output_dir, f"structured_{timestamp}.json")
    
    with open(json_output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total_segments": len(results),
            "speakers": list(set([r['speaker'] for r in results])),
            "conversation": results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Transcription saved to: {text_output_file}")
    print(f"✅ Structured data saved to: {json_output_file}")
    
    return text_output_file, json_output_file, results


def prepare_conversation_for_ai(results):
    """
    Format transcription results for AI processing.
    
    Args:
        results (list): List of transcription result dictionaries
        
    Returns:
        list: Formatted conversation data with merged utterances
    """
    # Filter empty texts
    valid_results = [r for r in results if r['text'].strip()]
    
    # Merge consecutive utterances from the same speaker
    merged_results = []
    current_speaker = None
    current_text = ""
    
    for result in valid_results:
        if result['speaker'] == current_speaker:
            # Same speaker, merge content
            current_text += " " + result['text']
        else:
            # New speaker, save previous content and start new
            if current_speaker:
                merged_results.append({
                    'speaker': current_speaker,
                    'text': current_text.strip()
                })
            current_speaker = result['speaker']
            current_text = result['text']
    
    # Add the last speaker's content
    if current_speaker:
        merged_results.append({
            'speaker': current_speaker,
            'text': current_text.strip()
        })
    
    return merged_results


def main():
    """Main function to execute the complete workflow."""
    total_start_time = time.time()
    create_output_dirs()
    print(f"📁 Created output directory: {OUTPUT_DIR}")
    
    # Use paths from config, adjusted to full paths
    audio_file = os.path.join(OUTPUT_DIR, AUDIO_FILE)
    rttm_file = os.path.join(OUTPUT_DIR, RTTM_FILE)
    segments_dir = SEGMENTS_DIR
    time_stats = {}

    # Step 1: Audio Recording
    print("🎤 Preparing to record...")
    recording_start_time = time.time()
    record_with_pyaudio(audio_file)
    recording_time = time.time() - recording_start_time
    time_stats["Recording"] = recording_time
    print(f"⏱️ Recording completed in {recording_time:.2f}s")

    if not os.path.exists(audio_file):
        raise FileNotFoundError(f"Recording file not generated: {audio_file}")

    # Step 2: Speaker Diarization
    print("\n👥 Running speaker diarization...")
    diarization_start_time = time.time()
    # Prompt for HuggingFace token or use environment variable
    hf_token = os.environ.get("HF_TOKEN") or HUGGINGFACE_TOKEN
    
    if not hf_token:
        print("\nPlease enter your HuggingFace Token:")
        print("(Get one at: https://huggingface.co/settings/tokens)")
        hf_token = input("Token: ").strip()

        if not hf_token:
            print("❌ No HuggingFace Token provided. Cannot continue.")
            return
    
    diarization_result = diarize(audio_file, rttm_file, hf_token)
    diarization_time = time.time() - diarization_start_time
    time_stats["Speaker Diarization"] = diarization_time
    print(f"⏱️ Speaker diarization completed in {diarization_time:.2f}s")

    # Step 3: Audio Segmentation
    print("\n✂️ Segmenting audio...")
    split_start_time = time.time()
    segments = split_audio_by_diarization(audio_file, diarization_result, output_dir=segments_dir)
    split_time = time.time() - split_start_time
    time_stats["Audio Segmentation"] = split_time
    print(f"⏱️ Audio segmentation completed in {split_time:.2f}s")

    # Step 4: Initialize ASR Model
    print("\n🧠 Loading speech recognition model...")
    model_load_start_time = time.time()
    asr = ASRTranscriber(model_name=WHISPER_MODEL_NAME, device=WHISPER_DEVICE)
    model_load_time = time.time() - model_load_start_time
    time_stats["Model Loading"] = model_load_time
    print(f"⏱️ Model loading completed in {model_load_time:.2f}s")

    # Step 5: Transcribe and Output Results
    print("\n🔊 Transcribing audio segments...")
    transcribe_start_time = time.time()
    text_file, json_file, results = save_transcription_results(segments, asr, OUTPUT_DIR, TIMESTAMP)
    transcribe_time = time.time() - transcribe_start_time
    time_stats["Transcription"] = transcribe_time
    print(f"⏱️ Transcription completed in {transcribe_time:.2f}s")

    # Step 6: Generate AI-friendly Format
    print("\n🤖 Preparing AI-friendly conversation format...")
    ai_format_start_time = time.time()
    ai_friendly_conversation = prepare_conversation_for_ai(results)
    
    # Save AI-friendly format
    ai_conversation_file = os.path.join(OUTPUT_DIR, f"ai_conversation_{TIMESTAMP}.json")
    with open(ai_conversation_file, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "conversation": ai_friendly_conversation
        }, f, ensure_ascii=False, indent=2)
    
    ai_format_time = time.time() - ai_format_start_time
    time_stats["Format Conversion"] = ai_format_time
    print(f"✅ AI-friendly format saved to: {ai_conversation_file}")

    # Step 7: Generate Conversation Summary
    summary_time = 0
    try:
        print("\n📝 Generating conversation summary...")
        summary_start_time = time.time()
        
        # Let user choose summarization method
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
        
        # Generate summary
        summary_file = summarizer.summarize_to_file(
            conversation_data=ai_conversation_file,
            output_dir=OUTPUT_DIR,
            timestamp=TIMESTAMP
        )
        
        summary_time = time.time() - summary_start_time
        time_stats["Summarization"] = summary_time
        print(f"✅ Summary saved to: {summary_file}")
        print(f"⏱️ Summary generation completed in {summary_time:.2f}s")
        
    except Exception as e:
        print(f"⚠️ Summary generation failed: {e}")

    # Calculate total time
    total_time = time.time() - total_start_time
    time_stats["Total Processing"] = total_time
    
    # Create process summary file
    summary_file = os.path.join(OUTPUT_DIR, "process_summary.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"AudioRobot Processing Summary\n")
        f.write(f"{'='*30}\n")
        f.write(f"Processing time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Original audio: {os.path.basename(audio_file)}\n")
        f.write(f"Speaker diarization: {os.path.basename(rttm_file)}\n")
        f.write(f"Audio segments: {len(segments)}\n")
        f.write(f"Transcript: {os.path.basename(text_file)}\n")
        f.write(f"Structured data: {os.path.basename(json_file)}\n")
        f.write(f"Output folder: {OUTPUT_DIR}\n\n")

        # Add time statistics
        f.write(f"Time Statistics\n")
        f.write(f"{'-'*30}\n")
        for step, duration in time_stats.items():
            f.write(f"{step}: {duration:.2f}s\n")
    
    # Print total time statistics
    print(f"\n⌛ Time Statistics:")
    print(f"{'='*30}")
    for step, duration in time_stats.items():
        print(f"🕒 {step}: {duration:.2f}s")
    
    print(f"\n🎉 All files saved to folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()