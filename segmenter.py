"""
Audio Segmentation Module

This module handles the segmentation of audio files based on speaker diarization results.
It splits audio files into segments corresponding to different speakers.
"""

import os
from pydub import AudioSegment


def split_audio_by_diarization(audio_file, diarization_result, output_dir="segments"):
    """
    Split audio file into segments based on pyannote.audio diarization results.
    
    Args:
        audio_file (str): Path to the original audio file
        diarization_result (pyannote.core.Annotation): Diarization result object
        output_dir (str): Directory to save the segmented files
        
    Returns:
        list: List of tuples containing (segment_filename, speaker_id)
    """
    os.makedirs(output_dir, exist_ok=True)

    sound = AudioSegment.from_wav(audio_file)
    segments = []

    for i, (turn, _, speaker) in enumerate(diarization_result.itertracks(yield_label=True)):
        start_ms = int(turn.start * 1000)
        end_ms = int(turn.end * 1000)

        sub_audio = sound[start_ms:end_ms]
        filename = os.path.join(output_dir, f"{i:03d}_{speaker}.wav")
        sub_audio.export(filename, format="wav")

        segments.append((filename, speaker))
        print(f"[{start_ms}ms - {end_ms}ms] -> {filename}")

    print(f"✅ Audio segmentation completed: {len(segments)} segments saved to {output_dir}")
    return segments