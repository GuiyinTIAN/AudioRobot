import os
from datetime import datetime

# Base directories and timestamps
OUTPUT_BASE_DIR = "outputs"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join(OUTPUT_BASE_DIR, f"session_{TIMESTAMP}")

# File path configurations
AUDIO_FILE = "recording.wav"
RTTM_FILE = "diarization.rttm"
SEGMENTS_DIR = os.path.join(OUTPUT_DIR, "segments")
MODEL_DIR = "./models/diarization"

HUGGINGFACE_TOKEN = "your_openai_api_key_here"   # Replace with your actual token

# Whisper model configurations
WHISPER_MODEL_NAME = "medium"  # Options: "tiny", "base", "small", "medium", "large-v2"
WHISPER_DEVICE = "auto"  # Options: "cpu", "cuda", "auto"

# Summarization configurations
SUMMARIZER_TYPE = "offline"  # Options: "openai", "deepseek", "offline"

# API keys (preferably set via environment variables)
OPENAI_API_KEY = "your_openai_api_key_here"  # Alternative: use environment variable OPENAI_API_KEY
DEEPSEEK_API_KEY = "your_openai_api_key_here"   # Alternative: use environment variable DEEPSEEK_API_KEY

# Model name configurations
OFFLINE_MODEL_NAME = "facebook/bart-large-cnn"  #optioal, can be set to any HuggingFace model
ONLINE_MODEL_NAME = ""  # Online API model name (empty uses default)


def create_output_dirs():
    """
    Create necessary output directories for the application.
    
    Returns:
        tuple: (output_directory, segments_directory)
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(SEGMENTS_DIR, exist_ok=True)
    return OUTPUT_DIR, SEGMENTS_DIR