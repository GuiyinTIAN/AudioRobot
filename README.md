# AudioRobot

AudioRobot is a real-time and batch audio transcription and summarization system with speaker diarization. It supports multi-speaker recognition, real-time speech-to-text, and automatic conversation summarization.


## Requirements
- See `requirements.txt` for Python dependencies

## Installation
```bash
# Clone the repository
git clone https://github.com/GuiyinTIAN/AudioRobot.git
cd AudioRobot

# (Recommended) Create and activate a virtual environment
conda create -n audiobot python=3.9
conda activate audiobot

# Install dependencies
pip install -r requirements.txt
```

## Configuration
- Edit `config.py` for default settings
- Create `config.local.py` (excluded from git) to override sensitive info (API keys, tokens)
- Set environment variables for API keys if preferred:
  ```bash
  export HF_TOKEN="your_huggingface_token"
  export OPENAI_API_KEY="your_openai_api_key"
  export DEEPSEEK_API_KEY="your_deepseek_api_key"
  ```

## How to Run

### 1. Real-time Transcription & Summarization

```bash
cd Realtime
python realtime_main
```
- Press Enter to start recording
- Speak into the microphone
- Press Enter again to stop recording
- Choose summarization method (OpenAI/DeepSeek/Offline)(Remind to configure API keys)
- Results will be saved in the `outputs/` directory

### 2. Batch/Standard Processing

```bash
python main.py
```
- Record or select an existing audio file (WAV format)
- The system will perform speaker diarization, segment audio, transcribe, and summarize
- Results saved in `outputs/`


## FAQ
**Q: Why do I need a HuggingFace token?**
A: Pyannote speaker diarization requires access to HuggingFace models.

**Q: How do I use OpenAI/DeepSeek summarization?**
A: Set your API key in `config.local.py` or as an environment variable.

**Q: How to run on GPU?**
A: Set `WHISPER_DEVICE = "cuda"` in your config and ensure CUDA drivers are installed.

**Q: Where are my results?**
A: All outputs are saved in the `outputs/` directory, organized by session.


## Acknowledgements
- [OpenAI Whisper](https://github.com/openai/whisper)
- [Faster-Whisper](https://github.com/guillaumekln/faster-whisper)
- [Pyannote Audio](https://github.com/pyannote/pyannote-audio)
- [Transformers](https://github.com/huggingface/transformers)

