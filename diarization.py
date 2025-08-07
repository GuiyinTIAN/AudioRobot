"""
Speaker Diarization Pipeline

This module provides a wrapper for pyannote.audio's speaker diarization capabilities,
allowing for identification of different speakers in an audio file.
"""

# Third-party imports
from pyannote.audio import Pipeline


class DiarizationPipeline:
    """
    Speaker diarization pipeline using pyannote.audio.
    
    Handles initialization of the diarization model and provides methods
    for running speaker diarization on audio files.
    """
    
    def __init__(self, model_name="pyannote/speaker-diarization-3.1", hf_token=None):
        """
        Initialize the speaker diarization pipeline.
        
        Args:
            model_name (str): Name of the pre-trained model to use
            hf_token (str): HuggingFace API token for model access
            
        Raises:
            ValueError: If no HuggingFace token is provided
        """
        if hf_token is None:
            raise ValueError("Must provide HuggingFace token")

        self.pipeline = Pipeline.from_pretrained(model_name, use_auth_token=hf_token)
        self.model_name = model_name

    def run(self, audio_path, output_rttm_path=None):
        """
        Run the diarization pipeline on the given audio file.
        
        Args:
            audio_path (str): Path to the audio file
            output_rttm_path (str, optional): Path to save the RTTM file
            
        Returns:
            pyannote.core.Annotation: Diarization result object
        """
        diarization = self.pipeline(audio_path)

        if output_rttm_path:
            with open(output_rttm_path, "w") as rttm_file:
                diarization.write_rttm(rttm_file)
            print(f"✅ RTTM saved to: {output_rttm_path}")

        return diarization


# =============================
# For testing purposes
# =============================
if __name__ == "__main__":
    HF_TOKEN = "your_huggingface_token_here"  # Replace with your actual HuggingFace token

    diarizer = DiarizationPipeline(hf_token=HF_TOKEN)
    result = diarizer.run("your_audio.wav", output_rttm_path="output.rttm")

    for turn, _, speaker in result.itertracks(yield_label=True):
        print(f"Time: {turn.start:.2f}s - {turn.end:.2f}s, Speaker: {speaker}")