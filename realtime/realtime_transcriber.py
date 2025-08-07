"""
Real-time Speech Transcription System with Speaker Diarization

This module implements a real-time audio transcription system that can
transcribe speech and identify different speakers in the conversation.
It uses Whisper for ASR and Pyannote for speaker diarization.
"""

import os
import json
import time
import wave
import threading
import queue
from collections import deque
from datetime import datetime

import numpy as np
import pyaudio
import torch
from faster_whisper import WhisperModel
from sklearn.metrics.pairwise import cosine_similarity

from diarization import DiarizationPipeline
from config import WHISPER_MODEL_NAME, WHISPER_DEVICE, OUTPUT_BASE_DIR


class RealTimeTranscriber: 
    def __init__(
        self, 
        model_name=None,
        device=None,
        chunk_duration=3.0,
        overlap_duration=0.5,
        rate=16000,
        channels=1,
        hf_token=None
    ):
        """
        Initialize the real-time transcription system.
        
        Args:
            model_name (str): Name of the Whisper model to use
            device (str): Device to run the model on ('cpu', 'cuda', or 'auto')
            chunk_duration (float): Duration of each audio chunk in seconds
            overlap_duration (float): Overlap between consecutive chunks in seconds
            rate (int): Audio sampling rate in Hz
            channels (int): Number of audio channels
            hf_token (str): HuggingFace API token for accessing Pyannote models
        
        Raises:
            ValueError: If no HuggingFace token is provided
            RuntimeError: If speaker diarization initialization fails
        """
        # Basic configuration
        self.model_name = model_name or WHISPER_MODEL_NAME
        device = device or WHISPER_DEVICE
        
        self.chunk_duration = chunk_duration
        self.overlap_duration = overlap_duration
        self.rate = rate
        self.channels = channels
        self.chunk_size = int(rate * chunk_duration)
        self.overlap_size = int(rate * overlap_duration)
        
        # Audio processing resources
        self.audio_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.is_recording = False
        self.is_processing = False
        self.stop_event = threading.Event()
        
        # Audio buffer for overlapping chunks
        self.audio_buffer = deque(maxlen=self.chunk_size + self.overlap_size)
        
        # Initialize speaker diarization
        if hf_token is None:
            raise ValueError("HuggingFace token is required for speaker diarization")
        
        try:
            print("🔧 Initializing speaker diarization model...")
            self.diarization_pipeline = DiarizationPipeline(hf_token=hf_token)
            print("✅ Speaker diarization model initialized successfully")
        except Exception as e:
            raise RuntimeError(f"Speaker diarization initialization failed: {e}")
        
        # Initialize Whisper model
        print(f"🔧 Initializing real-time transcription model: {self.model_name}")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        compute_type = "float16" if device == "cuda" else "int8"
        
        self.device = device
        self.compute_type = compute_type
        
        cpu_threads = 4 if device == "cpu" else 16
        num_workers = 1
        
        self.model = WhisperModel(
            self.model_name, 
            device=device, 
            compute_type=compute_type,
            cpu_threads=cpu_threads,
            num_workers=num_workers
        )
        
        # Speaker tracking
        self.current_speaker = None
        self.speaker_activity = {}
        self.speaker_mapping = {}      # Maps temporary pyannote IDs to persistent IDs
        self.speaker_counter = 0       # For assigning new persistent IDs
        self.max_speakers = 2
        
        # Results storage
        self.conversation_history = []
        self.segments_data = []
        self.segment_counter = 0

        # Voice profile feature storage
        self.speaker_features = {}     # Maps persistent IDs to voice feature vectors
        self.temp_features_cache = {}  # Temporary storage for current audio block features
        self.similarity_threshold = 0.98  # Threshold for voice similarity matching
        
        # Output setup
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(OUTPUT_BASE_DIR, f"realtime_session_{self.timestamp}")
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"✅ Real-time transcriber initialized")
        print(f"💻 Device: {device}, Compute type: {compute_type}")
        print(f"👥 Speaker recognition: pyannote diarization")
        print(f"📏 Processing block size: {chunk_duration}s, Overlap: {overlap_duration}s")
        print(f"📁 Output directory: {self.output_dir}")

    def has_speech(self, audio_data, threshold=0.01):
        """
        Detect if audio contains speech based on energy level.
        
        Args:
            audio_data (numpy.ndarray): Audio data to analyze
            threshold (float): Energy threshold for speech detection
            
        Returns:
            bool: True if speech is detected, False otherwise
        """
        rms = np.sqrt(np.mean(audio_data**2))
        return rms > threshold
    
    def transcribe_chunk(self, audio_data):
        """
        Transcribe an audio chunk using Whisper model.
        
        Args:
            audio_data (numpy.ndarray): Audio data to transcribe
            
        Returns:
            str: Transcribed text
        """
        try:
            # Save temporary file
            temp_file = os.path.join(self.output_dir, f"temp_chunk_{int(time.time() * 1000)}.wav")
            
            # Write WAV file
            with wave.open(temp_file, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.rate)
                # Convert to int16
                audio_int16 = (audio_data * 32768).astype(np.int16)
                wf.writeframes(audio_int16.tobytes())
            
            # Use Whisper for transcription
            segments, info = self.model.transcribe(
                temp_file,
                beam_size=1,
                language="zh",  # Specify Chinese language
                temperature=0.0,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=300),
                condition_on_previous_text=False
            )
            
            text = " ".join([seg.text for seg in segments]).strip()
            
            # Clean up temporary file
            try:
                os.remove(temp_file)
            except:
                pass
                
            return text
            
        except Exception as e:
            print(f"❌ Transcription failed: {e}")
            return ""
    
    def extract_audio_features(self, audio_data):
        """
        Extract acoustic features from audio for speaker recognition.
        
        Args:
            audio_data (numpy.ndarray): Audio data to analyze
            
        Returns:
            numpy.ndarray: Feature vector for speaker recognition
        """
        try:
            # Calculate basic acoustic features
            features = []
            
            # 1. Pitch feature (fundamental frequency estimation)
            def estimate_pitch(signal, sr):
                # Autocorrelation method
                correlation = np.correlate(signal, signal, mode='full')
                correlation = correlation[len(correlation)//2:]
                
                # Find the first peak (excluding zero point)
                min_period = int(sr / 500)  # Maximum 500Hz
                max_period = int(sr / 50)   # Minimum 50Hz
                
                if len(correlation) > max_period:
                    search_range = correlation[min_period:max_period]
                    if len(search_range) > 0:
                        peak_idx = np.argmax(search_range) + min_period
                        pitch = sr / peak_idx if peak_idx > 0 else 100
                        return pitch
                return 100
            
            pitch = estimate_pitch(audio_data, self.rate)
            features.append(pitch)
            
            # 2. Spectral centroid
            fft = np.fft.rfft(audio_data)
            magnitude = np.abs(fft)
            freqs = np.fft.rfftfreq(len(audio_data), 1/self.rate)
            
            if np.sum(magnitude) > 0:
                centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
            else:
                centroid = 0
            features.append(centroid)
            
            # 3. Spectral rolloff
            cumsum = np.cumsum(magnitude)
            rolloff_point = 0.85 * cumsum[-1]
            rolloff_idx = np.where(cumsum >= rolloff_point)[0]
            rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else freqs[-1]
            features.append(rolloff)
            
            # 4. Zero-crossing rate
            zero_crossings = np.sum(np.diff(np.sign(audio_data)) != 0)
            zcr = zero_crossings / len(audio_data)
            features.append(zcr)
            
            # 5. Energy feature
            energy = np.sum(audio_data ** 2) / len(audio_data)
            features.append(energy)
            
            # 6. Simplified MFCC - spectral envelope
            # Divide spectrum into frequency bands and calculate energy
            n_bands = 8
            band_energies = []
            band_size = len(magnitude) // n_bands
            
            for i in range(n_bands):
                start_idx = i * band_size
                end_idx = min((i + 1) * band_size, len(magnitude))
                band_energy = np.sum(magnitude[start_idx:end_idx])
                band_energies.append(band_energy)
            
            features.extend(band_energies)
            
            return np.array(features)
            
        except Exception as e:
            print(f"⚠️ Feature extraction failed: {e}")
            # Return default feature vector
            return np.zeros(13)  # 5 basic features + 8 frequency bands

    def find_matching_speaker_by_features(self, current_features):
        """
        Find the best matching speaker based on voice feature similarity.
        
        Args:
            current_features (numpy.ndarray): Voice features to match
            
        Returns:
            tuple: (speaker_id, similarity_score) or (None, 0) if no match
        """
        if len(self.speaker_features) == 0:
            return None, 0
        
        best_match = None
        best_similarity = 0
        
        try:
            # Compare with each known speaker
            for speaker_id, feature_list in self.speaker_features.items():
                if len(feature_list) == 0:
                    continue
                
                # Calculate similarity with all historical features of this speaker
                similarities = []
                for stored_features in feature_list:
                    # Use cosine similarity
                    sim = cosine_similarity(np.array([current_features]), np.array([stored_features]))[0][0]
                    similarities.append(sim)
                
                # Take the highest similarity
                max_sim = max(similarities) if similarities else 0
                
                if max_sim > best_similarity and max_sim > self.similarity_threshold:
                    best_similarity = max_sim
                    best_match = speaker_id
            
            return best_match, best_similarity
            
        except Exception as e:
            print(f"⚠️ Voice matching failed: {e}")
            return None, 0

    def register_speaker_features(self, speaker_id, features):
        """
        Register voice features for a speaker.
        
        Args:
            speaker_id (str): Speaker identifier
            features (numpy.ndarray): Voice feature vector to register
        """
        if speaker_id not in self.speaker_features:
            self.speaker_features[speaker_id] = []
        
        # Add new features (limit storage to avoid memory issues)
        self.speaker_features[speaker_id].append(features)
        
        # Keep only the 10 most recent feature vectors (reduce historical contamination)
        if len(self.speaker_features[speaker_id]) > 10:
            self.speaker_features[speaker_id] = self.speaker_features[speaker_id][-10:]

    def get_speaker_for_audio_chunk(self, audio_data):
        """
        Identify the speaker for an audio chunk using two-person mode.
        
        This method implements a fixed two-person mode for speaker diarization,
        combining pyannote output with voice feature matching to ensure consistent
        speaker identification across audio chunks.
        
        Args:
            audio_data (numpy.ndarray): Audio chunk to analyze
            
        Returns:
            str: Speaker identifier (SPEAKER_00 or SPEAKER_01)
        """
        try:
            # 1. Extract voice features for current audio
            current_features = self.extract_audio_features(audio_data.astype(np.float32))
            
            # 2. Save audio chunk to temporary file (for pyannote)
            temp_audio_file = os.path.join(self.output_dir, f"temp_diarization_{int(time.time() * 1000)}.wav")
            
            with wave.open(temp_audio_file, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.rate)
                if audio_data.dtype != np.int16:
                    audio_int16 = (audio_data * 32768).astype(np.int16)
                else:
                    audio_int16 = audio_data
                wf.writeframes(audio_int16.tobytes())
            
            # 3. Run pyannote (for verification)
            diarization = self.diarization_pipeline.run(temp_audio_file)
            
            # Analyze pyannote results
            speaker_durations = {}
            for segment, _, speaker in diarization.itertracks(yield_label=True):
                duration = segment.end - segment.start
                if speaker not in speaker_durations:
                    speaker_durations[speaker] = 0
                speaker_durations[speaker] += duration
            
            # Clean up temporary file
            try:
                os.remove(temp_audio_file)
            except:
                pass
            
            print(f"🔍 Pyannote detection results:")
            print(f"   Detected {len(speaker_durations)} speakers")
            for speaker, duration in speaker_durations.items():
                print(f"   {speaker}: {duration:.2f}s")
            
            # 4. Smart speaker recognition - two-person mode
            final_speaker = None
            
            # Case 1: No speakers registered yet
            if len(self.speaker_features) == 0:
                # First speaker
                final_speaker = "SPEAKER_00"
                self.register_speaker_features(final_speaker, current_features)
                print(f"🆕 Registering first speaker: {final_speaker}")
            
            # Case 2: Only one speaker registered
            elif len(self.speaker_features) == 1:
                existing_speaker = list(self.speaker_features.keys())[0]
                
                # Calculate similarity with the known speaker
                similarities = []
                for stored_features in self.speaker_features[existing_speaker]:
                    sim = cosine_similarity(np.array([current_features]), np.array([stored_features]))[0][0]
                    similarities.append(sim)
                
                best_similarity = max(similarities) if similarities else 0
                
                # Determine if this is the existing speaker or a new one
                if best_similarity > self.similarity_threshold:
                    # Still the first speaker
                    final_speaker = existing_speaker
                    self.register_speaker_features(final_speaker, current_features)
                else:
                    # New second speaker
                    final_speaker = "SPEAKER_01"
                    self.register_speaker_features(final_speaker, current_features)
                    print(f"🆕 Registering second speaker: {final_speaker} (similarity with known speaker: {best_similarity:.3f})")
            
            # Case 3: Two speakers already registered
            else:
                # Calculate maximum similarity with both known speakers
                best_match = None
                best_similarity = 0
                
                for speaker_id, feature_list in self.speaker_features.items():
                    if len(feature_list) == 0:
                        continue
                    
                    # Calculate similarity with all historical features of this speaker
                    speaker_similarities = []
                    for stored_features in feature_list:
                        sim = cosine_similarity(np.array([current_features]), np.array([stored_features]))[0][0]
                        speaker_similarities.append(sim)
                    
                    # Take the highest similarity
                    max_sim = max(speaker_similarities) if speaker_similarities else 0
                    
                    if max_sim > best_similarity:
                        best_similarity = max_sim
                        best_match = speaker_id
                
                # Always match current speaker to the most similar known speaker
                final_speaker = best_match
                print(f"🎯 Best match: {final_speaker} (similarity: {best_similarity:.3f})")
                
                # Only update feature database if similarity is high enough
                if best_similarity > self.similarity_threshold:
                    self.register_speaker_features(final_speaker, current_features)
                    print(f"   ✅ Similarity high enough, updating feature database")
                else:
                    print(f"   ⚠️ Low similarity ({best_similarity:.3f}), not updating database")
            
            # 5. Update statistics
            if final_speaker not in self.speaker_activity:
                self.speaker_activity[final_speaker] = 0
            self.speaker_activity[final_speaker] += 1
            
            # Display speaker change
            if self.current_speaker != final_speaker:
                print(f"🔄 Speaker change: {self.current_speaker} → {final_speaker}")
                self.current_speaker = final_speaker
            
            return final_speaker
            
        except Exception as e:
            print(f"❌ Speaker diarization failed: {e}")
            import traceback
            traceback.print_exc()
            return self.current_speaker or "SPEAKER_00"

    def start_recording(self):
        """
        Start the recording and real-time transcription process.
        
        This method initializes audio devices, starts recording and processing
        threads, and waits for user input to stop recording.
        """
        print("\n" + "="*50)
        print("🎤 AudioRobot Real-time Transcription System")
        print("💡 Press Enter to start recording...")
        print("="*50)
        
        # Wait for user to press Enter
        try:
            input()
        except KeyboardInterrupt:
            print("\n❌ User interrupted")
            return
        
        print("🔧 Initializing...")
        
        try:
            # Initialize PyAudio
            self.audio = pyaudio.PyAudio()
            
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=1024,
                stream_callback=None
            )
            
            print("✅ Audio device initialized")
            print(f"📊 Sample rate: {self.rate}Hz, Channels: {self.channels}")
            
        except Exception as e:
            print(f"❌ Audio device initialization failed: {e}")
            return
        
        # Start processing thread
        self.is_processing = True
        self.processing_thread = threading.Thread(target=self.process_audio_chunks)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        # Start recording
        self.is_recording = True
        print("\n" + "="*50)
        print("🎤 Recording started...")
        print("💡 Start speaking, the system will transcribe in real-time")
        print("⏸️  Press Enter again to stop recording")
        print("="*50)
        
        # Recording main loop
        recording_thread = threading.Thread(target=self.record_audio)
        recording_thread.daemon = True
        recording_thread.start()
        
        try:
            input()
        except KeyboardInterrupt:
            pass
        
        self.stop_recording()
    
    def record_audio(self):
        """
        Record audio data thread.
        
        This method continuously reads audio data from the microphone,
        divides it into chunks, and puts them in the processing queue.
        """
        while self.is_recording:
            try:
                data = self.stream.read(1024, exception_on_overflow=False)
                audio_chunk = np.frombuffer(data, dtype=np.int16)
                
                self.audio_buffer.extend(audio_chunk)
                
                # Check if we have enough data to process
                if len(self.audio_buffer) >= self.chunk_size:
                    # Extract a chunk of audio data
                    chunk_data = np.array(list(self.audio_buffer)[:self.chunk_size])
                    
                    # Remove processed data from buffer and keep overlap
                    overlap_start = self.chunk_size - self.overlap_size
                    self.audio_buffer = deque(
                        list(self.audio_buffer)[overlap_start:], 
                        maxlen=self.chunk_size + self.overlap_size
                    )
                    
                    try:
                        self.audio_queue.put(chunk_data, timeout=1.0)
                    except queue.Full:
                        print("⚠️ The audio queue is full, skipping this chunk")
                
            except Exception as e:
                if self.is_recording:
                    print(f"❌ Recording error: {e}")
                break
    
    def stop_recording(self):
        """
        Stop the recording and processing threads.
        
        This method ensures all remaining audio chunks are processed before
        completely shutting down the system.
        """
        print("🛑 Stopping recording...")
        
        # First, stop recording but keep processing
        self.is_recording = False
        
        print("⏳ Please wait while processing remaining chunks...")
        
        wait_start = time.time()
        max_wait_time = 20  # Maximum waiting time for remaining audio processing
        
        while not self.audio_queue.empty() and time.time() - wait_start < max_wait_time:
            remaining = self.audio_queue.qsize()
            print(f"   Queue has {remaining} audio chunks left to process...")
            time.sleep(1)
        
        # Now stop processing
        self.is_processing = False
        
        if hasattr(self, 'processing_thread'):
            self.processing_thread.join(timeout=2.0)
        
        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()
        
        if hasattr(self, 'audio'):
            self.audio.terminate()
        
        print("✅ Recording stopped, all audio has been processed")

    def process_audio_chunks(self):
        """
        Process audio chunks for transcription and speaker diarization.
        
        This thread continuously pulls audio chunks from the queue,
        transcribes them, and identifies the speakers.
        """
        print("🔄 Starting audio processing thread...")
        
        while self.is_processing:
            try:
                audio_chunk = self.audio_queue.get(timeout=3.0)
                
                audio_float = audio_chunk.astype(np.float32) / 32768.0
                
                if self.has_speech(audio_float):
                    start_time = time.time()
                    
                    text = self.transcribe_chunk(audio_float)
                    
                    if text and text.strip():
                        speaker = self.get_speaker_for_audio_chunk(audio_chunk)
                        
                        processing_time = time.time() - start_time
                        
                        segment_data = {
                            'segment_id': self.segment_counter,
                            'speaker': speaker,
                            'text': text.strip(),
                            'timestamp': datetime.now().isoformat(),
                            'processing_time': processing_time,
                            'language': 'zh',
                            'diarization_method': 'pyannote'
                        }
                        
                        self.segment_counter += 1
                        self.segments_data.append(segment_data)
                        self.result_queue.put(segment_data)
                        self.update_conversation_history(segment_data)
                        
                        print(f"🎤 [{speaker}]: {text.strip()}")
                        print(f"   ⏱️ Processing time: {processing_time:.2f}s")
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Audio processing error: {e}")
                continue

    def update_conversation_history(self, segment_data):
        """
        Update the conversation history with the latest segment data.
        
        This method merges consecutive segments from the same speaker.
        
        Args:
            segment_data (dict): Data for the current segment
        """
        speaker = segment_data['speaker']
        text = segment_data['text']
        
        # Check if the last entry is from the same speaker
        if (self.conversation_history and 
            self.conversation_history[-1]['speaker'] == speaker):
            # Combine with last entry
            self.conversation_history[-1]['text'] += " " + text
            self.conversation_history[-1]['segments'].append(segment_data['segment_id'])
        else:
            conversation_entry = {
                'speaker': speaker,
                'text': text,
                'start_time': segment_data['timestamp'],
                'segments': [segment_data['segment_id']]
            }
            self.conversation_history.append(conversation_entry)
    
    def save_results(self):
        """
        Save the transcription results to files.
        
        This method saves the results in three formats:
        1. Plain text (.txt)
        2. Structured data (.json)
        3. AI-friendly format (.json)
        
        Returns:
            tuple: Paths to the saved files (text_file, structured_file, ai_file)
        """
        try:
            # 1. Save text transcription
            text_file = os.path.join(self.output_dir, f"transcription_{self.timestamp}.txt")
            with open(text_file, 'w', encoding='utf-8') as f:
                f.write(f"AudioRobot Real-time Transcription Results\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Speaker identification: pyannote\n")
                f.write("=" * 50 + "\n\n")
                
                for entry in self.conversation_history:
                    f.write(f"[{entry['speaker']}]: {entry['text']}\n\n")
            
            # 2. Save structured data
            structured_file = os.path.join(self.output_dir, f"structured_{self.timestamp}.json")
            structured_data = {
                'metadata': {
                    'timestamp': self.timestamp,
                    'total_segments': len(self.segments_data),
                    'conversation_turns': len(self.conversation_history),
                    'diarization_method': 'pyannote',
                    'speaker_activity': self.speaker_activity
                },
                'segments': self.segments_data,
                'conversation': self.conversation_history
            }
            
            with open(structured_file, 'w', encoding='utf-8') as f:
                json.dump(structured_data, f, ensure_ascii=False, indent=2)
            
            # 3. Save AI-friendly format
            ai_file = os.path.join(self.output_dir, f"ai_conversation_{self.timestamp}.json")

            ai_friendly_conversation = [
                {
                    'speaker': entry['speaker'],
                    'text': entry['text']
                }
                for entry in self.conversation_history
            ]
            
            ai_data = {
                "timestamp": datetime.now().isoformat(),
                "conversation": ai_friendly_conversation
            }
            
            with open(ai_file, 'w', encoding='utf-8') as f:
                json.dump(ai_data, f, ensure_ascii=False, indent=2)
            
            print(f"✅ Transcription results saved:")
            print(f"  - Text version: {os.path.basename(text_file)}")
            print(f"  - Structured data: {os.path.basename(structured_file)}")
            print(f"  - AI-friendly format: {os.path.basename(ai_file)}")
            
            return text_file, structured_file, ai_file
        
        except Exception as e:
            print(f"❌ Failed to save results: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None