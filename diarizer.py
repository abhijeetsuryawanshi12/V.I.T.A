#!/usr/bin/env python3
"""
Module for handling speaker diarization using pyannote.audio.
"""
import os
os.environ["PYANNOTE_AUDIO_BACKEND"] = "soundfile"
import logging
import os
import warnings
from typing import List, Dict, Optional

import torch
import torchaudio
from pyannote.audio import Pipeline
from huggingface_hub import login
import librosa

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# --- MODIFIED FIX ---
# This explicit import helps resolve a NameError within the pyannote library.
# We have removed the try...except block. If this line fails with an ImportError,
# it indicates a critical problem with the torchaudio installation itself,
# and we want the program to stop with a clear error rather than failing later.
# from torchcodec.decoders import AudioDecoder


logger = logging.getLogger(__name__)

class Diarizer:
    """Handles speaker diarization using pyannote.audio."""

    def __init__(self, device: str = "cpu", auth_token: Optional[str] = None):
        self.device = device
        self.pipeline = self._load_pipeline(auth_token)
        self.is_available = self.pipeline is not None

    def _load_pipeline(self, auth_token: Optional[str]) -> Optional[Pipeline]:
        """Load the pyannote.audio pipeline."""
        try:
            warnings.filterwarnings("ignore", message=".*torchaudio._extension.*")
            token = auth_token or os.getenv("HF_TOKEN")

            if not token:
                logger.error("Hugging Face token not found. Please set HF_TOKEN in your .env file.")
                logger.warning("Diarization will be disabled.")
                return None

            # Login to Hugging Face Hub using the new method
            login(token=token)

            # Load the pipeline WITHOUT the 'use_auth_token' argument
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-community-1"
            )
            
            if self.device == "cuda" and torch.cuda.is_available():
                pipeline = pipeline.to(torch.device("cuda"))
            
            logger.info("Diarization pipeline loaded successfully.")
            return pipeline
        except Exception as e:
            error_msg = str(e).lower()
            if "authentication" in error_msg or "gated" in error_msg:
                logger.error("AUTHENTICATION ERROR for pyannote/speaker-diarization-3.1")
                logger.error("Please visit https://hf.co/pyannote/speaker-diarization-3.1, accept user conditions, and ensure your HF_TOKEN is correct.")
            else:
                logger.error(f"Failed to load diarization pipeline: {e}")
            logger.warning("Diarization will be disabled.")
            return None

    def diarize(self, audio_file: str, transcription_segments: List[Dict]) -> List[Dict]:
        """
        Perform diarization. Falls back to a simple method if the main pipeline is unavailable.
        """
        if self.is_available:
            return self._run_pipeline_diarization(audio_file)
        else:
            logger.info("Using simple voice activity based diarization fallback...")
            return self._simple_diarization(transcription_segments)

    

# Inside your Diarizer class, modify the method _run_pipeline_diarization:
    def _run_pipeline_diarization(self, audio_file: str) -> List[Dict]:
        """Run the main pyannote diarization pipeline using pre-loaded audio tensor."""
        logger.info("Starting speaker diarization with preloaded audio using librosa...")
        try:
            audio, sr = librosa.load(audio_file, sr=16000)
            waveform = torch.tensor(audio).unsqueeze(0)
            audio_dict = {
                'waveform': waveform,
                'sample_rate': sr
            }
            diarization = self.pipeline(audio_dict)
            segments = [
                {
                    "start": float(segment.start),
                    "end": float(segment.end),
                    "speaker": speaker,
                    "duration": float(segment.end - segment.start)
                }
                for segment, _, speaker in diarization.speaker_diarization.itertracks(yield_label=True)
            ]
            logger.info(f"Diarization completed: {len(segments)} segments found")
            return segments
        except Exception as e:
            logger.error(f"Diarization with pipeline failed: {e}")
            raise


    
    def _simple_diarization(self, transcription_segments: List[Dict]) -> List[Dict]:
        """
        Simple diarization based on silence gaps in transcription.
        """
        if not transcription_segments:
            return []
        
        diarization_segments = []
        current_speaker = "SPEAKER_00"
        segment_start = transcription_segments[0]["start"]
        SILENCE_THRESHOLD = 2.0  # seconds

        for i, segment in enumerate(transcription_segments):
            if i > 0:
                prev_end = transcription_segments[i-1]["end"]
                current_start = segment["start"]
                if current_start - prev_end > SILENCE_THRESHOLD:
                    diarization_segments.append({
                        "start": segment_start,
                        "end": prev_end,
                        "speaker": current_speaker,
                        "duration": prev_end - segment_start
                    })
                    current_speaker = "SPEAKER_01" if current_speaker == "SPEAKER_00" else "SPEAKER_00"
                    segment_start = current_start
        
        # Add the final segment
        last_end = transcription_segments[-1]["end"]
        diarization_segments.append({
            "start": segment_start,
            "end": last_end,
            "speaker": current_speaker,
            "duration": last_end - segment_start
        })

        logger.info(f"Simple diarization completed: {len(diarization_segments)} segments")
        return diarization_segments
