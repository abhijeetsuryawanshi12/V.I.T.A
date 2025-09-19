# transcriber.py
#!/usr/bin/env python3
"""
Module for handling speech transcription using faster-whisper.
"""

import logging
from typing import List, Dict

import torch
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

class Transcriber:
    """Handles audio transcription using the Whisper model."""

    def __init__(self, model_size: str = "large", device: str = "cpu", compute_type: str = "float16"):
        self.device = device
        self.model = self._load_model(model_size, compute_type)

    def _load_model(self, model_size: str, compute_type: str) -> WhisperModel:
        """Load the Whisper model."""
        try:
            # Adjust compute_type for CPU
            effective_compute_type = "int8" if self.device == "cpu" else compute_type
            
            model = WhisperModel(
                model_size,
                device=self.device,
                compute_type=effective_compute_type
            )
            logger.info(f"Whisper model '{model_size}' loaded successfully on {self.device} with compute_type {effective_compute_type}")
            return model
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise

    def transcribe(self, audio_file: str) -> List[Dict]:
        """
        Transcribe an audio file.
        """
        logger.info(f"Starting transcription for {audio_file}...")
        try:
            segments_gen, info = self.model.transcribe(
                audio_file,
                beam_size=5,
                language=None,  # Auto-detect
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=300) # Made VAD slightly more sensitive
            )

            results = [
                {
                    "start": float(seg.start),
                    "end": float(seg.end),
                    "text": seg.text.strip(),
                    "avg_logprob": float(seg.avg_logprob),
                    "no_speech_prob": float(seg.no_speech_prob)
                }
                for seg in segments_gen if seg.text.strip()
            ]
            
            if not results:
                logger.warning("Transcription resulted in zero segments. The audio might be silent or too short.")

            logger.info(f"Transcription completed. Language: {info.language}, Segments found: {len(results)}")
            return results
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise
