#!/usr/bin/env python3
"""
Module for handling speech transcription using a robust, multi-pass approach
for accurate multilingual detection with faster-whisper.
"""

import logging
from typing import List, Dict
import io
import numpy as np
from pydub import AudioSegment
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

class Transcriber:
    """
    Handles audio transcription using a multi-pass approach for high accuracy
    in mixed-language scenarios.
    1. VAD (Voice Activity Detection) to get speech timestamps.
    2. For each speech segment:
        a. Isolate the audio chunk.
        b. Detect its specific language.
        c. Transcribe the chunk using the detected language as a prompt.
    """

    def __init__(self, model_size: str = "small", device: str = "cpu", compute_type: str = "float32"):
        self.device = device
        self.model = self._load_model(model_size, compute_type)

    def _load_model(self, model_size: str, compute_type: str) -> WhisperModel:
        """Load the Whisper model."""
        try:
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
        Transcribes an audio file using a robust multi-pass method.
        """
        logger.info(f"Starting robust multilingual transcription for {audio_file}...")
        try:
            vad_segments, _ = self.model.transcribe(
                audio_file,
                beam_size=5,
                language=None,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )

            vad_timestamps = [{"start": seg.start, "end": seg.end} for seg in vad_segments]

            if not vad_timestamps:
                logger.warning("VAD found no speech segments in the audio.")
                return []
            
            logger.info(f"VAD pass complete. Found {len(vad_timestamps)} potential speech segments.")

            full_audio = AudioSegment.from_file(audio_file)
            final_results = []

            for i, segment_times in enumerate(vad_timestamps):
                start_ms = int(segment_times['start'] * 1000)
                end_ms = int(segment_times['end'] * 1000)
                
                logger.info(f"Processing segment {i+1}/{len(vad_timestamps)}: [{segment_times['start']:.2f}s -> {segment_times['end']:.2f}s]")

                audio_chunk = full_audio[start_ms:end_ms]

                buffer = io.BytesIO()
                audio_chunk.export(buffer, format="wav")
                buffer.seek(0)
                
                chunk_audio_segment = AudioSegment.from_file(buffer)
                samples = np.array(chunk_audio_segment.get_array_of_samples()).astype(np.float32) / 32768.0

                if samples.shape[0] < 100: 
                    logger.warning(f"Segment {i+1} is too short, skipping.")
                    continue
                
                lang_detections = self.model.detect_language(samples)

                if not lang_detections:
                    logger.warning(f"  -> Language detection failed for segment {i+1}. Skipping.")
                    continue

                # --- FIX IS HERE: Handle simple list of strings from detect_language ---
                # This version of the library returns a list of language codes, not tuples with probabilities.
                detected_lang = lang_detections[0]
                lang_prob = 0.0  # Assign a default value as probability is not provided.
                logger.info(f"  -> Detected language: {detected_lang}")
                # --- END OF FIX ---

                chunk_segments, _ = self.model.transcribe(
                    samples,
                    beam_size=5,
                    language=detected_lang
                )
                
                for seg in chunk_segments:
                    if seg.text.strip():
                        final_results.append({
                            "start": round(segment_times['start'] + seg.start, 2),
                            "end": round(segment_times['start'] + seg.end, 2),
                            "text": seg.text.strip(),
                            "language": detected_lang,
                            "language_probability": lang_prob, # Will be 0.0 but keeps structure consistent
                            "avg_logprob": float(seg.avg_logprob),
                            "no_speech_prob": float(seg.no_speech_prob)
                        })

            logger.info(f"Robust multilingual transcription completed. Found {len(final_results)} final segments.")
            return final_results

        except Exception as e:
            logger.error(f"Transcription failed during robust processing: {e}", exc_info=True)
            raise
