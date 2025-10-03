# transcriber.py
#!/usr/bin/env python3
"""
Module for handling speech transcription using faster-whisper and SpeechRecognition.
"""

import logging
from typing import List, Dict

import speech_recognition as sr
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

# --- Language Code Mapping ---
# Maps ISO 639-1 codes from Whisper to BCP-47 codes for Google Speech Recognition
# This is not an exhaustive list.
WHISPER_TO_GOOGLE_LANG = {
    "af": "af-ZA", "ar": "ar-SA", "bg": "bg-BG", "bn": "bn-BD",
    "ca": "ca-ES", "cs": "cs-CZ", "da": "da-DK", "de": "de-DE",
    "el": "el-GR", "en": "en-US", "es": "es-ES", "fi": "fi-FI",
    "fr": "fr-FR", "gu": "gu-IN", "he": "he-IL", "hi": "hi-IN",
    "hr": "hr-HR", "hu": "hu-HU", "id": "id-ID", "is": "is-IS",
    "it": "it-IT", "ja": "ja-JP", "kn": "kn-IN", "ko": "ko-KR",
    "lt": "lt-LT", "lv": "lv-LV", "ml": "ml-IN", "mr": "mr-IN",
    "ms": "ms-MY", "nb": "no-NO", "nl": "nl-NL", "pa": "pa-Guru-IN",
    "pl": "pl-PL", "pt": "pt-PT", "ro": "ro-RO", "ru": "ru-RU",
    "sk": "sk-SK", "sl": "sl-SI", "sr": "sr-RS", "sv": "sv-SE",
    "sw": "sw-TZ", "ta": "ta-IN", "te": "te-IN", "th": "th-TH",
    "tr": "tr-TR", "uk": "uk-UA", "ur": "ur-PK", "vi": "vi-VN",
    "zh": "zh-CN",
}

class Transcriber:
    """Handles audio transcription using the Whisper model for language detection
    and Google Speech Recognition for transcription of non-English languages."""

    def __init__(self, model_size: str = "small", device: str = "cpu", compute_type: str = "float32"):
        self.device = device
        self.model = self._load_model(model_size, compute_type)
        self.recognizer = sr.Recognizer()

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
        1. Use Whisper to detect language and get VAD segments.
        2. If language is not English and is supported by Google SR, use it for transcription.
        3. Otherwise, use Whisper's transcription.
        """
        logger.info(f"Starting transcription for {audio_file}...")
        try:
            # Use Whisper for language detection and initial segmentation
            segments_gen, info = self.model.transcribe(
                audio_file,
                beam_size=5,
                language=None,  # Auto-detect language
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=300)
            )

            # The generator needs to be consumed into a list to be reused.
            segments = list(segments_gen)

            detected_lang = info.language
            logger.info(f"Transcription language detected: {detected_lang} (Confidence: {info.language_probability:.2f})")

            google_lang_code = WHISPER_TO_GOOGLE_LANG.get(detected_lang)

            # If language is not English and is supported by Google SR, use it.
            if detected_lang != "en" and google_lang_code:
                logger.info(f"Language is '{detected_lang}', switching to Google Speech Recognition ({google_lang_code}).")
                return self._transcribe_with_google_sr(audio_file, segments, google_lang_code)
            else:
                if detected_lang == "en":
                    logger.info("Language is English, using faster-whisper for transcription.")
                else:
                    logger.info(f"Language '{detected_lang}' not mapped for Google SR, using faster-whisper for transcription.")
                return self._transcribe_with_whisper(segments)
        
        except Exception as e:
            logger.error(f"Transcription failed: {e}", exc_info=True)
            raise

    def _transcribe_with_whisper(self, segments: list) -> List[Dict]:
        """Format transcription results from faster-whisper."""
        results = [
            {
                "start": float(seg.start),
                "end": float(seg.end),
                "text": seg.text.strip(),
                "avg_logprob": float(seg.avg_logprob),
                "no_speech_prob": float(seg.no_speech_prob)
            }
            for seg in segments if seg.text.strip()
        ]
        if not results:
            logger.warning("Whisper transcription resulted in zero segments.")
        
        logger.info(f"Whisper transcription completed. Segments found: {len(results)}")
        return results

    def _transcribe_with_google_sr(self, audio_file: str, segments: list, lang_code: str) -> List[Dict]:
        """Transcribe audio segments using Google Speech Recognition."""
        results = []
        for seg in segments:
            start_time = float(seg.start)
            end_time = float(seg.end)
            duration = end_time - start_time
            
            if duration <= 0.2: # Skip very short segments
                continue

            # We need to open the file for each segment to handle the offset correctly
            with sr.AudioFile(audio_file) as source:
                try:
                    # Record the specific segment from the audio file
                    segment_audio = self.recognizer.record(source, duration=duration, offset=start_time)
                    text = self.recognizer.recognize_google(segment_audio, language=lang_code)
                    
                    if text.strip():
                        results.append({
                            "start": start_time,
                            "end": end_time,
                            "text": text.strip(),
                            "avg_logprob": 0.0, # Not available from Google SR
                            "no_speech_prob": 0.0 # Not available from Google SR
                        })
                        logger.info(f"Segment [{start_time:.2f}-{end_time:.2f}] transcribed with Google SR.")

                except sr.UnknownValueError:
                    logger.warning(f"Google SR could not understand audio for segment [{start_time:.2f}-{end_time:.2f}]. Using Whisper's text as fallback.")
                    if seg.text.strip():
                        results.append({
                            "start": start_time,
                            "end": end_time,
                            "text": seg.text.strip(),
                            "avg_logprob": float(seg.avg_logprob),
                            "no_speech_prob": float(seg.no_speech_prob)
                        })
                except sr.RequestError as e:
                    logger.error(f"Google SR request failed for segment [{start_time:.2f}-{end_time:.2f}]: {e}. Using Whisper's text as fallback.")
                    if seg.text.strip():
                         results.append({
                            "start": start_time,
                            "end": end_time,
                            "text": seg.text.strip(),
                            "avg_logprob": float(seg.avg_logprob),
                            "no_speech_prob": float(seg.no_speech_prob)
                        })
        
        if not results:
            logger.warning("Google SR transcription resulted in zero segments.")

        logger.info(f"Google SR transcription completed. Segments found: {len(results)}")
        return results
