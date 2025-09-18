# speaker_recognizer.py
#!/usr/bin/env python3
"""
Module for speaker recognition using Picovoice Eagle.
"""
import os
import struct
import wave
from typing import Dict

import pveagle
# We need to import the EagleProfile class to load the profile files
from pveagle import EagleProfile

class SpeakerRecognizer:
    """Handles speaker recognition using Picovoice Eagle."""

    def __init__(self, access_key: str, profiles_dir: str = "speaker_profiles"):
        if not os.path.exists(profiles_dir):
            raise FileNotFoundError(f"Speaker profiles directory not found at '{profiles_dir}'")

        profile_paths = [os.path.join(profiles_dir, f) for f in os.listdir(profiles_dir) if f.endswith('.pv')]
        if not profile_paths:
            raise ValueError(f"No speaker profiles (.pv files) found in '{profiles_dir}'. Please run enroll_speaker.py first.")

        try:
            print("Loading speaker profiles from files...")
            
            # --- THE CORRECT FIX IS HERE ---
            # We must manually open each profile file, read its binary content,
            # and then create the EagleProfile object from those bytes.
            speaker_profiles = []
            for path in profile_paths:
                with open(path, 'rb') as f:
                    profile_bytes = f.read()
                speaker_profiles.append(EagleProfile.from_bytes(profile_bytes))
            
            # This part remains correct
            self.eagle = pveagle.create_recognizer(
                access_key=access_key,
                speaker_profiles=speaker_profiles
            )

            self.speaker_labels = [os.path.basename(p).replace('.pv', '') for p in profile_paths]
            print("Speaker Recognizer initialized with profiles:", self.speaker_labels)

        except pveagle.EagleError as e:
            print(f"Error initializing Eagle Recognizer: {e}")
            raise

    def _read_wav_file(self, file_path: str):
        """Reads a WAV file and returns the audio samples."""
        with wave.open(file_path, 'rb') as wf:
            if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != self.eagle.sample_rate:
                raise ValueError(
                    f"Audio file must be mono, 16-bit, with a sample rate of {self.eagle.sample_rate} Hz."
                )
            buffer = wf.readframes(wf.getnframes())
            return struct.unpack(f"{wf.getnframes()}h", buffer)

    def recognize(self, audio_file: str) -> Dict[str, float]:
        """
        Recognize speakers in an audio file.

        Returns:
            A dictionary mapping speaker names to their confidence scores.
        """
        try:
            audio = self._read_wav_file(audio_file)
            scores = self.eagle.process(audio)
            
            return {self.speaker_labels[i]: scores[i] for i in range(len(scores))}
        except pveagle.EagleError as e:
            print(f"Eagle processing error: {e}")
            return {}
        
    def __del__(self):
        if hasattr(self, 'eagle') and self.eagle is not None:
            self.eagle.delete()