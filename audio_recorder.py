# audio_recorder.py
#!/usr/bin/env python3
"""
Module for recording audio from the microphone.
"""

import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import logging

logger = logging.getLogger(__name__)

class AudioRecorder:
    """A class to handle audio recording."""

    def __init__(self, filename: str = "recorded_audio.wav", sample_rate: int = 16000, channels: int = 1):
        """
        Initialize the audio recorder.

        Args:
            filename (str): The filename to save the recording to.
            sample_rate (int): The sample rate for the recording. Whisper prefers 16000.
            channels (int): The number of audio channels.
        """
        self.filename = filename
        self.sample_rate = sample_rate
        self.channels = channels
        self.recording = []

    def record(self) -> str:
        """
        Records audio from the microphone until the user presses Enter.

        Returns:
            str: The path to the saved audio file.
        """
        try:
            print("\n" + "="*80)
            print("🎙️  Press ENTER to start recording...")
            input()  # Wait for user to press Enter to start

            print("🔴  Recording... Press ENTER to stop.")
            
            self.recording = [] # Clear previous recording
            
            def callback(indata, frames, time, status):
                """This is called (from a separate thread) for each audio block."""
                if status:
                    print(status, flush=True)
                self.recording.append(indata.copy())

            with sd.InputStream(samplerate=self.sample_rate, channels=self.channels, callback=callback):
                input() # Wait for user to press Enter to stop

            print("👍  Recording finished.")
            print("="*80 + "\n")

            if not self.recording:
                logger.warning("No audio was recorded.")
                raise ValueError("Recording is empty. Please try again.")

            full_recording = np.concatenate(self.recording, axis=0)
            
            # --- FIX IS HERE ---
            # Convert the float32 audio data to int16 before saving
            # The float data is in the range [-1.0, 1.0], so we scale it to the int16 range [-32768, 32767]
            logger.info("Converting audio to 16-bit PCM format...")
            int16_recording = np.int16(full_recording * 32767)
            
            # Save the recording as a WAV file in the correct format
            logger.info(f"Saving recording to {self.filename}...")
            write(self.filename, self.sample_rate, int16_recording)
            logger.info("Recording saved successfully.")
            
            return self.filename

        except Exception as e:
            logger.error(f"An error occurred during recording: {e}")
            raise