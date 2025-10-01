# record_voice.py
# Author: Your AI Assistant
# Purpose: Records audio from your microphone and saves it as a WAV file.
# Instructions:
# 1. Install required libraries: pip install sounddevice scipy numpy
# 2. Run the script: python record_voice.py

import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np

def record_audio(filename="my_voice_recording.wav"):
    """
    Records audio from the default microphone and saves it to a file.
    """
    # --- Configuration ---
    # Whisper was trained on 16kHz audio, so we'll use that.
    SAMPLING_RATE = 16000  
    FILENAME = filename

    try:
        # --- Get Recording Duration from User ---
        duration_str = input("Enter recording duration in seconds (e.g., 10): ")
        duration_sec = int(duration_str)

        print("-" * 40)
        print(f"Starting a {duration_sec}-second recording.")
        print("Speak now...")

        # --- Record Audio ---
        # The rec() function records audio from the default input device.
        # It waits until the recording is finished (blocking=True is the default).
        myrecording = sd.rec(int(duration_sec * SAMPLING_RATE), samplerate=SAMPLING_RATE, channels=1, dtype='int16')
        sd.wait()  # Wait until recording is finished

        print("Recording complete.")
        print("-" * 40)

        # --- Save the Audio to a WAV file ---
        # The recording is a NumPy array. We use SciPy to write it to a .wav file.
        write(FILENAME, SAMPLING_RATE, myrecording)
        
        print(f"✅ Your voice has been saved successfully as '{FILENAME}'")
        print("You can now upload this file to the Google Colab notebook for transcription.")

    except ValueError:
        print("❌ Invalid input. Please enter a whole number for the duration.")
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure you have a microphone connected and have granted permission.")

if __name__ == "__main__":
    record_audio()