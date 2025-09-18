# enroll_speaker.py
#!/usr/bin/env python3
"""
Script for enrolling speakers using Picovoice Eagle.
"""
import os
import argparse
import wave
import struct
from dotenv import load_dotenv
import pveagle

def read_wav_file(file_path: str):
    """Reads a WAV file and returns the audio samples."""
    with wave.open(file_path, 'rb') as wf:
        if wf.getnchannels() != 1:
            raise ValueError("Audio file must be mono.")
        if wf.getsampwidth() != 2:
            raise ValueError("Audio file must be 16-bit.")
        # if wf.getframerate() != pveagle.SAMPLE_RATE:
        #     raise ValueError(f"Audio file must have a sample rate of {pveagle.SAMPLE_RATE} Hz.")
        
        buffer = wf.readframes(wf.getnframes())
        return struct.unpack(f"{wf.getnframes()}h", buffer)

def main():
    """Main enrollment function."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--name',
        help="The name of the speaker to enroll.",
        type=str,
        required=True)
    args = parser.parse_args()
    
    load_dotenv()
    access_key = os.getenv("PICOVOICE_ACCESS_KEY")
    if not access_key:
        print("Error: PICOVOICE_ACCESS_KEY not found in .env file.")
        return

    speaker_name = args.name
    enrollment_dir = "enrollment_audio"
    profiles_dir = "speaker_profiles"
    output_profile_path = os.path.join(profiles_dir, f"{speaker_name}.pv")

    if not os.path.exists(enrollment_dir):
        print(f"Error: Enrollment audio directory not found at '{enrollment_dir}'")
        print("Please create it and add WAV files for the speaker.")
        return
        
    os.makedirs(profiles_dir, exist_ok=True)

    enrollment_files = [os.path.join(enrollment_dir, f) for f in os.listdir(enrollment_dir) if f.lower().startswith(speaker_name.lower()) and f.endswith('.wav')]
    
    if not enrollment_files:
        print(f"Error: No enrollment audio files found for '{speaker_name}' in '{enrollment_dir}' directory.")
        print(f"Files should be named like '{speaker_name}_1.wav', '{speaker_name}_2.wav', etc.")
        return

    print(f"Found {len(enrollment_files)} audio files for enrolling '{speaker_name}':")
    for f in enrollment_files:
        print(f" - {f}")

    try:
        enroller = pveagle.create_profiler(access_key)
        print(f"Eagle version: {enroller.version}")
        
        enroll_percentage = 0.0
        for file_path in enrollment_files:
            audio = read_wav_file(file_path)
            enroll_percentage, feedback = enroller.enroll(audio)
            print(f"Enrolled '{file_path}': {feedback.name} ({enroll_percentage:.2f}%)")

        if enroll_percentage < 100.0:
            print(f"Warning: Enrollment is not complete ({enroll_percentage:.2f}%). Please provide more audio for this speaker.")
            return

        print("Enrollment complete. Exporting profile...")
        speaker_profile = enroller.export()
        with open(output_profile_path, 'wb') as f:
            f.write(speaker_profile.to_bytes())
        
        print(f"Speaker profile saved to '{output_profile_path}'")

    except pveagle.EagleError as e:
        print(f"An error occurred with Eagle: {e}")
    finally:
        if 'enroller' in locals() and enroller is not None:
            enroller.delete()

if __name__ == '__main__':
    main()