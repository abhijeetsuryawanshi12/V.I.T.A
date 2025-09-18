# record_enrollment.py
#!/usr/bin/env python3
"""
A dedicated script to record audio clips for speaker enrollment.
"""
import os
import argparse
import re

# We can reuse the AudioRecorder class from our existing module!
from audio_recorder import AudioRecorder

def get_next_file_index(directory: str, speaker_name: str) -> int:
    """
    Finds the next available index for a new recording file.
    For example, if Maria_1.wav and Maria_2.wav exist, this returns 3.
    """
    if not os.path.exists(directory):
        return 1
        
    pattern = re.compile(f"^{re.escape(speaker_name)}_(\d+)\.wav$")
    max_index = 0
    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            index = int(match.group(1))
            if index > max_index:
                max_index = index
    return max_index + 1

def main():
    """Main function to handle the recording loop."""
    parser = argparse.ArgumentParser(
        description="Record audio clips for enrolling a new speaker."
    )
    parser.add_argument(
        '--name',
        help="The name of the speaker to record audio for.",
        type=str,
        required=True
    )
    args = parser.parse_args()
    
    speaker_name = args.name
    enrollment_dir = "enrollment_audio"
    
    # Create the directory if it doesn't exist
    os.makedirs(enrollment_dir, exist_ok=True)
    
    print("="*80)
    print(f"🎤 Starting Enrollment Recording Session for: {speaker_name}")
    print("="*80)
    print("You will be prompted to record multiple short audio clips.")
    print("For best results, aim for a total of 30-60 seconds of clean speech.")
    print("Try to speak naturally in a quiet environment.")
    print("-" * 80)

    while True:
        next_index = get_next_file_index(enrollment_dir, speaker_name)
        output_filename = f"{speaker_name}_{next_index}.wav"
        output_path = os.path.join(enrollment_dir, output_filename)
        
        # Prompt the user
        user_input = input(f"\nPress ENTER to start recording clip #{next_index}, or type 'q' to quit: ")
        
        if user_input.lower() == 'q':
            print("Exiting recording session.")
            break
        
        try:
            # Create a new recorder instance for each clip
            recorder = AudioRecorder(filename=output_path)
            # The record method handles the start/stop and saving logic
            saved_file = recorder.record()
            print(f"✅ Clip saved successfully to '{saved_file}'")
        except Exception as e:
            print(f"⚠️ An error occurred during recording: {e}")
            print("Please try again.")
            
    print("\n" + "="*80)
    print("Recording session finished.")
    print(f"You can now enroll '{speaker_name}' by running:")
    print(f"  python enroll_speaker.py --name {speaker_name}")
    print("="*80)

if __name__ == '__main__':
    main()