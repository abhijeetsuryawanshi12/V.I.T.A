# main.py
#!/usr/bin/env python3
"""
Main script to record audio and perform transcription + speaker recognition.
"""

import logging
import os
from dotenv import load_dotenv

from audio_recorder import AudioRecorder
from transcriber import Transcriber
from speaker_recognizer import SpeakerRecognizer
from utils import print_results, save_results

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')
logger = logging.getLogger(__name__)

def show_setup_instructions():
    """Prints instructions for setting up the environment."""
    print("="*80)
    print("SPEAKER RECOGNITION WORKFLOW")
    print("="*80)
    print("This script uses Picovoice Eagle to identify enrolled speakers.")
    print("\nSTEP 1: ENROLL SPEAKERS (One-time setup)")
    print("  1. Create a folder named 'enrollment_audio'.")
    print("  2. Place WAV files of a speaker inside (e.g., 'Alice_1.wav', 'Alice_2.wav').")
    print("  3. Run the enrollment script: python enroll_speaker.py --name Alice")
    print("  4. Repeat for each person you want to recognize.")
    print("     This will create '.pv' profile files in the 'speaker_profiles' folder.")
    print("\nSTEP 2: RUN RECOGNITION (This script)")
    print("  - This script will record your voice and try to match it against")
    print("    the enrolled profiles.")
    print("="*80)

def main():
    """Main execution function."""
    load_dotenv()
    show_setup_instructions()

    # --- Configuration ---
    RECORDED_AUDIO_FILE = "my_voice_recording.wav"
    OUTPUT_FILE = "recognition_results.json"
    WHISPER_MODEL = "small"
    COMPUTE_TYPE = "float16"
    RECOGNITION_THRESHOLD = 0.5  # Confidence score needed to identify a speaker

    picovoice_access_key = os.getenv("PICOVOICE_ACCESS_KEY")
    if not picovoice_access_key:
        logger.error("PICOVOICE_ACCESS_KEY not found in .env file. Cannot start recognizer.")
        return

    try:
        # 1. Record Audio
        recorder = AudioRecorder(filename=RECORDED_AUDIO_FILE)
        audio_file = recorder.record()

        # 2. Initialize Models
        transcriber = Transcriber(model_size=WHISPER_MODEL, compute_type=COMPUTE_TYPE)
        recognizer = SpeakerRecognizer(access_key=picovoice_access_key)
        
        # 3. Transcribe Audio
        logger.info(f"Starting transcription for {audio_file}...")
        transcription_segments = transcriber.transcribe(audio_file)
        if not transcription_segments:
            logger.warning("Transcription returned no segments. Exiting.")
            return

        # 4. Recognize Speaker
        logger.info(f"Starting speaker recognition for {audio_file}...")
        speaker_scores = recognizer.recognize(audio_file)
        logger.info(f"Speaker scores: {speaker_scores}")
        
        # 5. Combine Results
        # Find the speaker with the highest score
        if speaker_scores:
            top_speaker = max(speaker_scores, key=speaker_scores.get)
            top_score = speaker_scores[top_speaker]
            
            if top_score >= RECOGNITION_THRESHOLD:
                identified_speaker = top_speaker
                logger.info(f"Identified speaker as '{identified_speaker}' with score {top_score:.2f}")
            else:
                identified_speaker = "Unknown Speaker"
                logger.warning(f"Top speaker '{top_speaker}' score ({top_score:.2f}) is below threshold.")
        else:
            identified_speaker = "Unknown Speaker"
        
        # Assign the identified speaker to all transcription segments
        final_results = []
        for segment in transcription_segments:
            final_results.append({
                "speaker": identified_speaker,
                "start": segment["start"],
                "end": segment["end"],
                "duration": segment["end"] - segment["start"],
                "text": segment["text"],
                "confidence": segment["avg_logprob"]
            })
            
        # 6. Display and Save Results
        print_results(final_results, show_confidence=True)
        save_results(final_results, OUTPUT_FILE)
        
        print("\nSUMMARY:")
        total_duration = transcription_segments[-1]['end'] if transcription_segments else 0
        print(f"Total duration processed: {total_duration:.2f} seconds")
        print(f"Identified Speaker: {identified_speaker}")
        print(f"Total segments: {len(final_results)}")
        print(f"Full results saved to: {OUTPUT_FILE}")

    except Exception as e:
        logger.error(f"An error occurred in the main process: {e}", exc_info=True)
        print(f"\nAn unexpected error occurred. Please check the logs for details.")

if __name__ == "__main__":
    main()
