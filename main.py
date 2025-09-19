# main.py
#!/usr/bin/env python3
"""
Main script to perform transcription with a choice of speaker analysis:
- Diarization (pyannote): Who spoke when? (unsupervised)
- Recognition (pveagle): Is this a known speaker? (supervised)
"""

import logging
import os
import argparse
from dotenv import load_dotenv

from audio_recorder import AudioRecorder
from utils import print_results, save_results

# Import processors for both modes
from audio_processor import AudioProcessor
from speaker_recognizer import SpeakerRecognizer
from transcriber import Transcriber

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main execution function."""
    load_dotenv()

    parser = argparse.ArgumentParser(description="A tool for speaker analysis and transcription.")
    parser.add_argument(
        '--mode',
        help="The mode of operation: 'diarize' or 'recognize'.",
        type=str,
        choices=['diarize', 'recognize'],
        default='diarize'
    )
    parser.add_argument(
        '--file',
        help="Path to an existing audio file. If not provided, will start live recording.",
        type=str
    )
    args = parser.parse_args()

    # --- Configuration ---
    RECORDED_AUDIO_FILE = "my_voice_recording.wav"
    WHISPER_MODEL = "large"
    COMPUTE_TYPE = "float16"
    
    try:
        # --- Step 1: Get Audio File ---
        if args.file:
            if not os.path.exists(args.file):
                logger.error(f"File not found: {args.file}"); return
            audio_file = args.file
            logger.info(f"Processing existing audio file: {audio_file}")
        else:
            logger.info("No file provided. Starting live recording session...")
            recorder = AudioRecorder(filename=RECORDED_AUDIO_FILE)
            audio_file = recorder.record()

        final_results = []
        
        # --- Step 2: Run Selected Mode ---
        if args.mode == 'diarize':
            print("\n--- Running in Speaker Diarization Mode (pyannote) ---")
            OUTPUT_FILE = "diarization_results.json"
            processor = AudioProcessor(whisper_model_size=WHISPER_MODEL, compute_type=COMPUTE_TYPE)
            final_results = processor.process_audio(audio_file)
            
            # Print diarization-specific summary
            print_results(final_results, show_confidence=True)
            save_results(final_results, OUTPUT_FILE)
            if final_results:
                total_duration = max(seg['end'] for seg in final_results)
                unique_speakers = set(seg['speaker'] for seg in final_results)
                print("\nSUMMARY (Diarization):")
                print(f"Total duration processed: {total_duration:.2f} seconds")
                print(f"Speakers identified: {len(unique_speakers)}")
                print(f"Diarization available (pyannote): {processor.diarization_available}")
            
        elif args.mode == 'recognize':
            print("\n--- Running in Speaker Recognition Mode (pveagle) ---")
            OUTPUT_FILE = "recognition_results.json"
            RECOGNITION_THRESHOLD = 0.5
            
            picovoice_access_key = os.getenv("PICOVOICE_ACCESS_KEY")
            if not picovoice_access_key:
                logger.error("PICOVOICE_ACCESS_KEY not found in .env file."); return

            transcriber = Transcriber(model_size=WHISPER_MODEL, compute_type=COMPUTE_TYPE)
            recognizer = SpeakerRecognizer(access_key=picovoice_access_key)
            
            transcription_segments = transcriber.transcribe(audio_file)
            if not transcription_segments:
                logger.warning("Transcription returned no segments."); return
            
            speaker_scores = recognizer.recognize(audio_file)
            logger.info(f"Speaker scores: {speaker_scores}")
            
            identified_speaker = "Unknown Speaker"
            if speaker_scores:
                top_speaker = max(speaker_scores, key=speaker_scores.get)
                if speaker_scores[top_speaker] >= RECOGNITION_THRESHOLD:
                    identified_speaker = top_speaker
            
            for segment in transcription_segments:
                segment['speaker'] = identified_speaker
            final_results = transcription_segments

            # Print recognition-specific summary
            print_results(final_results, show_confidence=True)
            save_results(final_results, OUTPUT_FILE)
            if final_results:
                total_duration = final_results[-1]['end']
                print("\nSUMMARY (Recognition):")
                print(f"Total duration processed: {total_duration:.2f} seconds")
                print(f"Identified Speaker: {identified_speaker}")

        print(f"Full results saved to: {OUTPUT_FILE}")

    except Exception as e:
        logger.error(f"An error occurred in the main process: {e}", exc_info=True)

if __name__ == "__main__":
    main()