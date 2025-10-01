#!/usr/bin/env python3
"""
Utility functions for audio processing, file handling, and result formatting.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict

import torchaudio

logger = logging.getLogger(__name__)

def validate_audio_file(audio_file: str) -> Path:
    """Validate audio file exists and is accessible."""
    audio_path = Path(audio_file)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_file}")
    
    if audio_path.stat().st_size == 0:
        raise ValueError(f"Audio file is empty: {audio_file}")
    
    try:
        info = torchaudio.info(str(audio_path))
        logger.info(f"Audio file info - Sample rate: {info.sample_rate}, Duration: {info.num_frames / info.sample_rate:.2f}s")
    except Exception as e:
        raise ValueError(f"Unable to read audio file {audio_file}: {e}")
        
    return audio_path

def merge_results(diarization: List[Dict], transcription: List[Dict]) -> List[Dict]:
    """
    Aligns transcription segments with diarization segments to assign a speaker to each text segment.

    This function iterates through each transcription segment and assigns it a speaker
    based on the diarization timeline. This ensures that each transcribed phrase
    appears as a separate entry, preventing them from being merged into a single cell.
    """
    logger.info("Aligning transcription to diarization results...")

    if not transcription:
        logger.warning("Transcription is empty. Cannot merge.")
        return []

    if not diarization:
        logger.warning("Diarization is empty. Assigning a default speaker to all transcription segments.")
        # Assign a default speaker and other necessary fields if diarization fails
        for segment in transcription:
            segment['speaker'] = "SPEAKER_00"
            segment['duration'] = segment.get('end', 0) - segment.get('start', 0)
            segment['confidence'] = segment.get('avg_logprob', 0)
        return transcription

    def get_speaker_for_time(time_point: float) -> str:
        """Find the speaker for a given time point."""
        for d_seg in diarization:
            if d_seg['start'] <= time_point < d_seg['end']:
                return d_seg['speaker']
        return "Unknown"  # Fallback for timepoints outside any diarization segment

    aligned_results = []
    for t_seg in transcription:
        # Determine the speaker at the midpoint of the transcription segment
        mid_point = t_seg['start'] + (t_seg['end'] - t_seg['start']) / 2
        speaker = get_speaker_for_time(mid_point)

        aligned_results.append({
            "speaker": speaker,
            "start": t_seg['start'],
            "end": t_seg['end'],
            "duration": t_seg['end'] - t_seg['start'],
            "text": t_seg['text'],
            # Preserve confidence if it exists in the original transcription
            "confidence": t_seg.get('avg_logprob', t_seg.get('confidence', 0)),
        })

    logger.info(f"Alignment completed: {len(aligned_results)} final segments created from {len(transcription)} transcription segments.")
    return aligned_results

def save_results(results: List[Dict], output_file: str):
    """Save results to JSON file."""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to: {output_file}")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
        raise

def print_results(results: List[Dict], show_confidence: bool = False):
    """Print formatted results."""
    print("\n" + "="*80)
    print("TRANSCRIPTION RESULTS")
    print("="*80)
    
    if not results:
        print("No results to display.")
        return

    for seg in results:
        timestamp = f"[{seg['start']:.2f}s - {seg['end']:.2f}s]"
        speaker_text = f"{seg['speaker']}: {seg['text']}"
        
        if show_confidence and 'confidence' in seg:
            confidence = f" (confidence: {seg['confidence']:.2f})"
            print(f"{timestamp} {speaker_text}{confidence}")
        else:
            print(f"{timestamp} {speaker_text}")
    print("="*80)
