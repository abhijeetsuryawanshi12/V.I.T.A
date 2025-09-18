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

def merge_results(diarization: List[Dict], transcription: List[Dict], 
                 overlap_threshold: float = 0.3) -> List[Dict]:
    """
    Merge diarization and transcription results with improved alignment.
    """
    logger.info("Merging diarization and transcription results...")
    
    if not diarization or not transcription:
        logger.warning("Either diarization or transcription is empty. Cannot merge.")
        return []

    def calculate_overlap(seg1: Dict, seg2: Dict) -> float:
        """Calculate overlap ratio between two segments."""
        overlap_start = max(seg1["start"], seg2["start"])
        overlap_end = min(seg1["end"], seg2["end"])
        overlap_duration = max(0, overlap_end - overlap_start)
        
        min_duration = min(seg1["end"] - seg1["start"], seg2["end"] - seg2["start"])
        return overlap_duration / min_duration if min_duration > 0 else 0
    
    output = []
    used_transcriptions = set()
    
    for d_seg in diarization:
        matching_transcriptions = []
        for i, t_seg in enumerate(transcription):
            if i in used_transcriptions:
                continue
            if calculate_overlap(d_seg, t_seg) >= overlap_threshold:
                matching_transcriptions.append((i, t_seg))
        
        if not matching_transcriptions:
            continue

        combined_text = " ".join([t["text"] for _, t in matching_transcriptions])
        used_indices = [i for i, _ in matching_transcriptions]
        used_transcriptions.update(used_indices)
        
        # Calculate an average confidence if available
        confidences = [t.get("avg_logprob", 0) for _, t in matching_transcriptions]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0

        output.append({
            "speaker": d_seg["speaker"],
            "start": d_seg["start"],
            "end": d_seg["end"],
            "duration": d_seg["duration"],
            "text": combined_text.strip(),
            "confidence": avg_confidence,
        })
    
    logger.info(f"Merge completed: {len(output)} final segments")
    return output

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