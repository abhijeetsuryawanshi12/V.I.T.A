#!/usr/bin/env python3
"""
Enhanced Audio Diarization and Transcription Script
Combines speaker diarization with speech transcription using faster-whisper and pyannote.audio
"""

import os
import logging
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import warnings

try:
    from faster_whisper import WhisperModel
    from pyannote.audio import Pipeline
    import torch
    import torchaudio
except ImportError as e:
    raise ImportError(f"Required dependency missing: {e}. Install with: pip install faster-whisper pyannote.audio torch torchaudio")

try:
    from huggingface_hub import login
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AudioProcessor:
    """Enhanced audio processing class with better error handling and optimization."""
    
    def __init__(self, 
                 whisper_model_size: str = "small",
                 device: Optional[str] = None,
                 auth_token: Optional[str] = None,
                 compute_type: str = "float16"):
        """
        Initialize the audio processor.
        
        Args:
            whisper_model_size: Whisper model size (tiny, base, small, medium, large-v3)
            device: Device to use (cuda/cpu), auto-detected if None
            auth_token: Hugging Face auth token for pyannote models
            compute_type: Computation type for Whisper (float16, int8, float32)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.whisper_model_size = whisper_model_size
        self.compute_type = compute_type
        self.auth_token = auth_token
        
        logger.info(f"Initializing AudioProcessor with device: {self.device}")
        
        # Initialize models
        self.whisper_model = self._load_whisper_model()
        self.diarization_pipeline = self._load_diarization_pipeline()
        
        # Check if diarization is available
        self.diarization_available = self.diarization_pipeline is not None
        if not self.diarization_available:
            logger.warning("Diarization is disabled. Only transcription will be performed.")
    
    def _load_whisper_model(self) -> WhisperModel:
        """Load Whisper model with error handling."""
        try:
            # Adjust compute_type based on device
            if self.device == "cpu":
                compute_type = "int8"  # More efficient for CPU
            else:
                compute_type = self.compute_type
                
            model = WhisperModel(
                self.whisper_model_size, 
                device=self.device,
                compute_type=compute_type
            )
            logger.info(f"Whisper model '{self.whisper_model_size}' loaded successfully")
            return model
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise
    
    def _load_diarization_pipeline(self) -> Optional[Pipeline]:
        """Load pyannote diarization pipeline with error handling and fallback options."""
        try:
            # Suppress pyannote warnings
            warnings.filterwarnings("ignore", message=".*torchaudio._extension.*")
            
            # Try to get auth token from environment if not provided
            auth_token = self.auth_token or os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
            
            # List of model versions to try (from newest to oldest)
            model_versions = [
                "pyannote/speaker-diarization-3.1",
                "pyannote/speaker-diarization",  # fallback to older version
            ]
            
            for model_name in model_versions:
                try:
                    logger.info(f"Attempting to load {model_name}...")
                    pipeline = Pipeline.from_pretrained(
                        model_name,
                        use_auth_token=auth_token
                    )
                    
                    # Move to appropriate device
                    if self.device == "cuda" and torch.cuda.is_available():
                        pipeline = pipeline.to(torch.device("cuda"))
                        
                    logger.info(f"Diarization pipeline '{model_name}' loaded successfully")
                    return pipeline
                    
                except Exception as model_error:
                    logger.warning(f"Failed to load {model_name}: {model_error}")
                    continue
            
            # If all models failed, provide detailed error message
            raise Exception("All diarization models failed to load")
            
        except Exception as e:
            error_msg = str(e)
            if any(keyword in error_msg.lower() for keyword in ["authentication", "gated", "private", "token"]):
                logger.error("AUTHENTICATION ERROR:")
                logger.error("The pyannote models require authentication. Please:")
                logger.error("1. Visit https://huggingface.co/pyannote/speaker-diarization-3.1")
                logger.error("2. Accept the user conditions")
                logger.error("3. Create a token at https://hf.co/settings/tokens")
                logger.error("4. Set your token using one of these methods:")
                logger.error("   - Set environment variable: export HF_TOKEN='your_token_here'")
                logger.error("   - Pass token to AudioProcessor: AudioProcessor(auth_token='your_token_here')")
                logger.error("   - Use huggingface_hub login: huggingface-cli login")
            else:
                logger.error(f"Failed to load diarization pipeline: {e}")
            
    def simple_voice_activity_diarization(self, transcription_segments: List[Dict]) -> List[Dict]:
        """
        Simple fallback diarization based on voice activity detection.
        Groups consecutive speech segments and assigns alternating speakers for gaps.
        """
        logger.info("Using simple voice activity based diarization fallback...")
        
        if not transcription_segments:
            return []
        
        diarization_segments = []
        current_speaker = "SPEAKER_00"
        segment_start = transcription_segments[0]["start"]
        
        # Parameters for speaker change detection
        SILENCE_THRESHOLD = 2.0  # seconds of silence to trigger speaker change
        
        for i, segment in enumerate(transcription_segments):
            # Check if there's a significant gap from the previous segment
            if i > 0:
                gap_duration = segment["start"] - transcription_segments[i-1]["end"]
                if gap_duration > SILENCE_THRESHOLD:
                    # End current speaker segment
                    diarization_segments.append({
                        "start": segment_start,
                        "end": transcription_segments[i-1]["end"],
                        "speaker": current_speaker,
                        "duration": transcription_segments[i-1]["end"] - segment_start
                    })
                    
                    # Switch speaker and start new segment
                    current_speaker = "SPEAKER_01" if current_speaker == "SPEAKER_00" else "SPEAKER_00"
                    segment_start = segment["start"]
        
        # Add final segment
        if transcription_segments:
            diarization_segments.append({
                "start": segment_start,
                "end": transcription_segments[-1]["end"],
                "speaker": current_speaker,
                "duration": transcription_segments[-1]["end"] - segment_start
            })
        
        logger.info(f"Simple diarization completed: {len(diarization_segments)} segments")
        return diarization_segments
    
    def validate_audio_file(self, audio_file: str) -> Path:
        """Validate audio file exists and is accessible."""
        audio_path = Path(audio_file)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
        
        # Check file size
        file_size = audio_path.stat().st_size
        if file_size == 0:
            raise ValueError(f"Audio file is empty: {audio_file}")
        
        # Check if file is readable
        try:
            info = torchaudio.info(str(audio_path))
            logger.info(f"Audio file info - Sample rate: {info.sample_rate}, Duration: {info.num_frames / info.sample_rate:.2f}s")
        except Exception as e:
            raise ValueError(f"Unable to read audio file {audio_file}: {e}")
            
        return audio_path
    
    def run_diarization(self, audio_file: str) -> List[Dict]:
        """
        Perform speaker diarization on audio file.
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            List of diarization segments with speaker labels
        """
        if not self.diarization_available:
            logger.warning("Diarization pipeline not available. Returning single speaker segment.")
            # Fallback: create a single segment covering the entire audio
            try:
                info = torchaudio.info(audio_file)
                duration = info.num_frames / info.sample_rate
                return [{
                    "start": 0.0,
                    "end": float(duration),
                    "speaker": "SPEAKER_00",
                    "duration": float(duration)
                }]
            except Exception as e:
                logger.error(f"Failed to get audio duration for fallback: {e}")
                # Default fallback segment
                return [{
                    "start": 0.0,
                    "end": 3600.0,  # 1 hour max
                    "speaker": "SPEAKER_00",
                    "duration": 3600.0
                }]
        
        logger.info("Starting speaker diarization...")
        try:
            diarization = self.diarization_pipeline(audio_file)
            segments = []
            
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append({
                    "start": float(turn.start),
                    "end": float(turn.end),
                    "speaker": speaker,
                    "duration": float(turn.end - turn.start)
                })
            
            logger.info(f"Diarization completed: {len(segments)} segments found")
            return segments
            
        except Exception as e:
            logger.error(f"Diarization failed: {e}")
            raise
    
    def run_transcription(self, audio_file: str) -> List[Dict]:
        """
        Perform speech transcription on audio file.
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            List of transcription segments with text
        """
        logger.info("Starting speech transcription...")
        try:
            segments, info = self.whisper_model.transcribe(
                audio_file, 
                beam_size=5,
                language=None,  # Auto-detect language
                vad_filter=True,  # Use voice activity detection
                vad_parameters=dict(min_silence_duration_ms=500)
            )
            
            results = []
            for segment in segments:
                if segment.text.strip():  # Skip empty transcriptions
                    results.append({
                        "start": float(segment.start),
                        "end": float(segment.end),
                        "text": segment.text.strip(),
                        "avg_logprob": float(segment.avg_logprob),
                        "no_speech_prob": float(segment.no_speech_prob)
                    })
            
            logger.info(f"Transcription completed: {len(results)} segments, Language: {info.language}")
            return results
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise
    
    def merge_results(self, diarization: List[Dict], transcription: List[Dict], 
                     overlap_threshold: float = 0.3) -> List[Dict]:
        """
        Merge diarization and transcription results with improved alignment.
        
        Args:
            diarization: Diarization segments
            transcription: Transcription segments  
            overlap_threshold: Minimum overlap ratio to consider segments matching
            
        Returns:
            Merged results with speaker labels and transcriptions
        """
        logger.info("Merging diarization and transcription results...")
        
        def calculate_overlap(seg1: Dict, seg2: Dict) -> float:
            """Calculate overlap ratio between two segments."""
            overlap_start = max(seg1["start"], seg2["start"])
            overlap_end = min(seg1["end"], seg2["end"])
            overlap_duration = max(0, overlap_end - overlap_start)
            
            # Calculate overlap as ratio of shorter segment
            min_duration = min(seg1["end"] - seg1["start"], seg2["end"] - seg2["start"])
            return overlap_duration / min_duration if min_duration > 0 else 0
        
        output = []
        used_transcriptions = set()
        
        for d_seg in diarization:
            matching_transcriptions = []
            
            for i, t_seg in enumerate(transcription):
                if i in used_transcriptions:
                    continue
                    
                overlap_ratio = calculate_overlap(d_seg, t_seg)
                if overlap_ratio >= overlap_threshold:
                    matching_transcriptions.append((i, t_seg, overlap_ratio))
            
            # Sort by overlap ratio (highest first)
            matching_transcriptions.sort(key=lambda x: x[2], reverse=True)
            
            # Combine text from matching transcriptions
            combined_text = ""
            avg_confidence = 0
            used_indices = []
            
            for idx, t_seg, _ in matching_transcriptions:
                combined_text += t_seg["text"] + " "
                avg_confidence += t_seg.get("avg_logprob", 0)
                used_indices.append(idx)
            
            # Mark transcriptions as used
            used_transcriptions.update(used_indices)
            
            if combined_text.strip():
                output.append({
                    "speaker": d_seg["speaker"],
                    "start": d_seg["start"],
                    "end": d_seg["end"],
                    "duration": d_seg["duration"],
                    "text": combined_text.strip(),
                    "confidence": avg_confidence / len(used_indices) if used_indices else 0,
                    "transcription_segments": len(used_indices)
                })
        
        logger.info(f"Merge completed: {len(output)} final segments")
        return output
    
    def process_audio(self, audio_file: str, output_file: Optional[str] = None, 
                     use_simple_diarization: bool = True) -> List[Dict]:
        """
        Complete processing pipeline: diarization + transcription + merging.
        
        Args:
            audio_file: Path to input audio file
            output_file: Optional path to save results as JSON
            use_simple_diarization: Use simple VAD-based diarization if pyannote fails
            
        Returns:
            Final processed results
        """
        # Validate input
        audio_path = self.validate_audio_file(audio_file)
        
        try:
            # Always run transcription first
            transcription_segments = self.run_transcription(str(audio_path))
            
            # Try proper diarization, fall back to simple method if needed
            if self.diarization_available:
                diarization_segments = self.run_diarization(str(audio_path))
            elif use_simple_diarization and transcription_segments:
                logger.info("Using simple diarization fallback method...")
                diarization_segments = self.simple_voice_activity_diarization(transcription_segments)
            else:
                # No diarization - create single speaker segment
                logger.info("No diarization available - assuming single speaker")
                if transcription_segments:
                    total_duration = max(seg["end"] for seg in transcription_segments)
                    diarization_segments = [{
                        "start": 0.0,
                        "end": total_duration,
                        "speaker": "SPEAKER_00",
                        "duration": total_duration
                    }]
                else:
                    diarization_segments = []
            
            # Merge results
            final_results = self.merge_results(diarization_segments, transcription_segments)
            
            # Save results if requested
            if output_file:
                self.save_results(final_results, output_file)
            
            return final_results
            
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            raise
    
    def save_results(self, results: List[Dict], output_file: str):
        """Save results to JSON file."""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to: {output_file}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            raise
    
    def print_results(self, results: List[Dict], show_confidence: bool = False):
        """Print formatted results."""
        print("\n" + "="*80)
        print("TRANSCRIPTION RESULTS")
        print("="*80)
        
        for seg in results:
            timestamp = f"[{seg['start']:.2f}s - {seg['end']:.2f}s]"
            speaker_text = f"{seg['speaker']}: {seg['text']}"
            
            if show_confidence and 'confidence' in seg:
                confidence = f" (confidence: {seg['confidence']:.2f})"
                print(f"{timestamp} {speaker_text}{confidence}")
            else:
                print(f"{timestamp} {speaker_text}")
        print("="*80)


def main():
    """Main execution function with example usage."""
    # Configuration
    AUDIO_FILE = "meeting_audio.wav"  # Change this to your audio file
    OUTPUT_FILE = "transcription_results.json"
    
    # Set up authentication for pyannote models
    print("SETUP INSTRUCTIONS:")
    print("For speaker diarization, you need to authenticate with Hugging Face:")
    print("1. Visit: https://huggingface.co/pyannote/speaker-diarization-3.1")
    print("2. Accept the user conditions")
    print("3. Get your token: https://hf.co/settings/tokens")
    print("4. Set it as environment variable: export HF_TOKEN='your_token_here'")
    print("   Or run: huggingface-cli login")
    print("\nWithout authentication, only transcription will be performed.\n")
    
    try:
        # Initialize processor
        processor = AudioProcessor(
            whisper_model_size="small",  # Options: tiny, base, small, medium, large-v3
            compute_type="float16"  # Options: float16, float32, int8
        )
        
        # Process audio
        results = processor.process_audio(AUDIO_FILE, OUTPUT_FILE)
        
        # Display results
        processor.print_results(results, show_confidence=True)
        
        # Print summary
        total_duration = max(seg['end'] for seg in results) if results else 0
        unique_speakers = set(seg['speaker'] for seg in results)
        
        print(f"\nSUMMARY:")
        print(f"Total duration: {total_duration:.2f} seconds")
        print(f"Speakers identified: {len(unique_speakers)}")
        print(f"Total segments: {len(results)}")
        print(f"Diarization available: {processor.diarization_available}")
        
    except FileNotFoundError as e:
        logger.error(f"File error: {e}")
        print(f"Please ensure your audio file '{AUDIO_FILE}' exists in the current directory")
    except Exception as e:
        logger.error(f"Processing error: {e}")
        print(f"An error occurred during processing: {e}")


if __name__ == "__main__":
    main()