#!/usr/bin/env python3
"""
Main audio processing pipeline orchestrator.
"""

import logging
from typing import List, Dict, Optional

import torch

from transcriber import Transcriber
from diarizer import Diarizer
import utils

logger = logging.getLogger(__name__)

class AudioProcessor:
    """Orchestrates the audio processing pipeline."""
    
    def __init__(self, 
                 whisper_model_size: str = "small",
                 auth_token: Optional[str] = None,
                 compute_type: str = "float16"):
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing AudioProcessor with device: {self.device}")
        
        self.transcriber = Transcriber(
            model_size=whisper_model_size, 
            device=self.device,
            compute_type=compute_type
        )
        self.diarizer = Diarizer(
            device=self.device,
            auth_token=auth_token
        )

    @property
    def diarization_available(self) -> bool:
        """Check if the diarization pipeline is available."""
        return self.diarizer.is_available

    def process_audio(self, audio_file: str, output_file: Optional[str] = None) -> List[Dict]:
        """
        Full processing pipeline: transcription -> diarization -> merge.
        """
        try:
            audio_path = utils.validate_audio_file(audio_file)
            
            transcription_segments = self.transcriber.transcribe(str(audio_path))
            
            diarization_segments = self.diarizer.diarize(str(audio_path), transcription_segments)

            final_results = utils.merge_results(diarization_segments, transcription_segments)
            
            if output_file:
                utils.save_results(final_results, output_file)
            
            return final_results
            
        except Exception as e:
            logger.error(f"Processing failed for {audio_file}: {e}")
            raise