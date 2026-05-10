"""
Unified AI detection orchestrator: synthetic generation + deepfakes.
Primary: Modern generative model detection (Sora, Veo, Seedance-style)
Fallback: Legacy deepfake detection
Note: this module focuses on technical visual/audio anomaly detection only,
not factual claim verification or semantic fact-checking.
"""

import glob
import logging
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


def detect_ai_generation_from_frames(
    frames_dir: str,
    audio_path: Optional[str] = None,
    *,
    url: str,
    max_frames: int = 16,
    video_title: Optional[str] = None,
    video_description: Optional[str] = None,
) -> Tuple[float, List[str]]:
    """
    Revamped AI detection focused on modern synthetic content.
    
    New approach:
    1. Vision Transformer embedding analysis for synthetic fingerprints
    2. Optical flow consistency to detect unnatural motion
    3. Frequency domain analysis for generation artifacts
    
    Args:
        frames_dir: Directory with frame_*.jpg files
        audio_path: Optional audio file
        url: Source video URL
        max_frames: Maximum frames to analyze
        video_title: Video title for context
        video_description: Video description for context
    
    Returns:
        (ai_score [0-1], explanation_list)
    """
    
    frame_paths = sorted(glob.glob(f"{frames_dir}/frame_*.jpg"))
    
    if not frame_paths:
        return 0.0, ["No frames found for AI detection."]
    
    # Try modern synthetic detection first
    try:
        from .synthetic_detector import detect_synthetic_generation
        
        synthetic_score, explanations = detect_synthetic_generation(
            frames_dir=frames_dir,
            audio_path=audio_path,
            url=url,
            max_frames=max_frames,
            video_title=video_title,
            video_description=video_description,
        )
        
        return float(synthetic_score), explanations
        
    except Exception as e:
        logger.warning(f"Synthetic detection failed, falling back to legacy: {e}")
    
    # Fallback: Legacy detection if modern approach fails
    try:
        from .deepfake_detector import analyze_frame_for_ai
        from .temporal_aggregator import aggregate_frame_scores
        import cv2
        import numpy as np
        
        if max_frames:
            frame_paths = frame_paths[:max_frames]
        
        frame_scores = []
        for frame_path in frame_paths:
            try:
                ai_score, details = analyze_frame_for_ai(frame_path)
                frame_scores.append(ai_score)
            except Exception as e:
                logger.debug(f"Frame analysis failed for {frame_path}: {e}")
                continue
        
        if not frame_scores:
            return 0.0, ["Frame analysis failed."]
        
        temporal_score = aggregate_frame_scores(frame_scores)
        
        return float(np.clip(temporal_score, 0.0, 1.0)), [
            f"Legacy detection score: {temporal_score:.1%} across {len(frame_scores)} frames",
        ]
        
    except Exception as e:
        logger.error(f"All detection methods failed: {e}")
        return 0.0, ["Detection system unavailable."]

