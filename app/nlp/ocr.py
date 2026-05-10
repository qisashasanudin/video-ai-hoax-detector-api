"""
Optical Character Recognition (OCR) for extracting text from video frames.
Analyzes text visible in frames for videos without voiceovers or with on-screen text.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def extract_text_from_frames(frames_dir: str, max_frames: int = 8) -> Tuple[str, Optional[str]]:
    """
    Extract text from video frames using EasyOCR.
    
    Args:
        frames_dir: Directory containing frame_*.jpg files
        max_frames: Maximum number of frames to analyze (for performance)
    
    Returns:
        (extracted_text, error_message) where extracted_text is concatenated text from frames
    """
    try:
        import glob
        import easyocr
        
        frame_paths = sorted(glob.glob(os.path.join(frames_dir, "frame_*.jpg")))
        
        if not frame_paths:
            return "", "No frames found for OCR"
        
        # Sample frames evenly if too many
        if len(frame_paths) > max_frames:
            frame_paths = frame_paths[::len(frame_paths) // max_frames][:max_frames]
        
        logger.info(f"OCR: Processing {len(frame_paths)} frames")
        
        # Initialize reader (will auto-download model on first use)
        reader = easyocr.Reader(['id', 'en'], gpu=False)
        
        extracted_texts = []
        
        for frame_path in frame_paths:
            try:
                # EasyOCR returns list of tuples: (text, confidence, bounding_box)
                results = reader.readtext(frame_path, detail=1)
                
                if results:
                    # Extract text with reasonable confidence (>0.3)
                    frame_text = " ".join([text for text, conf, _ in results if conf > 0.3])
                    if frame_text.strip():
                        extracted_texts.append(frame_text.strip())
                        logger.debug(f"Frame {os.path.basename(frame_path)}: extracted {len(frame_text)} chars")
                
            except Exception as e:
                logger.warning(f"OCR failed for frame {frame_path}: {e}")
                continue
        
        # Deduplicate and concatenate
        combined_text = " ".join(extracted_texts)
        
        # Remove excessive whitespace
        combined_text = " ".join(combined_text.split())
        
        if combined_text:
            logger.info(f"OCR: Extracted {len(combined_text)} total characters")
            return combined_text, None
        else:
            return "", "No text detected in frames"
            
    except ImportError:
        return "", "EasyOCR not installed. Install with: pip install easyocr"
    except Exception as e:
        error_msg = f"OCR extraction failed: {str(e)}"
        logger.error(error_msg)
        return "", error_msg


def _extract_text_from_frames_fallback(frames_dir: str, max_frames: int = 8) -> Tuple[str, Optional[str]]:
    """
    Fallback OCR using Tesseract (if EasyOCR not available).
    Requires system tesseract installation.
    """
    try:
        import pytesseract
        import glob
        from PIL import Image
        
        frame_paths = sorted(glob.glob(os.path.join(frames_dir, "frame_*.jpg")))
        
        if not frame_paths:
            return "", "No frames found"
        
        if len(frame_paths) > max_frames:
            frame_paths = frame_paths[::len(frame_paths) // max_frames][:max_frames]
        
        extracted_texts = []
        
        for frame_path in frame_paths:
            try:
                image = Image.open(frame_path)
                text = pytesseract.image_to_string(image, lang='ind+eng')
                if text.strip():
                    extracted_texts.append(text.strip())
            except Exception as e:
                logger.warning(f"Tesseract OCR failed for {frame_path}: {e}")
                continue
        
        combined_text = " ".join(extracted_texts)
        combined_text = " ".join(combined_text.split())
        
        if combined_text:
            return combined_text, None
        else:
            return "", "No text detected"
            
    except ImportError:
        return "", "Tesseract not installed or not configured"
    except Exception as e:
        return "", f"Fallback OCR failed: {str(e)}"


def extract_and_analyze_text(
    frames_dir: str,
    video_title: Optional[str] = None,
    video_description: Optional[str] = None,
) -> Dict[str, Optional[str]]:
    """
    Extract text from frames and prepare for LLM analysis.
    
    Returns:
        {
            "ocr_text": "extracted text from frames",
            "has_text": True/False,
            "extraction_note": "status message or error"
        }
    """
    ocr_text, ocr_error = extract_text_from_frames(frames_dir, max_frames=8)
    
    if not ocr_text and not ocr_error.startswith("No text"):
        # Try fallback
        ocr_text, ocr_error = _extract_text_from_frames_fallback(frames_dir, max_frames=8)
    
    has_text = len(ocr_text) > 20  # Require at least 20 characters to be meaningful
    
    result = {
        "ocr_text": ocr_text if has_text else "",
        "has_text": has_text,
        "extraction_note": ocr_error if not has_text else "OCR extraction successful",
    }
    
    return result
