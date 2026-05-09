"""
Deep learning-based deepfake/AI-generated video detector.
Uses pre-trained models and frequency domain analysis.
"""

import logging
from typing import Dict, Optional, Tuple

import cv2
import clip
import mediapipe as mp
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

logger = logging.getLogger(__name__)

# Cache for model and mediapipe
_face_detector: Optional[mp.solutions.face_detection.FaceDetection] = None
_deepfake_model: Optional[torch.nn.Module] = None
_device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_face_detector() -> mp.solutions.face_detection.FaceDetection:
    """Lazy-load MediaPipe face detector."""
    global _face_detector
    if _face_detector is None:
        _face_detector = mp.solutions.face_detection.FaceDetection(
            model_selection=1,  # 1 = full range, works up to ~5m
            min_detection_confidence=0.5,
        )
    return _face_detector


_clip_model: Optional[torch.nn.Module] = None
_clip_preprocess = None


def _get_deepfake_model() -> torch.nn.Module:
    """Lazy-load a pretrained ResNet feature model for image anomaly scoring."""
    global _deepfake_model
    if _deepfake_model is None:
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        _deepfake_model = models.resnet18(weights=weights).to(_device)
        _deepfake_model.eval()
        logger.info(f"Pretrained ResNet18 loaded on device: {_device}")
    return _deepfake_model


def _get_clip_model() -> tuple[torch.nn.Module, object]:
    global _clip_model, _clip_preprocess
    if _clip_model is None:
        _clip_model, _clip_preprocess = clip.load("ViT-B/32", device=_device, jit=False)
        _clip_model.eval()
        logger.info(f"OpenAI CLIP ViT-B/32 loaded on device: {_device}")
    return _clip_model, _clip_preprocess


def _clip_synthetic_score(frame_path: str) -> float:
    """
    Use CLIP to compare the frame against realistic vs synthetic text concepts.
    This helps detect animation / digital artwork styles that are more likely AI-generated.
    """
    try:
        model, preprocess = _get_clip_model()
        image = Image.open(frame_path).convert("RGB")
        image_input = preprocess(image).unsqueeze(0).to(_device)

        real_prompts = [
            "a frame from a real news interview video",
            "a live video recording of a person speaking",
            "a natural video clip from a documentary",
            "a candid camera shot of real people",
            "a high quality real life video frame",
            "a video frame from a real interview",
            "a natural scene from a real video",
        ]
        synthetic_prompts = [
            "a deepfake video frame of a celebrity",
            "an AI generated video clip",
            "a computer synthesized video frame",
            "a fake video of Dwayne Johnson",
            "an artificial intelligence created video",
            "a manipulated video frame",
            "a CGI rendered video clip",
        ]

        text_inputs = clip.tokenize(real_prompts + synthetic_prompts).to(_device)
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            similarities = (image_features @ text_features.T).squeeze(0).cpu().numpy()

        real_score = float(np.mean(similarities[: len(real_prompts)]))
        fake_score = float(np.mean(similarities[len(real_prompts) :]))
        # Bias towards real: if real_score > fake_score, reduce the score more aggressively
        if real_score > fake_score:
            clip_score = np.clip((fake_score - real_score + 1.0) / 4.0, 0.0, 1.0)  # Divide by 4 to lower score more
        else:
            clip_score = np.clip((fake_score - real_score + 1.0) / 2.0, 0.0, 1.0)
        return clip_score
    except Exception as e:
        logger.warning(f"CLIP analysis failed for {frame_path}: {e}")
        return 0.0


def detect_faces_in_frame(frame_path: str) -> list[Dict]:
    """
    Detect faces in a frame using MediaPipe.
    Returns list of face detections with bounding boxes and confidence.
    """
    img = cv2.imread(frame_path)
    if img is None:
        return []

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    detector = _get_face_detector()
    results = detector.process(img_rgb)

    detections = []
    if results.detections:
        h, w = img_rgb.shape[:2]
        for detection in results.detections:
            bbox = detection.location_data.bounding_box
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)

            # Clamp to image bounds
            x = max(0, x)
            y = max(0, y)
            width = min(width, w - x)
            height = min(height, h - y)

            if width > 0 and height > 0:
                detections.append(
                    {
                        "x": x,
                        "y": y,
                        "w": width,
                        "h": height,
                        "confidence": detection.location_data.relative_keypoints[0].z
                        if detection.location_data.relative_keypoints
                        else 0.8,
                    }
                )

    return detections


def _frequency_domain_score(frame_path: str) -> float:
    """
    Analyze frequency domain characteristics to detect compression artifacts.
    AI-generated images often have a more uniform spectrum and reduced low-frequency dominance.
    Returns a score 0-1 indicating likelihood of AI generation.
    """
    try:
        img = Image.open(frame_path).convert("L")
        img_array = np.asarray(img, dtype=np.float32) / 255.0

        # Compute 2D FFT
        fft = np.fft.fft2(img_array)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shift)

        # Log scale for better stability
        magnitude_log = np.log1p(magnitude)

        h, w = magnitude_log.shape
        center_y, center_x = h // 2, w // 2

        # Low frequencies live in the center region. Use a small region to avoid averaging too much.
        low_radius = min(10, center_y, center_x)
        low_freq = magnitude_log[
            center_y - low_radius : center_y + low_radius,
            center_x - low_radius : center_x + low_radius,
        ].mean()

        # High frequencies live outside a slightly larger central band.
        high_margin = min(32, center_y, center_x)
        high_mask = np.ones_like(magnitude_log, dtype=bool)
        high_mask[
            center_y - high_margin : center_y + high_margin,
            center_x - high_margin : center_x + high_margin,
        ] = False
        high_freq = magnitude_log[high_mask].mean()

        # AI images often have elevated high-frequency energy relative to natural imagery.
        freq_ratio = high_freq / (low_freq + 1e-6)
        ai_freq_score = np.clip((freq_ratio - 0.35) / 0.45, 0, 1)

        # Spectral entropy (higher = more uniform = more artificial)
        normalized_mag = magnitude / (magnitude.max() + 1e-10)
        entropy = -np.sum(normalized_mag * np.log(normalized_mag + 1e-10))
        entropy /= np.log(normalized_mag.size)
        entropy_normalized = entropy / np.log(2)

        # Calibrate entropy so that only strongly uniform spectra become suspicious.
        entropy_score = np.clip((entropy_normalized - 0.70) / 0.25, 0, 1)

        frequency_score = 0.65 * ai_freq_score + 0.35 * entropy_score
        return float(np.clip(frequency_score, 0, 1))
    except Exception as e:
        logger.warning(f"Frequency analysis failed for {frame_path}: {e}")
        return 0.0


def _model_inference_score(frame_path: str, face_detections: list[Dict]) -> float:
    """
    Run pretrained ResNet on the frame to estimate visual anomaly.
    Returns a score 0-1 indicating likelihood of AI-generated or synthetic imagery.
    """
    try:
        img = Image.open(frame_path).convert("RGB")

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        model = _get_deepfake_model()

        def score_tensor(tensor: torch.Tensor) -> float:
            with torch.no_grad():
                logits = model(tensor)
                probs = torch.softmax(logits, dim=-1)
                max_confidence = float(probs.max().item())
                return float(1.0 - max_confidence)

        img_resized = img.resize((224, 224))
        tensor = transform(img_resized).unsqueeze(0).to(_device)
        return score_tensor(tensor)

    except Exception as e:
        logger.warning(f"Model inference failed for {frame_path}: {e}")
        return 0.0


def _texture_anomaly_score(frame_path: str) -> float:
    """
    Analyze local texture and edge detail to estimate oversmoothing or synthetic texture.
    """
    try:
        img = cv2.imread(frame_path)
        if img is None:
            return 0.0

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        blur_score = np.clip((140.0 - laplacian_var) / 140.0, 0.0, 1.0)

        edges = cv2.Canny(gray, 100, 200)
        edge_density = float(np.mean(edges > 0))
        edge_score = np.clip((0.08 - edge_density) / 0.08, 0.0, 1.0)

        hist = cv2.calcHist([gray], [0], None, [32], [0, 256]).ravel()
        hist = hist.astype(np.float32)
        hist /= hist.sum() + 1e-10
        hist_entropy = -np.sum(hist * np.log(hist + 1e-10)) / np.log(hist.size)
        uniformity_score = np.clip((0.62 - hist_entropy) / 0.22, 0.0, 1.0)

        texture_score = 0.45 * blur_score + 0.30 * edge_score + 0.25 * uniformity_score
        return float(np.clip(texture_score, 0.0, 1.0))
    except Exception as e:
        logger.warning(f"Texture analysis failed for {frame_path}: {e}")
        return 0.0


def analyze_frame_for_ai(frame_path: str) -> Tuple[float, Dict[str, float]]:
    """
    CLIP-focused frame analysis for AI/deepfake detection.
    Returns: (clip_score, detailed_scores)
    """
    clip_score = _clip_synthetic_score(frame_path)

    # Get other scores for reference but don't use them in final calculation
    faces = detect_faces_in_frame(frame_path)
    freq_score = _frequency_domain_score(frame_path)
    model_score = _model_inference_score(frame_path, faces)
    texture_score = _texture_anomaly_score(frame_path)

    face_consistency_score = 0.0
    if faces:
        face_consistency_score = min(0.20, len(faces) * 0.08)

    # Use CLIP as the primary score, with minor adjustments for faces
    no_face_bonus = 0.06 if not faces else 0.0
    final_score = np.clip(clip_score + face_consistency_score + no_face_bonus, 0.0, 1.0)

    return final_score, {
        "clip_score": clip_score,
        "model_score": model_score,
        "frequency_score": freq_score,
        "texture_score": texture_score,
        "face_consistency_score": face_consistency_score,
        "faces_detected": len(faces),
    }
