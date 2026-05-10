"""
Synthetic/AI-generated video detection using modern methods.
Focuses on Sora, Veo, Seedance-style generation artifacts.
Core principles: temporal consistency, embedding anomalies, optical flow analysis.
"""

import logging
import glob
import numpy as np
import cv2
from typing import List, Tuple, Dict, Optional

logger = logging.getLogger(__name__)


def _load_frame(frame_path: str) -> Optional[np.ndarray]:
    """Load frame as RGB for consistent processing."""
    img = cv2.imread(frame_path)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _extract_vit_embeddings(frames: List[np.ndarray], model_name: str = "ViT-B/32") -> np.ndarray:
    """
    Extract Vision Transformer embeddings for each frame.
    Synthetic content clusters distinctly in embedding space.
    
    Returns: (N_frames, embedding_dim) array
    """
    try:
        import clip
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load(model_name, device=device)

        embeddings = []
        with torch.no_grad():
            for frame in frames:
                # Preprocess: convert to PIL, apply transformations
                from PIL import Image
                pil_img = Image.fromarray(frame.astype("uint8"))
                tensor = preprocess(pil_img).unsqueeze(0).to(device)
                embedding = model.encode_image(tensor).cpu().numpy()
                embeddings.append(embedding[0])

        return np.array(embeddings)
    except Exception as e:
        logger.warning(f"CLIP embedding extraction failed: {e}")
        return np.array([])


def _estimate_clip_prompt_synthetic_score(frames: List[np.ndarray]) -> Tuple[float, str]:
    """
    Estimate synthetic likelihood using CLIP similarity to real vs synthetic prompts.
    """
    try:
        import clip
        import torch
        from PIL import Image

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
        model.eval()

        real_prompts = [
            "a real video frame from a news broadcast",
            "a live action video frame from a real documentary",
            "a natural video frame recorded from a camera",
        ]
        synthetic_prompts = [
            "an AI generated video frame",
            "a computer generated video frame",
            "a deepfake video frame",
            "an artificial intelligence created video clip",
        ]
        text_inputs = clip.tokenize(real_prompts + synthetic_prompts).to(device)

        frame_features = []
        with torch.no_grad():
            for frame in frames[:8]:
                pil_img = Image.fromarray(frame.astype("uint8"))
                image_input = preprocess(pil_img).unsqueeze(0).to(device)
                image_features = model.encode_image(image_input)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                frame_features.append(image_features)

            if not frame_features:
                return 0.0, "No frames available for CLIP prompt analysis."

            image_features = torch.cat(frame_features, dim=0)
            text_features = model.encode_text(text_inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            similarities = (image_features @ text_features.T).cpu().numpy()

        real_score = float(np.mean(similarities[:, : len(real_prompts)]))
        fake_score = float(np.mean(similarities[:, len(real_prompts) :]))
        if real_score > fake_score:
            clip_score = np.clip((fake_score - real_score + 1.0) / 4.0, 0.0, 1.0)
        else:
            clip_score = np.clip((fake_score - real_score + 1.0) / 2.0, 0.0, 1.0)

        return clip_score, f"CLIP prompt anomaly={clip_score:.3f}, real_similarity={real_score:.3f}, synthetic_similarity={fake_score:.3f}"
    except Exception as e:
        logger.warning(f"CLIP prompt analysis failed: {e}")
        return 0.0, "CLIP prompt analysis unavailable."


def _detect_embedding_anomalies(embeddings: np.ndarray) -> Tuple[float, str]:
    """
    Detect synthetic fingerprints in embedding space.
    Real videos have natural drift; synthetic have characteristic clustering.
    
    Returns: (anomaly_score, explanation)
    """
    if len(embeddings) < 3:
        return 0.0, "Insufficient frames for embedding analysis."

    # Compute pairwise distances
    distances = []
    for i in range(len(embeddings) - 1):
        dist = np.linalg.norm(embeddings[i] - embeddings[i + 1])
        distances.append(dist)

    distances = np.array(distances)
    
    # Real videos: natural variability in embedding distance
    # Synthetic: either too uniform (repetitive) or jumpy (discontinuous generation)
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    
    # Detect both extremes
    uniformity_score = 1.0 - (std_dist / (mean_dist + 1e-6))  # High=uniform=bad
    uniformity_score = np.clip(uniformity_score, 0, 1)
    
    # Detect excessive jumps (frame-to-frame incoherence)
    quantile_90 = np.percentile(distances, 90)
    jump_ratio = np.sum(distances > quantile_90 * 1.5) / len(distances)
    
    anomaly_score = float(0.6 * uniformity_score + 0.4 * jump_ratio)
    
    return anomaly_score, f"Embedding uniformity={uniformity_score:.3f}, jumpy_frames={jump_ratio:.3f}"


def _analyze_optical_flow_consistency(frames: List[np.ndarray]) -> Tuple[float, str]:
    """
    Analyze optical flow for synthetic motion patterns.
    Real motion: smooth, coherent, follows physics.
    Synthetic: irregular, lacks smooth transitions, impossible accelerations.
    
    Returns: (motion_anomaly_score, explanation)
    """
    if len(frames) < 3:
        return 0.0, "Insufficient frames for motion analysis."

    frame_grays = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
    
    flow_magnitudes = []
    flow_angles = []
    flow_divergences = []
    
    for i in range(len(frame_grays) - 1):
        flow = cv2.calcOpticalFlowFarneback(
            frame_grays[i], frame_grays[i + 1],
            None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        flow_magnitudes.append(np.mean(mag))
        flow_angles.append(np.mean(ang))
        
        # Compute divergence: ∇·F = ∂u/∂x + ∂v/∂y
        u, v = flow[..., 0], flow[..., 1]
        div = np.gradient(u, axis=1).mean() + np.gradient(v, axis=0).mean()
        flow_divergences.append(abs(div))
    
    flow_magnitudes = np.array(flow_magnitudes)
    flow_angles = np.array(flow_angles)
    flow_divergences = np.array(flow_divergences)
    
    # Synthetic videos often have jerky motion (high acceleration)
    mag_acceleration = np.diff(flow_magnitudes)
    motion_jerkiness = np.std(mag_acceleration) + np.mean(np.abs(mag_acceleration))
    motion_jerkiness = np.clip(motion_jerkiness / 0.5, 0, 1)  # Normalize
    
    # Synthetic videos can have unrealistic divergence patterns
    avg_divergence = np.mean(flow_divergences)
    divergence_anomaly = np.clip(avg_divergence / 0.1, 0, 1)
    
    motion_score = float(0.6 * motion_jerkiness + 0.4 * divergence_anomaly)
    
    return motion_score, f"Jerkiness={motion_jerkiness:.3f}, divergence={avg_divergence:.3f}"


def _analyze_frequency_domain(frames: List[np.ndarray]) -> Tuple[float, str]:
    """
    Detect synthetic artifacts in frequency domain.
    Generative models produce characteristic frequency signatures.
    
    Returns: (frequency_anomaly_score, explanation)
    """
    if not frames:
        return 0.0, "No frames for frequency analysis."
    
    freq_scores = []
    
    for frame in frames:
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # FFT
        fft = np.fft.fft2(gray)
        magnitude = np.abs(fft)
        
        # Shift zero frequency to center
        magnitude_shifted = np.fft.fftshift(magnitude)
        
        # Log scale
        log_magnitude = np.log(magnitude_shifted + 1)
        
        # Analyze energy distribution
        # Real images: more energy at lower frequencies
        # Synthetic: characteristic ripple patterns and unnatural frequency spikes
        
        h, w = log_magnitude.shape
        center = (h // 2, w // 2)
        
        # Compute radial energy distribution
        y, x = np.ogrid[:h, :w]
        radius = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        
        # Energy in rings
        inner_energy = log_magnitude[radius < min(h, w) / 4].mean()
        middle_energy = log_magnitude[(radius >= min(h, w) / 4) & (radius < min(h, w) / 2)].mean()
        outer_energy = log_magnitude[radius >= min(h, w) / 2].mean()
        
        # Real images: smooth energy decay
        # Synthetic: irregular patterns
        energy_ratio = (middle_energy + outer_energy) / (inner_energy + 1e-6)
        energy_anomaly = np.clip(energy_ratio / 2.0, 0, 1)  # Normalize
        
        freq_scores.append(energy_anomaly)
    
    freq_anomaly = float(np.mean(freq_scores))
    
    return freq_anomaly, f"Frequency domain anomaly score={freq_anomaly:.3f}"



def detect_synthetic_generation(
    frames_dir: str,
    audio_path: Optional[str] = None,
    *,
    url: str,
    max_frames: int = 16,
    video_title: Optional[str] = None,
    video_description: Optional[str] = None,
) -> Tuple[float, List[str]]:
    """
    Detect modern AI-generated (synthetic) video content.
    
    Combines:
    1. Vision Transformer embedding analysis
    2. Optical flow consistency analysis
    3. Frequency domain anomalies
    4. CLIP prompt-based visual anomaly detection
    
    Args:
        frames_dir: Directory with extracted frames (frame_*.jpg)
        audio_path: Optional audio file path
        url: Source video URL
        max_frames: Maximum frames to analyze (for performance)
        video_title: Video title
        video_description: Video description
    
    Returns:
        (synthetic_score [0-1], explanation_list)
    """
    
    frame_paths = sorted(glob.glob(f"{frames_dir}/frame_*.jpg"))
    
    if not frame_paths:
        return 0.0, ["No frames found for synthetic detection."]
    
    if max_frames:
        frame_paths = frame_paths[::max(1, len(frame_paths) // max_frames)][:max_frames]
    
    # Load frames
    frames = []
    for fp in frame_paths:
        img = _load_frame(fp)
        if img is not None:
            frames.append(img)
    
    if not frames:
        return 0.0, ["Failed to load frames for analysis."]
    
    explanations = []
    scores = {}
    
    # 1. Embedding space analysis
    embeddings = _extract_vit_embeddings(frames)
    if len(embeddings) > 0:
        embedding_score, embedding_note = _detect_embedding_anomalies(embeddings)
        scores["embedding"] = embedding_score
        explanations.append(embedding_note)
    
    # 2. Optical flow analysis
    motion_score, motion_note = _analyze_optical_flow_consistency(frames)
    scores["motion"] = motion_score
    explanations.append(motion_note)
    
    # 3. Frequency domain analysis
    freq_score, freq_note = _analyze_frequency_domain(frames)
    scores["frequency"] = freq_score
    explanations.append(freq_note)

    # 4. CLIP prompt-based synthetic detection
    clip_prompt_score, clip_prompt_note = _estimate_clip_prompt_synthetic_score(frames)
    scores["clip_prompt"] = clip_prompt_score
    explanations.append(clip_prompt_note)

    # Weighted composite score using purely technical visual anomaly signals.
    weights = {
        "embedding": 0.35,
        "motion": 0.30,
        "frequency": 0.15,
        "clip_prompt": 0.20,
    }

    weighted_sum = 0.0
    total_weight = 0.0
    for key, weight in weights.items():
        score_value = scores.get(key)
        if score_value is None:
            continue
        weighted_sum += weight * score_value
        total_weight += weight

    synthetic_score = float(weighted_sum / total_weight) if total_weight else 0.0
    synthetic_score = np.clip(synthetic_score, 0.0, 1.0)

    return synthetic_score, explanations

