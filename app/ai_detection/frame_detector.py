"""
Frame-level AI/deepfake detection with temporal aggregation and Gemma analysis.
"""

import glob
import logging
import wave
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .deepfake_detector import analyze_frame_for_ai
from .temporal_aggregator import aggregate_frame_scores

logger = logging.getLogger(__name__)


def _analyze_content_with_gemma(url: str, video_title: Optional[str] = None, video_description: Optional[str] = None) -> Tuple[float, str]:
    """Use Gemma to analyze content for AI generation indicators."""
    try:
        import subprocess
        import json
        import re

        context_parts = []
        if video_title:
            context_parts.append(f"Judul: {video_title}")
        if video_description:
            context_parts.append(f"Deskripsi: {video_description}")
        context_parts.append(f"URL: {url}")

        context = "\n".join(context_parts)

        prompt = f"""Analisis konten video ini untuk indikasi bahwa ini dihasilkan oleh AI atau deepfake.

Konteks:
{context}

Pertanyaan:
1. Apakah konten ini menunjukkan pola yang khas AI-generated (seperti visualisasi mustahil, transisi yang tidak natural, efek visual yang tidak realistis)?
2. Apakah ada petunjuk tekstual tentang AI/generatif dalam judul/deskripsi?
3. Apakah konten menggambarkan skenario yang mustahil secara fisik atau ilmiah?
4. Berapa kemungkinan ini adalah konten AI-generated: 0 (manusia asli) sampai 1 (AI-generated)

Respons dalam JSON: {{"score": 0.75, "explanation": "Penjelasan dalam Bahasa Indonesia tentang mengapa konten ini terdeteksi sebagai AI-generated"}}"""

        completed = subprocess.run(
            ["ollama", "run", "gemma4", "--format", "json", "--nowordwrap", prompt],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )

        response = completed.stdout.strip()
        if "{" in response and "}" in response:
            json_text = response[response.find("{"): response.rfind("}") + 1]
            data = json.loads(json_text)
            score = data.get("score", 0.0)
            explanation = data.get("explanation", "Analisis Gemma tidak tersedia")
            return float(score), f"Gemma Analysis: {explanation}"

    except Exception as e:
        logger.warning(f"Gemma content analysis failed: {e}")

    return 0.0, "Gemma analysis unavailable"


def _analyze_motion_for_ai(frame_paths: List[str]) -> Tuple[float, str]:
    if len(frame_paths) < 2:
        return 0.0, "Insufficient frames for motion analysis."

    motion_magnitudes = []
    motion_stds = []
    prev_gray = None

    for frame_path in frame_paths:
        img = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        if prev_gray is not None and img.shape == prev_gray.shape:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray,
                img,
                None,
                0.5,
                3,
                15,
                3,
                5,
                1.2,
                0,
            )
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            motion_magnitudes.append(float(np.mean(mag)))
            motion_stds.append(float(np.std(mag)))

        prev_gray = img

    if not motion_magnitudes:
        return 0.0, "Motion analysis could not be performed on the extracted frames."

    avg_mag = float(np.mean(motion_magnitudes))
    avg_std = float(np.mean(motion_stds)) if motion_stds else 0.0

    if avg_mag < 0.18:
        motion_score = np.clip((0.18 - avg_mag) / 0.18 * 0.75 + min(avg_std, 0.25) * 0.25, 0.0, 1.0)
    elif avg_mag > 0.9:
        motion_score = np.clip((avg_mag - 0.9) / 1.5 * 0.55 + min(avg_std, 0.45) / 0.45 * 0.45, 0.0, 1.0)
    else:
        motion_score = np.clip(min(avg_std / 0.35, 1.0) * 0.55 + abs(avg_mag - 0.45) / 0.45 * 0.45, 0.0, 1.0)

    return motion_score, f"Motion signal: mean={avg_mag:.3f}, variability={avg_std:.3f}."


def _analyze_audio_for_ai(audio_path: str) -> Tuple[float, str]:
    try:
        with wave.open(audio_path, "rb") as wf:
            channels = wf.getnchannels()
            width = wf.getsampwidth()
            frames = wf.getnframes()
            audio_bytes = wf.readframes(frames)

        if frames == 0 or not audio_bytes:
            return 0.0, "Audio file is empty or unavailable."

        dtype = {1: np.int8, 2: np.int16, 4: np.int32}.get(width)
        if dtype is None:
            return 0.0, "Audio format not supported for artifact analysis."

        samples = np.frombuffer(audio_bytes, dtype=dtype).astype(np.float32)
        if channels > 1:
            samples = samples.reshape(-1, channels).mean(axis=1)

        max_val = float(np.iinfo(dtype).max)
        if max_val == 0:
            max_val = 1.0
        samples = samples / max_val
        samples -= np.mean(samples)

        if samples.size < 512:
            return 0.0, "Audio too short for meaningful analysis."

        window = samples[: min(len(samples), 16000 * 10)]
        zcr = float(np.mean(np.abs(np.diff(np.sign(window)))) / 2.0)

        spectrum = np.abs(np.fft.rfft(window))
        power = spectrum**2
        power /= np.sum(power) + 1e-10
        entropy = -np.sum(power * np.log(power + 1e-10)) / np.log(power.size)
        flatness = float(np.exp(np.mean(np.log(power + 1e-10))) / (np.mean(power) + 1e-10))

        flatness_score = np.clip((flatness - 0.25) / 0.5, 0.0, 1.0)
        entropy_score = np.clip((entropy - 0.60) / 0.25, 0.0, 1.0)
        zcr_score = np.clip(abs(zcr - 0.10) / 0.25, 0.0, 1.0)

        audio_score = float(np.clip(0.45 * flatness_score + 0.35 * entropy_score + 0.20 * zcr_score, 0.0, 1.0))
        return audio_score, f"Audio artifact score: flatness={flatness:.3f}, entropy={entropy:.3f}, zcr={zcr:.3f}."
    except Exception as e:
        logger.warning(f"Audio artifact analysis failed for {audio_path}: {e}")
        return 0.0, "Audio artifact analysis failed."


def detect_ai_generation_from_frames(
    frames_dir: str,
    audio_path: Optional[str] = None,
    *,
    url: str,
    max_frames: int = 12,
    video_title: Optional[str] = None,
    video_description: Optional[str] = None,
) -> Tuple[float, List[str]]:
    """
    Detect AI-generated/deepfake content from extracted frames.
    Uses pre-trained models, frequency analysis, motion, audio artifacts, and Gemma content analysis.

    Args:
        frames_dir: Directory containing extracted frame JPGs
        audio_path: Optional extracted audio file for artifact detection
        url: Source video URL (for contextual analysis)
        max_frames: Maximum number of frames to analyze (for performance)
        video_title: Video title for Gemma analysis
        video_description: Video description for Gemma analysis

    Returns:
        Tuple of (ai_score [0-1], explanation_drivers)
    """
    frame_paths = sorted(glob.glob(f"{frames_dir}/frame_*.jpg"))

    if not frame_paths:
        return 0.0, ["Tidak ada frame ditemukan untuk analisis deteksi AI."]

    if max_frames is not None:
        frame_paths = frame_paths[:max_frames]

    frame_scores: List[float] = []
    frame_details = {
        "frame_scores": [],
        "clip_scores": [],
        "model_scores": [],
        "frequency_scores": [],
        "texture_scores": [],
        "faces_per_frame": [],
    }

    for frame_path in frame_paths:
        try:
            ai_score, details = analyze_frame_for_ai(frame_path)
            frame_scores.append(ai_score)
            frame_details["frame_scores"].append(ai_score)
            frame_details["clip_scores"].append(details.get("clip_score", 0.0))
            frame_details["model_scores"].append(details.get("model_score", 0.0))
            frame_details["frequency_scores"].append(details.get("frequency_score", 0.0))
            frame_details["texture_scores"].append(details.get("texture_score", 0.0))
            frame_details["faces_per_frame"].append(details["faces_detected"])
        except Exception as e:
            logger.warning(f"Failed to analyze frame {frame_path}: {e}")
            continue

    if not frame_scores:
        return 0.0, ["Gagal menganalisis frame apa pun untuk deteksi AI."]

    temporal_score = aggregate_frame_scores(frame_scores)
    avg_clip_score = sum(frame_details["clip_scores"]) / len(frame_details["clip_scores"])
    avg_model_score = sum(frame_details["model_scores"]) / len(frame_details["model_scores"])
    avg_frequency_score = sum(frame_details["frequency_scores"]) / len(frame_details["frequency_scores"])
    avg_texture_score = sum(frame_details["texture_scores"]) / len(frame_details["texture_scores"])
    avg_faces = sum(frame_details["faces_per_frame"]) / len(frame_details["faces_per_frame"])

    motion_score, motion_note = _analyze_motion_for_ai(frame_paths)
    audio_score, audio_note = (0.0, "Audio tidak tersedia untuk analisis artefak.")
    if audio_path:
        audio_score, audio_note = _analyze_audio_for_ai(audio_path)

    # Gemma content analysis
    gemma_score, gemma_note = _analyze_content_with_gemma(url, video_title, video_description)

    # Multi-signal scoring: 25% CLIP, 20% model, 15% frequency, 10% texture, 10% temporal/motion/audio, 20% Gemma
    ai_score = float(
        0.25 * avg_clip_score
        + 0.20 * avg_model_score
        + 0.15 * avg_frequency_score
        + 0.10 * avg_texture_score
        + 0.05 * temporal_score
        + 0.02 * motion_score
        + 0.03 * audio_score
        + 0.20 * gemma_score
    )
    ai_score = max(0.0, min(1.0, ai_score))

    # Calibration: if score is between 0.2-0.8, bias downwards for natural content
    if 0.2 <= ai_score <= 0.8:
        ai_score = ai_score * 0.7  # Reduce by 30%

    drivers = [
        f"Analisis Semantik CLIP: {avg_clip_score * 100:.1f}% - Perbandingan gaya konten.",
        f"Deteksi Anomali Model: {avg_model_score * 100:.1f}% - Skor anomali visual.",
        f"Analisis Domain Frekuensi: {avg_frequency_score * 100:.1f}% - Deteksi artefak kompresi.",
        f"Anomali Tekstur: {avg_texture_score * 100:.1f}% - Analisis tepi dan blur.",
        f"Konsistensi Temporal: {temporal_score * 100:.1f}% di {len(frame_scores)} frame yang dianalisis.",
        f"Analisis Gerakan: {motion_score * 100:.1f}% - {motion_note}",
        f"Analisis Gemma Konten: {gemma_score * 100:.1f}% - {gemma_note}",
    ]

    if audio_path:
        drivers.append(f"Analisis Audio: {audio_score * 100:.1f}% - {audio_note}")
    else:
        drivers.append("Analisis audio tidak tersedia untuk video ini.")

    drivers.append(f"Deteksi Wajah: {avg_faces:.1f} wajah per frame rata-rata.")
    drivers.append(f"Skor Generasi AI Final: {ai_score * 100:.1f}% (analisis multi-sinyal dengan kalibrasi).")

    return ai_score, drivers

