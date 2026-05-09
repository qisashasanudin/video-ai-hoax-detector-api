from typing import Dict, Optional


def transcribe_audio(audio_wav_path: str) -> Dict[str, Optional[str]]:
    """
    ASR (Speech-to-Text) MVP interface.

    This repo environment does not ship with heavy ASR dependencies (e.g., torch/whisper),
    so for now we provide a graceful fallback that returns an empty transcript.
    """
    # Future: integrate Whisper / faster-whisper here.
    return {
        "transcript": "",
        "note": "ASR MVP fallback: transcript tidak tersedia (model ASR belum diintegrasikan).",
    }

