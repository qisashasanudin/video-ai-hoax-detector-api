import asyncio
import os
import pathlib
import subprocess
import time
from typing import Dict, Optional, Union

import re
from imageio_ffmpeg import get_ffmpeg_exe
from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError

ANDROID_USER_AGENT = (
    "Mozilla/5.0 (Linux; Android 14) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Mobile Safari/537.36"
)
WEB_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

class MediaExtractionError(RuntimeError):
    pass


def _ensure_dir(path: Union[str, os.PathLike[str]]) -> None:
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def _build_ytdlp_http_headers(url: str, user_agent: str) -> Dict[str, str]:
    return {
        "User-Agent": user_agent,
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": url,
        "Origin": "https://www.youtube.com",
    }





def _download_media(url: str, out_dir: str, max_download_seconds: int) -> str:
    _ensure_dir(out_dir)
    ffmpeg_exe = get_ffmpeg_exe()

    outtmpl = os.path.join(out_dir, "input.%(ext)s")

    attempts = [
        {
            "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "extractor_args": {"youtube": {"player_client": ["android"]}},
            "user_agent": ANDROID_USER_AGENT,
        },
        {
            "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "extractor_args": {"youtube": {"player_client": ["web"]}},
            "user_agent": WEB_USER_AGENT,
        },
        {
            "format": "best",
            "user_agent": WEB_USER_AGENT,
        },
    ]

    last_error: Optional[Exception] = None
    last_attempt: Optional[Dict[str, Union[str, Dict[str, list[str]]]]] = None

    for attempt_idx, attempt in enumerate(attempts):
        base_opts = {
            "outtmpl": outtmpl,
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "retries": 2,
            "fragment_retries": 2,
            "socket_timeout": 15,
            "timeout": max_download_seconds,
            "ffmpeg_location": ffmpeg_exe,
            "merge_output_format": "mp4",
            "http_headers": _build_ytdlp_http_headers(url, attempt["user_agent"]),
            "sleep_interval": 0.5,
            "max_sleep_interval": 2,
        }

        ydl_opts = {**base_opts, **attempt}
        ydl_opts.pop("user_agent", None)
        last_attempt = {"attempt": attempt}
        try:
            with YoutubeDL(ydl_opts) as ydl:
                ydl.extract_info(url, download=True)
            return _find_downloaded_file(out_dir, url)
        except DownloadError as e:
            last_error = e
            if attempt_idx < len(attempts) - 1:
                time.sleep(0.5 + attempt_idx * 0.5)
            continue

    raise MediaExtractionError(
        f"Unable to download media from {url} after fallback attempts. "
        f"Last attempt={last_attempt}. Last error: {last_error}"
    ) from last_error


def _find_downloaded_file(out_dir: str, url: str) -> str:
    candidates = sorted(pathlib.Path(out_dir).glob("input.*"))
    if not candidates:
        raise MediaExtractionError(f"Download succeeded but output file not found for url: {url}")
    mp4 = [c for c in candidates if c.suffix.lower() == ".mp4"]
    return str(mp4[0] if mp4 else candidates[0])


def _get_media_metadata(url: str) -> Dict[str, str]:
    import logging

    for extractor in (
        {"player_client": ["android"]},
        {"player_client": ["web"]},
    ):
        try:
            extract_args = {"youtube": extractor}
            base_opts = {
                "quiet": True,
                "no_warnings": True,
                "retries": 1,
                "socket_timeout": 10,
                "http_headers": _build_ytdlp_http_headers(url, ANDROID_USER_AGENT if extractor["player_client"] == ["android"] else WEB_USER_AGENT),
                "extractor_args": extract_args,
            }

            with YoutubeDL(base_opts) as ydl:
                info = ydl.extract_info(url, download=False)

            return {
                "title": str(info.get("title", "Video")) or "Video",
                "thumbnail_url": str(info.get("thumbnail", "")) or "",
                "description": str(info.get("description", "")) or "",
                "uploader": str(info.get("uploader", "")).strip(),
            }
        except Exception as e:
            logging.debug(f"Metadata extraction failed with {extractor}: {e}")
            continue

    return {"title": "Video", "thumbnail_url": "", "description": ""}


def _get_video_duration(input_video_path: str) -> float:
    ffmpeg_exe = get_ffmpeg_exe()
    try:
        completed = subprocess.run(
            [ffmpeg_exe, "-i", input_video_path],
            capture_output=True,
            text=True,
            check=False,
        )
        stderr = completed.stderr or ""
        duration_match = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", stderr)
        if duration_match:
            hours = int(duration_match.group(1))
            minutes = int(duration_match.group(2))
            seconds = float(duration_match.group(3))
            return hours * 3600.0 + minutes * 60.0 + seconds
    except Exception:
        pass
    return 0.0


def _extract_audio_samples(
    input_video_path: str,
    audio_wav_path: str,
    duration_seconds: float,
    ffmpeg_exe: str,
    segment_length: int = 30,
    max_segments: int = 3,
) -> None:
    if duration_seconds <= max_segments * segment_length or duration_seconds <= 120:
        cmd_audio = [
            ffmpeg_exe,
            "-y",
            "-i",
            input_video_path,
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "wav",
            audio_wav_path,
        ]
        subprocess.run(cmd_audio, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return

    positions = [0.0]
    mid_start = max((duration_seconds - segment_length) / 2.0, 0.0)
    end_start = max(duration_seconds - segment_length, 0.0)
    if mid_start > 0 and mid_start not in positions:
        positions.append(mid_start)
    if end_start > 0 and end_start not in positions:
        positions.append(end_start)

    segment_paths = []
    for idx, position in enumerate(positions):
        temp_path = os.path.join(os.path.dirname(audio_wav_path), f"audio_segment_{idx}.wav")
        cmd_segment = [
            ffmpeg_exe,
            "-y",
            "-ss",
            str(position),
            "-i",
            input_video_path,
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-t",
            str(segment_length),
            "-f",
            "wav",
            temp_path,
        ]
        subprocess.run(cmd_segment, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        segment_paths.append(temp_path)

    concat_list = os.path.join(os.path.dirname(audio_wav_path), "audio_segments.txt")
    with open(concat_list, "w", encoding="utf-8") as handle:
        for segment_path in segment_paths:
            handle.write(f"file '{segment_path}'\n")

    subprocess.run(
        [
            ffmpeg_exe,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            concat_list,
            "-c",
            "copy",
            audio_wav_path,
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    for temp_path in segment_paths + [concat_list]:
        try:
            os.remove(temp_path)
        except OSError:
            pass


def _extract_frames_and_audio(
    input_video_path: str,
    frames_dir: str,
    audio_wav_path: str,
    max_frames: int,
) -> int:
    _ensure_dir(frames_dir)
    ffmpeg_exe = get_ffmpeg_exe()

    duration = _get_video_duration(input_video_path)
    frame_rate = float(max_frames) / duration if duration > 0 else 1.0
    frame_rate = max(frame_rate, 0.1)

    frame_out = os.path.join(frames_dir, "frame_%05d.jpg")
    vf = f"fps={frame_rate},scale=320:-1"

    cmd_frames = [
        ffmpeg_exe,
        "-y",
        "-i",
        input_video_path,
        "-an",
        "-vf",
        vf,
        "-frames:v",
        str(max_frames),
        frame_out,
    ]
    subprocess.run(cmd_frames, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    _extract_audio_samples(input_video_path, audio_wav_path, duration, ffmpeg_exe)

    frame_files = list(pathlib.Path(frames_dir).glob("frame_*.jpg"))
    return len(frame_files)


async def extract_media(
    *,
    url: str,
    job_id: str,
    base_data_dir: str,
    max_download_seconds: int = 60,
    max_frames: int = 12,
) -> Dict[str, Union[Optional[str], int]]:
    """
    Downloads media (best effort) and extracts:
    - sampled frames (jpg)
    - mono 16kHz WAV audio
    - video metadata (title, thumbnail)
    """
    job_dir = os.path.join(base_data_dir, "jobs", job_id)
    frames_dir = os.path.join(job_dir, "frames")
    audio_wav_path = os.path.join(job_dir, "audio.wav")
    _ensure_dir(job_dir)

    loop = asyncio.get_running_loop()

    # Extract metadata first
    metadata = await loop.run_in_executor(
        None,
        _get_media_metadata,
        url,
    )

    input_video_path = await loop.run_in_executor(
        None,
        _download_media,
        url,
        job_dir,
        max_download_seconds,
    )

    frames_count = await loop.run_in_executor(
        None,
        _extract_frames_and_audio,
        input_video_path,
        frames_dir,
        audio_wav_path,
        max_frames,
    )

    try:
        os.remove(input_video_path)
    except OSError:
        pass

    return {
        "job_dir": job_dir,
        "video_path": None,
        "audio_path": audio_wav_path,
        "frames_count": frames_count,
        "video_title": metadata.get("title", ""),
        "video_thumbnail_url": metadata.get("thumbnail_url", ""),
        "video_description": metadata.get("description", ""),
        "video_channel": metadata.get("uploader", ""),
    }

