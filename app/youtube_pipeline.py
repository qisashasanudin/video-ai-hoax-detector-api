import asyncio
import os
import pathlib
import subprocess
import time
from typing import Dict, Optional, Union

from imageio_ffmpeg import get_ffmpeg_exe
from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError

# User-agent rotation to avoid bot detection
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4.1 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Linux; Android 14) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Mobile Safari/537.36",
]

COOKIE_BROWSERS = ["edge", "chrome", "firefox"]

def _detect_browser_cookie_sources() -> list[str]:
    # Prefer Microsoft Edge when present on the system.
    browsers = []
    if os.path.exists("/Applications/Microsoft Edge.app") or os.path.exists(
        os.path.expanduser("~/Applications/Microsoft Edge.app")
    ):
        browsers.append("edge")
    if os.path.exists("/Applications/Google Chrome.app") or os.path.exists(
        os.path.expanduser("~/Applications/Google Chrome.app")
    ):
        browsers.append("chrome")
    if os.path.exists("/Applications/Firefox.app") or os.path.exists(
        os.path.expanduser("~/Applications/Firefox.app")
    ):
        browsers.append("firefox")
    return browsers or COOKIE_BROWSERS

class YouTubeExtractionError(RuntimeError):
    pass


def _ensure_dir(path: Union[str, os.PathLike[str]]) -> None:
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def _download_youtube(url: str, out_dir: str, max_download_seconds: int) -> str:
    _ensure_dir(out_dir)
    ffmpeg_exe = get_ffmpeg_exe()

    outtmpl = os.path.join(out_dir, "input.%(ext)s")
    cookies_from_browser = os.getenv("YTDLP_COOKIES_FROM_BROWSER", "").strip().lower()
    cookie_file = os.getenv("YTDLP_COOKIE_FILE", "").strip()

    # Retry with progressively more permissive format/client settings.
    attempts = [
        {
            "format": "bestvideo+bestaudio/best",
            "extractor_args": {"youtube": {"player_client": ["web", "android", "ios"]}},
        },
        {
            "format": "best",
            "extractor_args": {"youtube": {"player_client": ["android"]}},
        },
        {
            "format": "best",
        },
    ]

    browser_options = [cookies_from_browser] if cookies_from_browser else _detect_browser_cookie_sources()
    last_error: Optional[Exception] = None
    for attempt_idx, extra in enumerate(attempts):
        for ua_idx, user_agent in enumerate(USER_AGENTS):
            for browser in browser_options:
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
                    "allow_unplayable_formats": True,
                    "http_headers": {"User-Agent": user_agent},
                    "sleep_interval": 0.5,
                    "max_sleep_interval": 2,
                }

                if cookie_file and os.path.exists(cookie_file):
                    base_opts["cookiefile"] = cookie_file
                else:
                    base_opts["cookiesfrombrowser"] = (browser,)

                ydl_opts = {**base_opts, **extra}
                try:
                    with YoutubeDL(ydl_opts) as ydl:
                        ydl.extract_info(url, download=True)
                    last_error = None
                    return _find_downloaded_file(out_dir, url)
                except DownloadError as e:
                    last_error = e
                    if ua_idx < len(USER_AGENTS) - 1 or attempt_idx < len(attempts) - 1 or browser != browser_options[-1]:
                        time.sleep(0.5 + attempt_idx * 0.5)  # Backoff delay
                    continue

    if last_error is not None:
        raise YouTubeExtractionError(f"unable to download video data: {last_error}") from last_error


def _find_downloaded_file(out_dir: str, url: str) -> str:
    """Locate the downloaded file. We expect exactly 1 'input.*'."""
    candidates = sorted(pathlib.Path(out_dir).glob("input.*"))
    if not candidates:
        raise YouTubeExtractionError(f"Download succeeded but output file not found for url: {url}")
    # Prefer mp4 if present.
    mp4 = [c for c in candidates if c.suffix.lower() == ".mp4"]
    picked = (mp4[0] if mp4 else candidates[0])
    return str(picked)


def _get_youtube_metadata(url: str) -> Dict[str, str]:
    """
    Extract video metadata (title and thumbnail) from YouTube URL.
    Returns dict with 'title' and 'thumbnail_url' keys.
    Gracefully degrades if extraction fails using user-agent rotation + browser cookies.
    """
    import logging
    
    cookies_from_browser = os.getenv("YTDLP_COOKIES_FROM_BROWSER", "").strip().lower()
    cookie_file = os.getenv("YTDLP_COOKIE_FILE", "").strip()
    browser_options = [cookies_from_browser] if cookies_from_browser else _detect_browser_cookie_sources()

    for ua_idx, user_agent in enumerate(USER_AGENTS):
        for browser in browser_options:
            try:
                base_opts = {
                    "quiet": True,
                    "no_warnings": True,
                    "retries": 1,
                    "socket_timeout": 10,
                    "http_headers": {"User-Agent": user_agent},
                }
                if cookie_file and os.path.exists(cookie_file):
                    base_opts["cookiefile"] = cookie_file
                else:
                    base_opts["cookiesfrombrowser"] = (browser,)
                    
                with YoutubeDL(base_opts) as ydl:
                    info = ydl.extract_info(url, download=False)
                    
                title = info.get("title", "Video")
                thumbnail = info.get("thumbnail", "")
                description = info.get("description", "")
                
                return {
                    "title": str(title) if title else "Video",
                    "thumbnail_url": str(thumbnail) if thumbnail else "",
                    "description": str(description) if description else "",
                }
            except Exception as e:
                if ua_idx < len(USER_AGENTS) - 1 or browser != browser_options[-1]:
                    time.sleep(0.3)
                    continue
                logging.warning(f"Failed to extract metadata for {url}: {e}")
    
    # Graceful fallback when all attempts fail
    return {
        "title": "Video",
        "thumbnail_url": "",
        "description": "",
    }


def _extract_frames_and_audio(
    input_video_path: str,
    frames_dir: str,
    audio_wav_path: str,
    max_frames: int,
) -> int:
    _ensure_dir(frames_dir)
    ffmpeg_exe = get_ffmpeg_exe()

    # Visual sampling: take frames at a low fixed rate and cap by -frames:v.
    # This avoids needing duration probing for MVP.
    frame_out = os.path.join(frames_dir, "frame_%05d.jpg")
    # Capture more temporal detail for AI detection while keeping cost bounded by max_frames.
    vf = "fps=1,scale=320:-1"

    # Extract frames.
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

    # Extract 16kHz mono WAV for ASR.
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

    frame_files = list(pathlib.Path(frames_dir).glob("frame_*.jpg"))
    return len(frame_files)


async def extract_youtube_media(
    *,
    url: str,
    job_id: str,
    base_data_dir: str,
    max_download_seconds: int = 60,
    max_frames: int = 12,
) -> Dict[str, Union[Optional[str], int]]:
    """
    Downloads YouTube video (best effort) and extracts:
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
        _get_youtube_metadata,
        url,
    )

    input_video_path = await loop.run_in_executor(
        None,
        _download_youtube,
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

    return {
        "job_dir": job_dir,
        "video_path": input_video_path,
        "audio_path": audio_wav_path,
        "frames_count": frames_count,
        "video_title": metadata.get("title", ""),
        "video_thumbnail_url": metadata.get("thumbnail_url", ""),
        "video_description": metadata.get("description", ""),
    }

