import asyncio
import os
import pathlib
import subprocess
import time
from typing import Dict, Optional, Union

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

class YouTubeExtractionError(RuntimeError):
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


def _detect_browser_cookie_sources() -> list[str]:
    sources = []
    if os.path.exists("/Applications/Microsoft Edge.app") or os.path.exists(
        os.path.expanduser("~/Applications/Microsoft Edge.app")
    ):
        sources.append("edge")
    if os.path.exists("/Applications/Google Chrome.app") or os.path.exists(
        os.path.expanduser("~/Applications/Google Chrome.app")
    ):
        sources.append("chrome")
    if os.path.exists("/Applications/Firefox.app") or os.path.exists(
        os.path.expanduser("~/Applications/Firefox.app")
    ):
        sources.append("firefox")
    return sources or ["edge", "chrome", "firefox"]


def _download_youtube(url: str, out_dir: str, max_download_seconds: int) -> str:
    _ensure_dir(out_dir)
    ffmpeg_exe = get_ffmpeg_exe()

    outtmpl = os.path.join(out_dir, "input.%(ext)s")
    cookies_from_browser = os.getenv("YTDLP_COOKIES_FROM_BROWSER", "").strip().lower()
    cookie_file = os.getenv("YTDLP_COOKIE_FILE", "").strip()
    browser_sources = [cookies_from_browser] if cookies_from_browser else _detect_browser_cookie_sources()

    attempts = [
        {
            "format": "best[ext=mp4][protocol=https]/best",
            "extractor_args": {"youtube": {"player_client": ["android"]}},
            "user_agent": ANDROID_USER_AGENT,
            "use_cookies": False,
        },
        {
            "format": "best[ext=mp4][protocol=https]/best",
            "extractor_args": {"youtube": {"player_client": ["web"]}},
            "user_agent": WEB_USER_AGENT,
            "use_cookies": True,
        },
        {
            "format": "best",
            "user_agent": WEB_USER_AGENT,
            "use_cookies": True,
        },
    ]

    last_error: Optional[Exception] = None
    last_attempt: Optional[Dict[str, Union[str, Dict[str, list[str]]]]] = None

    for attempt_idx, attempt in enumerate(attempts):
        use_cookies = attempt.get("use_cookies", False)
        for browser in browser_sources if use_cookies else [None]:
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

            if use_cookies:
                if cookie_file and os.path.exists(cookie_file):
                    base_opts["cookiefile"] = cookie_file
                elif browser:
                    base_opts["cookiesfrombrowser"] = (browser,)

            ydl_opts = {**base_opts, **attempt}
            ydl_opts.pop("user_agent", None)
            ydl_opts.pop("use_cookies", None)
            last_attempt = {"attempt": attempt, "browser": browser}
            try:
                with YoutubeDL(ydl_opts) as ydl:
                    ydl.extract_info(url, download=True)
                return _find_downloaded_file(out_dir, url)
            except DownloadError as e:
                last_error = e
                if attempt_idx < len(attempts) - 1 or (use_cookies and browser != browser_sources[-1]):
                    time.sleep(0.5 + attempt_idx * 0.5)
                continue

    raise YouTubeExtractionError(
        f"Unable to download YouTube media from {url} after fallback attempts. "
        f"Last attempt={last_attempt}. Last error: {last_error}"
    ) from last_error


def _find_downloaded_file(out_dir: str, url: str) -> str:
    candidates = sorted(pathlib.Path(out_dir).glob("input.*"))
    if not candidates:
        raise YouTubeExtractionError(f"Download succeeded but output file not found for url: {url}")
    mp4 = [c for c in candidates if c.suffix.lower() == ".mp4"]
    return str(mp4[0] if mp4 else candidates[0])


def _get_youtube_metadata(url: str) -> Dict[str, str]:
    import logging

    cookies_from_browser = os.getenv("YTDLP_COOKIES_FROM_BROWSER", "").strip().lower()
    cookie_file = os.getenv("YTDLP_COOKIE_FILE", "").strip()
    browser_sources = [cookies_from_browser] if cookies_from_browser else _detect_browser_cookie_sources()

    for browser in browser_sources:
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
                if cookie_file and os.path.exists(cookie_file):
                    base_opts["cookiefile"] = cookie_file
                elif browser:
                    base_opts["cookiesfrombrowser"] = (browser,)

                with YoutubeDL(base_opts) as ydl:
                    info = ydl.extract_info(url, download=False)

                return {
                    "title": str(info.get("title", "Video")) or "Video",
                    "thumbnail_url": str(info.get("thumbnail", "")) or "",
                    "description": str(info.get("description", "")) or "",
                }
            except Exception as e:
                logging.debug(f"Metadata extraction failed with {extractor}: {e}")
                continue

    return {"title": "Video", "thumbnail_url": "", "description": ""}


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

