import asyncio
import os
import time
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .store import (
    create_job,
    init_db,
    set_job_failed,
    set_job_progress,
    set_job_result,
    set_job_succeeded,
    set_job_status,
    get_job as get_job_record,
)

from .youtube_pipeline import extract_youtube_media
from .ai_detection.frame_detector import detect_ai_generation_from_frames
from .nlp.asr import transcribe_audio
from .nlp.claim_extractor import extract_claims_from_transcript
from .nlp.misinfo_scoring import score_misinformation


app = FastAPI(title="AI Hoax Video Platform API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


JobStatus = Literal["queued", "running", "succeeded", "failed", "extracted"]


class AnalyzeRequest(BaseModel):
    url: str = Field(..., min_length=1)
    source: Literal["youtube"] = "youtube"


class ClaimVerdict(str):
    # Pydantic-friendly enum-like alias (we validate using literals below).
    pass


class AIDetectionResult(BaseModel):
    score: float  # 0..1
    confidence: str  # "TINGGI/SEDANG/RENDAH"
    explanation: str


class MisinformationAnalysis(BaseModel):
    score: float  # 0..1
    risk_level: str  # "TINGGI/SEDANG/RENDAH/TIDAK ADA"
    explanation: str


class HoaxAnalysis(BaseModel):
    score: float  # 0..1
    risk_level: str  # "TINGGI/SEDANG/RENDAH/TIDAK ADA"
    explanation: str


class OverallAssessment(BaseModel):
    recommendation: str
    key_findings: list[str]


class ComprehensiveAnalysis(BaseModel):
    ai_detection: AIDetectionResult
    hoax_analysis: HoaxAnalysis
    misinformation_analysis: MisinformationAnalysis
    overall_assessment: OverallAssessment


class AnalysisResult(BaseModel):
    comprehensive_analysis: Optional[ComprehensiveAnalysis] = None
    analysis_error: Optional[str] = None
    video_title: Optional[str] = None
    video_description: Optional[str] = None
    video_thumbnail_url: Optional[str] = None
    claims: Optional[list[str]] = None


class JobResultResponse(BaseModel):
    status: JobStatus
    result: Optional[AnalysisResult] = None
    error: Optional[str] = None
    progress: Optional[str] = None


class AnalyzeResponse(BaseModel):
    job_id: str


class ExtractResponse(BaseModel):
    job_id: str
    video_title: Optional[str] = None
    video_description: Optional[str] = None
    video_thumbnail_url: Optional[str] = None


class StartAnalysisRequest(BaseModel):
    job_id: str


class StartAnalysisResponse(BaseModel):
    job_id: str
    status: str = "analysis_started"


db_lock = asyncio.Lock()

BASE_DATA_DIR = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "data")


@app.on_event("startup")
async def _startup() -> None:
    init_db()


def _hash_string(s: str) -> int:
    # Stable, lightweight hash (non-cryptographic).
    h = 2166136261
    for ch in s:
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return abs(h)


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


async def _extract_video(job_id: str, url: str) -> dict:
    """
    Synchronously extract video metadata (download video, extract frames, get metadata).
    This runs without creating a background task.
    Returns dict with video_title, video_description, video_thumbnail_url, job_dir, frames_count, audio_path.
    """
    try:
        set_job_progress(job_id, "Menganalisis video...")
        extraction = await extract_youtube_media(
            url=url,
            job_id=job_id,
            base_data_dir=BASE_DATA_DIR,
            max_download_seconds=20,
            max_frames=24,
        )
        return extraction
    except Exception as e:
        raise Exception(f"Extraction failed: {str(e)}")


async def _run_analysis_job(job_id: str) -> None:
    """
    Async background job for analysis only. Assumes video metadata is already extracted.
    """
    await asyncio.sleep(0.2)
    async with db_lock:
        record = get_job_record(job_id)
        if record is None:
            return
        set_job_status(job_id, "running")
        set_job_progress(job_id, "Starting comprehensive analysis...")

    async with db_lock:
        record = get_job_record(job_id)
        if record is None:
            return

    extraction_note: Optional[str] = None
    frames_count: Optional[int] = None
    job_dir: Optional[str] = None
    ai_score_override: Optional[float] = None
    ai_drivers_override: Optional[list[str]] = None
    misinfo_score_override: Optional[float] = None
    claims_override: Optional[list[AnalysisClaim]] = None
    transcript: str = ""
    asr_note: Optional[str] = None
    audio_path: Optional[str] = None
    video_title: Optional[str] = None
    video_description: Optional[str] = None
    video_thumbnail_url: Optional[str] = None

    # Get existing result to extract metadata that was stored during extraction
    result_json = record.get("result_json")
    if result_json:
        try:
            existing_result = AnalysisResult.model_validate_json(result_json)
            video_title = existing_result.video_title
            video_description = existing_result.video_description
            video_thumbnail_url = existing_result.video_thumbnail_url
        except:
            pass

    # Get job_dir and audio_path from the job directory structure
    url = record.get("url")
    if not url:
        set_job_failed(job_id, "No URL stored in job record")
        return

    # Reconstruct paths - we need to get these from somewhere
    # For now, we'll store them in the result during extraction
    job_dir_match = None
    audio_path = None
    frames_count = None

    try:
        # Scan the data directory to find the job directory
        jobs_dir = os.path.join(BASE_DATA_DIR, "jobs")
        if os.path.exists(jobs_dir):
            for entry in os.listdir(jobs_dir):
                if entry.startswith(f"job_{job_id.split('_')[1]}"):
                    job_dir_match = os.path.join(jobs_dir, entry)
                    audio_path = os.path.join(job_dir_match, "audio.wav")
                    frames_dir = os.path.join(job_dir_match, "frames")
                    if os.path.exists(frames_dir):
                        frames_count = len([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])
                    break
    except:
        pass

    if audio_path and os.path.exists(audio_path):
        set_job_progress(job_id, "Transcribing audio...")
        asr = transcribe_audio(audio_path)
        transcript = asr.get("transcript") or ""
        asr_note = asr.get("note")
    else:
        asr_note = "MVP: audio tidak tersedia untuk ASR."

    # Get CLIP analysis results
    clip_results = None
    if job_dir_match and frames_count:
        frames_dir = os.path.join(job_dir_match, "frames")
        set_job_progress(job_id, "Menganalisis tampilan visual untuk deteksi AI...")
        ai_score_override, ai_drivers_override = detect_ai_generation_from_frames(
            frames_dir,
            audio_path=audio_path,
            url=url,
            max_frames=24,
            video_title=video_title,
            video_description=video_description,
        )
        clip_results = {"ai_score": ai_score_override, "ai_drivers": ai_drivers_override}

    # Use Gemma orchestrator for comprehensive analysis
    from .nlp.misinfo_scoring import orchestrate_comprehensive_analysis

    # Create a progress callback that updates the job directly
    def update_analysis_progress(message: str) -> None:
        set_job_progress(job_id, message)

    comprehensive_result, analysis_error = orchestrate_comprehensive_analysis(
        url=url,
        video_title=video_title,
        video_description=video_description,
        transcript=transcript,
        frames_dir=os.path.join(job_dir_match, "frames") if job_dir_match else None,
        audio_path=audio_path,
        clip_results=clip_results,
        progress_callback=update_analysis_progress,
    )

    # Build result with comprehensive analysis
    if comprehensive_result:
        comprehensive_analysis = ComprehensiveAnalysis(
            ai_detection=AIDetectionResult(
                score=comprehensive_result["ai_detection"]["score"],
                confidence=comprehensive_result["ai_detection"]["confidence"],
                explanation=comprehensive_result["ai_detection"]["explanation"],
            ),
            hoax_analysis=HoaxAnalysis(
                score=comprehensive_result["hoax_analysis"]["score"],
                risk_level=comprehensive_result["hoax_analysis"]["risk_level"],
                explanation=comprehensive_result["hoax_analysis"]["explanation"],
            ),
            misinformation_analysis=MisinformationAnalysis(
                score=comprehensive_result["misinformation_analysis"]["score"],
                risk_level=comprehensive_result["misinformation_analysis"]["risk_level"],
                explanation=comprehensive_result["misinformation_analysis"]["explanation"],
            ),
            overall_assessment=OverallAssessment(
                recommendation=comprehensive_result["overall_assessment"]["recommendation"],
                key_findings=comprehensive_result["overall_assessment"]["key_findings"],
            ),
        )
        result = AnalysisResult(
            comprehensive_analysis=comprehensive_analysis,
            analysis_error=None,
            video_title=video_title,
            video_description=video_description,
            video_thumbnail_url=video_thumbnail_url,
            claims=comprehensive_result.get("claims", []),
        )
    else:
        result = AnalysisResult(
            comprehensive_analysis=None,
            analysis_error=analysis_error or "Comprehensive analysis failed",
            video_title=video_title,
            video_description=video_description,
            video_thumbnail_url=video_thumbnail_url,
            claims=[],
        )

    async with db_lock:
        record = get_job_record(job_id)
        if record is None:
            return
        set_job_progress(job_id, "Analisis selesai")
        set_job_succeeded(job_id, result.model_dump_json())




@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/extract", response_model=ExtractResponse)
async def extract(req: AnalyzeRequest) -> ExtractResponse:
    """Synchronously extract video metadata. Returns metadata immediately."""
    if not req.url.strip():
        raise HTTPException(status_code=400, detail="url is required")

    job_id = f"job_{int(time.time())}_{abs(_hash_string(req.url)) % 10_000}"

    async with db_lock:
        create_job(job_id, url=req.url, source=req.source)

    try:
        extraction = await _extract_video(job_id, req.url)
        video_title = extraction.get("video_title")
        video_description = extraction.get("video_description")
        video_thumbnail_url = extraction.get("video_thumbnail_url")

        # Store partial result with metadata
        partial_result = AnalysisResult(
            comprehensive_analysis=None,
            analysis_error=None,
            video_title=video_title,
            video_description=video_description,
            video_thumbnail_url=video_thumbnail_url,
        )
        async with db_lock:
            set_job_result(job_id, partial_result.model_dump_json())
            set_job_status(job_id, "extracted")

        return ExtractResponse(
            job_id=job_id,
            video_title=video_title,
            video_description=video_description,
            video_thumbnail_url=video_thumbnail_url,
        )
    except Exception as e:
        async with db_lock:
            set_job_failed(job_id, str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze", response_model=StartAnalysisResponse)
async def analyze(req: StartAnalysisRequest) -> StartAnalysisResponse:
    """Start analysis for an extracted job. Returns immediately, analysis runs in background."""
    job_id = req.job_id
    record = get_job_record(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Job not found")

    async with db_lock:
        current_status = record.get("status")
        if current_status not in ["extracted", "queued"]:
            raise HTTPException(status_code=400, detail=f"Cannot analyze job in status {current_status}")

    asyncio.create_task(_run_analysis_job(job_id=job_id))
    return StartAnalysisResponse(job_id=job_id, status="analysis_started")


@app.get("/jobs/{job_id}", response_model=JobResultResponse)
async def get_job(job_id: str) -> JobResultResponse:
    record = get_job_record(job_id)
    if record is None:
        return JobResultResponse(status="failed", result=None, error="Job not found", progress=None)

    status = record["status"]
    result_json = record.get("result_json")
    result_obj = AnalysisResult.model_validate_json(result_json) if result_json else None
    return JobResultResponse(status=status, result=result_obj, error=record.get("error"), progress=record.get("progress"))

