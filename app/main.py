import asyncio
import os
import time
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .store import (
    create_job,
    init_db,
    set_job_failed,
    set_job_progress,
    set_job_succeeded,
    set_job_status,
    get_job as get_job_record,
)

from .youtube_pipeline import extract_youtube_media
from .ai_detection.frame_detector import detect_ai_generation_from_frames
from .nlp.asr import transcribe_audio
from .nlp.claim_extractor import extract_claims_from_transcript
from .nlp.misinfo_scoring import score_misinformation


app = FastAPI(title="Video Misinfo Platform API", version="0.1.0")

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


JobStatus = Literal["queued", "running", "succeeded", "failed"]


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


class OverallAssessment(BaseModel):
    recommendation: str
    key_findings: list[str]


class ComprehensiveAnalysis(BaseModel):
    ai_detection: AIDetectionResult
    misinformation_analysis: MisinformationAnalysis
    overall_assessment: OverallAssessment


class AnalysisResult(BaseModel):
    comprehensive_analysis: Optional[ComprehensiveAnalysis] = None
    analysis_error: Optional[str] = None
    video_title: Optional[str] = None
    video_description: Optional[str] = None
    video_thumbnail_url: Optional[str] = None


class JobResultResponse(BaseModel):
    status: JobStatus
    result: Optional[AnalysisResult] = None
    error: Optional[str] = None
    progress: Optional[str] = None


class AnalyzeResponse(BaseModel):
    job_id: str


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


async def _run_mock_job(job_id: str, url: str) -> None:
    # Keep the same perceived latency pattern as the UI mock.
    await asyncio.sleep(0.2)
    async with db_lock:
        # If job was deleted/overwritten, just no-op.
        record = get_job_record(job_id)
        if record is None:
            return
        set_job_status(job_id, "running")
        set_job_progress(job_id, "Initializing analysis...")

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
    try:
        set_job_progress(job_id, "Extracting video and audio...")
        extraction = await extract_youtube_media(
            url=url,
            job_id=job_id,
            base_data_dir=BASE_DATA_DIR,
            max_download_seconds=20,
            max_frames=24,
        )
        frames_count = extraction.get("frames_count") if isinstance(extraction.get("frames_count"), int) else None
        job_dir = extraction.get("job_dir")
        audio_path = extraction.get("audio_path") if isinstance(extraction.get("audio_path"), str) else None
        video_title = extraction.get("video_title") if isinstance(extraction.get("video_title"), str) else None
        video_description = extraction.get("video_description") if isinstance(extraction.get("video_description"), str) else None
        video_thumbnail_url = extraction.get("video_thumbnail_url") if isinstance(extraction.get("video_thumbnail_url"), str) else None
        if isinstance(job_dir, str) and frames_count is not None:
            frames_dir = os.path.join(job_dir, "frames")
            set_job_progress(job_id, "Menganalisis video untuk deteksi AI...")
            ai_score_override, ai_drivers_override = detect_ai_generation_from_frames(
                frames_dir,
                audio_path=audio_path,
                url=url,
                max_frames=24,
                video_title=video_title,
                video_description=video_description,
            )
    except Exception as e:
        extraction_note = str(e)

    if audio_path and os.path.exists(audio_path):
        set_job_progress(job_id, "Transcribing audio...")
        asr = transcribe_audio(audio_path)
        transcript = asr.get("transcript") or ""
        asr_note = asr.get("note")
    else:
        asr_note = "MVP: audio tidak tersedia untuk ASR."

    if asr_note:
        extraction_note = f"{extraction_note}; {asr_note}" if extraction_note else asr_note

    # Get CLIP analysis results
    clip_results = None
    if isinstance(job_dir, str) and frames_count is not None:
        frames_dir = os.path.join(job_dir, "frames")
        set_job_progress(job_id, "Menganalisis tampilan visual untuk deteksi AI...")
        ai_score_override, ai_drivers_override = detect_ai_generation_from_frames(
            frames_dir,
            audio_path=audio_path,
            url=url,
            max_frames=24,
            video_title=video_title,
            video_description=video_description,
        )
        clip_results = {
            'ai_score': ai_score_override,
            'ai_drivers': ai_drivers_override
        }

    # Use Gemma orchestrator for comprehensive analysis
    set_job_progress(job_id, "Mencari bukti online untuk analisis...")
    from .nlp.misinfo_scoring import orchestrate_comprehensive_analysis
    set_job_progress(job_id, "Menganalisis informasi dan hoaks...")
    comprehensive_result, analysis_error = orchestrate_comprehensive_analysis(
        url=url,
        video_title=video_title,
        video_description=video_description,
        transcript=transcript,
        frames_dir=os.path.join(job_dir, "frames") if isinstance(job_dir, str) else None,
        audio_path=audio_path,
        clip_results=clip_results,
    )

    # Build result with comprehensive analysis
    if comprehensive_result:
        comprehensive_analysis = ComprehensiveAnalysis(
            ai_detection=AIDetectionResult(
                score=comprehensive_result['ai_detection']['score'],
                confidence=comprehensive_result['ai_detection']['confidence'],
                explanation=comprehensive_result['ai_detection']['explanation']
            ),
            misinformation_analysis=MisinformationAnalysis(
                score=comprehensive_result['misinformation_analysis']['score'],
                risk_level=comprehensive_result['misinformation_analysis']['risk_level'],
                explanation=comprehensive_result['misinformation_analysis']['explanation']
            ),
            overall_assessment=OverallAssessment(
                recommendation=comprehensive_result['overall_assessment']['recommendation'],
                key_findings=comprehensive_result['overall_assessment']['key_findings']
            )
        )
        result = AnalysisResult(
            comprehensive_analysis=comprehensive_analysis,
            analysis_error=None,
            video_title=video_title,
            video_description=video_description,
            video_thumbnail_url=video_thumbnail_url,
        )
    else:
        result = AnalysisResult(
            comprehensive_analysis=None,
            analysis_error=analysis_error or "Comprehensive analysis failed",
            video_title=video_title,
            video_description=video_description,
            video_thumbnail_url=video_thumbnail_url,
        )

    async with db_lock:
        record = get_job_record(job_id)
        if record is None:
            return
        set_job_progress(job_id, "Finalizing results...")
        set_job_succeeded(job_id, result.model_dump_json())


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    # Basic validation (MVP). Real pipeline will do deeper URL/media checks.
    if not req.url.strip():
        raise HTTPException(status_code=400, detail="url is required")

    job_id = f"job_{int(time.time())}_{abs(_hash_string(req.url)) % 10_000}"

    async with db_lock:
        create_job(job_id, url=req.url, source=req.source)

    asyncio.create_task(_run_mock_job(job_id=job_id, url=req.url))
    return AnalyzeResponse(job_id=job_id)


@app.get("/jobs/{job_id}", response_model=JobResultResponse)
async def get_job(job_id: str) -> JobResultResponse:
    record = get_job_record(job_id)
    if record is None:
        return JobResultResponse(status="failed", result=None, error="Job not found", progress=None)

    result_json = record.get("result_json")
    result_obj = AnalysisResult.model_validate_json(result_json) if result_json else None
    return JobResultResponse(status=record["status"], result=result_obj, error=record.get("error"), progress=record.get("progress"))

