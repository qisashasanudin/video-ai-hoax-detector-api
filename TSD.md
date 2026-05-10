# TSD: AI Hoax Video Platform API

## System summary

The API backend accepts YouTube URLs, queues asynchronous video analysis jobs, and returns structured results for AI generation detection and misinformation risk.

### Key outputs

- `ai_score`: normalized 0.0–1.0 likelihood of AI-generated or manipulated visual content.
- `hoax_analysis`: classification and score for content that is intentionally fabricated.
- `misinformation_analysis`: score and reasoning for misleading or inaccurate information.
- `overall_assessment`: recommendation and key findings based on evidence.

## Goals

- Provide a stable API for the frontend to submit video analysis requests.
- Persist job state and analysis results in SQLite.
- Run media extraction, frame analysis, transcription, and claim scoring asynchronously.
- Support optional local LLM scoring while retaining fallback logic when the model is unavailable.

## Non-goals

- Production-grade deployment.
- Full fact-checking or legal-grade verification.
- Long-term archival of processed video data beyond debugging artifacts.

## Architecture

### Core components

- `app/main.py`
  - FastAPI app exposing `POST /analyze`, `GET /jobs/{job_id}`, and `GET /health`.
  - Manages job creation, status updates, and persistence.

- `app/ai_detection/`
  - Visual and audio signal analysis for AI/deepfake detection.
  - `frame_detector.py` and `deepfake_detector.py` provide CLIP and frame-based scoring.

- `app/nlp/`
  - `asr.py`: audio transcription.
  - `claim_extractor.py`: claim extraction from title, description, and transcript.
  - `misinfo_scoring.py`: evidence orchestration, prompt building, and scoring.
  - `app/nlp/web_search.py`: search evidence retrieval using Bing HTML search first, with Yahoo HTML search as a fallback for resilience.

### Data flow

1. User submits a `POST /analyze` request with a YouTube URL.
2. Backend creates a new job record and marks it as `queued`.
3. Background worker extracts media, samples frames, transcribes audio, and generates claims.
4. Evidence is collected from web search results and used for scoring.
5. Results are persisted in `api/data/jobs.db` and returned by `GET /jobs/{job_id}`.

## API contract

### `POST /analyze`

Request body:

```json
{
  "url": "https://www.youtube.com/watch?v=VIDEO_ID",
  "source": "youtube"
}
```

Response:

```json
{
  "job_id": "job_123"
}
```

### `GET /jobs/{job_id}`

Response shape:

```json
{
  "status": "queued|running|succeeded|failed",
  "result": {
    "ai_detection": {
      "score": 0.0,
      "confidence": "TINGGI|SEDANG|RENDAH",
      "explanation": "string"
    },
    "hoax_analysis": {
      "score": 0.0,
      "risk_level": "TINGGI|SEDANG|RENDAH|TIDAK ADA",
      "explanation": "string"
    },
    "misinformation_analysis": {
      "score": 0.0,
      "risk_level": "TINGGI|SEDANG|RENDAH|TIDAK ADA",
      "explanation": "string"
    },
    "overall_assessment": {
      "recommendation": "string",
      "key_findings": ["string"]
    }
  } | null,
  "error": "string | null"
}
```

## Persistence

The backend uses SQLite to persist job state.

### Jobs table

- `id` TEXT PRIMARY KEY
- `status` TEXT
- `url` TEXT
- `source` TEXT
- `created_at` REAL
- `updated_at` REAL
- `result_json` TEXT
- `error` TEXT

Artifacts such as extracted frames and audio are stored under `api/data/jobs/` for debugging and analysis.

## Environment variables

- `MISINFO_LLM_MODEL`
  - Optional path or model ID for local LLM scoring.

- `OLLAMA_MODEL_NAME`
  - Optional Ollama alias for local model inference.

- `DATABASE_URL`
  - Optional override for the SQLite connection.

## Limitations

- This backend is an MVP and may fall back to heuristic scoring when models or media extraction fail. If the misinfo model is unavailable, AI detection will still use technical frame analysis if possible, while hoax/misinformation scores are marked as unavailable.
- Only YouTube video analysis is supported at this stage.
- The system is intended for developer experimentation rather than production deployment.
