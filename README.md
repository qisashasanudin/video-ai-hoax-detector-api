# AI Hoax Video Platform API

FastAPI backend for asynchronous YouTube video analysis, including AI-generation detection, transcript extraction, and misinformation scoring.

## Overview

This backend ingests a YouTube URL, queues an analysis job, and returns a job record that can be polled until completion. The analysis pipeline currently combines:

- frame-level AI detection and deepfake signal aggregation in `api/app/ai_detection/`
- audio transcription in `api/app/nlp/asr.py`
- claim extraction in `api/app/nlp/claim_extractor.py`
- misinfo scoring and evidence orchestration in `api/app/nlp/misinfo_scoring.py`

Analysis outputs are persisted in `api/data/jobs.db` and served by `GET /jobs/{job_id}`.

## Setup

1. Create or activate a virtual environment:

```bash
cd api
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the backend

```bash
cd api
source .venv/bin/activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Configuration

### Backend environment variables

- `MISINFO_LLM_MODEL`
  - Optional local model path or Hugging Face model ID.
  - When set, the backend will attempt to load this model for claim-level misinfo scoring.

- `OLLAMA_MODEL_NAME`
  - Optional Ollama model alias when using Ollama-based local inference.
  - Default alias is `gemma4`.

### Fallback behavior

If the configured model cannot be loaded, the backend falls back to keyword- and evidence-driven misinfo scoring rather than failing completely.

## API Endpoints

- `POST /analyze`
  - Request: `{"url": "https://www.youtube.com/watch?v=VIDEO_ID", "source": "youtube"}`
  - Response: `{"job_id": "job_..."}`

- `GET /jobs/{job_id}`
  - Response includes job status and the analysis result payload.

- `GET /health`
  - Response: `{"status": "ok"}`

## Architecture

### Backend structure

- `app/main.py`
  - API routes and analysis job orchestration.
  - Creates jobs, updates progress, and persists results.

- `app/ai_detection/`
  - `frame_detector.py` and `deepfake_detector.py` contain the visual AI detection pipeline.
  - Combines CLIP-based scoring, temporal consistency, and audio/visual heuristic signals.

- `app/nlp/`
  - `asr.py` transcribes audio from the video.
  - `claim_extractor.py` extracts candidate claims from title, description, and transcript.
  - `misinfo_scoring.py` builds evidence context and uses a local LLM or fallback logic to score misinformation.
  - `web_search.py` retrieves search evidence for claim verification using the Bing RSS search feed only.

### Data and persistence

- `api/data/jobs.db`
  - SQLite database storing job metadata, status, result JSON, and errors.

- `api/data/jobs/`
  - Stores extracted job artifacts such as frames and audio files.

## Frontend integration

The frontend is hosted in `web/` and depends on `NEXT_PUBLIC_API_BASE_URL` to target this API.

Example:

```bash
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

## Notes

- This API is designed as an MVP for developer experimentation and not production deployment.
- The system currently supports YouTube as the source input.
