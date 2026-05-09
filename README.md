# Video Misinfo Platform API (MVP)

## Run (mock backend)

1. Create a virtual environment (recommended).
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Start the server:
   - `uvicorn app.main:app --reload --port 8000 --host 0.0.0.0`

## Local LLM configuration

This backend can optionally use a local or explicitly configured LLM for claim-level misinfo scoring. Set the environment variable `MISINFO_LLM_MODEL` to a local model path or a Hugging Face model ID that you have access to.

Example:

- Local model path:
  - `export MISINFO_LLM_MODEL=/path/to/gemma-2b-it`
- Hugging Face model with access:
  - `export MISINFO_LLM_MODEL=google/gemma-2b-it`

If `MISINFO_LLM_MODEL` is not set or the model cannot be loaded, the system falls back to keyword-based misinfo scoring.

## Ollama / Gemma 4 automation

A helper script is available to provision Gemma 4 locally using Ollama:

- `./scripts/ensure_gemma4.sh`
- `./scripts/ensure_gemma4.sh gemma4`

If you have `ollama` installed, the script will pull the default local Ollama model alias `gemma4`.

By default, the backend will attempt to use `OLLAMA_MODEL_NAME` or the default local Ollama model alias `gemma4` if `MISINFO_LLM_MODEL` is not configured.

If you want to force a specific model path or ID, set `MISINFO_LLM_MODEL` explicitly.

## Endpoints

- `POST /analyze` -> returns `{ "job_id": "..." }`
- `GET /jobs/{job_id}` -> returns `{ "status": "...", "result": {...} | null, "error": "..." | null }`
- `GET /health` -> `{ "status": "ok" }`
