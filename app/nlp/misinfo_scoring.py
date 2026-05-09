import json
import glob
import math
import os
import re
import subprocess
from typing import Dict, List, Optional, Tuple
import logging

from .web_search import search_web

logger = logging.getLogger(__name__)

# Lazy load misinfo LLM model
_gemma_model = None
_gemma_tokenizer = None
_misinfo_model_error: Optional[str] = None
_LLM_MODEL_ENV = "MISINFO_LLM_MODEL"
_OLLAMA_MODEL_ENV = "OLLAMA_MODEL_NAME"
_OLLAMA_MODEL_PREFIX = "ollama://"
_OLLAMA_DEFAULT_MODEL = "gemma4"


def _can_use_ollama_model(model_alias: str) -> bool:
    try:
        completed = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        )
        return any(line.strip().startswith(model_alias) for line in completed.stdout.splitlines())
    except Exception as e:
        logger.debug(f"Ollama model availability check failed: {e}")
        return False


def _get_gemma_model():
    global _gemma_model, _gemma_tokenizer, _misinfo_model_error
    if _gemma_model is None:
        model_name = os.environ.get(_LLM_MODEL_ENV, "").strip()
        if not model_name:
            ollama_alias = os.environ.get(_OLLAMA_MODEL_ENV, "").strip() or _OLLAMA_DEFAULT_MODEL
            model_name = f"{_OLLAMA_MODEL_PREFIX}{ollama_alias}"

        if not model_name:
            logger.info(f"No {_LLM_MODEL_ENV} configured; using keyword-based misinfo fallback.")
            _misinfo_model_error = None
            return None, None

        if model_name.startswith(_OLLAMA_MODEL_PREFIX):
            model_alias = model_name[len(_OLLAMA_MODEL_PREFIX) :]
            if not _can_use_ollama_model(model_alias):
                _misinfo_model_error = (
                    f"Ollama model '{model_alias}' is not available locally. "
                    "Run the project provisioning script or set MISINFO_LLM_MODEL to a local model path."
                )
                logger.warning(_misinfo_model_error)
                _gemma_model = None
                _gemma_tokenizer = None
                return None, None

            logger.info(f"Using Ollama-based misinfo model: {model_alias}")
            _gemma_model = model_name
            _gemma_tokenizer = None
            _misinfo_model_error = None
            return _gemma_model, None

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            if os.path.isdir(model_name) or os.path.isfile(model_name):
                logger.info(f"Loading local LLM model from path: {model_name}")
            else:
                logger.info(f"Loading LLM model from Hugging Face ID: {model_name}")

            _gemma_tokenizer = AutoTokenizer.from_pretrained(model_name)
            _gemma_model = AutoModelForCausalLM.from_pretrained(
                model_name, device_map="auto", torch_dtype="auto"
            )
            logger.info("Loaded misinfo LLM model successfully.")
            _misinfo_model_error = None
        except Exception as e:
            _misinfo_model_error = str(e)
            logger.warning(
                f"Failed to load misinfo LLM model '{model_name}': {e}. "
                "Misinfo scoring is unavailable and will be reported as failed."
            )
            _gemma_model = None
            _gemma_tokenizer = None
    return _gemma_model, _gemma_tokenizer


def _clean_explanation_text(text: str) -> str:
    if not text:
        return ""

    cleaned = text.strip()
    cleaned = re.sub(r"^```json\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"```$", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"(?i)^thinking(?: process)?[:\-\s]*", "", cleaned).strip()
    cleaned = re.sub(r"(?i)^here(?:'s| is)?\s+.*?\s*\n", "", cleaned).strip()
    cleaned = re.sub(r"\s+\n\s+", " ", cleaned)
    return cleaned.strip()


def _build_search_query(video_title: Optional[str], video_description: Optional[str], transcript: Optional[str]) -> str:
    focused_parts = []
    if video_title:
        focused_parts.append(video_title.strip())
    elif video_description:
        focused_parts.append(video_description.strip())

    if transcript:
        transcript_text = transcript.strip()
        first_sentence = re.split(r"[\.\?!]\s+", transcript_text)[0]
        if first_sentence:
            focused_parts.append(first_sentence.strip())

    query = " ".join(focused_parts).strip()
    query = re.sub(r"\s+", " ", query)
    if len(query) > 220:
        query = query[:220].rsplit(" ", 1)[0]
    return query


def _parse_llm_response(response: str) -> tuple[Optional[float], str]:
    if not response:
        return None, ""

    parsed_score = None
    parsed_explanation = ""

    if "{" in response and "}" in response:
        try:
            json_text = response[response.find("{"): response.rfind("}") + 1]
            data = json.loads(json_text)
            parsed_score = data.get("score")
            parsed_explanation = (
                data.get("explanation")
                or data.get("analysis")
                or data.get("reason")
                or ""
            )
        except Exception:
            parsed_score = None

    if parsed_score is not None:
        try:
            return min(1.0, max(0.0, float(parsed_score))), _clean_explanation_text(str(parsed_explanation))
        except Exception:
            parsed_score = None

    match = re.search(r"(\d+\.?\d*)", response)
    if match:
        explanation = response[: match.start()].strip() or response.strip()
        return min(1.0, max(0.0, float(match.group(1)))), _clean_explanation_text(explanation)

    return None, _clean_explanation_text(response)


def _score_claim_with_ollama(model_name: str, prompt: str) -> tuple[Optional[float], str, Optional[str]]:
    model_alias = model_name[len(_OLLAMA_MODEL_PREFIX) :]
    try:
        completed = subprocess.run(
            ["ollama", "run", model_alias, "--format", "json", "--nowordwrap", prompt],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
        response = completed.stdout.strip()
        score, explanation = _parse_llm_response(response)
        if score is not None:
            return score, explanation or "(Gemma memberi analisis tetapi tidak mengembalikan penjelasan yang diparse).", None
        return None, explanation, "Could not parse a numeric score from the Ollama model response."
    except subprocess.CalledProcessError as e:
        logger.warning(f"Ollama scoring failed: {e.stderr or e}")
        return None, e.stderr.strip() if e.stderr else str(e), e.stderr.strip() if e.stderr else str(e)
    except Exception as e:
        logger.warning(f"Ollama scoring failed: {e}")
        return None, str(e), str(e)


def _score_claim_with_gemma(claim: str, context: str, search_evidence: str = "") -> tuple[Optional[float], str, Optional[str]]:
    model, tokenizer = _get_gemma_model()
    if model is None:
        return None, "", _misinfo_model_error

    full_context = f"{context}{search_evidence}" if search_evidence else context

    prompt = f"""Anda adalah analis misinformasi yang menggunakan bahasa sehari-hari. Jelaskan pakai kata yang mudah dimengerti semua orang.

Petunjuk:
1. Jawab berdasarkan bukti yang ada, jangan hanya tebak-tebakan.
2. Kalau ada bukti dari hasil pencarian web, ambil satu kutipan langsung dari judul atau deskripsi dan tuliskan dalam penjelasan.
3. Contoh kutipan bisa seperti: "Reuters bilang: '...'", "BBC menulis: '...'" atau "Menurut Politifact: '...'".
4. Gunakan bahasa yang ringan, tidak terlalu formal, dan jelaskan seolah kamu sedang ngomong ke teman.
5. Nilai kemungkinan misinformasi: 0 (berita benar/terpercaya) sampai 1 (palsu/hoax).
6. Buat penjelasan pendek tapi jelas, dalam Bahasa Indonesia.

Klaim: {claim}

Konteks dan Bukti:
{full_context}

Respons dengan objek JSON yang valid TUNGGAL saja. Tidak ada penjelasan proses berpikir, hanya jawaban final.
Gunakan kunci persis: "score" dan "explanation".
Contoh: {{"score": 0.85, "explanation": "Reuters bilang '...'; jadi kemungkinan isi ini hoax karena ..."}}"""

    if isinstance(model, str) and model.startswith(_OLLAMA_MODEL_PREFIX):
        return _score_claim_with_ollama(model, prompt)

    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=120, do_sample=False)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        score, explanation = _parse_llm_response(response)
        if score is not None:
            return score, explanation or "(Gemma memberi analisis tetapi tidak mengembalikan penjelasan yang diparse).", None
        return None, explanation, "Could not parse a numeric score from the misinfo model response."
    except Exception as e:
        logger.warning(f"Gemma scoring failed for claim: {e}")
        return None, str(e), str(e)


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _hash_string(s: str) -> int:
    # Stable, lightweight hash (non-cryptographic).
    h = 2166136261
    for ch in s:
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return abs(h)


def _extract_json_from_response(response: str) -> Tuple[Optional[Dict[str, any]], Optional[str]]:
    """Extract and parse JSON from model response, handling common formatting issues."""
    if not response:
        return None, "Empty response"

    response = response.strip()

    # Try direct parse first
    if response.startswith('{') and response.endswith('}'):
        try:
            return json.loads(response), None
        except json.JSONDecodeError:
            pass

    # Find JSON object using regex
    import re
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if json_match:
        json_str = json_match.group()
        try:
            return json.loads(json_str), None
        except json.JSONDecodeError as e:
            # Try to fix common issues: missing commas between keys
            fixed_json = re.sub(r'"\s*\n\s*"', '",\n"', json_str)  # Add comma between quoted strings on new lines
            fixed_json = re.sub(r'(\w)"\s*\n\s*"(\w)', r'\1",\n"\2', fixed_json)  # Add comma between unquoted keys
            try:
                return json.loads(fixed_json), None
            except json.JSONDecodeError:
                return None, f"Failed to parse JSON after fixes: {e}"

    return None, "No valid JSON object found in response"


def orchestrate_comprehensive_analysis(
    url: str,
    video_title: Optional[str] = None,
    video_description: Optional[str] = None,
    transcript: Optional[str] = None,
    frames_dir: Optional[str] = None,
    audio_path: Optional[str] = None,
    clip_results: Optional[Dict[str, any]] = None,
) -> Tuple[Optional[Dict[str, any]], Optional[str]]:
    """
    Comprehensive analysis orchestrator using Gemma LLM.
    AI detection proceeds regardless of web search.
    Hoax analysis requires web search evidence.
    """
    model, tokenizer = _get_gemma_model()
    if model is None:
        return None, _misinfo_model_error

    # Collect all available data for AI detection (no web search required)
    data_parts = []
    if video_title:
        data_parts.append(f"JUDUL VIDEO: {video_title}")
    if video_description:
        data_parts.append(f"DESKRIPSI VIDEO: {video_description}")
    if transcript:
        data_parts.append(f"TRANSCRIPT AUDIO: {transcript[:2000]}")  # Limit transcript length
    data_parts.append(f"URL VIDEO: {url}")

    # Add CLIP analysis results if available
    if clip_results:
        clip_score = clip_results.get('ai_score', 0.5)
        clip_drivers = clip_results.get('ai_drivers', [])
        data_parts.append(f"ANALISIS CLIP (AI Detection Score: {clip_score:.2f}):")
        data_parts.extend([f"- {driver}" for driver in clip_drivers])

    # Add frame information if available
    if frames_dir:
        try:
            import os
            frame_files = glob.glob(os.path.join(frames_dir, "*.jpg")) + glob.glob(os.path.join(frames_dir, "*.png"))
            data_parts.append(f"FRAME ANALYSIS: {len(frame_files)} frame diekstrak untuk analisis visual")
        except:
            pass

    # Add audio information if available
    if audio_path:
        try:
            import os
            if os.path.exists(audio_path):
                data_parts.append(f"AUDIO ANALYSIS: File audio tersedia untuk analisis ({os.path.getsize(audio_path)} bytes)")
        except:
            pass

    # Web search is only required for hoax analysis, not for AI detection
    search_evidence_section = ""
    search_query = _build_search_query(video_title, video_description, transcript)
    if search_query:
        search_results = search_web(search_query, max_results=5)
        if search_results:
            evidence_lines = []
            for idx, item in enumerate(search_results, start=1):
                evidence_lines.append(
                    f"{idx}. {item.get('title', '')}\n{item.get('snippet', '')}\nSumber: {item.get('url', '')}"
                )
            search_evidence_section = "\n\nBUKTI PENCARIAN WEB (gunakan URL ini dalam penjelasan):\n" + "\n\n".join(evidence_lines)
        else:
            search_evidence_section = (
                "\n\nKATERKANGAN: Web search tidak berhasil. "
                "Analisis misinformasi didasarkan pada data internal video saja."
            )

    context = "\n\n".join(data_parts) + search_evidence_section

    prompt = f"""Anda adalah analis konten video yang ramah dan menggunakan bahasa sehari-hari. Jelaskan seperti sedang ngobrol ke teman, supaya mudah dipahami.

DATA VIDEO YANG TERSEDIA:
{context}

TUGAS ANDA - BERIKAN ANALISIS YANG JELAS DALAM SATU RESPONS:

1. **DETEKSI AI-GENERATED CONTENT**: Jelaskan apakah video ini terkesan dibuat oleh AI/deepfake berdasarkan data yang tersedia
2. **ANALISIS MISINFORMASI**: Tulis apakah video ini mengandung informasi palsu, berlebihan, atau menyesatkan
3. **PENILAIAN KESELURUHAN**: Beri skor dan rekomendasi singkat yang mudah dimengerti

PETUNJUK TAMBAHAN:
- Gunakan bahasa sederhana dan tidak terlalu formal
- Jika ada bukti dari hasil pencarian web, kutip langsung satu frase atau judul dari hasil itu
- Contoh kutipan: "Reuters bilang: '...'", "BBC tulis: '...'", atau "Menurut sumber: '...'"
- Sertakan sumber dengan format [Sumber: URL] jika tersedia
- Fokus pada fakta yang nyata dari bukti, jangan bertele-tele
- Jika tidak ada bukti pencarian, analisis berdasarkan data video dan konteks yang tersedia

RESPONSI HARUS DALAM FORMAT JSON YANG VALID DENGAN STRUKTUR BERIKUT:

{{
  "ai_detection": {{
    "score": 0.0-1.0,
    "confidence": "TINGGI/SEDANG/RENDAH",
    "explanation": "Penjelasan singkat dalam bahasa Indonesia yang sederhana"
  }},
  "misinformation_analysis": {{
    "score": 0.0-1.0,
    "risk_level": "TINGGI/SEDANG/RENDAH/TIDAK ADA",
    "explanation": "Penjelasan singkat dalam bahasa Indonesia yang mudah dimengerti"
  }},
  "overall_assessment": {{
    "recommendation": "Rekomendasi singkat untuk penonton dalam bahasa Indonesia",
    "key_findings": ["Poin penting 1", "Poin penting 2", "Poin penting 3"]
  }}
}}

PENTING:
- Semua teks dalam Bahasa Indonesia saja
- Output HANYA objek JSON. Jangan sertakan teks pengantar, penjelasan proses, atau tambahan lain
- Jika ada bukti pencarian, gunakan setidaknya satu kutipan langsung dari judul atau deskripsi hasil pencarian
- Tulis [Sumber: URL] dalam penjelasan jika bukti web search tersedia
- Jangan gunakan bahasa formal berat
- Pastikan JSON valid"""

    if isinstance(model, str) and model.startswith(_OLLAMA_MODEL_PREFIX):
        return _orchestrate_with_ollama(model, prompt)

    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=800, do_sample=False, temperature=0.1)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract JSON from response
        result, parse_error = _extract_json_from_response(response)
        if result:
            return result, None
        else:
            return None, parse_error
    except Exception as e:
        logger.warning(f"Gemma orchestration failed: {e}")
        return None, str(e)


def _orchestrate_with_ollama(model_name: str, prompt: str) -> Tuple[Optional[Dict[str, any]], Optional[str]]:
    """Orchestrate analysis using Ollama Gemma model."""
    model_alias = model_name[len(_OLLAMA_MODEL_PREFIX):]
    try:
        completed = subprocess.run(
            ["ollama", "run", model_alias, "--format", "json", "--nowordwrap", prompt],
            capture_output=True,
            text=True,
            check=True,
            timeout=120,  # Longer timeout for comprehensive analysis
        )
        response = completed.stdout.strip()

        # Try to extract JSON from response
        result, parse_error = _extract_json_from_response(response)
        if result:
            return result, None
        else:
            return None, parse_error

    except subprocess.CalledProcessError as e:
        logger.warning(f"Ollama orchestration failed: {e.stderr or e}")
        return None, e.stderr.strip() if e.stderr else str(e)
    except Exception as e:
        logger.warning(f"Ollama orchestration failed: {e}")
        return None, str(e)


def score_misinformation(
    claims: List[str],
    transcript: str,
    url: str,
    *,
    ai_score: float,
    video_title: Optional[str] = None,
    video_description: Optional[str] = None,
) -> Tuple[Optional[float], List[Dict[str, str]], Optional[str]]:
    """
    Enhanced misinfo scoring using Gemma LLM for claim analysis.
    Falls back to keyword-based if model unavailable.
    """
    claim_outputs: List[Dict[str, str]] = []
    gemma_scores: List[float] = []
    misinfo_error: Optional[str] = None

    summary_parts = [f"Transcript: {transcript[:1000]}", f"URL: {url}"]
    if video_title:
        summary_parts.append(f"Title: {video_title}")
    if video_description:
        summary_parts.append(f"Description: {video_description[:1000]}")

    context = "\n".join(summary_parts)
    search_query = " ".join(part for part in [video_title or "", video_description or ""] if part).strip()
    search_results = []
    if search_query:
        search_results = search_web(search_query, max_results=3)

    evidence_section = ""
    if search_results:
        evidence_lines = []
        for idx, item in enumerate(search_results, start=1):
            evidence_lines.append(f"{idx}. {item.get('title', '')}\n{item.get('snippet', '')}\nURL: {item.get('url', '')}")
        evidence_section = "\n\nSearch evidence:\n" + "\n\n".join(evidence_lines)

    for c in claims:
        evidence_urls = [item["url"] for item in search_results if item.get("url")]
        prompt_context = f"{context}{evidence_section}\n\nClaim: {c}"
        gemma_score, gemma_explanation, claim_error = _score_claim_with_gemma(c, prompt_context, search_evidence=evidence_section)
        if claim_error and misinfo_error is None:
            misinfo_error = claim_error

        if gemma_score is not None:
            gemma_scores.append(gemma_score)

        evidence_urls = [item["url"] for item in search_results if item.get("url")]
        if gemma_score is None:
            verdict = "TIDAK PASTI"
            reason = (
                "Misinformasi tidak dapat dievaluasi karena penilaian model gagal. "
                "Periksa konfigurasi model lokal atau akses repositori lokal Gemma/GEMMA."
            )
        elif gemma_score >= 0.6:
            verdict = "HOAX"
            reason = "Analisis LLM menunjukkan konten ini sangat mungkin mengandung misinformasi atau informasi menyesatkan."
        elif gemma_score >= 0.3:
            verdict = "TIDAK PASTI"
            reason = "Analisis LLM menunjukkan ketidakpastian sedang terhadap kebenaran konten ini."
        else:
            verdict = "BENAR"
            reason = "Analisis LLM menunjukkan konten ini tampak kredibel berdasarkan konteks saat ini."

        claim_outputs.append(
            {
                "claim": c,
                "verdict": verdict,
                "evidence_urls": evidence_urls,
                "reason": reason,
                "gemma_explanation": gemma_explanation or "Tidak ada penjelasan tambahan tersedia.",
            }
        )

    if misinfo_error is not None:
        return None, claim_outputs, misinfo_error

    if gemma_scores:
        misinfo_score = sum(gemma_scores) / len(gemma_scores)
    else:
        u = (url or "").lower()
        mis_keywords = ["hoaks", "penipuan", "kabar bohong", "bencana", "virus", "konflik", "krisis", "dinas", "viral", "darurat"]
        url_signal = _keyword_strength(u, mis_keywords)
        transcript_signal = _keyword_strength(transcript, mis_keywords)
        title_signal = _keyword_strength(video_title or "", mis_keywords)
        description_signal = _keyword_strength(video_description or "", mis_keywords)
        jitter = (_hash_string(u) % 1000) / 1000.0
        combined = (
            0.16
            + 0.40 * max(url_signal, transcript_signal)
            + 0.18 * max(title_signal, description_signal)
            + 0.10 * min(1.0, url_signal + transcript_signal + title_signal + description_signal)
            + 0.06 * _clamp01(ai_score)
        )
        misinfo_score = _clamp01(combined + (jitter - 0.5) * 0.12)

    return misinfo_score, claim_outputs, None

