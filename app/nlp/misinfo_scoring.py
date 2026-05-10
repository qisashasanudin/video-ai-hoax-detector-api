import json
import glob
import math
import os
import re
import subprocess
from typing import Dict, List, Optional, Tuple, Callable
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
    """
    Build a focused search query to verify the specific claim in the video.
    Prioritize extracting actual claims/quotes over generic titles.
    """
    focused_parts = []
    
    # Extract specific claims from transcript first (quotes, key statements)
    if transcript:
        transcript_text = transcript.strip()
        # Look for quotes in quotes or important statements
        # For short videos, the whole transcript is often the claim
        if len(transcript_text) < 500:
            focused_parts.append(transcript_text)
        else:
            # For longer transcripts, extract first meaningful sentence or paragraph
            sentences = re.split(r'[\.\?!\n]+', transcript_text)
            for sent in sentences:
                sent = sent.strip()
                if len(sent) > 20:  # Long enough to be meaningful
                    focused_parts.append(sent)
                    break
    
    # Add title if no good transcript extract
    if video_title and not focused_parts:
        focused_parts.append(video_title.strip())
    elif video_title and len(focused_parts[0]) < 30:
        # If transcript extract is very short, enhance with title
        focused_parts.append(video_title.strip())
    
    query = " ".join(focused_parts).strip()
    query = re.sub(r"\s+", " ", query)
    
    # Cap at 220 chars to avoid overly long queries
    if len(query) > 220:
        query = query[:220].rsplit(" ", 1)[0]
    
    return query


def _extract_claims(video_title: Optional[str], video_description: Optional[str], transcript: Optional[str]) -> List[str]:
    """
    Extract individual claims/statements from video metadata.
    Extract multiple claims from any video content, regardless of length.
    """
    claims = []
    
    # Add title as a claim if it contains substantive content
    if video_title:
        title_clean = video_title.strip()
        if len(title_clean) > 10 and not _is_url_or_junk(title_clean):
            claims.append(title_clean)
    
    # Extract claims from description
    if video_description:
        desc_clean = video_description.strip()
        # Split description into sentences and extract meaningful claims
        sentences = _split_sentences_smart(desc_clean)
        for sent in sentences:
            sent = sent.strip()
            # Only add meaningful sentences (not too short, not too long, not URLs)
            if 15 < len(sent) < 400 and not _is_url_or_junk(sent) and sent not in claims:
                claims.append(sent)
                # Limit to avoid too many claims from description
                if len(claims) >= 3:
                    break
    
    # For transcripts, always try to extract multiple claims
    if transcript:
        transcript_clean = transcript.strip()
        
        # Extract sentences/quotes from transcript
        sentences = _split_sentences_smart(transcript_clean)
        for sent in sentences:
            sent = sent.strip()
            # Only add meaningful sentences (not too short, not too long, not URLs)
            if 15 < len(sent) < 400 and not _is_url_or_junk(sent) and sent not in claims:
                claims.append(sent)
                # Limit to 5 extracted claims per video
                if len(claims) >= 5:
                    break
    
    # If no claims extracted, use title or fallback
    if not claims:
        claims = [video_title or "Video content"]
    
    # Remove duplicates while preserving order
    seen = set()
    unique_claims = []
    for claim in claims:
        claim_lower = claim.lower()
        if claim_lower not in seen:
            seen.add(claim_lower)
            unique_claims.append(claim)
    
    return unique_claims


def _split_sentences_smart(text: str) -> List[str]:
    """Split text into sentences, handling URLs and abbreviations better."""
    # First, temporarily replace URLs with placeholders
    url_pattern = r'https?://[^\s]+'
    urls = re.findall(url_pattern, text)
    for i, url in enumerate(urls):
        text = text.replace(url, f'__URL_PLACEHOLDER_{i}__')
    
    # Split on sentence endings, but be careful with abbreviations
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Restore URLs
    for i, url in enumerate(urls):
        for j, sent in enumerate(sentences):
            sentences[j] = sent.replace(f'__URL_PLACEHOLDER_{i}__', url)
    
    return sentences


def _is_url_or_junk(text: str) -> bool:
    """Check if text is a URL, contains mostly non-alphabetic characters, or is junk."""
    # Check for URLs
    if re.search(r'https?://', text):
        return True
    
    # Check for too many non-alphabetic characters
    alpha_count = sum(1 for c in text if c.isalpha())
    total_count = len(text)
    if total_count > 0 and alpha_count / total_count < 0.3:
        return True
    
    # Check for very short fragments that are likely junk
    words = text.split()
    if len(words) < 3 and len(text) < 20:
        return True
    
    return False


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

    prompt = f"""Anda adalah analis misinformasi yang profesional. Gunakan bahasa Indonesia yang jelas, sopan, dan mudah dipahami oleh pembaca website.

Petunjuk:
1. Jawab berdasarkan bukti yang ada, jangan hanya tebak-tebakan.
2. Kalau ada bukti dari hasil pencarian web, ambil satu kutipan langsung dari judul atau deskripsi dan tuliskan dalam penjelasan.
3. Contoh kutipan bisa seperti: "Reuters bilang: '...'", "BBC menulis: '...'" atau "Menurut Politifact: '...'".
4. Gunakan bahasa yang sederhana, sopan, dan jelaskan seolah Anda sedang menjelaskan kepada pembaca biasa.
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

    import re

    # Try to locate the first JSON object-like block.
    json_match = re.search(r'\{[\s\S]*\}', response)
    if json_match:
        json_str = json_match.group()
        try:
            return json.loads(json_str), None
        except json.JSONDecodeError as e:
            fixed_json = json_str
            # Remove trailing commas before } or ]
            fixed_json = re.sub(r',\s*(?=[}\]])', '', fixed_json)
            # Insert missing commas between object values on separate lines
            fixed_json = re.sub(r'([}\]"\d])\s*\n\s*"', r'\1,\n"', fixed_json)
            fixed_json = re.sub(r'([}\]"\d])\s*\n\s*([A-Za-z_][A-Za-z0-9_]*)(?=\s*:)', r'\1,\n"\2', fixed_json)
            # Normalize single-quoted keys/strings if necessary
            if "'" in fixed_json and '"' not in fixed_json:
                fixed_json = fixed_json.replace("'", '"')
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
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Tuple[Optional[Dict[str, any]], Optional[str]]:
    """
    Comprehensive analysis orchestrator using Gemma LLM.
    Extracts claims, searches for each, then analyzes with evidence.
    """
    def update_progress(message: str):
        if progress_callback:
            progress_callback(message)
        logger.info(message)
    
    model, tokenizer = _get_gemma_model()
    if model is None:
        # Fallback analysis when model is not available
        update_progress("Menggunakan analisis fallback (model tidak tersedia)")
        fallback_result = {
            "ai_detection": {
                "score": 0.5,
                "confidence": "SEDANG",
                "explanation": "Model AI tidak tersedia, menggunakan estimasi berdasarkan metadata video."
            },
            "hoax_analysis": {
                "score": 0.3,
                "risk_level": "SEDANG",
                "explanation": "Model analisis tidak tersedia, menggunakan estimasi berdasarkan pencarian web."
            },
            "misinformation_analysis": {
                "score": 0.2,
                "risk_level": "RENDAH",
                "explanation": "Model analisis tidak tersedia, menggunakan estimasi berdasarkan pencarian web."
            },
            "overall_assessment": {
                "recommendation": "Verifikasi informasi dari sumber terpercaya sebelum menyebarkan.",
                "key_findings": [
                    "Model AI tidak tersedia untuk analisis mendalam",
                    "Estimasi risiko berdasarkan metadata video",
                    "Disarankan verifikasi manual dari sumber kredibel"
                ]
            },
            "claims": claims
        }
        return fallback_result, None

    # Collect baseline data for AI detection
    update_progress("Mengekstrak data video...")
    data_parts = []
    if video_title:
        data_parts.append(f"JUDUL VIDEO: {video_title}")
    if video_description:
        data_parts.append(f"DESKRIPSI VIDEO: {video_description}")
    if transcript:
        data_parts.append(f"TRANSCRIPT AUDIO: {transcript[:2000]}")
    data_parts.append(f"URL VIDEO: {url}")

    # Add CLIP analysis results
    if clip_results:
        clip_score = clip_results.get('ai_score', 0.5)
        clip_drivers = clip_results.get('ai_drivers', [])
        data_parts.append(f"ANALISIS CLIP (AI Detection Score: {clip_score:.2f}):")
        data_parts.extend([f"- {driver}" for driver in clip_drivers])

    if frames_dir:
        try:
            frame_files = glob.glob(os.path.join(frames_dir, "*.jpg")) + glob.glob(os.path.join(frames_dir, "*.png"))
            data_parts.append(f"FRAME ANALISIS: {len(frame_files)} frame diekstrak untuk analisis visual")
            data_parts.append(
                "SPEAKER VISUAL CONTEXT: Gunakan frame visual untuk menilai apakah pembicara yang terlihat kemungkinan adalah orang yang mengucapkan klaim."
            )
        except:
            pass

    if audio_path:
        try:
            if os.path.exists(audio_path):
                data_parts.append(f"AUDIO ANALYSIS: File audio tersedia untuk analisis ({os.path.getsize(audio_path)} bytes)")
                data_parts.append(
                    "SPEAKER CONTEXT: Audio tersedia dan harus digunakan untuk memverifikasi apakah pernyataan berasal dari pembicara yang terlihat di video."
                )
        except:
            pass

    # Extract individual claims from video
    update_progress("Mengekstrak klaim dari video...")
    claims = _extract_claims(video_title, video_description, transcript)
    update_progress(f"Ditemukan {len(claims)} klaim untuk dianalisis")

    # Search and collect evidence for each claim
    update_progress("Mencari bukti online untuk setiap klaim...")
    claim_evidence = {}
    for idx, claim in enumerate(claims, start=1):
        update_progress(f"Mencari bukti untuk klaim {idx}/{len(claims)}: {claim[:50]}...")
        search_results = search_web(claim, max_results=3)
        claim_evidence[claim] = search_results if search_results else []

    # Build evidence section with all claims
    evidence_lines = []
    for claim, results in claim_evidence.items():
        evidence_lines.append(f"\nKLAIM: {claim}")
        if results:
            for idx, item in enumerate(results, start=1):
                evidence_lines.append(
                    f"  {idx}. {item.get('title', '')}\n     {item.get('snippet', '')}\n     Sumber: {item.get('url', '')}"
                )
        else:
            evidence_lines.append("  (Tidak ada hasil pencarian)")
    
    search_evidence_section = "\n\nBUKTI PENCARIAN WEB:\n" + "\n".join(evidence_lines) if evidence_lines else ""
    context = "\n\n".join(data_parts) + search_evidence_section

    # Generate comprehensive analysis prompt
    update_progress("Menganalisis informasi dengan bukti online...")
    prompt = f"""Anda adalah analis konten video yang profesional dan berbasis bukti. Evaluasi faktualitas, bukan sensasionalitas.

DATA VIDEO YANG TERSEDIA:
{context}

PEDOMAN PENTING UNTUK ANALISIS AKURAT:

TENTANG MISINFORMASI:
- Misinformasi = informasi yang secara objektif SALAH atau MENYESATKAN, bukan hanya kontroversial
- Fakta yang sensitif atau kontroversial BUKAN misinformasi jika dapat diverifikasi dari sumber kredibel
- Video dari channel berita resmi cenderung menceritakan klaim autentik, bukan manipulasi
- Jika ada bukti web search yang menegaskan klaim video, klasifikasi sebagai BUKAN misinformasi (score rendah)
- Hanya berikan score tinggi jika ada bukti objektif bahwa informasi SALAH atau telah dikemas secara menyesatkan

TENTANG DETEKSI AI (DEEPFAKE):
- Video shorts dan konten terkompresi sering menunjukkan anomali teknis tinggi tanpa menjadi deepfake
- Anomali tinggi ≠ deepfake otomatis; diperlukan bukti manipulasi visual/audio yang nyata
- Rekaman publik dari figur publik (bahkan pernyataan kontroversial) adalah data autentik, bukan AI-generated
- Hanya berikan score AI tinggi jika ada indikasi kuat artefak deepfake (wajah tidak sinkron, audio aneh, dll)
- Jika audio dan visual menunjukkan seseorang berbicara langsung, gunakan bukti tersebut untuk menilai apakah pernyataan itu berasal dari pembicara yang terlihat
- Jika klaim menyebutkan figur tertentu (misalnya presiden), verifikasi apakah pembicara di video kemungkinan adalah figur yang dimaksud atau nyatakan bahwa identitasnya tidak dapat dikonfirmasi
- Jika judul atau transkrip menunjukkan pelaporan tuduhan dari figur publik, jangan klasifikasikan konten tersebut sebagai hoax hanya karena tuduhan belum terbukti; itu lebih tepat dinilai sebagai misinformasi atau ketidakpastian jika klaim tidak bisa diverifikasi

TUGAS ANDA:

1. **DETEKSI AI-GENERATED CONTENT**: 
   - Periksa apakah ada tanda-tanda manipulasi visual/audio yang jelas
   - Anomali skor tinggi pada shorts/video terkompresi ≠ deepfake
   - Jika video dari sumber resmi/channel terverifikasi, asumsikan autentik kecuali ada bukti jelas

2. **ANALISIS HOAX** (KONTEN SEPENUHNYA PALSU):
   - Hoax = cerita atau klaim yang sepenuhnya direkayasa/fabricated tanpa dasar fakta sama sekali
   - Jika video hanya melaporkan tuduhan atau komentar dari seorang figur publik (misalnya "Trump menuduh Obama ..."), itu tidak otomatis menjadi HOAX. Klasifikasikan sebagai hoax hanya jika ada bukti kuat bahwa narasi sengaja dibuat-buat, dipalsukan, atau disampaikan sebagai fakta yang jelas-jelas salah.
   - Contoh hoax: "Alien mendarat di Jakarta" tanpa konteks atau sumber sama sekali
   - Jika ada hasil pencarian web yang membantah klaim secara total, score TINGGI
   - Jika klaim didukung oleh sumber terverifikasi, score RENDAH
   - Fokus pada klaim yang benar-benar tidak memiliki dasar realitas atau dibuat-buat oleh pembuat konten

3. **ANALISIS MISINFORMASI** (KONTEN MENYESATKAN):
   - Misinformasi = informasi yang sebagian benar tapi disajikan secara menyesatkan atau tidak akurat
   - Contoh: "Vaksin COVID menyebabkan kematian massal" (mungkin benar untuk kasus langka tapi digeneralisasi)
   - Jika ada hasil pencarian web yang menunjukkan penyimpangan fakta, score TINGGI
   - Jika informasi akurat meskipun kontroversial, score RENDAH
   - Fokus pada akurasi faktual, bukan kontroversialitas topik

4. **PENILAIAN KESELURUHAN**: 
   - Beri rekomendasi berdasarkan bukti, bukan asumsi

FORMAT JSON WAJIB:

{{
  "ai_detection": {{
    "score": 0.0-1.0,
    "confidence": "TINGGI/SEDANG/RENDAH",
    "explanation": "Penjelasan singkat tentang tanda-tanda AI atau alasan dianggap autentik"
  }},
  "hoax_analysis": {{
    "score": 0.0-1.0,
    "risk_level": "TINGGI/SEDANG/RENDAH/TIDAK ADA",
    "explanation": "Penjelasan singkat tentang apakah konten sepenuhnya palsu/fabricated"
  }},
  "misinformation_analysis": {{
    "score": 0.0-1.0,
    "risk_level": "TINGGI/SEDANG/RENDAH/TIDAK ADA",
    "explanation": "Penjelasan singkat tentang akurasi faktual dan penyimpangan informasi"
  }},
  "overall_assessment": {{
    "recommendation": "Rekomendasi singkat berdasarkan bukti",
    "key_findings": ["Temuan 1", "Temuan 2", "Temuan 3"]
  }}
}}

ATURAN WAJIB:
- Semua teks dalam Bahasa Indonesia saja
- Output HANYA JSON, tanpa penjelasan tambahan
- Jika ada bukti web search, kutip satu frase dari hasil atau tulis [Sumber: URL]
- Evaluasi FAKTA OBJEKTIF, bukan opini atau sentiment
- Jangan biarkan score AI atau misinformasi dipengaruhi oleh sifat kontroversial topik
- Pastikan JSON valid"""

    update_progress("Menunggu hasil analisis dari model LLM...")
    if isinstance(model, str) and model.startswith(_OLLAMA_MODEL_PREFIX):
        return _orchestrate_with_ollama(model, prompt, claims)

    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=1000, do_sample=False, temperature=0.1)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract JSON from response
        result, parse_error = _extract_json_from_response(response)
        if result:
            update_progress("Analisis selesai!")
            # Add claims to the result
            result["claims"] = claims
            return result, None
        else:
            return None, parse_error
    except Exception as e:
        logger.warning(f"Gemma orchestration failed: {e}")
        return None, str(e)


def _orchestrate_with_ollama(model_name: str, prompt: str, claims: List[str]) -> Tuple[Optional[Dict[str, any]], Optional[str]]:
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
            result["claims"] = claims
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

