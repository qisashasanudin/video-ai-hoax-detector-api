from typing import List


def extract_claims_from_transcript(transcript: str, *, url: str) -> List[str]:
    """
    MVP claim extraction (no ML deps):
    - If transcript exists: pick top 3-ish sentences as candidate claims.
    - If transcript is empty: derive plausible claim templates from URL keywords.
    """
    t = (transcript or "").strip()
    if t:
        # Split into rough sentences and keep non-trivial ones.
        parts = []
        for chunk in t.replace("\n", " ").split("."):
            s = chunk.strip()
            if len(s) >= 25:
                parts.append(s)
        if not parts:
            parts = [x.strip() for x in t.split(" ") if x.strip()]
        selected = parts[:3]
        return selected if selected else [t[:120] + "..." if len(t) > 120 else t]

    u = url.lower()

    claims: List[str] = []
    if "bencana" in u or "gempa" in u:
        claims.append("Video mengandung klaim peristiwa bencana (mis. gempa/kejadian tertentu).")
    else:
        claims.append("Video memuat klaim/narasi yang berpotensi memengaruhi persepsi penonton.")

    if any(k in u for k in ["hoaks", "penipuan", "kabar bohong", "viral", "darurat", "krisis"]):
        claims.append("Narasi berisi pernyataan yang bernuansa kepanikan/urgensi (berpotensi menyesatkan).")
    else:
        claims.append("Narasi berisi klaim yang perlu diverifikasi karena konteks berpotensi terdistorsi.")

    claims.append("Video mungkin menyajikan cuplikan tanpa konteks lengkap sehingga interpretasi dapat berubah.")
    return claims[:3]

