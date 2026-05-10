import logging
import re
import unicodedata
from typing import Dict, List, Optional
from urllib.parse import quote_plus, unquote

import requests

logger = logging.getLogger(__name__)

MAX_SEARCH_QUERY_LENGTH = 220
SEARCH_TIMEOUT_SECONDS = 20


def _sanitize_search_query(query: str) -> str:
    if not query:
        return ""

    normalized = unicodedata.normalize("NFKC", query)
    normalized = normalized.replace("|", " ")
    normalized = normalized.replace("‘", "'").replace("’", "'")
    normalized = normalized.replace("“", '"').replace("”", '"')
    normalized = normalized.replace("—", "-").replace("–", "-")
    normalized = re.sub(r"[^\x00-\x7F]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _normalize_search_query(query: str, max_length: int = MAX_SEARCH_QUERY_LENGTH) -> str:
    cleaned = " ".join(query.strip().split())
    if len(cleaned) <= max_length:
        return cleaned
    trimmed = cleaned[:max_length]
    return trimmed.rsplit(" ", 1)[0]


def _strip_html_tags(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"<[^>]+>", "", text).strip()

def _query_variations(query: str) -> List[str]:
    normalized = query.strip()
    if not normalized:
        return []

    query_lower = normalized.lower()
    variations = [normalized]

    if "news" not in query_lower:
        variations.append(f"{normalized} news")

    trusted_outlets = [
        "Reuters",
        "BBC News",
        "Al Jazeera",
        "AP News",
        "CNN",
        "The Guardian",
        "Kompas",
        "Detik",
        "Tempo",
        "The Jakarta Post",
        "Channel NewsAsia",
        "The Straits Times",
        "South China Morning Post",
        "Nikkei Asia",
        "Bloomberg",
        "Al Arabiya",
    ]

    outlet_queries = [f"{normalized} {outlet}" for outlet in trusted_outlets if outlet.lower() not in query_lower]
    variations.extend(outlet_queries[:6])

    if len(normalized) < 150 and '"' not in normalized:
        variations.append(f'"{normalized}"')

    seen = set()
    unique_variants: List[str] = []
    for variant in variations:
        cleaned = _sanitize_search_query(_normalize_search_query(variant))
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            unique_variants.append(cleaned)
    return unique_variants


def _build_search_query(query: str) -> str:
    normalized = _sanitize_search_query(_normalize_search_query(query))
    return normalized


def _extract_results(search_results, max_results: int) -> List[Dict[str, str]]:
    results = []
    if not search_results:
        return results

    for item in search_results[:max_results]:
        title = _strip_html_tags(item.get("title") or item.get("heading") or item.get("body") or "")
        snippet = _strip_html_tags(
            item.get("summary") or item.get("content") or item.get("body") or item.get("snippet") or item.get("excerpt") or title
        )
        url = item.get("link") or item.get("url") or item.get("href") or ""
        if url:
            results.append({"title": title[:120], "snippet": snippet, "url": url})
    return results


def _score_search_result(item: Dict[str, str], query: str) -> int:
    title = (item.get("title") or "").lower()
    snippet = (item.get("snippet") or "").lower()
    url = (item.get("url") or "").lower()

    score = 0

    trusted_domains = [
        "reuters.com",
        "bbc.com",
        "aljazeera.com",
        "apnews.com",
        "kompas.com",
        "detik.com",
        "cnn.com",
        "theguardian.com",
        "nytimes.com",
        "cbsnews.com",
        "npr.org",
        "straitstimes.com",
        "channelnewsasia.com",
        "scmp.com",
        "asia.nikkei.com",
        "bloomberg.com",
        "alarabiya.net",
    ]
    for trusted in trusted_domains:
        if trusted in url:
            score += 25

    if any(blocked in url for blocked in ["aliexpress", "alibaba", "amazon", "shop", "esquire"]):
        score -= 80

    query_terms = set(re.findall(r"\w+", query.lower()))
    overlap = 0
    for term in query_terms:
        if len(term) < 3:
            continue
        if term in title or term in snippet:
            overlap += 1
    score += min(overlap, 8) * 3

    return score


def _decode_yahoo_redirect(href: str) -> str:
    match = re.search(r"RU=([^/]+)", href)
    if not match:
        return ""
    decoded = unquote(match.group(1))
    return decoded


def _search_yahoo_html(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    if not query:
        return []

    url = f"https://search.yahoo.com/search?p={quote_plus(query)}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    }

    try:
        response = requests.get(url, headers=headers, timeout=SEARCH_TIMEOUT_SECONDS)
        response.raise_for_status()
        html = response.text

        results = []
        for anchor_match in re.finditer(
            r'(<a [^>]*target="_blank"[^>]*referrerpolicy="origin"[^>]*>.*?</a>)',
            html,
            re.IGNORECASE | re.DOTALL,
        ):
            anchor = anchor_match.group(1)
            href_match = re.search(r'href="([^"]+)"', anchor)
            if not href_match:
                continue

            href = href_match.group(1)
            if "r.search.yahoo.com" not in href:
                continue

            url = _decode_yahoo_redirect(href)
            if not url or "yahoo.com" in url:
                continue

            title_match = re.search(r'<h3[^>]*>(.*?)</h3>', anchor, re.IGNORECASE | re.DOTALL)
            title = _strip_html_tags(title_match.group(1)) if title_match else ""

            snippet = ""
            after_anchor = html[anchor_match.end() : anchor_match.end() + 500]
            snippet_match = re.search(r'<div[^>]*class="compText[^"]*"[^>]*>.*?<p[^>]*>(.*?)</p>', after_anchor, re.IGNORECASE | re.DOTALL)
            if snippet_match:
                snippet = _strip_html_tags(snippet_match.group(1))

            results.append({"title": title[:120], "snippet": snippet, "url": url})
            if len(results) >= max_results:
                break

        return results
    except Exception as e:
        logger.debug(f"Yahoo HTML search failed for query '{query}': {e}")
        return []


def search_web(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    if not query or not query.strip():
        return []

    normalized_query = _build_search_query(query)
    if not normalized_query:
        return []

    queries = _query_variations(normalized_query)
    all_results: List[Dict[str, str]] = []
    seen_urls = set()

    for current_query in queries:
        search_results = _search_yahoo_html(current_query, max_results=max_results * 2)
        candidate_results = _extract_results(search_results, max_results=max_results * 2)

        for item in candidate_results:
            url = item.get("url")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            item_score = _score_search_result(item, current_query)
            all_results.append({"score": item_score, **item})

    if not all_results:
        logger.debug(f"No search results for query: {normalized_query}")
        return []

    all_results.sort(key=lambda item: item["score"], reverse=True)
    ranked_results = [
        {"title": item["title"], "snippet": item["snippet"], "url": item["url"]}
        for item in all_results[:max_results]
    ]
    return ranked_results
