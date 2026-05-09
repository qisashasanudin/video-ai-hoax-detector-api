import html
import logging
import re
import unicodedata
import warnings
from typing import Dict, List

import requests
from duckduckgo_search.duckduckgo_search import DDGS, DuckDuckGoSearchException

logger = logging.getLogger(__name__)

MAX_SEARCH_QUERY_LENGTH = 220
SEARX_INSTANCES = [
    "https://searx.org/search",
    "https://searx.tiekoetter.com/search",
    "https://searx.privacy.computer/search",
    "https://searx.garudalinux.org/search",
]


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


def _search_ddg(query: str, max_results: int = 5):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        original_showwarning = warnings.showwarning
        warnings.showwarning = lambda *args, **kwargs: None
        try:
            with DDGS() as ddgs:
                return ddgs.text(query, max_results=max_results)
        finally:
            warnings.showwarning = original_showwarning


def _search_searx(query: str, max_results: int = 5):
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; VideoAIHoaxDetector/1.0; +https://github.com)"
    }
    params = {
        "q": query,
        "format": "json",
        "engines": "google,duckduckgo,bing",
        "language": "en-US",
        "categories": "general",
        "safesearch": 1,
        "count": max_results,
    }

    for url in SEARX_INSTANCES:
        try:
            response = requests.get(url, params=params, headers=headers, timeout=20)
            if response.status_code != 200:
                logger.warning(f"SearX endpoint {url} returned {response.status_code}")
                continue
            data = response.json()
            results = data.get("results", [])
            if results:
                return results
        except requests.exceptions.JSONDecodeError:
            logger.warning(f"SearX endpoint {url} returned invalid JSON")
        except requests.exceptions.RequestException as e:
            logger.warning(f"SearX endpoint {url} request failed: {e}")
        except Exception as e:
            logger.warning(f"SearX search failed for {url}: {e}")
    return []


def _build_search_variants(query: str) -> list[str]:
    variants = []
    normalized = _sanitize_search_query(_normalize_search_query(query))
    if normalized:
        variants.append(normalized)

    if len(normalized.split()) > 8:
        short_variant = " ".join(normalized.split()[:20])
        if short_variant and short_variant not in variants:
            variants.append(short_variant)

    stripped = normalized
    for token in [" youtube", " youtube short", " youtube shorts", " short", " shorts", " video"]:
        stripped = stripped.replace(token, "")
    stripped = " ".join(stripped.split())
    if stripped and stripped != normalized:
        variants.append(stripped)

    if " hoax" not in normalized.lower():
        variants.extend([f"{stripped} hoax", f"{stripped} fact check", f"{stripped} cek fakta", f"{stripped} hoaks"])

    # Keep unique variants while preserving order
    seen = set()
    unique_variants = []
    for v in variants:
        if v and v not in seen:
            seen.add(v)
            unique_variants.append(v)
    return unique_variants


def _search_ddg_html(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; VideoAIHoaxDetector/1.0; +https://github.com)"}
        response = requests.post("https://html.duckduckgo.com/html/", data={"q": query}, headers=headers, timeout=20)
        response.raise_for_status()
        html_content = response.text
        titles = re.findall(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', html_content, flags=re.S)
        snippets = re.findall(r'<a[^>]+class="result__snippet"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', html_content, flags=re.S)
        results = []
        for idx, (url, title_html) in enumerate(titles[:max_results]):
            title = html.unescape(re.sub(r'<[^>]+>', '', title_html).strip())
            snippet = ""
            if idx < len(snippets) and snippets[idx][0] == url:
                snippet = html.unescape(re.sub(r'<[^>]+>', '', snippets[idx][1]).strip())
            results.append({"title": title, "snippet": snippet, "url": url})
        return results
    except Exception as e:
        logger.warning(f"DuckDuckGo HTML search failed: {e}")
        return []


def _extract_results(search_results, max_results: int) -> List[Dict[str, str]]:
    results = []
    if not search_results:
        return results

    for item in search_results[:max_results]:
        title = item.get("title") or item.get("heading") or item.get("body") or ""
        snippet = item.get("content") or item.get("body") or item.get("snippet") or item.get("excerpt") or title
        url = item.get("url") or item.get("href") or item.get("link") or ""
        if url:
            results.append({"title": title[:120], "snippet": snippet, "url": url})
    return results


def search_web(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    if not query or not query.strip():
        return []

    search_results: List[Dict[str, str]] = []
    variants = _build_search_variants(query)

    for variant in variants:
        # Try SearXNG first
        searx_results = _search_searx(variant, max_results=max_results)
        candidate_results = _extract_results(searx_results, max_results)
        if candidate_results:
            return candidate_results[:max_results]

        # Next try direct DuckDuckGo HTML scraping
        html_results = _search_ddg_html(variant, max_results=max_results)
        candidate_results = _extract_results(html_results, max_results)
        if candidate_results:
            return candidate_results[:max_results]

        # Fallback to DuckDuckGo library as last resort
        try:
            ddg_results = _search_ddg(variant, max_results=max_results)
        except DuckDuckGoSearchException as e:
            logger.warning(f"DuckDuckGo search library failed for query '{variant}': {e}")
            ddg_results = []
        except Exception as e:
            logger.warning(f"Unexpected DuckDuckGo search error for query '{variant}': {e}")
            ddg_results = []

        candidate_results = _extract_results(ddg_results, max_results)
        if candidate_results:
            return candidate_results[:max_results]

    logger.info(f"No search results for query variants: {variants}")
    return []
