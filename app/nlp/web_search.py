import logging
import re
import unicodedata
import warnings
from typing import Dict, List, Optional
from urllib.parse import quote_plus
from xml.etree import ElementTree as ET

import requests
from duckduckgo_search.duckduckgo_search import DDGS, DuckDuckGoSearchException

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


def _build_search_query(query: str) -> str:
    normalized = _sanitize_search_query(_normalize_search_query(query))
    return normalized


def _extract_results(search_results, max_results: int) -> List[Dict[str, str]]:
    results = []
    if not search_results:
        return results

    for item in search_results[:max_results]:
        title = item.get("title") or item.get("heading") or item.get("body") or ""
        snippet = item.get("summary") or item.get("content") or item.get("body") or item.get("snippet") or item.get("excerpt") or title
        url = item.get("link") or item.get("url") or item.get("href") or item.get("link") or ""
        if url:
            results.append({"title": title[:120], "snippet": snippet, "url": url})
    return results


def _search_bing_rss(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    if not query:
        return []

    url = f"https://www.bing.com/search?format=rss&q={quote_plus(query)}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    }

    try:
        response = requests.get(url, headers=headers, timeout=SEARCH_TIMEOUT_SECONDS)
        response.raise_for_status()
        root = ET.fromstring(response.text)
        items = root.findall(".//item")
        results = []

        for item in items[:max_results]:
            title = item.findtext("title") or ""
            link = item.findtext("link") or ""
            description = item.findtext("description") or ""
            results.append({"title": title, "snippet": description, "url": link})

        return results
    except Exception as e:
        logger.debug(f"Bing RSS search failed for query '{query}': {e}")
        return []


def _search_ddg(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    if not query:
        return []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        original_showwarning = warnings.showwarning
        warnings.showwarning = lambda *args, **kwargs: None
        try:
            with DDGS() as ddgs:
                return ddgs.text(query, max_results=max_results)
        except DuckDuckGoSearchException as e:
            logger.debug(f"DuckDuckGo search library failed for query '{query}': {e}")
            return []
        except Exception as e:
            logger.debug(f"Unexpected DuckDuckGo search error for query '{query}': {e}")
            return []
        finally:
            warnings.showwarning = original_showwarning


def search_web(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    if not query or not query.strip():
        return []

    normalized_query = _build_search_query(query)
    if not normalized_query:
        return []

    bing_results = _search_bing_rss(normalized_query, max_results=max_results)
    candidate_results = _extract_results(bing_results, max_results)
    if candidate_results:
        return candidate_results[:max_results]

    ddg_results = _search_ddg(normalized_query, max_results=max_results)
    candidate_results = _extract_results(ddg_results, max_results)
    if candidate_results:
        return candidate_results[:max_results]

    logger.debug(f"No search results for query: {normalized_query}")
    return []
