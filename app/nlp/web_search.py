import base64
import html as html_lib
import logging
import re
import unicodedata
from typing import Dict, List, Optional
from urllib.parse import quote_plus, unquote, urlparse

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

    source_hints = [
        "news",
        "official news",
        "press release",
        "news article",
        "media report",
        "breaking news",
    ]

    outlet_queries = [f"{normalized} {hint}" for hint in source_hints if hint not in query_lower]
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


def _has_authority_markers(text: str) -> bool:
    markers = [
        "news",
        "press",
        "media",
        "report",
        "article",
        "official",
        "broadcast",
        "statement",
        "breaking",
        "journal",
        "daily",
        "times",
        "post",
        "tribune",
        "herald",
        "gazette",
        "insider",
        "bulletin",
    ]
    return any(marker in text for marker in markers)


def _is_authoritative_url(url: str) -> bool:
    if not url:
        return False
    parsed = urlparse(url.lower())
    host = parsed.netloc or url.lower()
    if any(marker in host for marker in [
        "news",
        "press",
        "media",
        "times",
        "daily",
        "post",
        "journal",
        "broadcast",
        "tribune",
        "herald",
        "gazette",
        "insider",
        "bulletin",
    ]):
        return True
    return False


def _score_search_result(item: Dict[str, str], query: str) -> int:
    title = (item.get("title") or "").lower()
    snippet = (item.get("snippet") or "").lower()
    url = (item.get("url") or "").lower()

    score = 0
    query_terms = set(re.findall(r"\w+", query.lower()))
    overlap = 0
    for term in query_terms:
        if len(term) < 3:
            continue
        if term in title or term in snippet:
            overlap += 1

    if _is_authoritative_url(url):
        score += 18
    if _has_authority_markers(url):
        score += 8
    if _has_authority_markers(title) or _has_authority_markers(snippet):
        score += 8

    if any(blocked in url for blocked in ["aliexpress", "alibaba", "amazon", "shop", "esquire"]):
        score -= 80

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
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
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
        logger.warning("Yahoo HTML search failed for query '%s': %s", query, e)
        return []


def _decode_bing_redirect(href: str) -> str:
    match = re.search(r"u=a1([^&]+)", href)
    if not match:
        return href
    try:
        encoded = match.group(1)
        # Add padding if necessary
        encoded += "=" * ((4 - len(encoded) % 4) % 4)
        return base64.urlsafe_b64decode(encoded).decode("utf-8")
    except Exception:
        return href


def _search_bing_html(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    if not query:
        return []

    url = f"https://www.bing.com/search?q={quote_plus(query)}"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    try:
        response = requests.get(url, headers=headers, timeout=SEARCH_TIMEOUT_SECONDS)
        response.raise_for_status()
        html = response.text

        results = []
        for match in re.finditer(
            r'<h2[^>]*>\s*<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>\s*</h2>',
            html,
            re.IGNORECASE | re.DOTALL,
        ):
            href = html_lib.unescape(match.group(1))
            href = _decode_bing_redirect(href)
            title = _strip_html_tags(match.group(2))
            snippet = ""
            after_anchor = html[match.end() : match.end() + 400]
            snippet_match = re.search(r'<p[^>]*>(.*?)</p>', after_anchor, re.IGNORECASE | re.DOTALL)
            if snippet_match:
                snippet = _strip_html_tags(snippet_match.group(1))
            results.append({"title": title[:120], "snippet": snippet, "url": href})
            if len(results) >= max_results:
                break

        return results
    except Exception as e:
        logger.warning("Bing HTML search failed for query '%s': %s", query, e)
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
        search_results = _search_bing_html(current_query, max_results=max_results * 2)
        if search_results:
            logger.info("Bing returned results for query '%s'", current_query)
        else:
            logger.info("Bing returned no results for query '%s'; falling back to Yahoo", current_query)
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
        logger.info("Web search returned no results for query: %s", normalized_query)
        print(f"[WEB SEARCH] No results for query: {normalized_query}", flush=True)
        return []

    all_results.sort(key=lambda item: item["score"], reverse=True)
    ranked_results = [
        {"title": item["title"], "snippet": item["snippet"], "url": item["url"]}
        for item in all_results[:max_results]
    ]

    if ranked_results:
        logger.info("Web search results for query '%s':", normalized_query)
        print(f"[WEB SEARCH] Query: {normalized_query}", flush=True)
        for idx, item in enumerate(ranked_results, start=1):
            title = item.get("title", "(no title)")
            url = item.get("url", "")
            logger.info("  %d. %s -> %s", idx, title, url)
            print(f"[WEB SEARCH] {idx}. {title} -> {url}", flush=True)
    else:
        logger.info("Web search returned no results for query '%s'", normalized_query)
        print(f"[WEB SEARCH] No results for query: {normalized_query}")

    return ranked_results
