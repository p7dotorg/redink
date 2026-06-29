"""Citation verification via Semantic Scholar, Crossref, and arXiv."""
import time

import httpx

from paper.paper7 import paper7_search


def _search_semantic_scholar(query: str, timeout: int = 10) -> list[dict]:
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {"query": query, "limit": 3, "fields": "title,authors,year,externalIds"}
    try:
        r = httpx.get(url, params=params, timeout=timeout)
        if r.status_code == 429:
            time.sleep(2)
            r = httpx.get(url, params=params, timeout=timeout)
        if r.status_code == 200:
            return r.json().get("data", [])
    except Exception:
        pass
    return []


def _search_crossref(query: str, timeout: int = 10) -> list[dict]:
    url = "https://api.crossref.org/works"
    params = {"query": query, "rows": 3, "select": "title,author,published"}
    try:
        r = httpx.get(url, params=params, timeout=timeout,
                      headers={"User-Agent": "p7-reviewer/0.1 (mailto:review@example.com)"})
        if r.status_code == 200:
            return r.json().get("message", {}).get("items", [])
    except Exception:
        pass
    return []


def _title_match(ref_lower: str, title: str) -> bool:
    words = [w for w in title.lower().split() if len(w) > 4]
    if not words:
        return False
    return sum(1 for w in words[:6] if w in ref_lower) >= min(3, len(words))


def check_citation(reference: str) -> dict:
    """Verify a bibliographic reference exists. Returns {status, source, details}."""
    ref_lower = reference.lower()

    for r in _search_semantic_scholar(reference):
        title = r.get("title") or ""
        if _title_match(ref_lower, title):
            return {"status": "found", "source": "Semantic Scholar",
                    "details": f"'{title}' ({r.get('year', '')})"}

    for r in _search_crossref(reference):
        titles = r.get("title", [])
        title = titles[0] if titles else ""
        if _title_match(ref_lower, title):
            return {"status": "found", "source": "Crossref", "details": f"'{title}'"}

    for r in paper7_search(reference):
        if _title_match(ref_lower, r.get("title", "")):
            return {"status": "found", "source": "arXiv",
                    "details": f"[{r['id']}] {r['title']}"}

    is_old = any(str(y) in reference for y in range(1900, 2000))
    has_google = any(kw in ref_lower for kw in ["google", "ga4", "ads data hub"])
    if is_old:
        reason = "Pre-arXiv publication (book/chapter). Check Google Scholar."
    elif has_google:
        reason = "Likely Google internal tech report. Cite as technical documentation with URL."
    else:
        reason = "Not indexed in arXiv, Semantic Scholar, or Crossref. May be ACM/KDD/WSDM proceedings, tech report, or incorrect citation."

    return {"status": "not_found", "source": "none", "details": reason}


def find_related_work(query: str, limit: int = 5) -> list[dict]:
    """Find related papers via paper7 + Semantic Scholar fallback."""
    p7 = paper7_search(query, max_results=limit)
    if p7:
        return p7
    results = _search_semantic_scholar(query)
    return [
        {"id": r.get("externalIds", {}).get("ArXiv", ""),
         "title": r.get("title"),
         "year": r.get("year"),
         "authors": [a.get("name") for a in r.get("authors", [])[:3]]}
        for r in results[:limit]
    ]
