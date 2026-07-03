"""paper7 CLI integration + Semantic Scholar fallback for search."""
import re
import subprocess
import time
import urllib.parse

import httpx


def _paper7(args: list[str], timeout: int = 20) -> str:
    try:
        r = subprocess.run(
            ["paper7"] + args,
            capture_output=True, timeout=timeout,
        )
        return r.stdout.decode("utf-8", errors="replace")
    except (FileNotFoundError, Exception):
        return ""


def paper7_search(query: str, max_results: int = 5) -> list[dict]:
    """Search arXiv via paper7. Returns [{id, title}]."""
    output = _paper7(["search", query[:200]])
    results = []
    for line in output.splitlines():
        line = line.strip()
        if not line.startswith("["):
            continue
        try:
            arxiv_id = line.split("]")[0].strip("[")
            rest = line.split("]", 1)[1].strip()
            title = rest.split("  ")[0].strip() if "  " in rest else rest
            results.append({"id": arxiv_id, "title": title})
        except Exception:
            continue
        if len(results) >= max_results:
            break
    return results


def paper7_get(arxiv_id: str) -> str:
    """Fetch abstract and metadata for an arXiv paper."""
    return _paper7(["get", arxiv_id])


def paper7_refs(arxiv_id: str) -> str:
    """List references of a paper via Semantic Scholar."""
    return _paper7(["refs", arxiv_id])


def arxiv_api_search(query: str, max_results: int = 5) -> list[dict]:
    """Semantic Scholar API search — fallback when paper7 CLI search fails."""
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    for attempt in range(3):
        try:
            time.sleep(1.5 * (attempt + 1))  # 1.5s, 3s, 4.5s — respect S2 rate limit
            r = httpx.get(
                url,
                params={"query": query[:200], "fields": "title,year,externalIds", "limit": max_results},
                timeout=15,
                headers={"User-Agent": "redink/0.1"},
            )
            if r.status_code == 429:
                continue
            if r.status_code != 200:
                return []
            results = []
            for item in r.json().get("data", []):
                title = item.get("title", "")
                if not title:
                    continue
                arxiv_id = item.get("externalIds", {}).get("ArXiv", "")
                year = item.get("year", "")
                paper_id = arxiv_id or item.get("paperId", "")[:12]
                results.append({"id": paper_id, "title": f"{title} ({year})" if year else title})
            return results[:max_results]
        except Exception:
            return []
    return []
