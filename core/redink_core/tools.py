"""LangChain tools for the reviewer nodes — each does exactly one thing."""
import os
import re
from contextvars import ContextVar

import httpx
from langchain_core.tools import tool

from redink_core.paper7 import paper7_search, paper7_get, arxiv_api_search

# Publication date of the paper under review (YYMM int, e.g. 1706) — search
# results dated after it are prior-work anachronisms and get filtered out.
_PAPER_YYMM: ContextVar[int | None] = ContextVar("paper_yymm", default=None)

_ARXIV_YYMM_RE = re.compile(r"^(\d{4})\.\d{4,5}")
_TITLE_YEAR_RE = re.compile(r"\((\d{4})\)\s*$")


def set_paper_cutoff(arxiv_id: str | None) -> None:
    """Set the temporal cutoff from the reviewed paper's arXiv ID (YYMM.NNNNN)."""
    if arxiv_id:
        m = _ARXIV_YYMM_RE.match(arxiv_id)
        if m:
            _PAPER_YYMM.set(int(m.group(1)))
            return
    _PAPER_YYMM.set(None)


def _is_after_cutoff(result_id: str, title: str, cutoff: int) -> bool:
    m = _ARXIV_YYMM_RE.match(result_id or "")
    if m:
        return int(m.group(1)) > cutoff
    m = _TITLE_YEAR_RE.search(title or "")
    if m:
        return int(m.group(1)) > 2000 + cutoff // 100
    return False  # undated result — keep, the model can judge


def _temporal_filter(results: list[dict]) -> tuple[list[dict], int]:
    """Drop results published after the paper under review."""
    cutoff = _PAPER_YYMM.get()
    if cutoff is None:
        return results, 0
    kept = [r for r in results
            if not _is_after_cutoff(r.get("id", ""), r.get("title", ""), cutoff)]
    return kept, len(results) - len(kept)


def _format_results(results: list[dict], dropped: int, empty_msg: str) -> str:
    lines = [f"[{r['id']}] {r['title']}" for r in results]
    if dropped:
        lines.append(
            f"({dropped} resultado(s) publicados DEPOIS do paper sob revisão "
            "foram excluídos — não são prior work)"
        )
    return "\n".join(lines) if lines else empty_msg


@tool
def search_papers(query: str) -> str:
    """Search Semantic Scholar and arXiv for papers matching a query.
    Covers CS, psychology, philosophy, medicine, and all scientific disciplines.
    Use this to verify if a cited paper exists, find related work, or discover
    prior work that challenges novelty claims. Returns paper IDs and titles."""
    base = os.getenv("PAPER7_API_URL", "")
    if base:
        try:
            r = httpx.get(f"{base}/api/search", params={"q": query}, timeout=10)
            if r.status_code == 200:
                raw = r.json()
                if raw:
                    results = [
                        {"id": item.get("id", item.get("arxivId", "")),
                         "title": item.get("title", "")}
                        for item in raw[:5]
                    ]
                    kept, dropped = _temporal_filter(results)
                    return _format_results(kept, dropped, "No papers found for this query.")
        except Exception:
            pass

    results = arxiv_api_search(query, max_results=5)
    if not results:
        results = paper7_search(query, max_results=5)
    kept, dropped = _temporal_filter(results)
    return _format_results(kept, dropped, "No papers found for this query.")


@tool
def search_arxiv(query: str) -> str:
    """Search arXiv for CS/AI/ML papers via the paper7 CLI.
    Faster than search_papers and better for finding prior work in computer science.
    Use this for novelty checks — finding papers that already do what the paper claims.
    Returns arXiv IDs and titles."""
    results = paper7_search(query, max_results=5)
    if not results:
        results = arxiv_api_search(query, max_results=5)
    kept, dropped = _temporal_filter(results)
    return _format_results(kept, dropped, "No papers found on arXiv for this query.")


@tool
def get_paper(arxiv_id: str) -> str:
    """Fetch the abstract and metadata of an arXiv paper by its ID (e.g. '2303.08774').
    Use this after search_papers to read the full abstract and compare claims,
    methods, or results with the paper under review."""
    text = paper7_get(arxiv_id)
    if not text or not text.strip():
        return f"Paper {arxiv_id} not found on arXiv."
    return text[:800]


@tool
def verify_doi(doi: str) -> str:
    """Look up a paper by its DOI in the Crossref database.
    Use this to verify citations that are NOT on arXiv (Nature, ACM, IEEE, books, etc.)
    or to confirm publication details (journal, year, authors) for any reference.
    DOI format: '10.XXXX/...' — strip any 'https://doi.org/' prefix first."""
    doi = doi.strip().removeprefix("https://doi.org/").removeprefix("http://doi.org/")
    try:
        r = httpx.get(
            f"https://api.crossref.org/works/{doi}",
            timeout=10,
            headers={"User-Agent": "redink/0.1 (mailto:review@example.com)"},
        )
        if r.status_code == 404:
            return f"DOI {doi} not found in Crossref — possible hallucinated citation."
        if r.status_code != 200:
            return f"Crossref returned {r.status_code} for DOI {doi}."
        w       = r.json().get("message", {})
        title   = (w.get("title") or [""])[0]
        authors = ", ".join(
            f"{a.get('given', '')} {a.get('family', '')}".strip()
            for a in w.get("author", [])[:3]
        )
        journal = (w.get("container-title") or [""])[0]
        year    = (w.get("published", {}).get("date-parts") or [[""]])[0][0]
        return f"FOUND: '{title}' by {authors} — {journal} ({year})"
    except Exception as e:
        return f"Error looking up DOI: {e}"


# Tool lists passed to bind_tools() in reviewer nodes
REVIEWER_TOOLS = [search_papers, get_paper, verify_doi]
NOVELTY_TOOLS  = [search_arxiv, get_paper]
