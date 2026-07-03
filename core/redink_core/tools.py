"""LangChain tools for the reviewer nodes — each does exactly one thing."""
import os

import httpx
from langchain_core.tools import tool

from redink_core.paper7 import paper7_search, paper7_get, arxiv_api_search


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
                results = r.json()
                if results:
                    return "\n".join(
                        f"[{r.get('id', r.get('arxivId', ''))}] {r.get('title', '')}"
                        for r in results[:5]
                    )
        except Exception:
            pass

    results = arxiv_api_search(query, max_results=5)
    if not results:
        results = paper7_search(query, max_results=5)
    if not results:
        return "No papers found for this query."
    return "\n".join(f"[{r['id']}] {r['title']}" for r in results)


@tool
def search_arxiv(query: str) -> str:
    """Search arXiv for CS/AI/ML papers via the paper7 CLI.
    Faster than search_papers and better for finding prior work in computer science.
    Use this for novelty checks — finding papers that already do what the paper claims.
    Returns arXiv IDs and titles."""
    results = paper7_search(query, max_results=5)
    if not results:
        results = arxiv_api_search(query, max_results=5)
    if not results:
        return "No papers found on arXiv for this query."
    return "\n".join(f"[{r['id']}] {r['title']}" for r in results)


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
