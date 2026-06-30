"""Tools for paper reviewer — each tool does exactly one thing."""
import os
import re

import httpx
from langchain_core.tools import tool

from paper.paper7 import paper7_search, paper7_get


# ---------------------------------------------------------------------------
# @tool definitions — single responsibility each
# ---------------------------------------------------------------------------

@tool
def search_papers(query: str) -> str:
    """Search arXiv for papers matching a query (title, method name, topic, or author).
    Use this to find related work, verify if a cited paper exists on arXiv,
    or discover prior work that challenges the novelty of the paper under review.
    Returns a list of arXiv IDs and titles."""
    base = os.getenv("PAPER7_API_URL", "")
    if base:
        try:
            r = httpx.get(f"{base}/api/search", params={"q": query}, timeout=10)
            if r.status_code == 200:
                results = r.json()
                if results:
                    lines = [f"[{r.get('id', r.get('arxivId', ''))}] {r.get('title', '')}" for r in results[:5]]
                    return "\n".join(lines)
        except Exception:
            pass

    results = paper7_search(query, max_results=5)
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
            headers={"User-Agent": "p7-reviewer/0.1 (mailto:review@example.com)"},
        )
        if r.status_code == 404:
            return f"DOI {doi} not found in Crossref — possible hallucinated citation."
        if r.status_code != 200:
            return f"Crossref returned {r.status_code} for DOI {doi}."
        w = r.json().get("message", {})
        title = (w.get("title") or [""])[0]
        authors = ", ".join(
            f"{a.get('given', '')} {a.get('family', '')}".strip()
            for a in w.get("author", [])[:3]
        )
        journal = (w.get("container-title") or [""])[0]
        year = (w.get("published", {}).get("date-parts") or [[""]])[0][0]
        return f"FOUND: '{title}' by {authors} — {journal} ({year})"
    except Exception as e:
        return f"Error looking up DOI: {e}"


# Tools passed to bind_tools() in reviewer nodes
REVIEWER_TOOLS = [search_papers, get_paper, verify_doi]


# ---------------------------------------------------------------------------
# Figure extraction (vision pipeline — not a LangChain tool)
# ---------------------------------------------------------------------------

_AR5IV_BASE = "https://ar5iv.labs.arxiv.org/html"


def extract_figures(arxiv_id: str, max_figures: int = 6) -> list[dict]:
    """Fetch ar5iv HTML and extract figure URLs + captions. Returns [{url, caption}]."""
    url = f"{_AR5IV_BASE}/{arxiv_id}"
    try:
        r = httpx.get(url, timeout=20, follow_redirects=True,
                      headers={"User-Agent": "p7-reviewer/0.1"})
        if r.status_code != 200:
            return []
    except Exception:
        return []

    html = r.text
    figures = []

    for fig_html in re.findall(r"<figure[^>]*>(.*?)</figure>", html, re.DOTALL | re.IGNORECASE):
        img_match = re.search(r'<img[^>]+src=["\']([^"\']+)["\']', fig_html, re.IGNORECASE)
        if not img_match:
            continue
        src = img_match.group(1)

        if src.endswith(".svg") or "icon" in src.lower():
            continue

        if src.startswith("http"):
            img_url = src
        elif src.startswith("//"):
            img_url = "https:" + src
        elif src.startswith("/"):
            img_url = "https://ar5iv.labs.arxiv.org" + src
        else:
            img_url = f"{_AR5IV_BASE}/{arxiv_id}/{src.lstrip('/')}"

        cap_match = re.search(r"<figcaption[^>]*>(.*?)</figcaption>", fig_html, re.DOTALL | re.IGNORECASE)
        caption = ""
        if cap_match:
            caption = re.sub(r"<[^>]+>", " ", cap_match.group(1)).strip()
            caption = re.sub(r"\s+", " ", caption)[:300]

        figures.append({"url": img_url, "caption": caption})
        if len(figures) >= max_figures:
            break

    return figures
