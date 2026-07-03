"""fetch_paper node — fetches paper content from a URL if paper is not set."""
import re

import httpx
from langchain_core.runnables import RunnableConfig

_GITHUB_RE = re.compile(r"https?://github\.com/([^/]+)/([^/]+?)(?:\.git)?(?:/.*)?$")
_ARXIV_RE  = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})", re.IGNORECASE)
_CANDIDATES = ["README.md", "paper.md", "PAPER.md", "docs/paper.md"]

_STRIP_BLOCKS = re.compile(
    r"<(script|style|noscript|nav|footer|header)[^>]*>.*?</\1>",
    re.DOTALL | re.IGNORECASE,
)

_MATH_RE = re.compile(
    r'<math\b([^>]*)>.*?</math>',
    re.DOTALL | re.IGNORECASE,
)

def _replace_math(m: re.Match) -> str:
    """Swap <math> elements for their LaTeX alttext so MathJax can render it."""
    attrs = m.group(1)
    alttext_m = re.search(r'\balttext="([^"]*)"', attrs)
    if not alttext_m:
        return " "
    latex = alttext_m.group(1)
    is_block = 'display="block"' in attrs or "display='block'" in attrs
    return f" $${latex}$$ " if is_block else f" ${latex}$ "


def _html_to_text(html: str) -> str:
    """Extract readable text from HTML; preserve LaTeX math as $...$ tokens."""
    text = _MATH_RE.sub(_replace_math, html)
    text = _STRIP_BLOCKS.sub(" ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _fetch_github_readme(url: str) -> str | None:
    m = _GITHUB_RE.match(url.strip())
    if not m:
        return None
    owner, repo = m.group(1), m.group(2)
    for branch in ("main", "master"):
        for fname in _CANDIDATES:
            raw = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{fname}"
            try:
                r = httpx.get(raw, timeout=10, follow_redirects=True)
                if r.status_code == 200 and len(r.text) > 500:
                    return r.text
            except Exception:
                continue
    return None


def _fetch_arxiv(url: str) -> str | None:
    m = _ARXIV_RE.search(url)
    if not m:
        return None
    arxiv_id = m.group(1)

    # Try ar5iv for full HTML text first
    try:
        r = httpx.get(f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}", timeout=20, follow_redirects=True)
        if r.status_code == 200 and len(r.text) > 2000:
            text = _html_to_text(r.text)
            if len(text) > 1000:
                return f"arXiv:{arxiv_id}\n\n{text}"
    except Exception:
        pass

    # Fallback: abstract page
    try:
        r = httpx.get(f"https://arxiv.org/abs/{arxiv_id}", timeout=15, follow_redirects=True)
        if r.status_code == 200:
            return f"arXiv:{arxiv_id}\n\n{_html_to_text(r.text)}"
    except Exception:
        pass

    return None


def fetch_paper(state, config: RunnableConfig = None):
    """Pass-through if paper is already set; else fetch from github_url (GitHub or arXiv)."""
    if state.get("paper"):
        return {}
    url = state.get("github_url")
    if not url:
        return {}

    content = _fetch_arxiv(url) or _fetch_github_readme(url)
    return {"paper": content} if content else {}
