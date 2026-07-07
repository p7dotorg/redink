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


_TABLE_RE = re.compile(r"<table\b[^>]*>(.*?)</table>", re.DOTALL | re.IGNORECASE)
_ROW_RE   = re.compile(r"<tr\b[^>]*>(.*?)</tr>", re.DOTALL | re.IGNORECASE)
_CELL_RE  = re.compile(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", re.DOTALL | re.IGNORECASE)


def _cell_text(cell_html: str) -> str:
    txt = re.sub(r"<[^>]+>", " ", cell_html)
    return re.sub(r"\s+", " ", txt).strip()


def _table_to_text(m: re.Match) -> str:
    """Render a <table> as pipe-delimited rows so numeric data survives the
    tag strip — reviewers need the actual BLEU/F1/resistivity numbers, not
    a word-soup of concatenated cells."""
    rows = []
    for row_html in _ROW_RE.findall(m.group(1)):
        cells = [_cell_text(c) for c in _CELL_RE.findall(row_html)]
        if any(cells):
            rows.append("| " + " | ".join(cells) + " |")
    if not rows:
        return " "
    return "\n\n[TABLE]\n" + "\n".join(rows) + "\n[/TABLE]\n\n"


def _html_to_text(html: str) -> str:
    """Extract readable text from HTML; preserve LaTeX math as $...$ tokens
    and tables as pipe-delimited rows."""
    text = _MATH_RE.sub(_replace_math, html)
    text = _TABLE_RE.sub(_table_to_text, text)
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


# ar5iv redirects to the arXiv abstract page when it can't render a paper —
# these markers identify that we got the abstract shell, not the body.
_ABSTRACT_ONLY_MARKERS = ("View a PDF of the paper titled", "Submission history", "view email")
_ABSTRACT_NOTICE = (
    "[AVISO AO REVISOR: apenas o ABSTRACT deste paper pôde ser recuperado — "
    "o corpo completo (métodos, experimentos, tabelas, figuras) NÃO está "
    "disponível. NUNCA reporte como problema a ausência de dados, experimentos "
    "ou seções que existem no paper mas não foram recuperados aqui. Avalie "
    "apenas o que o abstract afirma.]\n\n"
)


def _looks_abstract_only(text: str) -> bool:
    """True when the fetched text is just the arXiv abstract page, not the body."""
    if len(text) >= 12000:
        return False
    return sum(mk in text for mk in _ABSTRACT_ONLY_MARKERS) >= 2


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
            if len(text) > 1000 and not _looks_abstract_only(text):
                return f"arXiv:{arxiv_id}\n\n{text}"
    except Exception:
        pass

    # Fallback: abstract page — flag it so reviewers don't fault the paper for
    # sections they never received.
    try:
        r = httpx.get(f"https://arxiv.org/abs/{arxiv_id}", timeout=15, follow_redirects=True)
        if r.status_code == 200:
            return f"arXiv:{arxiv_id}\n\n{_ABSTRACT_NOTICE}{_html_to_text(r.text)}"
    except Exception:
        pass

    return None


def fetch_paper(state, config: RunnableConfig = None):
    """Pass-through if paper is already set; else fetch from github_url (GitHub or arXiv)."""
    if state.get("paper"):
        # Re-emit so downstream consumers that read node updates (the CLI's
        # stream accumulator, the HTML annotator) see the paper for local
        # inputs — not just for fetched arXiv/GitHub ones.
        return {"paper": state["paper"]}
    url = state.get("github_url")
    if not url:
        return {}

    content = _fetch_arxiv(url) or _fetch_github_readme(url)
    return {"paper": content} if content else {}
