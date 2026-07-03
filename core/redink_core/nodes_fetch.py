"""fetch_paper node — fetches paper content from a GitHub URL if paper is not set."""
import re

import httpx
from langchain_core.runnables import RunnableConfig

_GITHUB_RE = re.compile(r"https?://github\.com/([^/]+)/([^/]+?)(?:\.git)?(?:/.*)?$")
_CANDIDATES = ["README.md", "paper.md", "PAPER.md", "docs/paper.md"]


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


def fetch_paper(state, config: RunnableConfig = None):
    """Pass-through if paper is already set; else fetch from github_url."""
    if state.get("paper"):
        return {}
    github_url = state.get("github_url")
    if not github_url:
        return {}
    content = _fetch_github_readme(github_url)
    return {"paper": content} if content else {}
