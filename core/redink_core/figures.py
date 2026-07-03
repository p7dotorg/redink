"""Figure extraction from ar5iv — vision pipeline (not a LangChain tool)."""
import re
import httpx

_AR5IV_BASE = "https://ar5iv.labs.arxiv.org/html"


def extract_figures(arxiv_id: str, max_figures: int = 6) -> list[dict]:
    """Fetch ar5iv HTML and extract figure URLs + captions. Returns [{url, caption}]."""
    url = f"{_AR5IV_BASE}/{arxiv_id}"
    try:
        r = httpx.get(url, timeout=20, follow_redirects=True,
                      headers={"User-Agent": "redink/0.1"})
        if r.status_code != 200:
            return []
    except Exception:
        return []

    html    = r.text
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
            caption = re.sub(r"\s+", " ", caption)
            if len(caption) > 1000:
                # mark the cut so the reviewer doesn't mistake truncation for a paper flaw
                caption = caption[:1000] + " [legenda truncada]"

        figures.append({"url": img_url, "caption": caption})
        if len(figures) >= max_figures:
            break

    return figures
