"""Dataset source scanners.

- HuggingFace: public API, no auth.
- Kaggle: public list endpoint works anonymously (auth only needed for
  downloads); uses KAGGLE_USERNAME/KAGGLE_KEY if present for higher limits.
- OpenML: public JSON API, no auth. Replaces Papers With Code, whose API was
  shut down (paperswithcode.com/api now redirects to Hugging Face).
"""
import os

import httpx

_HF_API = "https://huggingface.co/api/datasets"
_KAGGLE_API = "https://www.kaggle.com/api/v1/datasets/list"
_OPENML_API = "https://www.openml.org/api/v1/json/data/list"


def _normalize_hf(d: dict) -> dict:
    return {
        "source": "hf",
        "id": d.get("id", ""),
        "title": (d.get("cardData") or {}).get("pretty_name") or d.get("id", ""),
        "url": f"https://huggingface.co/datasets/{d.get('id','')}",
        "description": (d.get("description") or "").strip(),
        "tags": d.get("tags", []) or [],
        "downloads": d.get("downloads", 0) or 0,
        "likes": d.get("likes", 0) or 0,
        "last_modified": d.get("lastModified", ""),
        "gated": bool(d.get("gated")),
        "private": bool(d.get("private")),
        "disabled": bool(d.get("disabled")),
    }


def scan_hf(query: str = "", limit: int = 50) -> list[dict]:
    """Fetch datasets from HuggingFace, most-downloaded first."""
    params = {"limit": limit, "full": "true", "sort": "downloads", "direction": -1}
    if query:
        params["search"] = query
    try:
        r = httpx.get(_HF_API, params=params, timeout=25)
        if r.status_code != 200:
            return []
        return [_normalize_hf(d) for d in r.json()]
    except Exception:
        return []


def _normalize_kaggle(d: dict) -> dict:
    tags = [t.get("nameNullable") or t.get("name", "") for t in (d.get("tags") or [])]
    return {
        "source": "kaggle",
        "id": d.get("ref", ""),
        "title": d.get("title") or d.get("ref", ""),
        "url": d.get("url", ""),
        "description": (d.get("subtitle") or d.get("description") or "").strip(),
        "tags": [t for t in tags if t],
        "downloads": d.get("downloadCount", 0) or 0,
        "likes": d.get("voteCount", 0) or 0,
        "last_modified": d.get("lastUpdated", ""),
        "gated": False,
        "private": bool(d.get("isPrivate")),
        "disabled": False,
    }


def scan_kaggle(query: str = "", limit: int = 50) -> list[dict]:
    """Fetch datasets from Kaggle (anonymous list; creds used if present)."""
    params = {"pageSize": min(limit, 100), "sortBy": "hottest"}
    if query:
        params["search"] = query
    user, key = os.getenv("KAGGLE_USERNAME"), os.getenv("KAGGLE_KEY")
    auth = (user, key) if user and key else None
    try:
        r = httpx.get(_KAGGLE_API, params=params, auth=auth, timeout=25,
                      headers={"User-Agent": "drl/0.1"})
        if r.status_code != 200 or "json" not in r.headers.get("content-type", ""):
            return []
        return [_normalize_kaggle(d) for d in r.json()[:limit]]
    except Exception:
        return []


def _normalize_openml(d: dict) -> dict:
    did = d.get("did", "")
    fmt = d.get("format", "")
    tags = [f"format:{fmt.lower()}"] if fmt else []
    # size hint from qualities, if present
    quals = {q.get("name"): q.get("value") for q in (d.get("quality") or [])} \
        if isinstance(d.get("quality"), list) else {}
    return {
        "source": "openml",
        "id": str(did),
        "title": d.get("name", str(did)),
        "url": f"https://www.openml.org/d/{did}",
        "description": "",  # list endpoint is thin; enrichment is a later increment
        "tags": tags,
        "downloads": 0,
        "likes": 0,
        "last_modified": "",
        "gated": False,
        "private": d.get("status") not in ("active", None),
        "disabled": False,
        "_instances": quals.get("NumberOfInstances"),
    }


def scan_openml(query: str = "", limit: int = 50) -> list[dict]:
    """Fetch active datasets from OpenML; client-side name filter for query."""
    window = max(limit * 8, 200)
    try:
        r = httpx.get(f"{_OPENML_API}/limit/{window}/status/active", timeout=25,
                      headers={"Accept": "application/json"})
        if r.status_code != 200 or "json" not in r.headers.get("content-type", ""):
            return []
        rows = r.json().get("data", {}).get("dataset", [])
    except Exception:
        return []
    recs = [_normalize_openml(d) for d in rows]
    if query:
        q = query.lower()
        recs = [d for d in recs if q in d["title"].lower()]
    return recs[:limit]


SCANNERS = {"hf": scan_hf, "kaggle": scan_kaggle, "openml": scan_openml}
