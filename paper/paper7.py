"""paper7 CLI integration — fetch/search arXiv papers."""
import subprocess


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
