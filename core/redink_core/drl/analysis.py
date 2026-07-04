"""Read-side analytics over an OKF bundle. Per the OKF spec, storage/query is
a non-goal — consumers synthesize views by scanning frontmatter at read time.
That is exactly what this does: load every Dataset concept's frontmatter and
compute rankings/gaps in memory. No DB, no index files beyond OKF's own.
"""
from datetime import datetime, timedelta, timezone
from pathlib import Path

from redink_core.drl import okf

# tag namespaces that carry signal for similarity/gaps (drop library:/region:/format: noise)
_SIGNAL_NS = ("task_categories:", "task_ids:", "modality:", "language:", "size_categories:")
_TASK_NS = ("task_categories:", "task_ids:")


def _read_full(path: Path) -> dict:
    """Parse an OKF concept: frontmatter dict + raw body."""
    text = path.read_text(encoding="utf-8", errors="replace")
    fm = okf._read_frontmatter(path)
    body = ""
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            body = text[end + 4:].strip()
    # tags come back as a bracketed string from the flat parser — split it
    raw_tags = fm.get("tags", "")
    tags = []
    if raw_tags.startswith("["):
        tags = [t.strip().strip('"') for t in raw_tags[1:-1].split(",") if t.strip()]
    fm["_tags"] = tags
    fm["_body"] = body
    fm["_path"] = str(path)
    return fm


def load_datasets(root: Path = None) -> list[dict]:
    """Every Dataset concept in the bundle, with parsed tags."""
    root = root or okf.bundle_dir()
    ddir = root / "datasets"
    if not ddir.exists():
        return []
    out = []
    for p in sorted(ddir.glob("*.md")):
        fm = _read_full(p)
        if fm.get("type") == "Dataset":
            fm["_id"] = str(p.relative_to(root).with_suffix(""))
            out.append(fm)
    return out


def _signal_tags(tags: list[str]) -> set[str]:
    return {t for t in tags if t.startswith(_SIGNAL_NS)}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def rank(root: Path = None, top: int = 10) -> list[tuple[str, float, dict]]:
    """Personalized PageRank over the tag-similarity graph, biased toward
    high-opportunity datasets. Pure power iteration, no deps."""
    ds = load_datasets(root)
    n = len(ds)
    if n == 0:
        return []
    if n == 1:
        return [(ds[0].get("title") or ds[0]["_id"], 1.0, ds[0])]

    sig = [_signal_tags(d["_tags"]) for d in ds]
    # weighted adjacency from tag Jaccard
    adj = [[_jaccard(sig[i], sig[j]) if i != j else 0.0 for j in range(n)] for i in range(n)]
    row_sum = [sum(row) or 1.0 for row in adj]
    # personalization: opportunity + 1 (so a dataset with no edges still gets mass)
    pers = [float(d.get("opportunity", 0) or 0) + 1.0 for d in ds]
    ps = sum(pers)
    pers = [p / ps for p in pers]

    r = [1.0 / n] * n
    damp = 0.85
    for _ in range(60):
        nxt = [(1 - damp) * pers[i] for i in range(n)]
        for i in range(n):
            if r[i] == 0:
                continue
            share = damp * r[i]
            for j in range(n):
                if adj[i][j]:
                    nxt[j] += share * adj[i][j] / row_sum[i]
        # redistribute dangling mass via personalization
        leaked = 1.0 - sum(nxt)
        if leaked > 1e-9:
            for i in range(n):
                nxt[i] += leaked * pers[i]
        r = nxt

    ranked = sorted(zip(ds, r), key=lambda t: t[1], reverse=True)[:top]
    return [(d.get("title") or d["_id"], score, d) for d, score in ranked]


def task_gaps(root: Path = None, top: int = 10) -> list[tuple[str, int]]:
    """Task categories ranked by fewest datasets in the catalog (under-served).
    Real gap mining needs a reference task universe; this reports relative
    under-coverage within what has been scanned."""
    ds = load_datasets(root)
    coverage: dict[str, int] = {}
    for d in ds:
        for t in d["_tags"]:
            if t.startswith(_TASK_NS):
                task = t.split(":", 1)[1]
                coverage[task] = coverage.get(task, 0) + 1
    return sorted(coverage.items(), key=lambda kv: kv[1])[:top]


def spikes(root: Path = None, days: int = 7, top: int = 10) -> list[dict]:
    """Recency proxy for velocity: datasets modified within `days`, by downloads.
    True velocity needs multiple scan snapshots over time (not yet accumulated) —
    this is a first-pass 'recently active & popular' signal."""
    ds = load_datasets(root)
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    hot = []
    for d in ds:
        body = d.get("_body", "")
        # last_modified is in the Stats body line
        lm = ""
        for line in body.splitlines():
            if line.startswith("- Last modified:"):
                lm = line.split(":", 1)[1].strip()
                break
        try:
            dt = datetime.fromisoformat(lm.replace("Z", "+00:00")) if lm else None
        except ValueError:
            dt = None
        if dt and dt >= cutoff:
            hot.append(d)
    def _dl(d):
        try:
            return int(d.get("downloads", 0))
        except (ValueError, TypeError):
            return 0
    return sorted(hot, key=_dl, reverse=True)[:top]


def read_concept(slug: str, root: Path = None) -> str | None:
    """Return the raw markdown of a concept by id/slug (with or without
    'datasets/' prefix and .md suffix)."""
    root = root or okf.bundle_dir()
    candidates = [slug, f"{slug}.md", f"datasets/{slug}.md",
                  f"{slug.removesuffix('.md')}.md"]
    for c in candidates:
        p = root / c
        if p.exists():
            return p.read_text(encoding="utf-8", errors="replace")
    # fuzzy: any concept whose filename contains the slug
    for p in root.rglob("*.md"):
        if slug.lower() in p.stem.lower():
            return p.read_text(encoding="utf-8", errors="replace")
    return None
