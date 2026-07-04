"""Build a labeled evaluation set for redink from ASAP-Review.

ASAP-Review (Yuan et al., 2021, "Can We Automate Scientific Reviewing?",
Apache-2.0) packages ICLR 2017-2020 + NIPS 2016-2019 papers with their real
reviews, meta-reviews and accept/reject decisions. We use it to MEASURE the
harness instead of eyeballing it: for each paper we keep the full text (so
redink can review it) and the human reviewers' weaknesses + decision (the
label to score redink against).

IMPORTANT — do NOT optimize redink to predict the accept/reject decision. That
label is confounded by novelty, timing and committee politics; fitting it would
teach the harness conference politics, not rigor. The decision is kept as a
coarse signal only. The real target is overlap between redink's findings and
the weaknesses human reviewers actually raised (computed by a separate metric
step, not here). NIPS ships reviews for accepted papers only, so all rejects
come from ICLR — pull rejects from ICLR venues.

Each ASAP paper is three JSON files joined by id:
  {VENUE}_{YEAR}_{N}_paper.json    -> id, conference, decision, title, authors
  {VENUE}_{YEAR}_{N}_content.json  -> metadata.{title, abstractText, sections[]}
  {VENUE}_{YEAR}_{N}_review.json   -> reviews[{review, rating, confidence}], metaReview

Usage:
  uv run python eval/collect_asap.py --n 300 --balance --out eval/data/asap_300.jsonl
  uv run python eval/collect_asap.py --zip /path/to/dataset.zip --venues ICLR_2019,ICLR_2020
"""
import argparse
import json
import random
import re
import sys
import zipfile
from pathlib import Path

_GDRIVE_ID = "1nJdljy468roUcKLbVwWUhMs7teirah75"  # ASAP-Review dataset.zip
_RATING_RE = re.compile(r"^\s*(\d+)")
_CONS_RE = re.compile(r"(?:cons|weakness(?:es)?|con)\s*[:\-]", re.IGNORECASE)
_BULLET_RE = re.compile(r"^\s*[-*•+]\s*(.+)$")


def _ensure_zip(zip_path: Path) -> Path:
    if zip_path.exists():
        return zip_path
    try:
        import gdown
    except ImportError:
        sys.exit("gdown not installed — run: uv run --with gdown python eval/collect_asap.py ...")
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading ASAP-Review (~235MB) to {zip_path} ...", file=sys.stderr)
    gdown.download(id=_GDRIVE_ID, output=str(zip_path), quiet=False)
    return zip_path


def _normalize_decision(raw: str) -> str | None:
    low = (raw or "").lower()
    if "accept" in low:
        return "accept"
    if "reject" in low:
        return "reject"
    return None  # "Invite to Workshop", withdrawn, unknown -> skip


def _rating_int(raw: str) -> int | None:
    m = _RATING_RE.match(raw or "")
    return int(m.group(1)) if m else None


def _sections_to_text(sections: list, cap: int = 60000) -> str:
    parts = []
    for s in sections or []:
        head = (s.get("heading") or "").strip()
        body = (s.get("text") or "").strip()
        if body:
            parts.append(f"{head}\n{body}" if head else body)
    return "\n\n".join(parts)[:cap]


def _extract_weaknesses(meta_review: str) -> list[str]:
    """Best-effort weaknesses from the meta-review's Cons/Weaknesses block.
    Not exhaustive — the full review text is kept for the LLM-based overlap
    metric; this is a convenience signal."""
    if not meta_review:
        return []
    m = _CONS_RE.search(meta_review)
    if not m:
        return []
    tail = meta_review[m.end():]
    weaknesses = []
    for line in tail.splitlines():
        bm = _BULLET_RE.match(line)
        if bm:
            weaknesses.append(bm.group(1).strip())
        elif weaknesses and line.strip() and not line.strip().endswith(":"):
            weaknesses[-1] += " " + line.strip()  # continuation
    return [w for w in weaknesses if len(w) > 10]


def _load(z: zipfile.ZipFile, name: str) -> dict | None:
    try:
        with z.open(name) as f:
            return json.load(f)
    except (KeyError, json.JSONDecodeError):
        return None


def _paper_records(z: zipfile.ZipFile, venues: set[str]):
    """Yield joined records for every paper whose 3 files are present."""
    paper_files = [
        n for n in z.namelist()
        if n.endswith("_paper.json") and any(f"/{v}/" in n for v in venues)
    ]
    for pf in paper_files:
        base = pf.replace("_paper/", "_content/").replace("_paper.json", "_content.json")
        rf = pf.replace("_paper/", "_review/").replace("_paper.json", "_review.json")
        meta = _load(z, pf)
        if not meta:
            continue
        decision = _normalize_decision(meta.get("decision", ""))
        if decision is None:
            continue
        content = _load(z, base) or {}
        review = _load(z, rf) or {}
        cmeta = content.get("metadata", {})
        abstract = (cmeta.get("abstractText") or "").strip()
        full_text = _sections_to_text(cmeta.get("sections", []))
        reviews = []
        for rv in review.get("reviews", []):
            reviews.append({
                "rating": _rating_int(rv.get("rating", "")),
                "confidence": _rating_int(rv.get("confidence", "")),
                "text": (rv.get("review") or "").strip(),
            })
        if not full_text or not reviews:
            continue  # unreviewable or unparsed — useless as a label
        meta_review = (review.get("metaReview") or "").strip()
        yield {
            "id": meta.get("id"),
            "venue": meta.get("conference"),
            "decision": decision,
            "raw_decision": meta.get("decision"),
            "title": meta.get("title") or cmeta.get("title"),
            "abstract": abstract,
            "full_text": full_text,
            "reviews": reviews,
            "avg_rating": round(
                sum(r["rating"] for r in reviews if r["rating"] is not None)
                / max(1, sum(1 for r in reviews if r["rating"] is not None)), 2),
            "meta_review": meta_review,
            "weaknesses": _extract_weaknesses(meta_review),
        }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", type=Path, default=Path("eval/data/asap_dataset.zip"),
                    help="path to ASAP-Review dataset.zip (downloaded if absent)")
    ap.add_argument("--venues", default="ICLR_2018,ICLR_2019,ICLR_2020",
                    help="comma-separated venues; rejects only exist for ICLR")
    ap.add_argument("--n", type=int, default=300, help="total papers to sample")
    ap.add_argument("--balance", action="store_true",
                    help="sample 50/50 accept/reject (else natural distribution)")
    ap.add_argument("--min-fulltext", type=int, default=2000,
                    help="drop papers whose parsed text is shorter than this")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("eval/data/asap_sample.jsonl"))
    args = ap.parse_args()

    zip_path = _ensure_zip(args.zip)
    venues = {v.strip() for v in args.venues.split(",") if v.strip()}
    rng = random.Random(args.seed)

    accepts, rejects = [], []
    with zipfile.ZipFile(zip_path) as z:
        for rec in _paper_records(z, venues):
            if len(rec["full_text"]) < args.min_fulltext:
                continue
            (accepts if rec["decision"] == "accept" else rejects).append(rec)

    print(f"pool: {len(accepts)} accept · {len(rejects)} reject (venues={sorted(venues)})",
          file=sys.stderr)

    if args.balance:
        half = args.n // 2
        rng.shuffle(accepts); rng.shuffle(rejects)
        chosen = accepts[:half] + rejects[:half]
        if len(chosen) < args.n:
            print(f"warning: only {len(chosen)} papers available for balanced n={args.n}",
                  file=sys.stderr)
    else:
        pool = accepts + rejects
        rng.shuffle(pool)
        chosen = pool[:args.n]
    rng.shuffle(chosen)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for rec in chosen:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n_acc = sum(1 for r in chosen if r["decision"] == "accept")
    with_weak = sum(1 for r in chosen if r["weaknesses"])
    print(f"wrote {len(chosen)} papers to {args.out} "
          f"({n_acc} accept / {len(chosen) - n_acc} reject; "
          f"{with_weak} have extracted weaknesses)", file=sys.stderr)


if __name__ == "__main__":
    main()
