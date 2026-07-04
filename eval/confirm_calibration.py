"""Confirm the PRODUCTION judge_panel reproduces the calibrated verdict.

The overlap metric caches the pipeline verdict per paper — those caches predate
the anchor promotion, so re-running it would show stale verdicts. Instead this
re-runs the ACTUAL production judge_panel node (prompts.JUDGE_PANEL_PROMPT +
CALIBRATION_ANCHORS + the FAIL-needs-sustained backstop) over the cached
findings, and tallies verdict × human decision.

Findings are unchanged by the anchor promotion (only the judge changed), so
recall/noise are unaffected; this isolates the verdict.

  uv run python eval/confirm_calibration.py --n 50
"""
import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "core")

from dotenv import load_dotenv
load_dotenv(".env")  # importing judge_panel directly bypasses graph.py's load
os.environ["JUDGE_MODEL"] = "openai/gpt-4o-mini"  # cheap confirmation

from redink_core.nodes_synthesis import judge_panel
from redink_core.schemas import Finding, Classification

CACHE = Path("eval/data/cache")


def _reconstruct(rec: dict, cached: dict) -> dict:
    findings = [
        Finding(
            dimension=f["dimension"], persona="skeptic", severity=f["severity"],
            issue=f["issue"], evidence="", suggestion="",
            debate_outcome=f.get("debate_outcome"),
        )
        for f in cached["findings"]
    ]
    clf = Classification(
        area="Machine Learning", paper_type="empirical",
        dimensions=["methodology"], citations=[],
        claims=[l.strip() for l in rec["full_text"].splitlines() if l.strip()][:5] or ["-"],
    )
    return {"deduped_findings": findings, "classification": clf, "paper": rec["full_text"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", type=Path, default=Path("eval/data/asap_300.jsonl"))
    ap.add_argument("--n", type=int, default=50)
    args = ap.parse_args()

    recs = {json.loads(l)["id"]: json.loads(l)
            for l in args.inp.read_text().split("\n") if l.strip()}
    ids = [Path(p).stem for p in (CACHE / "redink").glob("*.json")][:args.n]

    vb = {}
    for i, pid in enumerate(ids, 1):
        rec = recs.get(pid)
        if not rec:
            continue
        cached = json.loads((CACHE / "redink" / f"{pid}.json").read_text())
        out = judge_panel(_reconstruct(rec, cached))
        panel = out.get("judge_votes")
        verdict = panel.verdict if panel else "?"
        dec = rec["decision"]
        vb.setdefault(verdict, Counter())[dec] += 1
        print(f"[{i}/{len(ids)}] {pid} ({dec}, avg {rec.get('avg_rating')}): {verdict}",
              file=sys.stderr)

    n = sum(sum(c.values()) for c in vb.values())
    print("\n" + "=" * 52)
    print(f"PRODUCTION judge_panel × human decision (n={n})")
    print("=" * 52)
    for v in ("PASS", "REVISE", "FAIL"):
        if v in vb:
            print(f"  {v:7} accept={vb[v].get('accept',0):2}  reject={vb[v].get('reject',0):2}")
    fail_acc = vb.get("FAIL", Counter()).get("accept", 0)
    acc = sum(c.get("accept", 0) for c in vb.values())
    print(f"\n  accepted papers FAILed: {fail_acc}/{acc}")


if __name__ == "__main__":
    main()
