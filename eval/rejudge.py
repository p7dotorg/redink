"""Re-judge cached redink findings with calibration anchors, cheaply.

The overlap metric showed the verdict layer is miscalibrated: redink FAILs
~80% of papers regardless of whether ICLR accepted them. The findings are fine
(recall ~0.72) — only the PASS/REVISE/FAIL call is broken, because the judges
evaluate in a vacuum ("does this have flaws?") and every real paper has flaws.

Fix under test (technique 1): give the judges a REFERENCE FRAME — a few real
papers with their finding-profiles and the verdict their human rating implies,
so the judge places the current paper on that scale instead of against an
implicit standard of perfection.

This script does NOT re-run the ~50-call pipeline. It re-runs only the judge
panel over already-cached findings (3 calls/paper), so the anchor experiment
costs minutes, not hours. It prints the baseline verdict distribution (from
cache) next to the anchored one, both against the human accept/reject.

Anchors are drawn from papers OUTSIDE the scored set (no leakage); their
finding-profiles are produced by running redink on them once (cached).

Usage (fire after the n=50 baseline finishes):
  uv run python eval/rejudge.py --in eval/data/asap_300.jsonl --n 50 --anchors 4
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "core")

from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.messages import SystemMessage, HumanMessage

from redink_core.graph import graph_runner
from redink_core.nodes_helpers import make_model

CACHE = Path("eval/data/cache")


class Vote(BaseModel):
    verdict: Literal["PASS", "REVISE", "FAIL"]
    rationale: str = Field(description="2-3 sentences")


ANCHORED_JUDGE = """You are one judge on a panel deciding a paper's verdict.

Lens: {lens}

You are given REFERENCE PAPERS with their finding-profiles and the verdict
implied by their real reviewer ratings. Use them to calibrate — place the paper
under review on the SAME scale, not against an ideal of perfection.

CALIBRATION — what the verdicts mean in practice:
- FAIL is RARE. It means a central conclusion does not hold (a sustained
  critical that survived the author's rebuttal). A long list of major
  weaknesses is NOT a FAIL — real accepted papers carry many majors.
- REVISE is the COMMON verdict: real weaknesses that are fixable and do not
  invalidate the contribution.
- PASS: solid, few addressable issues.

A paper with 0 sustained criticals and a wall of majors is almost always REVISE
(the reference accepted papers look exactly like that).

REFERENCE PAPERS:
{anchors}

Now judge the paper under review. Vote PASS / REVISE / FAIL with a rationale."""


def _profile(findings: list[dict]) -> str:
    sev = Counter(f["severity"] for f in findings)
    sustained = sum(1 for f in findings if f.get("debate_outcome") == "sustained")
    dims = sorted(set(f["dimension"] for f in findings))
    return (f"{sustained} sustained critical(s), "
            f"{sev.get('critical',0)} critical / {sev.get('major',0)} major / "
            f"{sev.get('minor',0)} minor across {', '.join(dims)}")


def _target_verdict(avg_rating: float | None) -> str:
    if avg_rating is None:
        return "REVISE"
    if avg_rating >= 6.5:
        return "PASS"
    if avg_rating < 5.0:
        return "FAIL"
    return "REVISE"


def _run_redink(rec: dict) -> dict:
    cache_f = CACHE / "redink" / f"{rec['id']}.json"
    if cache_f.exists():
        return json.loads(cache_f.read_text())
    final: dict = {}
    for chunk in graph_runner.stream(
        {"paper": rec["full_text"], "findings": [], "classification": None,
         "deduped_findings": None, "contradiction_map": None,
         "blind_spots": None, "judge_votes": None, "verdict": None},
        stream_mode="updates"):
        for _, upd in (chunk or {}).items():
            if not upd:
                continue
            for k, v in upd.items():
                if k == "findings" and isinstance(v, list):
                    final.setdefault("findings", []).extend(v)
                else:
                    final[k] = v
    verdict = final.get("verdict")
    findings = verdict.findings if verdict else final.get("findings", [])
    out = {"status": verdict.status if verdict else None,
           "findings": [{"dimension": f.dimension, "severity": f.severity,
                         "issue": f.issue, "debate_outcome": getattr(f, "debate_outcome", None)}
                        for f in findings]}
    cache_f.parent.mkdir(parents=True, exist_ok=True)
    cache_f.write_text(json.dumps(out, ensure_ascii=False))
    return out


def _build_anchors(records: list[dict], scored_ids: set[str], k: int) -> str:
    """Pick k held-out papers spanning the rating range, run redink for their
    profiles, format as few-shot anchors. k=0 → use the frozen production set."""
    if k == 0:
        from redink_core.prompts import CALIBRATION_ANCHORS
        return CALIBRATION_ANCHORS
    held = [r for r in records if r["id"] not in scored_ids and r.get("avg_rating")]
    held.sort(key=lambda r: r["avg_rating"])
    if len(held) < k:
        picks = held
    else:  # spread across the rating spectrum
        idxs = [round(i * (len(held) - 1) / (k - 1)) for i in range(k)]
        picks = [held[i] for i in idxs]
    blocks = []
    for r in picks:
        rk = _run_redink(r)
        blocks.append(
            f"- Paper (reviewer avg {r['avg_rating']}, human decision "
            f"{r['decision'].upper()}): {_profile(rk['findings'])}. "
            f"Implied verdict: {_target_verdict(r['avg_rating'])}."
        )
    return "\n".join(blocks)


LENSES = {
    "rigor": "methodological and statistical rigor — do the experiments support the central claims?",
    "contribution": "contribution vs flaws — does the core contribution survive the weaknesses?",
    "standards": "real peer-review standards of the venue and era — accept, revise, or reject?",
}


def _anchored_verdict(findings: list[dict], paper_head: str, anchors: str, model) -> str:
    sustained = sum(1 for f in findings if f.get("debate_outcome") == "sustained")
    profile = _profile(findings)
    case = (f"PAPER UNDER REVIEW\nFinding profile: {profile}\n"
            f"Sustained criticals: {sustained}\n\nPaper opening:\n{paper_head[:4000]}")
    votes = []
    for lens_key, lens in LENSES.items():
        try:
            v = model.with_structured_output(Vote).invoke([
                SystemMessage(content=ANCHORED_JUDGE.format(lens=lens, anchors=anchors)),
                HumanMessage(content=case)])
            if isinstance(v, Vote):
                votes.append(v.verdict)
        except Exception:
            pass
    if not votes:
        return "REVISE"
    top, n = Counter(votes).most_common(1)[0]
    verdict = top if n >= 2 else "REVISE"
    if verdict == "FAIL" and sustained == 0:  # same backstop as the real panel
        verdict = "REVISE"
    return verdict


def _dist_vs_human(pairs: list[tuple[str, str]]) -> None:
    """pairs = [(verdict, decision)]"""
    table: dict[str, Counter] = {}
    for verdict, decision in pairs:
        table.setdefault(verdict, Counter())[decision] += 1
    for v in ("PASS", "REVISE", "FAIL"):
        if v in table:
            c = table[v]
            print(f"    {v:7} accept={c.get('accept',0):3}  reject={c.get('reject',0):3}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--n", type=int, default=50, help="how many scored papers (must already be cached)")
    ap.add_argument("--anchors", type=int, default=4)
    args = ap.parse_args()

    records = [json.loads(l) for l in args.inp.read_text().split("\n") if l.strip()]
    scored = records[:args.n]
    scored_ids = {r["id"] for r in scored}

    # only papers whose redink pipeline is already cached
    scored = [r for r in scored if (CACHE / "redink" / f"{r['id']}.json").exists()]
    print(f"scored papers cached: {len(scored)}/{args.n}", file=sys.stderr)

    model = make_model("METRIC_MODEL", "openai/gpt-4o-mini", max_tokens=800)
    print(f"building {args.anchors} calibration anchors (held-out)...", file=sys.stderr)
    anchors = _build_anchors(records, scored_ids, args.anchors)
    print("ANCHORS:\n" + anchors + "\n", file=sys.stderr)

    base_pairs, anch_pairs = [], []
    for i, r in enumerate(scored, 1):
        rk = json.loads((CACHE / "redink" / f"{r['id']}.json").read_text())
        base_pairs.append((rk["status"], r["decision"]))
        av = _anchored_verdict(rk["findings"], r.get("full_text", ""), anchors, model)
        anch_pairs.append((av, r["decision"]))
        print(f"[{i}/{len(scored)}] {r['id']} ({r['decision']}, avg {r.get('avg_rating')}): "
              f"{rk['status']} -> {av}", file=sys.stderr)

    print("\n" + "=" * 56)
    print(f"VERDICT × HUMAN DECISION  (n={len(scored)})")
    print("=" * 56)
    print("\nBASELINE (no anchors):")
    _dist_vs_human(base_pairs)
    print("\nANCHORED:")
    _dist_vs_human(anch_pairs)


if __name__ == "__main__":
    main()
