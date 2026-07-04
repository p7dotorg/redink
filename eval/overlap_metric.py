"""Measure redink against human reviewers on an ASAP-Review labeled set.

Runs redink over each paper's full text, then an LLM judge compares redink's
findings to the weaknesses the human reviewers actually raised. Produces the
first objective numbers for the harness:

  recall     — fraction of human-raised weaknesses redink also surfaced
               (does redink SEE what expert reviewers see?)
  noise_rate — fraction of redink findings the judge calls noise / wrong /
               pedantry (does redink hallucinate or nitpick?)
  novel_rate — fraction of redink findings that are legitimate concerns the
               humans simply did not raise (bonus: does it catch misses?)
  verdict×rating — does redink's PASS/REVISE/FAIL trend with the human avg rating?

A redink finding that does not match a human weakness is NOT automatically
noise — human reviewers miss things too. So the judge classifies each unmatched
finding as `plausible_unraised` vs `noise`, and we report both separately.

Every stage is cached per paper under eval/data/cache/ so re-running after a
matcher-prompt tweak does not re-run the expensive redink pipeline.

Usage:
  uv run python eval/overlap_metric.py --in eval/data/asap_300.jsonl --n 10
  uv run python eval/overlap_metric.py --in eval/data/asap_300.jsonl --n 50 --out eval/data/metric_50.jsonl

Cost warning: each paper runs the full redink pipeline (~50-70 LLM calls) plus
2 judge calls. Start small (--n 10). Papers already cached are skipped.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "core")

from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.messages import SystemMessage, HumanMessage

from redink_core.graph import graph_runner
from redink_core.nodes_helpers import make_model

CACHE = Path("eval/data/cache")


# ── LLM stage schemas ────────────────────────────────────────────────────────
class HumanWeaknesses(BaseModel):
    weaknesses: list[str] = Field(
        description="Critérios negativos distintos levantados pelos revisores — "
                    "um problema por item, sem redundância, sem elogios."
    )


class Match(BaseModel):
    human_covered: list[bool] = Field(
        description="Para cada fraqueza humana (na ordem dada), True se ALGUM "
                    "finding do redink cobre o mesmo problema subjacente."
    )
    redink_labels: list[Literal["matches_human", "plausible_unraised", "noise"]] = Field(
        description="Para cada finding do redink (na ordem dada): matches_human "
                    "se corresponde a uma fraqueza humana; plausible_unraised se "
                    "é uma preocupação legítima que os humanos não levantaram; "
                    "noise se é errado, alucinado, pedante ou trivial."
    )


# ── stage 1: run redink (cached) ─────────────────────────────────────────────
def run_redink(rec: dict) -> dict:
    cache_f = CACHE / "redink" / f"{rec['id']}.json"
    if cache_f.exists():
        return json.loads(cache_f.read_text())

    final: dict = {}
    for chunk in graph_runner.stream(
        {"paper": rec["full_text"], "findings": [], "classification": None,
         "deduped_findings": None, "contradiction_map": None,
         "blind_spots": None, "judge_votes": None, "verdict": None},
        stream_mode="updates",
    ):
        for _, update in (chunk or {}).items():
            if not update:
                continue
            for k, v in update.items():
                if k == "findings" and isinstance(v, list):
                    final.setdefault("findings", []).extend(v)
                else:
                    final[k] = v

    verdict = final.get("verdict")
    findings = verdict.findings if verdict else final.get("findings", [])
    out = {
        "status": verdict.status if verdict else None,
        "findings": [
            {"dimension": f.dimension, "severity": f.severity,
             "issue": f.issue, "debate_outcome": getattr(f, "debate_outcome", None)}
            for f in findings
        ],
    }
    cache_f.parent.mkdir(parents=True, exist_ok=True)
    cache_f.write_text(json.dumps(out, ensure_ascii=False))
    return out


# ── stage 2: extract human weaknesses (cached) ───────────────────────────────
def human_weaknesses(rec: dict, model) -> list[str]:
    cache_f = CACHE / "human" / f"{rec['id']}.json"
    if cache_f.exists():
        return json.loads(cache_f.read_text())["weaknesses"]

    review_text = "\n\n".join(
        f"[REVIEW {i+1}, rating {r.get('rating')}]\n{r['text']}"
        for i, r in enumerate(rec["reviews"])
    )
    if rec.get("meta_review"):
        review_text += f"\n\n[META-REVIEW]\n{rec['meta_review']}"

    res = model.with_structured_output(HumanWeaknesses).invoke([
        SystemMessage(content=(
            "Extraia a lista de FRAQUEZAS distintas que os revisores levantaram "
            "sobre este paper. Um problema por item. Ignore elogios, resumos e "
            "perguntas neutras. Funda itens redundantes entre revisores."
        )),
        HumanMessage(content=review_text[:16000]),
    ])
    weaknesses = res.weaknesses if isinstance(res, HumanWeaknesses) else []
    cache_f.parent.mkdir(parents=True, exist_ok=True)
    cache_f.write_text(json.dumps({"weaknesses": weaknesses}, ensure_ascii=False))
    return weaknesses


# ── stage 3: match redink findings ↔ human weaknesses (cached) ───────────────
def match(rec: dict, human: list[str], findings: list[dict], model) -> Match:
    cache_f = CACHE / "match" / f"{rec['id']}.json"
    if cache_f.exists():
        d = json.loads(cache_f.read_text())
        return Match(**d)

    h_list = "\n".join(f"[H{i}] {w}" for i, w in enumerate(human)) or "(nenhuma)"
    r_list = "\n".join(f"[R{i}] ({f['severity']}/{f['dimension']}) {f['issue']}"
                       for i, f in enumerate(findings)) or "(nenhum)"
    res = model.with_structured_output(Match).invoke([
        SystemMessage(content=(
            "Você compara as FRAQUEZAS levantadas por revisores humanos com os "
            "FINDINGS de um revisor automático sobre o mesmo paper.\n"
            "1) Para cada fraqueza humana, decida se algum finding cobre o mesmo "
            "problema subjacente (não exija wording igual).\n"
            "2) Para cada finding, rotule: matches_human (corresponde a uma "
            "fraqueza humana), plausible_unraised (preocupação legítima que os "
            "humanos não citaram) ou noise (errado, alucinado, pedante, trivial).\n"
            "human_covered deve ter exatamente o número de fraquezas humanas; "
            "redink_labels exatamente o número de findings."
        )),
        HumanMessage(content=f"FRAQUEZAS HUMANAS:\n{h_list}\n\nFINDINGS REDINK:\n{r_list}"),
    ])
    if not isinstance(res, Match):
        res = Match(human_covered=[False] * len(human),
                    redink_labels=["noise"] * len(findings))
    # guard against length drift from the model
    res.human_covered = (res.human_covered + [False] * len(human))[:len(human)]
    res.redink_labels = (res.redink_labels + ["noise"] * len(findings))[:len(findings)]
    cache_f.parent.mkdir(parents=True, exist_ok=True)
    cache_f.write_text(res.model_dump_json())
    return res


def _rating_bucket(avg: float | None) -> str | None:
    if avg is None:
        return None
    if avg >= 6.0:
        return "accept-ish"
    if avg <= 4.5:
        return "reject-ish"
    return "borderline"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", type=Path, required=True, help="asap jsonl from collect_asap.py")
    ap.add_argument("--n", type=int, default=10, help="papers to score (start small — cost)")
    ap.add_argument("--out", type=Path, default=Path("eval/data/metric_results.jsonl"))
    args = ap.parse_args()

    model = make_model("METRIC_MODEL", "openai/gpt-4o", max_tokens=2000)
    # split on "\n" only — str.splitlines() also breaks on U+2028/U+2029 which
    # json.dumps(ensure_ascii=False) leaves raw inside paper text, corrupting records
    records = [json.loads(l) for l in args.inp.read_text().split("\n") if l.strip()][:args.n]

    rows = []
    agg = {"human_total": 0, "human_covered": 0,
           "redink_total": 0, "matches": 0, "plausible": 0, "noise": 0}
    verdict_by_bucket: dict[str, dict[str, int]] = {}

    for i, rec in enumerate(records, 1):
        print(f"[{i}/{len(records)}] {rec['id']} ({rec['decision']}, "
              f"avg {rec.get('avg_rating')})...", file=sys.stderr)
        rk = run_redink(rec)
        human = human_weaknesses(rec, model)
        m = match(rec, human, rk["findings"], model)

        covered = sum(m.human_covered)
        matches = m.redink_labels.count("matches_human")
        plausible = m.redink_labels.count("plausible_unraised")
        noise = m.redink_labels.count("noise")
        n_find = len(rk["findings"])

        agg["human_total"] += len(human); agg["human_covered"] += covered
        agg["redink_total"] += n_find; agg["matches"] += matches
        agg["plausible"] += plausible; agg["noise"] += noise

        bucket = _rating_bucket(rec.get("avg_rating"))
        if bucket and rk["status"]:
            verdict_by_bucket.setdefault(bucket, {}).setdefault(rk["status"], 0)
            verdict_by_bucket[bucket][rk["status"]] += 1

        rows.append({
            "id": rec["id"], "decision": rec["decision"], "avg_rating": rec.get("avg_rating"),
            "redink_status": rk["status"], "n_findings": n_find,
            "n_human": len(human), "recall": round(covered / max(1, len(human)), 3),
            "noise": noise, "plausible": plausible, "matches": matches,
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    ht, rt = max(1, agg["human_total"]), max(1, agg["redink_total"])
    print("\n" + "=" * 64)
    print(f"OVERLAP METRIC  ({len(records)} papers)")
    print("=" * 64)
    print(f"recall      {agg['human_covered']/ht:.3f}  "
          f"({agg['human_covered']}/{agg['human_total']} human weaknesses caught)")
    print(f"noise_rate  {agg['noise']/rt:.3f}  ({agg['noise']}/{agg['redink_total']} findings)")
    print(f"novel_rate  {agg['plausible']/rt:.3f}  ({agg['plausible']}/{agg['redink_total']} legit-unraised)")
    print(f"match_rate  {agg['matches']/rt:.3f}  ({agg['matches']}/{agg['redink_total']} align w/ humans)")
    print("\nverdict × human rating:")
    for bucket in ("reject-ish", "borderline", "accept-ish"):
        dist = verdict_by_bucket.get(bucket, {})
        if dist:
            print(f"  {bucket:12} " + "  ".join(f"{k}={v}" for k, v in sorted(dist.items())))
    print(f"\nper-paper rows -> {args.out}")


if __name__ == "__main__":
    main()
