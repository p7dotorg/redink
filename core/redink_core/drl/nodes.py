"""DRL graph nodes: scan → merge → prescore → score → catalog → digest.

Mirrors the reviewer graph's shape: a source fan-out (like the persona
reviewers) and a per-dataset scoring fan-out, converging on a catalog writer
that emits OKF concepts.
"""
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from redink_core.nodes_helpers import make_model
from redink_core.drl.scanners import SCANNERS
from redink_core.drl import okf

_TASK_PREFIXES = ("task_categories:", "task_ids:")


class OpportunityScore(BaseModel):
    score: int = Field(description="Opportunity 0-3: 0 trivial/saturated, 3 high-value & underexploited")
    rationale: str = Field(description="One sentence, in English")


def scan_source(state, config: RunnableConfig = None):
    """One source, one Send. Returns its raw datasets (accumulated via add)."""
    src = state["source"]
    scanner = SCANNERS.get(src)
    if not scanner:
        return {"raw": []}
    return {"raw": scanner(state.get("query", ""), state.get("limit", 50))}


def merge(state, config: RunnableConfig = None):
    """Dedupe by (source, id), keep the richer record."""
    seen: dict[tuple, dict] = {}
    for d in state.get("raw", []):
        key = (d["source"], d["id"])
        if key not in seen or d.get("downloads", 0) > seen[key].get("downloads", 0):
            seen[key] = d
    return {"merged": list(seen.values())}


def _quality(d: dict) -> int:
    """Rule-based pre-score 0/1/2 — cheap gate before the LLM. Source-aware:
    OpenML's list endpoint carries no download/like counts, but every entry is
    a curated ML dataset, so it floors at 1 (bumped by instance count)."""
    if d["gated"] or d["private"] or d["disabled"]:
        return 0
    if d["source"] == "openml":
        try:
            return 2 if float(d.get("_instances") or 0) >= 10000 else 1
        except (ValueError, TypeError):
            return 1
    has_task = any(t.startswith(_TASK_PREFIXES) for t in d["tags"])
    if d["downloads"] >= 1000 or d["likes"] >= 5:
        return 2 if has_task else 1
    if d["downloads"] >= 50 or d["description"]:
        return 1
    return 0


def prescore(state, config: RunnableConfig = None):
    """Annotate quality and drop score-0 datasets."""
    kept = []
    for d in state.get("merged", []):
        q = _quality(d)
        if q > 0:
            kept.append({**d, "quality": q})
    kept.sort(key=lambda d: (d["quality"], d["downloads"]), reverse=True)
    return {"merged": kept}


def score_one(state, config: RunnableConfig = None):
    """LLM opportunity score for a single dataset (one Send)."""
    d = state["dataset"]
    model = make_model("DRL_SCORE_MODEL", "openai/gpt-4o-mini", OpportunityScore,
                       max_tokens=300, config=config)
    tags = ", ".join(d["tags"][:20])
    try:
        res = model.invoke([
            SystemMessage(content=(
                "You rate the OPPORTUNITY of an ML dataset for someone looking to "
                "build something valuable on top of it. 0 = trivial, saturated, or "
                "toy. 3 = high-value and underexploited (useful, real, and not yet "
                "over-served by existing work). Answer in English."
            )),
            HumanMessage(content=(
                f"Dataset: {d['title']} ({d['id']})\n"
                f"Downloads: {d['downloads']}  Likes: {d['likes']}\n"
                f"Tags: {tags}\n"
                f"Description: {d['description'][:600]}"
            )),
        ])
        score = res.score if isinstance(res, OpportunityScore) else d["quality"]
        rationale = res.rationale if isinstance(res, OpportunityScore) else ""
    except Exception:
        score, rationale = d["quality"], ""
    return {"scored": [{**d, "opportunity": score, "rationale": rationale}]}


def _dataset_body(d: dict) -> str:
    tag_lines = "\n".join(f"- `{t}`" for t in d["tags"][:30]) or "- (none)"
    return (
        f"{d['description'][:1000] or '_No description provided._'}\n\n"
        f"# Stats\n\n"
        f"- Downloads: {d['downloads']}\n- Likes: {d['likes']}\n"
        f"- Last modified: {d['last_modified']}\n"
        f"- Pre-score (quality): {d['quality']}/2\n"
        f"- Opportunity: {d['opportunity']}/3 — {d.get('rationale','')}\n\n"
        f"# Tags\n\n{tag_lines}\n\n"
        f"# Citations\n\n[1] [{d['source'].upper()} dataset page]({d['url']})\n"
    )


def catalog(state, config: RunnableConfig = None):
    """Emit an OKF concept per scored dataset, then rebuild index + log."""
    scored = state.get("scored", [])
    written = []
    for d in scored:
        cid = f"datasets/{d['source']}--{okf.slugify(d['id'])}"
        okf.write_concept(
            cid, type="Dataset", body=_dataset_body(d),
            title=d["title"], description=d["description"][:160],
            resource=d["url"], tags=d["tags"][:12],
            extra={"source": d["source"], "opportunity": d["opportunity"],
                   "downloads": d["downloads"]},
        )
        written.append(cid)
    if written:
        okf.append_log(f"scanned {state.get('query','') or 'all'} — wrote {len(written)} dataset concept(s)")
        okf.rebuild_index()
    return {"catalog": {"written": written, "count": len(written)}}


def digest(state, config: RunnableConfig = None):
    """Summarize the run as an OKF Digest concept."""
    scored = sorted(state.get("scored", []), key=lambda d: d["opportunity"], reverse=True)
    top = scored[:10]
    lines = [f"- [{d['title']}](/datasets/{d['source']}--{okf.slugify(d['id'])}.md) "
             f"— opportunity {d['opportunity']}/3" for d in top]
    body = (
        f"Scan for query `{state.get('query','') or 'all'}` across "
        f"{', '.join(state.get('sources', []))}.\n\n"
        f"- Scanned: {len(state.get('merged', []))} after pre-score\n"
        f"- Scored: {len(scored)}\n\n"
        f"# Top opportunities\n\n" + ("\n".join(lines) or "_none_") + "\n"
    )
    q = okf.slugify(state.get("query", "") or "all")
    cid = f"digests/{okf.now_iso()[:10]}-{q}"
    okf.write_concept(cid, type="Digest", body=body,
                      title=f"Scan {okf.now_iso()[:10]} — {state.get('query','') or 'all'}")
    okf.rebuild_index()
    return {"digest": cid}
