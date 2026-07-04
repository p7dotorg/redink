"""debate node — adversarial rebuttal pass over critical findings.

Every critical finding gets a defender (argues the author's side from the
paper text) and a judge (sustain / downgrade / dismiss). Criticals that only
existed because nobody pushed back die here, before they can drive the verdict.
"""
from concurrent.futures import ThreadPoolExecutor

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from redink_core.nodes_helpers import make_model
from redink_core.nodes_synthesis import _dedup_findings
from redink_core.prompts import DEFENDER_PROMPT, REBUTTAL_JUDGE_PROMPT
from redink_core.schemas import Finding, Rebuttal

_DEBATE_EXCERPT = 60000


def _debate_one(finding: Finding, paper_excerpt: str, config) -> Rebuttal | None:
    try:
        defender = make_model("REVIEWER_MODEL", "deepseek/deepseek-v4-flash",
                              max_tokens=1000, config=config)
        defense = defender.invoke([
            SystemMessage(content=DEFENDER_PROMPT),
            HumanMessage(content=(
                f"CRÍTICA ({finding.dimension}/{finding.persona}):\n{finding.issue}\n\n"
                f"EVIDÊNCIA CITADA:\n{finding.evidence}\n\n"
                f"PAPER:\n{paper_excerpt}"
            )),
        ]).content

        # nuanced call (refutes vs merely recontextualizes) — worth the stronger model
        judge = make_model("JUDGE_MODEL", "openai/gpt-4o",
                           Rebuttal, max_tokens=1000, config=config)
        ruling = judge.invoke([
            SystemMessage(content=REBUTTAL_JUDGE_PROMPT),
            HumanMessage(content=(
                f"FINDING ({finding.dimension}):\n{finding.issue}\n"
                f"Evidência: {finding.evidence}\n\n"
                f"DEFESA DO AUTOR:\n{defense}"
            )),
        ])
        if isinstance(ruling, Rebuttal):
            if not ruling.defense_summary:
                ruling.defense_summary = defense[:300]
            return ruling
    except Exception:
        pass
    return None  # debate failed → finding passes through untouched


def debate(state, config: RunnableConfig = None):
    """Dedup all findings, then put every critical through defender vs judge."""
    findings = _dedup_findings(state["findings"], config)
    criticals = [f for f in findings if f.severity == "critical"]
    if not criticals:
        return {"deduped_findings": findings}

    paper_excerpt = (state.get("paper") or "")[:_DEBATE_EXCERPT]
    with ThreadPoolExecutor(max_workers=4) as pool:
        rulings = list(pool.map(lambda f: _debate_one(f, paper_excerpt, config), criticals))
    outcome_by_id = {id(f): r for f, r in zip(criticals, rulings)}

    kept: list[Finding] = []
    for f in findings:
        ruling = outcome_by_id.get(id(f))
        if ruling is None:
            kept.append(f)  # non-critical, or debate errored → untouched
            continue
        f.defense = ruling.defense_summary[:300]
        if ruling.ruling == "dismiss":
            f.debate_outcome = "dismissed"
            continue  # judged factually wrong or already addressed — drop
        if ruling.ruling == "downgrade":
            f.severity = "major"
            f.debate_outcome = "downgraded"
        else:
            f.debate_outcome = "sustained"
            f.confidence = max(f.confidence, 8)  # survived adversarial contest
        kept.append(f)
    return {"deduped_findings": kept}
