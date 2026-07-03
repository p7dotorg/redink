"""contradiction_map, blind_spot, and synthesize nodes."""
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from redink_core.nodes_helpers import make_model
from redink_core.prompts import CONTRADICTION_MAP_PROMPT, BLIND_SPOT_PROMPT, DEDUP_PROMPT
from redink_core.schemas import Finding, Verdict, ContradictionMap, BlindSpot, DedupMap

_SEV_RANK = {"critical": 0, "major": 1, "minor": 2}


def contradiction_map(state, config: RunnableConfig = None):
    findings = state["findings"]
    clf = state["classification"]
    # Cap at 30 most severe findings to keep prompt manageable
    ranked = sorted(findings, key=lambda x: {"critical": 0, "major": 1, "minor": 2}[x.severity])[:30]
    findings_text = "\n\n".join(
        f"[{f.persona.upper()} / {f.dimension}] severity={f.severity}\n"
        f"Issue: {f.issue[:300]}\nEvidence: {f.evidence[:200]}"
        for f in ranked
    )
    # DeepSeek V4 Flash (reasoning) returns plain text for structured output — use gpt-4o-mini
    model = make_model("STRUCTURED_MODEL", "openai/gpt-4o-mini", ContradictionMap, max_tokens=6000, config=config)
    result = model.invoke([
        SystemMessage(content=CONTRADICTION_MAP_PROMPT),
        HumanMessage(content=f"Paper: {clf.area} — {clf.paper_type}\n\nFindings:\n\n{findings_text}"),
    ])
    if not isinstance(result, ContradictionMap):
        result = ContradictionMap(contradictions=[], consensus=[], most_disputed_dimension=None)
    return {"contradiction_map": result}


def blind_spot(state, config: RunnableConfig = None):
    findings = state["findings"]
    clf = state["classification"]
    covered = "\n".join(f"- [{f.persona}/{f.dimension}] {f.issue[:80]}" for f in findings)
    model = make_model("STRUCTURED_MODEL", "openai/gpt-4o-mini", BlindSpot, max_tokens=4000, config=config)
    result = model.invoke([
        SystemMessage(content=BLIND_SPOT_PROMPT.format(area=clf.area, paper_type=clf.paper_type)),
        HumanMessage(content=f"Claims: {'; '.join(clf.claims)}\n\nCobertos:\n{covered}"),
    ])
    if not isinstance(result, BlindSpot):
        result = BlindSpot(topics_not_covered=[], highest_priority=None)
    return {"blind_spots": result}


def _apply_clusters(findings: list[Finding], clusters) -> list[Finding]:
    """Collapse each cluster to one representative finding.

    Representative = the highest-severity member (ties broken toward the
    model's pick). Multi-persona clusters get confidence 9 — cross-persona
    agreement is the strongest signal the pipeline produces.
    """
    n = len(findings)
    assigned: set[int] = set()
    deduped: list[Finding] = []

    for cluster in clusters:
        members = [i for i in cluster.members if 0 <= i < n and i not in assigned]
        if not members:
            continue
        assigned.update(members)
        rep_idx = min(
            members,
            key=lambda i: (_SEV_RANK[findings[i].severity], i != cluster.representative),
        )
        rep = findings[rep_idx]
        personas = {findings[i].persona for i in members}
        if len(personas) >= 2:
            rep.confidence = 9
        deduped.append(rep)

    for i in range(n):
        if i not in assigned:
            deduped.append(findings[i])
    return deduped


def _dedup_findings(findings: list[Finding], config: RunnableConfig = None) -> list[Finding]:
    """Semantic dedup: cluster findings that report the same underlying issue
    (across personas AND dimensions) before any counting happens."""
    if len(findings) <= 1:
        return findings
    listing = "\n".join(
        f"[{i}] ({f.severity}/{f.dimension}/{f.persona}) {f.issue[:200]}"
        for i, f in enumerate(findings)
    )
    model = make_model("STRUCTURED_MODEL", "openai/gpt-4o-mini", DedupMap, max_tokens=4000, config=config)
    try:
        result = model.invoke([
            SystemMessage(content=DEDUP_PROMPT),
            HumanMessage(content=listing),
        ])
    except Exception:
        result = None
    if not isinstance(result, DedupMap) or not result.clusters:
        return findings
    return _apply_clusters(findings, result.clusters)


def synthesize(state, config: RunnableConfig = None):
    clf = state["classification"]
    c_map = state.get("contradiction_map")
    b_spots = state.get("blind_spots")

    findings = _dedup_findings(state["findings"], config)

    critical = [f for f in findings if f.severity == "critical"]
    major = [f for f in findings if f.severity == "major"]
    minor = [f for f in findings if f.severity == "minor"]

    high_confidence = [f.issue for f in findings if f.confidence >= 9]
    consensus_criticals = [f for f in critical if f.confidence >= 9]

    if len(consensus_criticals) >= 2 or len(critical) >= 3:
        status = "FAIL"
    elif critical or len(major) >= 3:
        status = "REVISE"
    else:
        status = "PASS"

    findings_text = "\n\n".join(
        f"[{f.severity.upper()}][{f.persona}] {f.dimension}: {f.issue}\n"
        f"Evidência: {f.evidence}\nSugestão: {f.suggestion}"
        for f in findings
    )
    extra = ""
    if c_map and c_map.contradictions:
        extra += "\n\nCONTRADIÇÕES:\n" + "\n".join(
            f"- {c.persona_a} vs {c.persona_b} em '{c.dimension}': {c.claim_a} ≠ {c.claim_b}"
            for c in c_map.contradictions[:3]
        )
        if c_map.consensus:
            extra += "\nCONSENSO: " + "; ".join(c_map.consensus[:3])
    if b_spots and b_spots.topics_not_covered:
        extra += "\n\nBLIND SPOTS: " + "; ".join(b_spots.topics_not_covered[:3])

    model = make_model("SYNTHESIZE_MODEL", "deepseek/deepseek-v4-flash", config=config)
    summary = model.invoke([
        SystemMessage(content=(
            "Você é um meta-revisor STORM. Escreva um parágrafo de veredito integrando "
            "findings, contradições entre personas, consensos e blind spots."
        )),
        HumanMessage(content=(
            f"Paper: {clf.area} — {clf.paper_type}\nStatus: {status}\n"
            f"Críticos: {len(critical)} | Maiores: {len(major)} | Menores: {len(minor)}\n\n"
            f"{findings_text}{extra}"
        )),
    ])

    return {"verdict": Verdict(
        status=status, summary=summary.content,
        critical_count=len(critical), major_count=len(major), minor_count=len(minor),
        findings=sorted(findings, key=lambda f: _SEV_RANK[f.severity]),
        contradiction_map=c_map, blind_spots=b_spots,
        high_confidence_issues=high_confidence,
    )}
