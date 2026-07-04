"""contradiction_map, blind_spot, judge_panel, and synthesize nodes."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from redink_core.nodes_helpers import make_model, extract_arxiv_id
from redink_core.prompts import (
    CONTRADICTION_MAP_PROMPT, BLIND_SPOT_PROMPT, DEDUP_PROMPT,
    JUDGE_LENSES, JUDGE_PANEL_PROMPT, OUTPUT_LANGUAGE,
)
from redink_core.schemas import (
    Finding, Verdict, ContradictionMap, BlindSpot, DedupMap, JudgeVote, JudgePanel,
)

_SEV_RANK = {"critical": 0, "major": 1, "minor": 2}


def contradiction_map(state, config: RunnableConfig = None):
    findings = state.get("deduped_findings") or state["findings"]
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
    findings = state.get("deduped_findings") or state["findings"]
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


def _dedup_pass(findings: list[Finding], config: RunnableConfig = None) -> list[Finding]:
    """One clustering call over a list of findings."""
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


def _dedup_findings(findings: list[Finding], config: RunnableConfig = None) -> list[Finding]:
    """Two-pass semantic dedup. A single call over 70+ findings clusters
    poorly; per-dimension passes are small enough to cluster well, and a
    final global pass catches cross-dimension duplicates."""
    if len(findings) <= 1:
        return findings

    by_dim: dict[str, list[Finding]] = {}
    for f in findings:
        by_dim.setdefault(f.dimension, []).append(f)

    with ThreadPoolExecutor(max_workers=4) as pool:
        stage1_groups = list(pool.map(lambda fs: _dedup_pass(fs, config), by_dim.values()))
    stage1 = [f for group in stage1_groups for f in group]

    return _dedup_pass(stage1, config)


def judge_panel(state, config: RunnableConfig = None):
    """Three judges with distinct lenses vote PASS/REVISE/FAIL on the
    post-debate findings. Majority wins; a three-way split lands on REVISE."""
    findings = state.get("deduped_findings") or state.get("findings", [])
    clf = state["classification"]

    paper = state.get("paper") or ""
    arxiv_id = extract_arxiv_id(paper)
    year = 2000 + int(arxiv_id[:2]) if arxiv_id and arxiv_id[:2].isdigit() else None
    year_note = (
        f"o paper foi publicado em {year}."
        if year else "o ano do paper é desconhecido — infira a época pelo conteúdo."
    )

    n_crit = sum(1 for f in findings if f.severity == "critical")
    n_maj  = sum(1 for f in findings if f.severity == "major")
    n_min  = sum(1 for f in findings if f.severity == "minor")
    sustained  = sum(1 for f in findings if f.debate_outcome == "sustained")
    downgraded = sum(1 for f in findings if f.debate_outcome == "downgraded")

    listing = "\n".join(
        f"- [{f.severity.upper()}] {f.dimension}"
        + (f" · debate: {f.debate_outcome}" if f.debate_outcome else "")
        + f" · {f.issue[:200]}"
        for f in sorted(findings, key=lambda x: _SEV_RANK[x.severity])[:40]
    )
    case = (
        f"Paper: {clf.area} — {clf.paper_type}\n"
        f"Claims centrais: {'; '.join(clf.claims[:5])}\n\n"
        f"DEBATE ADVERSARIAL: {sustained} critical(s) sustentados, "
        f"{downgraded} rebaixados para major.\n"
        f"CONTAGEM PÓS-DEBATE: {n_crit} critical · {n_maj} major · {n_min} minor\n\n"
        f"FINDINGS CONSOLIDADOS:\n{listing}\n\n"
        f"INÍCIO DO PAPER:\n{paper[:6000]}"
    )

    def _vote(lens_key: str) -> JudgeVote | None:
        # verdict-deciding call — worth a stronger model than the structurers
        model = make_model("JUDGE_MODEL", "openai/gpt-4o",
                           JudgeVote, max_tokens=800, config=config)
        try:
            v = model.invoke([
                SystemMessage(content=JUDGE_PANEL_PROMPT.format(
                    lens=JUDGE_LENSES[lens_key], year_note=year_note)),
                HumanMessage(content=case),
            ])
        except Exception:
            return None
        if isinstance(v, JudgeVote):
            v.lens = lens_key
            return v
        return None

    with ThreadPoolExecutor(max_workers=3) as pool:
        votes = [v for v in pool.map(_vote, list(JUDGE_LENSES)) if v]

    if not votes:
        return {"judge_votes": None}
    top, top_n = Counter(v.vote for v in votes).most_common(1)[0]
    verdict = top if top_n >= 2 else "REVISE"
    if verdict == "FAIL" and sustained == 0:
        # rubric backstop: FAIL requires a critical that survived the debate —
        # a majority voting FAIL over majors alone is miscalibration, not signal
        verdict = "REVISE"
    return {"judge_votes": JudgePanel(votes=votes, verdict=verdict)}


def synthesize(state, config: RunnableConfig = None):
    clf = state["classification"]
    c_map = state.get("contradiction_map")
    b_spots = state.get("blind_spots")
    panel = state.get("judge_votes")

    # debate node already deduped; fall back for direct invocations (/rerun, tests)
    findings = state.get("deduped_findings") or _dedup_findings(state["findings"], config)

    critical = [f for f in findings if f.severity == "critical"]
    major = [f for f in findings if f.severity == "major"]
    minor = [f for f in findings if f.severity == "minor"]

    high_confidence = [f.issue for f in findings if f.confidence >= 9]

    if panel:
        status = panel.verdict
    else:
        # threshold fallback when the panel didn't run
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
    if panel:
        extra += "\n\nPAINEL DE JUÍZES:\n" + "\n".join(
            f"- [{v.lens}] {v.vote}: {v.rationale}" for v in panel.votes
        )
    debated = [f for f in findings if f.debate_outcome]
    if debated:
        extra += "\n\nDEBATE (criticals contestados pela defesa do autor):\n" + "\n".join(
            f"- [{f.debate_outcome}] {f.issue[:120]}" for f in debated
        )

    model = make_model("SYNTHESIZE_MODEL", "deepseek/deepseek-v4-flash", config=config)
    summary = model.invoke([
        SystemMessage(content=(
            "Você é um meta-revisor STORM. Escreva um parágrafo de veredito integrando "
            "findings, o resultado do debate adversarial, os votos do painel de juízes, "
            "consensos e blind spots. O status já foi decidido pelo painel — justifique-o."
            + OUTPUT_LANGUAGE
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
        contradiction_map=c_map, blind_spots=b_spots, judge_panel=panel,
        high_confidence_issues=high_confidence,
    )}
