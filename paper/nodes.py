"""LangGraph node implementations — STORM-enhanced multi-persona reviewer."""
import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from paper.prompts import (
    build_reviewer_prompt, FINDING_SCHEMA_PROMPT,
    CONTRADICTION_MAP_PROMPT, BLIND_SPOT_PROMPT,
)
from paper.reviewers import run_cli_reviewers
from paper.schemas import (
    Classification, Finding, FindingsList, Verdict,
    ContradictionMap, BlindSpot,
)
from paper.tools import check_citation, find_related_work
from paper.paper7 import paper7_get


def _make_model(model_env_key: str, default: str, structured_schema=None, max_tokens: int = None):
    kwargs = dict(
        model=os.getenv(model_env_key, default),
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        default_headers={"HTTP-Referer": "http://localhost:2024", "X-Title": "p7-reviewer"},
        temperature=0,
    )
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    m = ChatOpenAI(**kwargs)
    return m.with_structured_output(structured_schema) if structured_schema else m


def classify(state):
    model = _make_model("CLASSIFY_MODEL", "qwen/qwen3-8b", Classification)
    result = model.invoke([
        SystemMessage(content="Você é um especialista em análise de papers acadêmicos. Classifique com precisão."),
        HumanMessage(content=f"Classifique este paper:\n\n{state['paper']}"),
    ])
    return {"classification": result}


def reviewer(state):
    dim = state["dimension"]
    persona = state.get("persona", "skeptic")
    clf = state["classification"]
    paper = state["paper"]

    system_prompt = build_reviewer_prompt(dim, persona)
    extra_context = ""

    if dim == "citations":
        results = []
        for ref in clf.citations[:15]:
            r = check_citation(ref)
            line = f"Referência: {ref}\nStatus: {r['status']} (fonte: {r.get('source','?')})\nDetalhe: {r['details']}"
            if r["status"] == "found" and r.get("source") == "arXiv":
                arxiv_id = r["details"].split("]")[0].strip("[") if "[" in r["details"] else ""
                if arxiv_id:
                    line += f"\nAbstract: {paper7_get(arxiv_id)[:400]}"
            results.append(line)
        extra_context = "\n\nResultados de citações:\n\n" + "\n\n".join(results)

    elif dim == "novelty":
        query = " ".join(clf.claims[:2])
        related = find_related_work(query)
        if related:
            details = []
            for r in related[:3]:
                arxiv_id = r.get("id", "")
                title = r.get("title", "")
                if arxiv_id:
                    abstract = paper7_get(arxiv_id)[:300]
                    details.append(f"- [{arxiv_id}] {title}\n  {abstract}")
                else:
                    details.append(f"- {title} ({r.get('year','')})")
            extra_context = "\n\nTrabalhos relacionados:\n" + "\n".join(details)

    conciseness = "\n\nIMPORTANTE: Máximo 4 findings, cada um com no máximo 3 frases."
    full_prompt = (
        f"{system_prompt}\n\n{FINDING_SCHEMA_PROMPT}{conciseness}\n\n"
        f"Área: {clf.area} | Tipo: {clf.paper_type}\n"
        f"Claims: {'; '.join(clf.claims)}\nPersona: {persona}\n\nPAPER:\n{paper}{extra_context}"
    )

    analysis_text = run_cli_reviewers(full_prompt)
    if not analysis_text:
        model = _make_model("REVIEWER_MODEL", "google/gemini-2.5-flash", max_tokens=3000)
        response = model.invoke([
            SystemMessage(content=system_prompt + "\n\n" + FINDING_SCHEMA_PROMPT + conciseness),
            HumanMessage(content=(
                f"Área: {clf.area} | Tipo: {clf.paper_type}\n"
                f"Claims: {'; '.join(clf.claims)}\nPersona: {persona}\n\n"
                f"PAPER:\n{paper}{extra_context}"
            )),
        ])
        analysis_text = response.content

    structured = _make_model("STRUCTURED_MODEL", "openai/gpt-4o-mini", FindingsList, max_tokens=4000)
    result = structured.invoke([
        SystemMessage(content=(
            "Converta a análise em findings estruturados. "
            "Severity: critical, major ou minor. Máximo 4 findings. "
            f"O campo dimension deve ser sempre '{dim}'. "
            f"O campo persona deve ser sempre '{persona}'."
        )),
        HumanMessage(content=f"Dimensão: {dim}\nPersona: {persona}\n\nAnálise:\n{analysis_text[:5000]}"),
    ])

    findings = result.findings if isinstance(result, FindingsList) else []
    for f in findings:
        f.persona = persona  # garantir que persona está correta
    if not findings:
        findings = [Finding(
            dimension=dim, persona=persona, severity="minor",
            issue="Análise não retornou findings estruturados.",
            evidence=analysis_text[:200],
            suggestion="Revisar manualmente esta dimensão.",
        )]
    return {"findings": findings}


def contradiction_map(state):
    """STORM Phase 2: mapeie onde as personas discordam."""
    findings = state["findings"]
    clf = state["classification"]

    findings_text = "\n\n".join(
        f"[{f.persona.upper()} / {f.dimension}] severity={f.severity}\n"
        f"Issue: {f.issue}\nEvidence: {f.evidence}"
        for f in findings
    )

    model = _make_model("SYNTHESIZE_MODEL", "openai/gpt-4o-mini", ContradictionMap, max_tokens=3000)
    result = model.invoke([
        SystemMessage(content=CONTRADICTION_MAP_PROMPT),
        HumanMessage(content=(
            f"Paper: {clf.area} — {clf.paper_type}\n\n"
            f"Findings de todas as personas:\n\n{findings_text}"
        )),
    ])

    if not isinstance(result, ContradictionMap):
        result = ContradictionMap(contradictions=[], consensus=[], most_disputed_dimension=None)

    # Boost confidence on consensus findings
    consensus_issues = {c.lower() for c in result.consensus}
    for f in findings:
        for c in consensus_issues:
            if any(word in f.issue.lower() for word in c.split()[:3]):
                f.confidence = 9
                break

    return {"contradiction_map": result}


def blind_spot(state):
    """STORM Phase 4 variant: o que nenhum revisor mencionou?"""
    findings = state["findings"]
    clf = state["classification"]

    covered = "\n".join(
        f"- [{f.persona}/{f.dimension}] {f.issue[:80]}"
        for f in findings
    )

    prompt = BLIND_SPOT_PROMPT.format(
        area=clf.area,
        paper_type=clf.paper_type,
    )

    model = _make_model("SYNTHESIZE_MODEL", "openai/gpt-4o-mini", BlindSpot, max_tokens=1500)
    result = model.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=(
            f"Claims do paper: {'; '.join(clf.claims)}\n\n"
            f"Tópicos já cobertos pelos revisores:\n{covered}"
        )),
    ])

    if not isinstance(result, BlindSpot):
        result = BlindSpot(topics_not_covered=[], highest_priority=None)

    return {"blind_spots": result}


def synthesize(state):
    findings = state["findings"]
    clf = state["classification"]
    c_map = state.get("contradiction_map")
    b_spots = state.get("blind_spots")

    # Severity decision: usar consenso para elevar confiança
    critical = [f for f in findings if f.severity == "critical"]
    major = [f for f in findings if f.severity == "major"]
    minor = [f for f in findings if f.severity == "minor"]

    # High-confidence issues = consensus (aparecem em 2+ personas)
    from collections import Counter
    issue_personas: dict[str, set] = {}
    for f in findings:
        key = f.dimension + ":" + f.issue[:40]
        issue_personas.setdefault(key, set()).add(f.persona)
    high_confidence = [k.split(":")[1] for k, v in issue_personas.items() if len(v) >= 2]

    # Status: critical consensuais são mais graves
    consensus_criticals = [f for f in critical if len(issue_personas.get(f.dimension + ":" + f.issue[:40], set())) >= 2]
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

    contradiction_text = ""
    if c_map and c_map.contradictions:
        contradiction_text = "\n\nCONTRADIÇÕES ENTRE PERSONAS:\n" + "\n".join(
            f"- {c.persona_a} vs {c.persona_b} em '{c.dimension}': {c.claim_a} ≠ {c.claim_b}"
            for c in c_map.contradictions[:3]
        )
        if c_map.consensus:
            contradiction_text += "\n\nCONSENSO (alta confiança): " + "; ".join(c_map.consensus[:3])

    blind_text = ""
    if b_spots and b_spots.topics_not_covered:
        blind_text = "\n\nBLIND SPOTS: " + "; ".join(b_spots.topics_not_covered[:3])

    model = _make_model("SYNTHESIZE_MODEL", "openai/gpt-4o-mini")
    summary = model.invoke([
        SystemMessage(content=(
            "Você é um meta-revisor STORM. Escreva um parágrafo de veredito integrando: "
            "findings individuais, contradições entre personas, consensos de alta confiança, "
            "e blind spots. Destaque onde as personas concordam (certeza alta) vs discordam."
        )),
        HumanMessage(content=(
            f"Paper: {clf.area} — {clf.paper_type}\nStatus: {status}\n"
            f"Críticos: {len(critical)} | Maiores: {len(major)} | Menores: {len(minor)}\n"
            f"High-confidence issues: {len(high_confidence)}\n\n"
            f"{findings_text}{contradiction_text}{blind_text}"
        )),
    ])

    # Deduplicate findings for final verdict — keep highest severity per (dimension, issue_key)
    seen: dict[str, Finding] = {}
    for f in sorted(findings, key=lambda x: {"critical": 0, "major": 1, "minor": 2}[x.severity]):
        key = f.dimension + ":" + f.issue[:40]
        if key not in seen:
            seen[key] = f
    deduped = list(seen.values())

    return {"verdict": Verdict(
        status=status,
        summary=summary.content,
        critical_count=len(critical),
        major_count=len(major),
        minor_count=len(minor),
        findings=deduped,
        contradiction_map=c_map,
        blind_spots=b_spots,
        high_confidence_issues=high_confidence,
    )}
