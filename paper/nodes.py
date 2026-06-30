"""LangGraph node implementations — STORM-enhanced multi-persona reviewer."""
import os
import re

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
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
from paper.tools import REVIEWER_TOOLS, extract_figures
from paper.paper7 import paper7_get


_ARXIV_ID_RE = re.compile(r"(?:arXiv[:/])?(\d{4}\.\d{4,5})(?:v\d+)?", re.IGNORECASE)


def _extract_arxiv_id(text: str) -> str | None:
    """Pull the first arXiv ID (YYMM.NNNNN) from paper text or metadata header."""
    m = _ARXIV_ID_RE.search(text[:2000])
    return m.group(1) if m else None


_TOOL_MAP = {t.name: t for t in REVIEWER_TOOLS}


def _tool_loop(model_with_tools, messages: list, max_rounds: int = 5) -> str:
    """Run a tool-calling loop until the model stops calling tools or max_rounds hit."""
    response = None
    for _ in range(max_rounds):
        response = model_with_tools.invoke(messages)
        messages.append(response)
        if not getattr(response, "tool_calls", None):
            break
        for tc in response.tool_calls:
            tool = _TOOL_MAP.get(tc["name"])
            result = tool.invoke(tc["args"]) if tool else f"Unknown tool: {tc['name']}"
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
    return response.content if response else ""


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


_CLASSIFY_SYSTEM = """\
Você é um especialista em análise de papers acadêmicos. Classifique com precisão.

Regras para o campo dimensions — inclua SEMPRE:
  citations, methodology, novelty, writing

Inclua também quando aplicável:
  statistics      — se houver tabelas, métricas, p-values, benchmarks ou comparações numéricas
  reproducibility — se for ML / computação / software
  ethics          — se envolver dados de pessoas ou aplicações de alto risco
  figures         — se o paper tiver gráficos, curvas de aprendizado, figuras de resultados,
                    plots de comparação ou qualquer resultado apresentado visualmente
                    (ML, CV, medicina, física experimental, economia empírica: SEMPRE inclua figures)
"""


def classify(state):
    model = _make_model("CLASSIFY_MODEL", "qwen/qwen3-8b", Classification)
    result = model.invoke([
        SystemMessage(content=_CLASSIFY_SYSTEM),
        HumanMessage(content=f"Classifique este paper:\n\n{state['paper']}"),
    ])
    return {"classification": result}


def reviewer(state):
    dim = state["dimension"]
    persona = state.get("persona", "skeptic")
    clf = state["classification"]
    paper = state["paper"]

    system_prompt = build_reviewer_prompt(dim, persona)
    conciseness = "\n\nIMPORTANTE: Máximo 4 findings, cada um com no máximo 3 frases."
    header = (
        f"Área: {clf.area} | Tipo: {clf.paper_type}\n"
        f"Claims: {'; '.join(clf.claims)}\nPersona: {persona}"
    )

    # Tool-calling dimensions — model decides what to look up
    if dim in ("citations", "novelty"):
        tool_instructions = {
            "citations": (
                "Você tem 3 ferramentas:\n"
                "• search_papers(query) — busca no arXiv pelo título ou autores da referência\n"
                "• get_paper(arxiv_id) — lê o abstract de um paper arXiv para confirmar conteúdo\n"
                "• verify_doi(doi) — confirma publicação via Crossref (para papers fora do arXiv)\n\n"
                "Estratégia: para cada referência suspeita, tente search_papers primeiro. "
                "Se não achar e a referência tiver DOI, use verify_doi. "
                "Verifique no mínimo 5 referências — priorize títulos vagos ou autores desconhecidos."
            ),
            "novelty": (
                "Você tem 3 ferramentas:\n"
                "• search_papers(query) — busca no arXiv por método, problema ou baseline\n"
                "• get_paper(arxiv_id) — lê o abstract para comparar com as claims do paper\n"
                "• verify_doi(doi) — verifica papers de conferências fora do arXiv\n\n"
                "Estratégia: faça pelo menos 3 buscas com queries diferentes "
                "(nome do método, problema central, baseline principal). "
                "Para cada resultado relevante, leia o abstract com get_paper e compare diretamente com as claims."
            ),
        }
        model = _make_model("REVIEWER_MODEL", "google/gemini-2.5-flash", max_tokens=4000)
        model_with_tools = model.bind_tools(REVIEWER_TOOLS)
        messages = [
            SystemMessage(content=f"{system_prompt}\n\n{tool_instructions[dim]}\n\n{FINDING_SCHEMA_PROMPT}{conciseness}"),
            HumanMessage(content=f"{header}\n\nCitações extraídas: {'; '.join(clf.citations[:20])}\n\nPAPER:\n{paper}"),
        ]
        analysis_text = _tool_loop(model_with_tools, messages, max_rounds=6)

    else:
        full_prompt = (
            f"{system_prompt}\n\n{FINDING_SCHEMA_PROMPT}{conciseness}\n\n"
            f"{header}\n\nPAPER:\n{paper}"
        )
        analysis_text = run_cli_reviewers(full_prompt)
        if not analysis_text:
            model = _make_model("REVIEWER_MODEL", "google/gemini-2.5-flash", max_tokens=3000)
            response = model.invoke([
                SystemMessage(content=system_prompt + "\n\n" + FINDING_SCHEMA_PROMPT + conciseness),
                HumanMessage(content=f"{header}\n\nPAPER:\n{paper}"),
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


def figure_reviewer(state):
    """Vision node — fetches ar5iv figures and runs Gemini 2.5 Flash image analysis."""
    clf = state["classification"]
    paper = state["paper"]

    arxiv_id = _extract_arxiv_id(paper)
    figures = extract_figures(arxiv_id) if arxiv_id else []

    if not figures:
        return {"findings": [Finding(
            dimension="figures", persona="skeptic", severity="minor",
            issue="Figuras não disponíveis para análise visual (paper não está no ar5iv ou não tem figuras).",
            evidence="ar5iv retornou lista vazia ou paper não tem arXiv ID.",
            suggestion="Verifique manualmente os gráficos do PDF original.",
        )]}

    system_prompt = build_reviewer_prompt("figures", "skeptic")
    conciseness = "\n\nIMPORTANTE: Máximo 4 findings, cada um com no máximo 3 frases."

    vision_content = []
    for fig in figures:
        vision_content.append({"type": "image_url", "image_url": {"url": fig["url"]}})
        if fig["caption"]:
            vision_content.append({"type": "text", "text": f"[Caption: {fig['caption']}]"})

    vision_content.append({"type": "text", "text": (
        f"{system_prompt}\n\n{FINDING_SCHEMA_PROMPT}{conciseness}\n\n"
        f"Área: {clf.area} | Tipo: {clf.paper_type}\n"
        f"Claims: {'; '.join(clf.claims)}\n\n"
        f"Analise as {len(figures)} figuras acima. Detecte desonestidade visual, "
        f"cherry-picking, ausência de barras de erro, eixos truncados, captions enganosas."
    )})

    model = _make_model("FIGURE_MODEL", "google/gemini-2.5-flash", max_tokens=3000)
    response = model.invoke([HumanMessage(content=vision_content)])
    analysis_text = response.content

    structured = _make_model("STRUCTURED_MODEL", "openai/gpt-4o-mini", FindingsList, max_tokens=4000)
    result = structured.invoke([
        SystemMessage(content=(
            "Converta a análise visual em findings estruturados. "
            "Severity: critical, major ou minor. Máximo 4 findings. "
            "O campo dimension deve ser sempre 'figures'. "
            "O campo persona deve ser sempre 'skeptic'."
        )),
        HumanMessage(content=f"Dimensão: figures\nPersona: skeptic\n\nAnálise:\n{analysis_text[:5000]}"),
    ])

    findings = result.findings if isinstance(result, FindingsList) else []
    for f in findings:
        f.persona = "skeptic"
    if not findings:
        findings = [Finding(
            dimension="figures", persona="skeptic", severity="minor",
            issue="Análise visual não retornou findings estruturados.",
            evidence=analysis_text[:200],
            suggestion="Revisar figuras manualmente.",
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
