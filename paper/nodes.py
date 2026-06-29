"""LangGraph node implementations."""
import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from paper.prompts import (
    DEFAULT_REVIEWER_PROMPT, FINDING_SCHEMA_PROMPT, REVIEWER_PROMPTS,
)
from paper.reviewers import run_cli_reviewers
from paper.schemas import Classification, Finding, FindingsList, Verdict
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
    clf = state["classification"]
    paper = state["paper"]
    system_prompt = REVIEWER_PROMPTS.get(dim, DEFAULT_REVIEWER_PROMPT)
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

    conciseness = "\n\nIMPORTANTE: Máximo 5 findings, cada um com no máximo 3 frases."
    full_prompt = (
        f"{system_prompt}\n\n{FINDING_SCHEMA_PROMPT}{conciseness}\n\n"
        f"Área: {clf.area} | Tipo: {clf.paper_type}\n"
        f"Claims: {'; '.join(clf.claims)}\n\nPAPER:\n{paper}{extra_context}"
    )

    analysis_text = run_cli_reviewers(full_prompt)
    if not analysis_text:
        model = _make_model("REVIEWER_MODEL", "google/gemini-2.5-flash", max_tokens=3000)
        response = model.invoke([
            SystemMessage(content=system_prompt + "\n\n" + FINDING_SCHEMA_PROMPT + conciseness),
            HumanMessage(content=f"Área: {clf.area} | Tipo: {clf.paper_type}\nClaims: {'; '.join(clf.claims)}\n\nPAPER:\n{paper}{extra_context}"),
        ])
        analysis_text = response.content

    structured = _make_model("STRUCTURED_MODEL", "openai/gpt-4o-mini", FindingsList, max_tokens=4000)
    result = structured.invoke([
        SystemMessage(content=(
            "Converta a análise em findings estruturados. "
            "Severity: critical, major ou minor. Máximo 5 findings. "
            f"O campo dimension deve ser sempre '{dim}'."
        )),
        HumanMessage(content=f"Dimensão: {dim}\n\nAnálise:\n{analysis_text[:5000]}"),
    ])

    findings = result.findings if isinstance(result, FindingsList) else []
    if not findings:
        findings = [Finding(
            dimension=dim, severity="minor",
            issue="Análise não retornou findings estruturados.",
            evidence=analysis_text[:200],
            suggestion="Revisar manualmente esta dimensão.",
        )]
    return {"findings": findings}


def synthesize(state):
    findings = state["findings"]
    clf = state["classification"]
    critical = [f for f in findings if f.severity == "critical"]
    major = [f for f in findings if f.severity == "major"]
    minor = [f for f in findings if f.severity == "minor"]

    if len(critical) >= 2:
        status = "FAIL"
    elif critical or len(major) >= 3:
        status = "REVISE"
    else:
        status = "PASS"

    findings_text = "\n\n".join(
        f"[{f.severity.upper()}] {f.dimension}: {f.issue}\nEvidência: {f.evidence}\nSugestão: {f.suggestion}"
        for f in findings
    )

    model = _make_model("SYNTHESIZE_MODEL", "openai/gpt-4o-mini")
    summary = model.invoke([
        SystemMessage(content="Você é um meta-revisor. Escreva um parágrafo de veredito geral."),
        HumanMessage(content=(
            f"Paper: {clf.area} — {clf.paper_type}\nStatus: {status}\n"
            f"Críticos: {len(critical)} | Maiores: {len(major)} | Menores: {len(minor)}\n\n{findings_text}"
        )),
    ])

    return {"verdict": Verdict(
        status=status,
        summary=summary.content,
        critical_count=len(critical),
        major_count=len(major),
        minor_count=len(minor),
        findings=findings,
    )}
