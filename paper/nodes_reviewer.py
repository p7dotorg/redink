"""reviewer and figure_reviewer nodes."""
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from paper.nodes_helpers import make_model, tool_loop, extract_arxiv_id
from paper.prompts import build_reviewer_prompt, FINDING_SCHEMA_PROMPT
from paper.schemas import Finding, FindingsList
from paper.tools import REVIEWER_TOOLS, extract_figures

_TOOL_INSTRUCTIONS = {
    "citations": (
        "Você tem 3 ferramentas:\n"
        "• search_papers(query) — busca no arXiv pelo título ou autores da referência\n"
        "• get_paper(arxiv_id) — lê o abstract de um paper arXiv para confirmar conteúdo\n"
        "• verify_doi(doi) — confirma publicação via Crossref (para papers fora do arXiv)\n\n"
        "Estratégia: verifique ≥5 referências — priorize títulos vagos ou autores desconhecidos. "
        "Para cada referência: search_papers primeiro; se não achar e houver DOI, use verify_doi.\n\n"
        "FORMATO OBRIGATÓRIO — cada finding DEVE conter o título exato verificado:\n"
        "  • VERIFICADO: [título] — abstract confirma o que o paper afirma\n"
        "  • NÃO ENCONTRADO: [título] — suspeita de citação alucinada ou pré-print não indexado\n"
        "  • MISMATCH: [título] — encontrado mas conteúdo diverge do citado\n\n"
        "NUNCA gere crítica genérica sem citar título exato. "
        "NUNCA inclua na resposta o processo de busca — apenas os resultados."
    ),
    "novelty": (
        "Você tem 3 ferramentas:\n"
        "• search_papers(query) — busca no arXiv por método, problema ou baseline\n"
        "• get_paper(arxiv_id) — lê o abstract para comparar com as claims do paper\n"
        "• verify_doi(doi) — verifica papers de conferências fora do arXiv\n\n"
        "Estratégia: faça ≥3 buscas sobre as contribuições CENTRAIS: "
        "nome do benchmark/método, problema principal, baselines usados.\n\n"
        "FORMATO dos findings:\n"
        "  • Se encontrou prior work: 'PRIOR WORK [título] já faz X → claim Y é incremental'\n"
        "  • Se NÃO encontrou: 'Busca por [query] não retornou prior work → claim parece válida'\n\n"
        "NUNCA critique metodologia geral, falta de peer review ou generalidades. "
        "NUNCA inclua na resposta tentativas ou descrições de busca — apenas os resultados."
    ),
}

_CONCISENESS = "\n\nIMPORTANTE: Máximo 4 findings, cada um com no máximo 3 frases."


def _structured_findings(
    analysis_text: str, dim: str, persona: str, config: RunnableConfig = None
) -> list[Finding]:
    structured = make_model("STRUCTURED_MODEL", "openai/gpt-4o-mini", FindingsList, max_tokens=2000, config=config)
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
        f.persona = persona
    if not findings:
        findings = [Finding(
            dimension=dim, persona=persona, severity="minor",
            issue="Análise não retornou findings estruturados.",
            evidence=analysis_text[:200],
            suggestion="Revisar manualmente esta dimensão.",
        )]
    return findings


def _reviewer_excerpt(paper: str, dim: str) -> str:
    """Truncate long papers to keep prompts manageable.

    citations: front (abstract+intro) + tail (references).
    Others: first 20k chars covers abstract + methods + results for most papers.
    """
    if len(paper) <= 20000:
        return paper
    if dim == "citations":
        return paper[:6000] + "\n\n[... body omitted ...]\n\n" + paper[-6000:]
    return paper[:20000]


def reviewer(state, config: RunnableConfig = None):
    dim = state["dimension"]
    persona = state.get("persona", "skeptic")
    clf = state["classification"]
    paper = _reviewer_excerpt(state["paper"], dim)
    system_prompt = build_reviewer_prompt(dim, persona)
    header = (
        f"Área: {clf.area} | Tipo: {clf.paper_type}\n"
        f"Claims: {'; '.join(clf.claims)}\nPersona: {persona}"
    )

    if dim == "citations":
        real_citations = [c for c in clf.citations if not c.startswith("@")]
        if not real_citations:
            if persona != "skeptic":
                return {"findings": []}
            return {"findings": [Finding(
                dimension="citations", persona="skeptic", severity="minor",
                issue="Paper não possui referências bibliográficas verificáveis.",
                evidence="Nenhuma citação acadêmica encontrada — documento não tem seção References/Bibliography.",
                suggestion="Adicione uma seção References citando os trabalhos que embasam as claims.",
                confidence=10,
            )]}

    if dim in ("citations", "novelty"):
        # Reasoning models (DeepSeek V4 Flash) return empty content in tool loops —
        # use a non-reasoning model for reliable tool calling
        real_citations = [c for c in clf.citations if not c.startswith("@")]
        model = make_model("TOOL_MODEL", "openai/gpt-4o-mini", max_tokens=4000, config=config)
        model_with_tools = model.bind_tools(REVIEWER_TOOLS)
        messages = [
            SystemMessage(content=f"{system_prompt}\n\n{_TOOL_INSTRUCTIONS[dim]}\n\n{FINDING_SCHEMA_PROMPT}{_CONCISENESS}"),
            HumanMessage(content=f"{header}\n\nCitações extraídas: {'; '.join(real_citations[:20])}\n\nPAPER:\n{paper}"),
        ]
        analysis_text = tool_loop(model_with_tools, messages, max_rounds=6)
    else:
        model = make_model("REVIEWER_MODEL", "deepseek/deepseek-v4-flash", max_tokens=3000, config=config)
        response = model.invoke([
            SystemMessage(content=system_prompt + "\n\n" + FINDING_SCHEMA_PROMPT + _CONCISENESS),
            HumanMessage(content=f"{header}\n\nPAPER:\n{paper}"),
        ])
        analysis_text = response.content

    return {"findings": _structured_findings(analysis_text, dim, persona, config)}


def figure_reviewer(state, config: RunnableConfig = None):
    """Vision node — fetches ar5iv figures and runs image analysis."""
    clf = state["classification"]
    paper = state["paper"]
    arxiv_id = extract_arxiv_id(paper)
    figures = extract_figures(arxiv_id) if arxiv_id else []

    if not figures:
        return {"findings": [Finding(
            dimension="figures", persona="skeptic", severity="minor",
            issue="Figuras não disponíveis (paper não está no ar5iv ou sem arXiv ID).",
            evidence="ar5iv retornou lista vazia.",
            suggestion="Verifique manualmente os gráficos do PDF original.",
        )]}

    system_prompt = build_reviewer_prompt("figures", "skeptic")
    vision_content = []
    for fig in figures:
        vision_content.append({"type": "image_url", "image_url": {"url": fig["url"]}})
        if fig["caption"]:
            vision_content.append({"type": "text", "text": f"[Caption: {fig['caption']}]"})
    vision_content.append({"type": "text", "text": (
        f"{system_prompt}\n\n{FINDING_SCHEMA_PROMPT}{_CONCISENESS}\n\n"
        f"Área: {clf.area} | Tipo: {clf.paper_type}\n"
        f"Claims: {'; '.join(clf.claims)}\n\n"
        f"Analise as {len(figures)} figuras. Detecte desonestidade visual, "
        "cherry-picking, ausência de barras de erro, eixos truncados, captions enganosas."
    )})

    model = make_model("FIGURE_MODEL", "google/gemini-2.5-flash", max_tokens=3000, config=config)
    analysis_text = model.invoke([HumanMessage(content=vision_content)]).content
    return {"findings": _structured_findings(analysis_text, "figures", "skeptic", config)}
