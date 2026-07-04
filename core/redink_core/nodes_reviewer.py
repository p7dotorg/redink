"""reviewer and figure_reviewer nodes."""
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from redink_core.evidence import verify_findings
from redink_core.nodes_helpers import make_model, tool_loop, extract_arxiv_id, reviewer_excerpt
from redink_core.prompts import build_reviewer_prompt, FINDING_SCHEMA_PROMPT, OUTPUT_LANGUAGE
from redink_core.reviewer_prompts import TOOL_INSTRUCTIONS, CONCISENESS
from redink_core.schemas import Finding, FindingsList
from redink_core.tools import REVIEWER_TOOLS, NOVELTY_TOOLS, set_paper_cutoff
from redink_core.figures import extract_figures


def _structured_findings(
    analysis_text: str, dim: str, persona: str, config: RunnableConfig = None
) -> list[Finding]:
    structured = make_model(
        "STRUCTURED_MODEL", "openai/gpt-4o-mini", FindingsList, max_tokens=2000, config=config
    )
    result = structured.invoke([
        SystemMessage(content=(
            "Converta a análise em findings estruturados. Máximo 4 findings. "
            f"O campo dimension deve ser sempre '{dim}'. "
            f"O campo persona deve ser sempre '{persona}'.\n\n"
            "RUBRICA DE SEVERIDADE — aplique com rigor:\n"
            "• critical — erro que INVALIDA uma conclusão central do paper, com "
            "evidência citável do texto (quote literal no campo evidence). "
            "Ex: resultado principal sem controle, contradição factual demonstrada.\n"
            "• major — falha real que enfraquece o trabalho mas não invalida a "
            "conclusão central. Ex: ablação ausente, comparação assimétrica, "
            "métrica sem intervalo de confiança.\n"
            "• minor — melhoria de clareza, completude ou estilo.\n\n"
            "Regras: na dúvida entre dois níveis, escolha o MAIS BRANDO. "
            "Ausência de algo que pode estar em seção não mostrada NUNCA é critical. "
            "Crítica de tom/linguagem (título forte, claim otimista) é no máximo minor. "
            "O campo evidence deve conter um trecho LITERAL do paper, não paráfrase.\n\n"
            "NÃO GERE FINDING quando a análise conclui que uma claim parece "
            "original, válida, ou que a busca NÃO encontrou prior work nem "
            "problemas — isso é resultado positivo, não é problema. "
            "Retornar lista vazia é correto quando a análise não aponta problemas reais."
            + OUTPUT_LANGUAGE
        )),
        HumanMessage(content=f"Dimensão: {dim}\nPersona: {persona}\n\nAnálise:\n{analysis_text[:5000]}"),
    ])
    findings = result.findings if isinstance(result, FindingsList) else []
    for f in findings:
        f.persona = persona
    # empty is a valid outcome: the analysis found no real problems
    return findings


def reviewer(state, config: RunnableConfig = None):
    dim     = state["dimension"]
    persona = state.get("persona", "skeptic")
    clf     = state["classification"]
    set_paper_cutoff(extract_arxiv_id(state["paper"]))
    paper   = reviewer_excerpt(state["paper"], dim)
    system_prompt = build_reviewer_prompt(dim, persona)
    header  = (
        f"Área: {clf.area} | Tipo: {clf.paper_type}\n"
        f"Claims: {'; '.join(clf.claims)}\nPersona: {persona}"
    )

    # Early exit for citations with no references
    if dim == "citations":
        real_citations = [c for c in clf.citations if not c.startswith("@")]
        if not real_citations:
            if persona != "skeptic":
                return {"findings": []}
            return {"findings": [Finding(
                dimension="citations", persona="skeptic", severity="minor",
                issue="Paper não possui referências bibliográficas verificáveis.",
                evidence="Nenhuma citação acadêmica encontrada.",
                suggestion="Adicione uma seção References.",
                confidence=10,
            )]}

    if dim == "citations" and persona != "skeptic":
        real_citations = [c for c in clf.citations if not c.startswith("@")]
        model    = make_model("REVIEWER_MODEL", "deepseek/deepseek-v4-flash", max_tokens=3000, config=config)
        response = model.invoke([
            SystemMessage(content=system_prompt + "\n\n" + FINDING_SCHEMA_PROMPT + CONCISENESS),
            HumanMessage(content=(
                f"{header}\n\n"
                f"Citações listadas: {'; '.join(real_citations[:20])}\n\n"
                "Avalie: as claims centrais estão adequadamente suportadas pelas referências? "
                f"Há claims importantes sem citação?\n\nPAPER:\n{paper}"
            )),
        ])
        analysis_text = response.content

    elif dim == "novelty":
        model            = make_model("TOOL_MODEL", "openai/gpt-4o-mini", max_tokens=6000, config=config)
        model_with_tools = model.bind_tools(NOVELTY_TOOLS)
        messages = [
            SystemMessage(content=f"{system_prompt}\n\n{TOOL_INSTRUCTIONS['novelty']}\n\n{FINDING_SCHEMA_PROMPT}{CONCISENESS}"),
            HumanMessage(content=f"{header}\n\nPAPER:\n{paper}"),
        ]
        analysis_text = tool_loop(model_with_tools, messages, max_rounds=8)

    elif dim == "citations":
        real_citations   = [c for c in clf.citations if not c.startswith("@")]
        model            = make_model("TOOL_MODEL", "openai/gpt-4o-mini", max_tokens=6000, config=config)
        model_with_tools = model.bind_tools(REVIEWER_TOOLS)
        messages = [
            SystemMessage(content=f"{system_prompt}\n\n{TOOL_INSTRUCTIONS['citations']}\n\n{FINDING_SCHEMA_PROMPT}{CONCISENESS}"),
            HumanMessage(content=f"{header}\n\nCitações extraídas: {'; '.join(real_citations[:20])}\n\nPAPER:\n{paper}"),
        ]
        analysis_text = tool_loop(model_with_tools, messages, max_rounds=6)

    else:
        model    = make_model("REVIEWER_MODEL", "deepseek/deepseek-v4-flash", max_tokens=3000, config=config)
        response = model.invoke([
            SystemMessage(content=system_prompt + "\n\n" + FINDING_SCHEMA_PROMPT + CONCISENESS),
            HumanMessage(content=f"{header}\n\nPAPER:\n{paper}"),
        ])
        analysis_text = response.content

    findings = _structured_findings(analysis_text, dim, persona, config)
    return {"findings": verify_findings(findings, state["paper"])}


def figure_reviewer(state, config: RunnableConfig = None):
    """Vision node — fetches ar5iv figures and runs image analysis."""
    clf      = state["classification"]
    paper    = state["paper"]
    arxiv_id = extract_arxiv_id(paper)
    figures  = extract_figures(arxiv_id) if arxiv_id else []

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
        f"{system_prompt}\n\n{FINDING_SCHEMA_PROMPT}{CONCISENESS}\n\n"
        f"Área: {clf.area} | Tipo: {clf.paper_type}\n"
        f"Claims: {'; '.join(clf.claims)}\n\n"
        f"Analise as {len(figures)} figuras. Detecte cherry-picking, "
        "ausência de barras de erro, eixos truncados, captions enganosas."
    )})

    model         = make_model("FIGURE_MODEL", "google/gemini-2.5-flash", max_tokens=3000, config=config)
    analysis_text = model.invoke([HumanMessage(content=vision_content)]).content
    return {"findings": _structured_findings(analysis_text, "figures", "skeptic", config)}
