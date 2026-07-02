"""reviewer and figure_reviewer nodes."""
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from paper.nodes_helpers import make_model, tool_loop, extract_arxiv_id
from paper.prompts import build_reviewer_prompt, FINDING_SCHEMA_PROMPT
from paper.schemas import Finding, FindingsList
from paper.tools import REVIEWER_TOOLS, NOVELTY_TOOLS, extract_figures

_TOOL_INSTRUCTIONS = {
    "citations": (
        "Você tem 3 ferramentas:\n"
        "• search_papers(query) — busca no Semantic Scholar (cobertura ampla: CS, psicologia, filosofia, medicina)\n"
        "• get_paper(arxiv_id) — lê abstract de paper arXiv (use se resultado tiver ID arXiv)\n"
        "• verify_doi(doi) — confirma via Crossref (use se DOI estiver explícito no texto)\n\n"
        "PASSO 0 — OBRIGATÓRIO antes de qualquer busca:\n"
        "  Localize a seção 'References' no texto do paper abaixo.\n"
        "  Para cada citação curta (ex: 'Flavell (1979)'), encontre a entrada COMPLETA na seção References.\n"
        "  Ex: 'Flavell [1979] John H Flavell. Metacognition and cognitive monitoring: A new area...\n"
        "  American psychologist, 34(10):906'\n"
        "  Use SEMPRE palavras-chave do título completo na busca — NUNCA busque 'Autor (ano)' isolado.\n\n"
        "REGRA DE OURO — referência COM periódico + volume + página = quase certamente REAL:\n"
        "  Se a entrada na seção References tem formato 'Título. Periódico, vol(num):pp, ano',\n"
        "  a citação é provavelmente válida. Use search_papers apenas para confirmar se o conteúdo\n"
        "  bate com o claim, não para provar existência.\n"
        "  Foque buscas em citações SEM detalhes (sem periódico, sem página, só 'Autor, ano').\n\n"
        "QUANDO marcar NÃO ENCONTRADO — APENAS se:\n"
        "  (a) A entrada na seção References está incompleta (sem título ou periódico) E\n"
        "  (b) Semantic Scholar não retorna o paper após 2 buscas com palavras do título E\n"
        "  (c) O autor ou título parece implausível (genérico demais, inconsistente com a área)\n"
        "  Papers de psicologia/filosofia/medicina pré-2000 são reais mas raramente estão no arXiv.\n\n"
        "FORMATO OBRIGATÓRIO — cada finding DEVE conter título completo verificado:\n"
        "  • VERIFICADO: [título] — encontrado (ou referência completa com periódico/página confirma existência)\n"
        "  • MISMATCH: [título] — encontrado mas conteúdo diverge do que o paper afirma\n"
        "  • NÃO ENCONTRADO: [título] — incompleto na referência E ausente no Semantic Scholar após ≥2 buscas\n\n"
        "NUNCA critique citação genérica sem título exato. "
        "NUNCA marque clássicos pré-2000 com periódico+página como suspeitos. "
        "NUNCA inclua processo de busca — apenas resultados."
    ),
    "novelty": (
        "Você tem 2 ferramentas:\n"
        "• search_arxiv(query) — busca no arXiv via paper7 CLI (rápido, cobertura completa de CS/AI/ML)\n"
        "• get_paper(arxiv_id) — lê o abstract para comparar com as claims do paper\n\n"
        "Estratégia: para cada claim central do paper, faça ≥1 busca específica.\n"
        "Use queries técnicas com verbos e conceitos precisos — NÃO use o título do paper como query.\n"
        "Ex: para 'memory management como skill treinável em LLMs' → busque:\n"
        "  'LLM agent memory optimization fine-tuning'\n"
        "  'automated scaffold revision LLM agent'\n"
        "  'meta-learning memory management language model'\n\n"
        "FORMATO OBRIGATÓRIO dos findings:\n"
        "  • PRIOR WORK: [título arXiv:XXXX] já faz X → claim Y do paper é incremental ou redundante\n"
        "  • NÃO ENCONTRADO: busca por [query] não retornou prior work → claim parece original\n\n"
        "NUNCA critique metodologia geral, falta de peer review ou generalidades. "
        "NUNCA inclua tentativas de busca — apenas resultados com título e arXiv ID. "
        "NUNCA marque como incremental sem citar paper específico encontrado."
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

    if dim == "citations" and persona != "skeptic":
        # Practitioner and academic do contextual analysis without S2 tool calls —
        # running 3 parallel tool-calling agents hammers S2 rate limit (1 req/sec)
        # and causes all searches to fail, producing generic off-topic findings.
        # Skeptic does the web verification; others judge claim-support quality.
        real_citations = [c for c in clf.citations if not c.startswith("@")]
        model = make_model("REVIEWER_MODEL", "deepseek/deepseek-v4-flash", max_tokens=3000, config=config)
        response = model.invoke([
            SystemMessage(content=system_prompt + "\n\n" + FINDING_SCHEMA_PROMPT + _CONCISENESS),
            HumanMessage(content=(
                f"{header}\n\n"
                f"Citações listadas: {'; '.join(real_citations[:20])}\n\n"
                "Avalie: as claims centrais do paper estão adequadamente suportadas pelas referências citadas? "
                "Há claims importantes sem citação? As referências são relevantes para o que afirmam?\n\n"
                f"PAPER:\n{paper}"
            )),
        ])
        analysis_text = response.content
    elif dim == "novelty":
        tools = NOVELTY_TOOLS  # paper7/arXiv — faster, full CS coverage, no S2 rate limit
        model = make_model("TOOL_MODEL", "openai/gpt-4o-mini", max_tokens=6000, config=config)
        model_with_tools = model.bind_tools(tools)
        messages = [
            SystemMessage(content=f"{system_prompt}\n\n{_TOOL_INSTRUCTIONS['novelty']}\n\n{FINDING_SCHEMA_PROMPT}{_CONCISENESS}"),
            HumanMessage(content=f"{header}\n\nPAPER:\n{paper}"),
        ]
        analysis_text = tool_loop(model_with_tools, messages, max_rounds=8)
    elif dim == "citations":
        # Reasoning models (DeepSeek V4 Flash) return empty content in tool loops —
        # use a non-reasoning model for reliable tool calling
        real_citations = [c for c in clf.citations if not c.startswith("@")]
        model = make_model("TOOL_MODEL", "openai/gpt-4o-mini", max_tokens=6000, config=config)
        model_with_tools = model.bind_tools(REVIEWER_TOOLS)
        messages = [
            SystemMessage(content=f"{system_prompt}\n\n{_TOOL_INSTRUCTIONS['citations']}\n\n{FINDING_SCHEMA_PROMPT}{_CONCISENESS}"),
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
