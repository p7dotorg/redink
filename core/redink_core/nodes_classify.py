"""classify node."""
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from redink_core.nodes_helpers import make_model
from redink_core.schemas import Classification

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

Regras OBRIGATÓRIAS para o campo claims — extraia 5-8 contribuições TÉCNICAS específicas:
  FORMATO: sujeito-técnico + verbo-de-ação + objeto-específico
  INCLUA: nome do método/sistema, componentes técnicos, benchmarks, métricas, baselines
  ERRADO (vago): 'primeiro a tratar memória como skill treinável'
  CERTO: 'meta-LLM revisa trajetórias completas de episódios para reescrever iterativamente o scaffold'
  CERTO: 'fine-tune de modelo de memória com exemplos curados pelo meta-LLM a partir de episódios'
  CERTO: 'operações file-system (read/write/search/append) como ações de primeira classe no espaço de ação'
  CERTO: 'benchmark em Crafter, MiniHack, NetHack com métrica progression rate vs Qwen2.5-72B baseline'
"""


# Matches "REFERENCES" / "BIBLIOGRAPHY" even when a PDF extractor letter-spaces
# the drop-cap header ("R EFERENCES", "R E F E R E N C E S").
_REFS_RE = re.compile(
    r"R\s*E\s*F\s*E\s*R\s*E\s*N\s*C\s*E\s*S|B\s*I\s*B\s*L\s*I\s*O\s*G\s*R\s*A\s*P\s*H\s*Y",
    re.IGNORECASE,
)


def _classify_excerpt(paper: str) -> str:
    """First 12k + the References section (if stranded in the omitted middle,
    e.g. before an appendix) + last 3k. Without the refs chunk, papers with a
    post-references appendix look citation-less to classify."""
    if len(paper) <= 15000:
        return paper
    parts = [paper[:12000]]
    m = _REFS_RE.search(paper, 12000, len(paper) - 3000)
    if m:
        parts.append("\n\n[... body omitted ...]\n\n" + paper[m.start():m.start() + 4000])
    parts.append("\n\n[... omitted ...]\n\n" + paper[-3000:])
    return "".join(parts)


_ML_AREAS = ("machine learning", "deep learning", "computer vision", "nlp",
             "natural language", "artificial intelligence", "computação", "software")


def _drop_self_citations(result, paper: str) -> None:
    """Remove citations that are actually the paper citing itself.

    The schema forbids self-citations but structuring models keep including
    them (e.g. 'Vaswani et al. (2017)' extracted from the Attention paper),
    which later produces false 'unverifiable reference' findings.
    """
    head = paper[:3000].lower()
    kept = []
    for c in result.citations:
        m = re.match(r"^.*?\(\d{4}\)[.,]?\s*(.+)$", c)  # 'Autor et al. (ano). Título.'
        title_part = (m.group(1) if m else c).strip(" .").lower()
        if len(title_part) >= 15 and title_part[:40] in head:
            continue  # citation title appears in the paper's own header — self-citation
        kept.append(c)
    result.citations = kept


def classify(state, config: RunnableConfig = None):
    model = make_model("CLASSIFY_MODEL", "qwen/qwen3-8b", Classification, max_tokens=2000, config=config)
    result = model.invoke([
        SystemMessage(content=_CLASSIFY_SYSTEM),
        HumanMessage(content=f"Classifique este paper:\n\n{_classify_excerpt(state['paper'])}"),
    ])
    # Enforce in code what the prompt only asks for
    _drop_self_citations(result, state["paper"])
    if any(a in result.area.lower() for a in _ML_AREAS) and "reproducibility" not in result.dimensions:
        result.dimensions.append("reproducibility")
    return {"classification": result}
