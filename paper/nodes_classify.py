"""classify node."""
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from paper.nodes_helpers import make_model
from paper.schemas import Classification

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


def _classify_excerpt(paper: str) -> str:
    """Send first 12k + last 3k chars — covers abstract + intro + start of methods + references."""
    if len(paper) <= 15000:
        return paper
    return paper[:12000] + "\n\n[... seções intermediárias omitidas ...]\n\n" + paper[-3000:]


def classify(state, config: RunnableConfig = None):
    model = make_model("CLASSIFY_MODEL", "qwen/qwen3-8b", Classification, max_tokens=2000, config=config)
    result = model.invoke([
        SystemMessage(content=_CLASSIFY_SYSTEM),
        HumanMessage(content=f"Classifique este paper:\n\n{_classify_excerpt(state['paper'])}"),
    ])
    return {"classification": result}
