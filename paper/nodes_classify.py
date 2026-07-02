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
"""


def classify(state, config: RunnableConfig = None):
    model = make_model("CLASSIFY_MODEL", "qwen/qwen3-8b", Classification, max_tokens=800, config=config)
    result = model.invoke([
        SystemMessage(content=_CLASSIFY_SYSTEM),
        HumanMessage(content=f"Classifique este paper:\n\n{state['paper']}"),
    ])
    return {"classification": result}
