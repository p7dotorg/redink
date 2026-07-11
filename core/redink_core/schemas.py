from typing import Literal, Optional
from pydantic import BaseModel, Field


class Classification(BaseModel):
    area: str = Field(description="Área acadêmica do paper em 2-4 palavras (ex: Machine Learning, Ciências Sociais, Saúde Pública)")
    paper_type: Literal["empirical", "benchmark", "survey", "theoretical", "system", "position"] = Field(
        description=(
            "Tipo do paper: empirical=tem experimentos/resultados, benchmark=avalia modelos/sistemas, "
            "survey=revisão da literatura, theoretical=contribuição teórica, "
            "system=descreve um sistema/ferramenta, position=argumento/opinião"
        )
    )
    dimensions: list[Literal["citations", "methodology", "novelty", "writing", "statistics", "reproducibility", "ethics", "figures"]] = Field(
        description=(
            "Dimensões de revisão a acionar. USE EXATAMENTE esses valores: "
            "citations, methodology, novelty, writing, statistics, reproducibility, ethics, figures. "
            "Sempre inclua: citations, methodology, novelty, writing. "
            "Inclua statistics se houver análise quantitativa. "
            "Inclua reproducibility se for ML/computação/software. "
            "Inclua ethics se envolver dados de pessoas. "
            "Inclua figures se o paper mencionar gráficos, experimentos visuais ou resultados em tabelas/figuras — "
            "especialmente para papers de ML, CV, medicina, ou qualquer paper com avaliação quantitativa visual."
        )
    )
    citations: list[str] = Field(
        description=(
            "Lista de até 20 referências extraídas da seção References/Bibliography do paper. "
            "Formato obrigatório: 'Autor et al. (ano). Título.' "
            "RETORNE [] (lista vazia) se: o documento não tem seção References/Bibliography, "
            "é um README de repositório, ou só contém auto-citação BibTeX. "
            "NUNCA inclua: chaves BibTeX (@software, @article, @misc...), URLs soltas, DOIs isolados, "
            "auto-citações do próprio paper, ou linhas de código. "
            "Apenas títulos reais de papers, livros ou artigos que o autor cita para embasar o trabalho."
        )
    )
    claims: list[str] = Field(
        description=(
            "Liste 5-8 contribuições TÉCNICAS e ESPECÍFICAS do paper — o que o método FAZ, não o que afirma.\n"
            "Formato obrigatório: sujeito-técnico + verbo-de-ação + objeto-específico.\n"
            "RUIM (vago): 'primeiro a tratar memória como skill treinável'\n"
            "BOM (técnico): 'meta-LLM revisa trajetórias completas de episódios para reescrever iterativamente o scaffold do agente'\n"
            "BOM (técnico): 'fine-tune de modelo de memória especializado com exemplos curados pelo meta-LLM a partir de episódios'\n"
            "BOM (técnico): 'operações file-system (read/write/search/append) como ações de primeira classe no espaço de ação'\n"
            "BOM (técnico): 'benchmark em três jogos procedurais (Crafter, MiniHack, NetHack) com métrica progression rate'\n"
            "Inclua: nome do método/sistema, componentes técnicos, benchmarks, métricas, baselines comparados."
        )
    )
    code_repo: Optional[str] = Field(
        default=None,
        description=(
            "URL do repositório de código OFICIAL do paper (o que os autores "
            "publicaram: 'code/implementation available at github.com/...'). "
            "NÃO é um repo citado como baseline ou dataset de terceiros. "
            "null se o paper não linka repositório próprio."
        ),
    )


class Finding(BaseModel):
    dimension: str
    persona: Literal["skeptic", "practitioner", "academic"] = "skeptic"
    severity: Literal["critical", "major", "minor"]
    issue: str = Field(description="Descrição do problema encontrado")
    evidence: str = Field(description="Evidência ou trecho do paper que suporta o problema")
    suggestion: str = Field(description="O que o autor deve fazer para corrigir")
    confidence: int = Field(default=5, description="Confiança 1-10 baseada em consenso entre personas")
    evidence_verified: bool = Field(
        default=True,
        description="True se o trecho de evidence foi localizado no texto do paper"
    )
    debate_outcome: Optional[Literal["sustained", "downgraded", "dismissed"]] = Field(
        default=None,
        description="Resultado do debate adversarial (apenas findings critical passam por ele)"
    )
    defense: Optional[str] = Field(
        default=None,
        description="Melhor argumento da defesa do autor durante o debate"
    )
    grounded: bool = Field(
        default=False,
        description=(
            "True se o finding foi verificado por EXECUÇÃO (repro_check rodou "
            "o código). Findings grounded bypassam o debate adversarial e a "
            "dedup — um fato de execução não é argumentável."
        ),
    )


class FindingsList(BaseModel):
    findings: list[Finding]


class FindingCluster(BaseModel):
    representative: int = Field(
        description="Índice do finding que melhor expressa o problema do cluster"
    )
    members: list[int] = Field(
        description="Índices de TODOS os findings deste cluster (incluindo o representative)"
    )


class DedupMap(BaseModel):
    clusters: list[FindingCluster] = Field(
        description=(
            "Partição dos findings em clusters de problema idêntico. "
            "Cada índice aparece em exatamente um cluster; clusters de um "
            "único finding são normais."
        )
    )


class Contradiction(BaseModel):
    dimension: str
    claim_a: str = Field(description="Posição de uma persona")
    persona_a: str
    claim_b: str = Field(description="Posição conflitante de outra persona")
    persona_b: str
    significance: Literal["high", "medium", "low"] = Field(
        description="Quão importante é essa contradição para o veredito final"
    )


class ContradictionMap(BaseModel):
    contradictions: list[Contradiction]
    consensus: list[str] = Field(
        description="Afirmações em que TODAS as personas concordam — alta confiança"
    )
    most_disputed_dimension: Optional[str] = Field(
        default=None,
        description="Dimensão com mais discordância entre personas"
    )


class BlindSpot(BaseModel):
    topics_not_covered: list[str] = Field(
        description="Aspectos críticos do paper que NENHUM revisor de nenhuma persona mencionou"
    )
    highest_priority: Optional[str] = Field(
        default=None,
        description="O blind spot mais importante — o que mudaria o veredito se fosse encontrado"
    )


class Rebuttal(BaseModel):
    ruling: Literal["sustain", "downgrade", "dismiss"] = Field(
        description=(
            "dismiss = finding factualmente errado ou já endereçado pelo paper; "
            "downgrade = problema real mas não invalida conclusão central (não é critical); "
            "sustain = crítica sólida no nível critical mesmo após a defesa"
        )
    )
    defense_summary: str = Field(description="O ponto mais forte da defesa, em 1-2 frases")
    reasoning: str = Field(description="Justificativa da decisão em 1-2 frases")


class JudgeVote(BaseModel):
    lens: str = ""
    vote: Literal["PASS", "REVISE", "FAIL"] = Field(
        description="PASS = aceitar; REVISE = major revision; FAIL = rejeitar"
    )
    rationale: str = Field(description="Justificativa do voto em 2-3 frases")


class JudgePanel(BaseModel):
    votes: list[JudgeVote]
    verdict: Literal["PASS", "REVISE", "FAIL"]


class Verdict(BaseModel):
    status: Literal["PASS", "REVISE", "FAIL"]
    summary: str = Field(description="Parágrafo resumindo o veredito geral com foco nas contradições")
    critical_count: int
    major_count: int
    minor_count: int
    findings: list[Finding]
    contradiction_map: Optional[ContradictionMap] = None
    blind_spots: Optional[BlindSpot] = None
    judge_panel: Optional[JudgePanel] = None
    high_confidence_issues: list[str] = Field(
        default_factory=list,
        description="Problemas que todas as personas concordaram — certeza máxima"
    )
