from typing import Literal, Optional
from pydantic import BaseModel, Field


class Classification(BaseModel):
    area: str = Field(description="Área acadêmica do paper (ex: Machine Learning, Ciências Sociais, Saúde)")
    paper_type: str = Field(description="Tipo do paper (ex: empírico, teórico, survey, caso de estudo)")
    dimensions: list[Literal["citations", "methodology", "novelty", "writing", "statistics", "reproducibility", "ethics"]] = Field(
        description=(
            "Dimensões de revisão a acionar. USE EXATAMENTE esses valores: "
            "citations, methodology, novelty, writing, statistics, reproducibility, ethics. "
            "Sempre inclua: citations, methodology, novelty, writing. "
            "Inclua statistics se houver análise quantitativa. "
            "Inclua reproducibility se for ML/computação/software. "
            "Inclua ethics se envolver dados de pessoas."
        )
    )
    citations: list[str] = Field(
        description="Lista de referências bibliográficas extraídas do paper, cada uma como string completa"
    )
    claims: list[str] = Field(
        description="Principais afirmações/contribuições do paper (3-7 claims centrais)"
    )


class Finding(BaseModel):
    dimension: str
    persona: Literal["skeptic", "practitioner", "academic"] = "skeptic"
    severity: Literal["critical", "major", "minor"]
    issue: str = Field(description="Descrição do problema encontrado")
    evidence: str = Field(description="Evidência ou trecho do paper que suporta o problema")
    suggestion: str = Field(description="O que o autor deve fazer para corrigir")
    confidence: int = Field(default=5, description="Confiança 1-10 baseada em consenso entre personas")


class FindingsList(BaseModel):
    findings: list[Finding]


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


class Verdict(BaseModel):
    status: Literal["PASS", "REVISE", "FAIL"]
    summary: str = Field(description="Parágrafo resumindo o veredito geral com foco nas contradições")
    critical_count: int
    major_count: int
    minor_count: int
    findings: list[Finding]
    contradiction_map: Optional[ContradictionMap] = None
    blind_spots: Optional[BlindSpot] = None
    high_confidence_issues: list[str] = Field(
        default_factory=list,
        description="Problemas que todas as personas concordaram — certeza máxima"
    )
