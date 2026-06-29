from typing import Literal
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
    severity: Literal["critical", "major", "minor"]
    issue: str = Field(description="Descrição do problema encontrado")
    evidence: str = Field(description="Evidência ou trecho do paper que suporta o problema")
    suggestion: str = Field(description="O que o autor deve fazer para corrigir")


class FindingsList(BaseModel):
    findings: list[Finding]


class Verdict(BaseModel):
    status: Literal["PASS", "REVISE", "FAIL"]
    summary: str = Field(description="Parágrafo resumindo o veredito geral")
    critical_count: int
    major_count: int
    minor_count: int
    findings: list[Finding]
