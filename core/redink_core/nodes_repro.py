"""repro_check node — roda o código do repo do paper e vira o resultado em
Finding grounded. Determinístico, sem LLM. Ver docs/superpowers/specs/
2026-07-10-repro-check-design.md."""
from dataclasses import asdict

from langchain_core.runnables import RunnableConfig

from redink_core.repro import (
    ReproResult, docker_available, resolve_repo_url, run_repro_check,
)
from redink_core.schemas import Finding

# status → (severity, texto do issue). 'ok'/'no_docker' não emitem finding.
_ISSUE = {
    "repo_missing": ("critical", "O repositório de código linkado pelo paper não existe, está vazio ou retornou erro no clone."),
    "install_fail": ("major", "Clonei o repositório oficial do paper, mas o código NÃO instala (deps/build quebram)."),
    "import_fail": ("major", "O código instala, mas o pacote principal NÃO importa."),
    "timeout": ("major", "A instalação/import do repositório passou do tempo limite (180s)."),
}

_SUGGESTION = {
    "repo_missing": "Publique o repositório e confirme que a URL no paper está correta e acessível.",
    "install_fail": "Fixe as dependências (requirements/pyproject) e teste a instalação num ambiente limpo.",
    "import_fail": "Corrija os imports top-level do pacote; teste `import` num ambiente limpo pós-install.",
    "timeout": "Reduza o custo de instalação ou documente o tempo/recursos necessários.",
}


def _to_finding(r: ReproResult) -> Finding:
    severity, issue = _ISSUE[r.status]
    return Finding(
        dimension="reproducibility",
        persona="skeptic",
        severity=severity,
        issue=issue,
        evidence=(r.log or "")[:1500] or f"repro_check status={r.status}",
        suggestion=_SUGGESTION[r.status],
        confidence=9,
        evidence_verified=True,
        grounded=True,
    )


def repro_check(state, config: RunnableConfig = None):
    """Clona+instala+importa o repo do paper num sandbox; emite Finding grounded
    só em caso de falha. Em sucesso, só registra repro_result pro veredito."""
    raw_url = state.get("code_repo")
    if not raw_url:
        return {"findings": [], "repro_result": {"status": "no_docker", "skipped": True}}
    if not docker_available():
        return {"findings": [], "repro_result": {"status": "no_docker", "repo_url": raw_url}}

    # Normaliza+valida a URL contra o GitHub (URLs de paper vêm quebradas do
    # PDF: http, truncadas, hífen->espaço). Só clonamos o que existe.
    repo_url = resolve_repo_url(raw_url, state.get("paper", ""))
    if not repo_url:
        result = ReproResult("repo_missing", raw_url,
                             log="URL do repo não resolve pra um repositório existente no GitHub")
        return {"findings": [_to_finding(result)], "repro_result": asdict(result)}

    result = run_repro_check(repo_url)
    repro_result = asdict(result)
    if result.status in ("ok", "no_docker"):
        return {"findings": [], "repro_result": repro_result}
    return {"findings": [_to_finding(result)], "repro_result": repro_result}
