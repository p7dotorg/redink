# repro_check Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Substituir a heurística de texto do reviewer de `reproducibility` por verificação real — clonar o repo do paper num sandbox Docker e testar se instala e importa.

**Architecture:** Um node `repro_check` entra no fan-out do grafo em paralelo aos reviewers de texto (via `Send`). Ele chama `run_repro_check()`, que orquestra dois containers Docker efêmeros (fase 1 clone+install com rede; fase 2 import sem rede) compartilhando um volume. O `ReproResult` vira um `Finding` `grounded` que bypassa o debate adversarial. Opt-in via env `REDINK_REPRO`.

**Tech Stack:** Python 3.11+, LangGraph, Pydantic, Docker CLI (via `subprocess`), pytest. **Zero dependências Python novas** — o Docker é chamado por subprocess.

## Global Constraints

- Escopo v1: **clone → install → import**, determinístico, **sem LLM**. Sem rodar testes, sem reproduzir números.
- Isolamento: Docker local; `--network=none` na fase de import; sem secrets/env montados; fs efêmero; `--rm`.
- Limites do container: `--memory=2g` · `--cpus=2` · `--pids-limit=512` · timeout global 180s.
- Imagem base: `python:3.11-slim`.
- Ativação: opt-in via env `REDINK_REPRO` (default off).
- Severity: `repo_missing` → `critical`; `install_fail`/`import_fail`/`timeout` → `major`; `ok`/`no_docker` → sem finding.
- Findings de execução: `dimension="reproducibility"`, `grounded=True`, `evidence_verified=True`, evidence = cauda do log real.
- Todos os `git commit` terminam com o trailer padrão do repositório.

---

## File Structure

| Arquivo | Responsabilidade | Ação |
|---------|------------------|------|
| `core/redink_core/schemas.py` | `Classification.code_repo`, `Finding.grounded` | Modify |
| `core/redink_core/repro.py` | `ReproResult`, `docker_available`, `run_repro_check` (orquestração Docker) | Create |
| `core/redink_core/nodes_repro.py` | node `repro_check` + mapa `ReproResult → Finding` | Create |
| `core/redink_core/nodes.py` | re-exportar `repro_check` | Modify |
| `core/redink_core/nodes_debate.py` | bypass de findings `grounded` | Modify |
| `core/redink_core/nodes_synthesis.py` | surfacing de `repro_result` no veredito | Modify |
| `core/redink_core/graph.py` | node + edge + `Send` no route + state field | Modify |
| `core/pyproject.toml` | pytest no dev extra | Modify |
| `core/tests/test_*.py` | testes unitários | Create |

---

## Task 1: Schema fields + infra de teste

**Files:**
- Modify: `core/redink_core/schemas.py`
- Modify: `core/pyproject.toml:23`
- Create: `core/tests/__init__.py`
- Test: `core/tests/test_schemas_repro.py`

**Interfaces:**
- Produces: `Classification.code_repo: Optional[str]` (default `None`); `Finding.grounded: bool` (default `False`).

- [ ] **Step 1: Instalar pytest e registrar no dev extra**

Editar `core/pyproject.toml`, linha 23, de:
```toml
dev = ["langgraph-cli>=0.1.0"]
```
para:
```toml
dev = ["langgraph-cli>=0.1.0", "pytest>=8"]
```
Depois rodar:
```bash
uv pip install pytest
```
Expected: `Installed ... pytest-8.x`.

- [ ] **Step 2: Criar o pacote de testes**

Criar `core/tests/__init__.py` vazio:
```python
```

- [ ] **Step 3: Escrever o teste que falha**

Criar `core/tests/test_schemas_repro.py`:
```python
from redink_core.schemas import Classification, Finding


def test_classification_has_optional_code_repo():
    clf = Classification(
        area="Machine Learning", paper_type="empirical",
        dimensions=["reproducibility"], citations=[], claims=[],
    )
    assert clf.code_repo is None

    clf2 = Classification(
        area="ML", paper_type="empirical", dimensions=["reproducibility"],
        citations=[], claims=[], code_repo="https://github.com/foo/bar",
    )
    assert clf2.code_repo == "https://github.com/foo/bar"


def test_finding_grounded_defaults_false():
    f = Finding(
        dimension="reproducibility", severity="major",
        issue="x", evidence="y", suggestion="z",
    )
    assert f.grounded is False

    f2 = Finding(
        dimension="reproducibility", severity="major",
        issue="x", evidence="y", suggestion="z", grounded=True,
    )
    assert f2.grounded is True
```

- [ ] **Step 4: Rodar o teste e ver falhar**

Run: `python -m pytest core/tests/test_schemas_repro.py -v`
Expected: FAIL — `Classification` não aceita `code_repo` (ou `code_repo` inexistente) / `Finding` não aceita `grounded`.

- [ ] **Step 5: Adicionar os campos**

Em `core/redink_core/schemas.py`, no fim da classe `Classification` (depois do campo `claims`, ~linha 48), adicionar:
```python
    code_repo: Optional[str] = Field(
        default=None,
        description=(
            "URL do repositório de código OFICIAL do paper (o que os autores "
            "publicaram: 'code/implementation available at github.com/...'). "
            "NÃO é um repo citado como baseline ou dataset de terceiros. "
            "null se o paper não linka repositório próprio."
        ),
    )
```

Na classe `Finding`, depois do campo `defense` (~linha 70), adicionar:
```python
    grounded: bool = Field(
        default=False,
        description=(
            "True se o finding foi verificado por EXECUÇÃO (repro_check rodou "
            "o código). Findings grounded bypassam o debate adversarial e a "
            "dedup — um fato de execução não é argumentável."
        ),
    )
```

- [ ] **Step 6: Rodar o teste e ver passar**

Run: `python -m pytest core/tests/test_schemas_repro.py -v`
Expected: PASS (2 passed).

- [ ] **Step 7: Commit**

```bash
git add core/redink_core/schemas.py core/pyproject.toml core/tests/__init__.py core/tests/test_schemas_repro.py
git commit -m "$(cat <<'EOF'
feat: schema fields para repro_check (code_repo, grounded)

Classification.code_repo (repo oficial do paper) e Finding.grounded
(verificado por execução, bypassa debate). Infra de teste com pytest.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_012HuCydr63ssfhGkTBzPMyF
EOF
)"
```

---

## Task 2: `repro.py` — orquestração Docker

**Files:**
- Create: `core/redink_core/repro.py`
- Test: `core/tests/test_repro.py`

**Interfaces:**
- Produces:
  - `ReproResult` dataclass: campos `status: str`, `repo_url: str`, `package: Optional[str] = None`, `log: str = ""`. `status` ∈ `{"ok","install_fail","import_fail","repo_missing","timeout","no_docker"}`.
  - `docker_available() -> bool`.
  - `run_repro_check(repo_url: str, *, timeout: int = 180, mem: str = "2g", cpus: int = 2) -> ReproResult`.
  - `_docker_run(args: list[str], timeout: int) -> tuple[int, str]` — wrapper mockável de `subprocess.run(["docker", *args])`, retorna `(returncode, output_tail)`; propaga `subprocess.TimeoutExpired`.

- [ ] **Step 1: Escrever os testes que falham**

Criar `core/tests/test_repro.py`:
```python
import subprocess
from unittest.mock import patch

import pytest

from redink_core.repro import ReproResult, docker_available, run_repro_check

REPO = "https://github.com/foo/bar"


def test_docker_available_true():
    with patch("redink_core.repro._docker_run", return_value=(0, "Docker version 27")):
        assert docker_available() is True


def test_docker_available_false_when_missing():
    with patch("redink_core.repro._docker_run", side_effect=FileNotFoundError):
        assert docker_available() is False


def test_clone_failure_is_repo_missing():
    # phase 1 returns rc 10 (git clone failed)
    with patch("redink_core.repro._docker_run", side_effect=[(10, "fatal: not found"), (0, "")]):
        r = run_repro_check(REPO)
    assert r.status == "repo_missing"
    assert r.repo_url == REPO


def test_install_failure_is_install_fail():
    with patch("redink_core.repro._docker_run", side_effect=[(20, "ERROR: No matching distribution torch"), (0, "")]):
        r = run_repro_check(REPO)
    assert r.status == "install_fail"
    assert "torch" in r.log


def test_no_manifest_is_install_fail():
    with patch("redink_core.repro._docker_run", side_effect=[(30, "no manifest"), (0, "")]):
        r = run_repro_check(REPO)
    assert r.status == "install_fail"


def test_import_failure_is_import_fail():
    # phase 1 ok (rc 0), phase 2 import fails (rc 1), volume rm ok
    with patch("redink_core.repro._docker_run", side_effect=[(0, "installed"), (1, "IMPORT_FAIL:ImportError(x)"), (0, "")]):
        r = run_repro_check(REPO)
    assert r.status == "import_fail"


def test_nothing_importable_is_import_fail():
    with patch("redink_core.repro._docker_run", side_effect=[(0, "installed"), (40, ""), (0, "")]):
        r = run_repro_check(REPO)
    assert r.status == "import_fail"


def test_success_is_ok_with_package():
    with patch("redink_core.repro._docker_run", side_effect=[(0, "installed"), (0, "IMPORT_OK:bar"), (0, "")]):
        r = run_repro_check(REPO)
    assert r.status == "ok"
    assert r.package == "bar"


def test_timeout_is_timeout():
    with patch("redink_core.repro._docker_run", side_effect=subprocess.TimeoutExpired(cmd="docker", timeout=180)):
        r = run_repro_check(REPO)
    assert r.status == "timeout"


def test_volume_is_removed_on_success():
    calls = []

    def fake(args, timeout):
        calls.append(args)
        if args[0] == "run" and "--network=none" in args:
            return (0, "IMPORT_OK:bar")
        if args[0] == "run":
            return (0, "installed")
        return (0, "")

    with patch("redink_core.repro._docker_run", side_effect=fake):
        run_repro_check(REPO)

    assert any(a[0] == "volume" and a[1] == "rm" for a in calls), "volume must be removed"
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `python -m pytest core/tests/test_repro.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'redink_core.repro'`.

- [ ] **Step 3: Implementar `repro.py`**

Criar `core/redink_core/repro.py`:
```python
"""repro.py — verificação por execução: clona o repo do paper num sandbox
Docker efêmero e testa se instala e importa. Zero deps novas: chama o CLI do
Docker por subprocess. Todo código do repo roda dentro do container, com rede
cortada na fase de import."""
import subprocess
from dataclasses import dataclass
from typing import Optional

_IMAGE = "python:3.11-slim"
_LOG_TAIL = 2000

# Fase 1 (rede ligada): clona, instala, e grava os pacotes top-level do repo.
# Exit codes distintos deixam run_repro_check mapear a causa da falha.
_PHASE1 = r"""
set -o pipefail
cd /work
rm -rf repo
git clone --depth 1 "$REPO_URL" repo 2>&1 || exit 10
cd repo
if [ -f pyproject.toml ] || [ -f setup.py ]; then
  pip install --quiet . 2>&1 || exit 20
elif [ -f requirements.txt ]; then
  pip install --quiet -r requirements.txt 2>&1 || exit 20
else
  exit 30
fi
python - <<'PY'
import pathlib
root = pathlib.Path('/work/repo')
cands = []
for base in (root, root / 'src'):
    if base.is_dir():
        for d in sorted(base.iterdir()):
            if (d / '__init__.py').exists():
                cands.append(d.name)
pathlib.Path('/work/candidates.txt').write_text("\n".join(cands))
PY
"""

# Fase 2 (rede cortada): importa o pacote instalado. NÃO faz cd em /work/repo —
# importa o que o pip instalou no site-packages, não a árvore de fonte.
_PHASE2 = r"""
cd /work
python - <<'PY'
import pathlib, sys
p = pathlib.Path('/work/candidates.txt')
cands = p.read_text().split() if p.exists() else []
if not cands:
    sys.exit(40)
err = None
for name in cands:
    try:
        __import__(name)
        print("IMPORT_OK:" + name)
        sys.exit(0)
    except Exception as e:
        err = e
print("IMPORT_FAIL:" + repr(err))
sys.exit(1)
PY
"""


@dataclass
class ReproResult:
    status: str  # ok | install_fail | import_fail | repo_missing | timeout | no_docker
    repo_url: str
    package: Optional[str] = None
    log: str = ""


def _docker_run(args: list[str], timeout: int) -> tuple[int, str]:
    """Run `docker <args>` and return (returncode, combined-output-tail).

    Isolated so tests can mock the whole Docker boundary. Raises
    subprocess.TimeoutExpired on timeout; FileNotFoundError if docker is absent.
    """
    proc = subprocess.run(
        ["docker", *args],
        capture_output=True, text=True, timeout=timeout,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, out[-_LOG_TAIL:]


def docker_available() -> bool:
    try:
        rc, _ = _docker_run(["version", "--format", "{{.Server.Version}}"], timeout=10)
        return rc == 0
    except (FileNotFoundError, subprocess.SubprocessError):
        return False


def _run_args(volume: str, network: str, script: str, repo_url: str,
              mem: str, cpus: int) -> list[str]:
    return [
        "run", "--rm",
        f"--network={network}",
        f"--memory={mem}", f"--cpus={cpus}", "--pids-limit=512",
        "-v", f"{volume}:/work",
        "-e", f"REPO_URL={repo_url}",
        _IMAGE, "bash", "-c", script,
    ]


def run_repro_check(repo_url: str, *, timeout: int = 180,
                    mem: str = "2g", cpus: int = 2) -> ReproResult:
    """Clone+install (network on) then import (network off) in ephemeral Docker.

    Deterministic, no LLM. Maps phase exit codes to a ReproResult status.
    """
    # Ephemeral named volume; derived from the repo so a run is reproducible
    # and no wall-clock/random is needed.
    volume = "repro_" + "".join(c if c.isalnum() else "_" for c in repo_url)[-48:]
    try:
        rc1, out1 = _docker_run(
            _run_args(volume, "bridge", _PHASE1, repo_url, mem, cpus), timeout)
        if rc1 == 10:
            return ReproResult("repo_missing", repo_url, log=out1)
        if rc1 in (20, 30):
            return ReproResult("install_fail", repo_url, log=out1)
        if rc1 != 0:
            return ReproResult("install_fail", repo_url, log=out1)

        rc2, out2 = _docker_run(
            _run_args(volume, "none", _PHASE2, repo_url, mem, cpus), timeout)
        if rc2 == 0:
            pkg = None
            for line in out2.splitlines():
                if line.startswith("IMPORT_OK:"):
                    pkg = line.split(":", 1)[1].strip()
            return ReproResult("ok", repo_url, package=pkg, log=out2)
        return ReproResult("import_fail", repo_url, log=out2)
    except subprocess.TimeoutExpired:
        return ReproResult("timeout", repo_url, log=f"passou de {timeout}s")
    finally:
        try:
            _docker_run(["volume", "rm", "-f", volume], timeout=15)
        except Exception:
            pass
```

- [ ] **Step 4: Rodar e ver passar**

Run: `python -m pytest core/tests/test_repro.py -v`
Expected: PASS (10 passed).

- [ ] **Step 5: Commit**

```bash
git add core/redink_core/repro.py core/tests/test_repro.py
git commit -m "$(cat <<'EOF'
feat: repro.py — clone/install/import num sandbox Docker

Duas fases em containers efêmeros (fase 1 rede ON: clone+install;
fase 2 rede OFF: import) compartilhando um volume. Mapeia exit codes
para ReproResult. Zero deps novas (subprocess + docker CLI).

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_012HuCydr63ssfhGkTBzPMyF
EOF
)"
```

---

## Task 3: node `repro_check` — mapa ReproResult → Finding

**Files:**
- Create: `core/redink_core/nodes_repro.py`
- Modify: `core/redink_core/nodes.py`
- Test: `core/tests/test_nodes_repro.py`

**Interfaces:**
- Consumes: `ReproResult`, `run_repro_check`, `docker_available` de `redink_core.repro`; `Finding` de `redink_core.schemas`.
- Produces: `repro_check(state, config=None) -> dict` com chaves `findings: list[Finding]` e `repro_result: dict`. Lê `state["code_repo"]`.

- [ ] **Step 1: Escrever os testes que falham**

Criar `core/tests/test_nodes_repro.py`:
```python
from unittest.mock import patch

from redink_core.nodes_repro import repro_check
from redink_core.repro import ReproResult

STATE = {"code_repo": "https://github.com/foo/bar"}


def _run(result=None, docker=True):
    with patch("redink_core.nodes_repro.docker_available", return_value=docker), \
         patch("redink_core.nodes_repro.run_repro_check", return_value=result):
        return repro_check(STATE)


def test_ok_emits_no_finding_but_records_result():
    out = _run(ReproResult("ok", STATE["code_repo"], package="bar", log="IMPORT_OK:bar"))
    assert out["findings"] == []
    assert out["repro_result"]["status"] == "ok"


def test_no_docker_emits_no_finding():
    out = _run(docker=False)
    assert out["findings"] == []
    assert out["repro_result"]["status"] == "no_docker"


def test_install_fail_is_major_grounded_finding():
    out = _run(ReproResult("install_fail", STATE["code_repo"], log="No matching distribution torch"))
    assert len(out["findings"]) == 1
    f = out["findings"][0]
    assert f.dimension == "reproducibility"
    assert f.severity == "major"
    assert f.grounded is True
    assert f.evidence_verified is True
    assert "torch" in f.evidence


def test_import_fail_is_major():
    out = _run(ReproResult("import_fail", STATE["code_repo"], log="ImportError"))
    assert out["findings"][0].severity == "major"


def test_timeout_is_major():
    out = _run(ReproResult("timeout", STATE["code_repo"], log="passou de 180s"))
    assert out["findings"][0].severity == "major"


def test_repo_missing_is_critical():
    out = _run(ReproResult("repo_missing", STATE["code_repo"], log="not found"))
    assert out["findings"][0].severity == "critical"
    assert out["findings"][0].grounded is True


def test_no_code_repo_skips():
    out = repro_check({})
    assert out["findings"] == []
    assert out["repro_result"]["status"] == "no_docker" or out["repro_result"].get("skipped")
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `python -m pytest core/tests/test_nodes_repro.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'redink_core.nodes_repro'`.

- [ ] **Step 3: Implementar `nodes_repro.py`**

Criar `core/redink_core/nodes_repro.py`:
```python
"""repro_check node — roda o código do repo do paper e vira o resultado em
Finding grounded. Determinístico, sem LLM. Ver docs/superpowers/specs/
2026-07-10-repro-check-design.md."""
from dataclasses import asdict

from langchain_core.runnables import RunnableConfig

from redink_core.repro import ReproResult, docker_available, run_repro_check
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
    repo_url = state.get("code_repo")
    if not repo_url:
        return {"findings": [], "repro_result": {"status": "no_docker", "skipped": True}}
    if not docker_available():
        return {"findings": [], "repro_result": {"status": "no_docker", "repo_url": repo_url}}

    result = run_repro_check(repo_url)
    repro_result = asdict(result)
    if result.status in ("ok", "no_docker"):
        return {"findings": [], "repro_result": repro_result}
    return {"findings": [_to_finding(result)], "repro_result": repro_result}
```

- [ ] **Step 4: Re-exportar o node**

Em `core/redink_core/nodes.py`, adicionar o import (depois da linha 5) e o `__all__`:
```python
from redink_core.nodes_repro import repro_check
```
E incluir `"repro_check"` na lista `__all__`:
```python
__all__ = [
    "fetch_paper", "classify", "reviewer", "figure_reviewer",
    "debate", "contradiction_map", "blind_spot", "judge_panel", "synthesize",
    "repro_check",
]
```

- [ ] **Step 5: Rodar e ver passar**

Run: `python -m pytest core/tests/test_nodes_repro.py -v`
Expected: PASS (7 passed).

- [ ] **Step 6: Commit**

```bash
git add core/redink_core/nodes_repro.py core/redink_core/nodes.py core/tests/test_nodes_repro.py
git commit -m "$(cat <<'EOF'
feat: node repro_check — ReproResult vira Finding grounded

Falha de execução vira finding grounded (dimension=reproducibility,
severity major, repo_missing=critical). Sucesso não emite finding, só
grava repro_result. Docker ausente / sem repo: skip sem punir o paper.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_012HuCydr63ssfhGkTBzPMyF
EOF
)"
```

---

## Task 4: debate bypassa findings grounded

**Files:**
- Modify: `core/redink_core/nodes_debate.py:53-82`
- Test: `core/tests/test_debate_bypass.py`

**Interfaces:**
- Consumes: `Finding` (com `.grounded`), `debate(state, config=None) -> {"deduped_findings": list[Finding]}`.

- [ ] **Step 1: Escrever o teste que falha**

Criar `core/tests/test_debate_bypass.py`:
```python
from unittest.mock import patch

from redink_core.nodes_debate import debate
from redink_core.schemas import Finding, Rebuttal


def _crit(issue, grounded=False):
    return Finding(
        dimension="reproducibility" if grounded else "methodology",
        severity="critical", issue=issue, evidence="e", suggestion="s",
        grounded=grounded,
    )


def test_grounded_critical_bypasses_debate():
    grounded = _crit("não instala", grounded=True)
    textual = _crit("método frágil", grounded=False)
    state = {"findings": [grounded, textual], "paper": "corpo do paper"}

    # dedup: passthrough; _debate_one: sempre dismiss (mataria um crítico normal)
    dismiss = Rebuttal(ruling="dismiss", defense_summary="d", reasoning="r")
    with patch("redink_core.nodes_debate._dedup_findings", side_effect=lambda f, c: f), \
         patch("redink_core.nodes_debate._debate_one", return_value=dismiss) as m:
        out = debate(state)

    kept = out["deduped_findings"]
    # o grounded sobrevive intocado; o textual foi dismissado (removido)
    assert any(f.grounded and f.issue == "não instala" for f in kept)
    assert all(not (f.issue == "método frágil") for f in kept)
    # _debate_one só foi chamado para o finding NÃO-grounded
    assert m.call_count == 1


def test_grounded_finding_keeps_no_debate_outcome():
    grounded = _crit("repo vazio", grounded=True)
    state = {"findings": [grounded], "paper": "x"}
    with patch("redink_core.nodes_debate._dedup_findings", side_effect=lambda f, c: f):
        out = debate(state)
    kept = out["deduped_findings"]
    assert len(kept) == 1
    assert kept[0].debate_outcome is None
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `python -m pytest core/tests/test_debate_bypass.py -v`
Expected: FAIL — hoje o grounded critical entra em `criticals`, é dismissado, e some (`test_grounded_critical_bypasses_debate` falha) e/ou `_debate_one` é chamado 2x.

- [ ] **Step 3: Modificar `debate`**

Em `core/redink_core/nodes_debate.py`, substituir o corpo da função `debate` (linhas 53-58, a parte do dedup e seleção de criticals) por:
```python
def debate(state, config: RunnableConfig = None):
    """Dedup all findings, then put every critical through defender vs judge.
    Findings grounded (verificados por execução) bypassam dedup E debate — um
    fato de execução não é argumentável nem clusterizável."""
    all_findings = state["findings"]
    grounded = [f for f in all_findings if f.grounded]
    debatable = [f for f in all_findings if not f.grounded]
    findings = _dedup_findings(debatable, config) + grounded
    criticals = [f for f in findings if f.severity == "critical" and not f.grounded]
    if not criticals:
        return {"deduped_findings": findings}
```
O resto da função (de `paper_excerpt = ...` até o `return`) permanece igual — ele já itera sobre `findings` e só consulta `outcome_by_id` para os criticals selecionados, então os grounded passam intocados.

- [ ] **Step 4: Rodar e ver passar**

Run: `python -m pytest core/tests/test_debate_bypass.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add core/redink_core/nodes_debate.py core/tests/test_debate_bypass.py
git commit -m "$(cat <<'EOF'
feat: debate bypassa findings grounded

Findings verificados por execução (grounded=True) não entram no dedup
nem no debate adversarial — um stack trace real não pode ser dismissado
por um LLM defender. São anexados verbatim pós-dedup.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_012HuCydr63ssfhGkTBzPMyF
EOF
)"
```

---

## Task 5: wiring no grafo (node + edge + Send + state)

**Files:**
- Modify: `core/redink_core/graph.py`
- Test: `core/tests/test_route_repro.py`

**Interfaces:**
- Consumes: `repro_check` (via `redink_core.nodes`), `_GITHUB_RE` de `redink_core.nodes_fetch`.
- Produces: `ReviewState.repro_result: NotRequired[Optional[dict]]`; `route_to_reviewers` emite `Send("repro_check", {"code_repo": <url>})` quando aplicável.

- [ ] **Step 1: Escrever o teste que falha**

Criar `core/tests/test_route_repro.py`:
```python
from redink_core.graph import route_to_reviewers
from redink_core.schemas import Classification


def _state(dims, code_repo=None, github_url=None):
    clf = Classification(
        area="Machine Learning", paper_type="empirical",
        dimensions=dims, citations=[], claims=[], code_repo=code_repo,
    )
    s = {"classification": clf, "paper": "corpo"}
    if github_url:
        s["github_url"] = github_url
    return s


def _has_repro(sends):
    return any(getattr(s, "node", None) == "repro_check" for s in sends)


def test_no_repro_send_when_flag_off(monkeypatch):
    monkeypatch.delenv("REDINK_REPRO", raising=False)
    sends = route_to_reviewers(_state(["reproducibility"], code_repo="https://github.com/foo/bar"))
    assert not _has_repro(sends)


def test_repro_send_when_flag_and_code_repo(monkeypatch):
    monkeypatch.setenv("REDINK_REPRO", "1")
    sends = route_to_reviewers(_state(["reproducibility"], code_repo="https://github.com/foo/bar"))
    assert _has_repro(sends)
    repro = next(s for s in sends if s.node == "repro_check")
    assert repro.arg["code_repo"] == "https://github.com/foo/bar"


def test_no_repro_send_without_reproducibility_dim(monkeypatch):
    monkeypatch.setenv("REDINK_REPRO", "1")
    sends = route_to_reviewers(_state(["methodology"], code_repo="https://github.com/foo/bar"))
    assert not _has_repro(sends)


def test_no_repro_send_without_repo(monkeypatch):
    monkeypatch.setenv("REDINK_REPRO", "1")
    sends = route_to_reviewers(_state(["reproducibility"]))
    assert not _has_repro(sends)


def test_falls_back_to_github_url_input(monkeypatch):
    monkeypatch.setenv("REDINK_REPRO", "1")
    sends = route_to_reviewers(_state(
        ["reproducibility"], github_url="https://github.com/foo/bar"))
    assert _has_repro(sends)
    repro = next(s for s in sends if s.node == "repro_check")
    assert repro.arg["code_repo"] == "https://github.com/foo/bar"


def test_arxiv_url_is_not_used_as_repo(monkeypatch):
    monkeypatch.setenv("REDINK_REPRO", "1")
    sends = route_to_reviewers(_state(
        ["reproducibility"], github_url="https://arxiv.org/abs/1706.03762"))
    assert not _has_repro(sends)
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `python -m pytest core/tests/test_route_repro.py -v`
Expected: FAIL — `route_to_reviewers` ainda não emite `Send("repro_check")`.

- [ ] **Step 3: Modificar `graph.py`**

Em `core/redink_core/graph.py`:

(a) Nos imports do topo, adicionar `os` e o regex de repo:
```python
import operator
import os
```
E na linha de import dos nodes (linha 13-16), adicionar `repro_check`:
```python
from redink_core.nodes import (
    fetch_paper, classify, reviewer, figure_reviewer,
    debate, contradiction_map, blind_spot, judge_panel, synthesize,
    repro_check,
)
from redink_core.nodes_fetch import _GITHUB_RE
```

(b) Em `ReviewState`, adicionar o campo (depois de `verdict`, ~linha 30):
```python
    repro_result: Optional[dict]
```

(c) Adicionar um helper antes de `route_to_reviewers` (~linha 45):
```python
def _repro_url(state: ReviewState) -> Optional[str]:
    """Repo oficial do paper: code_repo do classify, ou o input se já for um
    repo GitHub (arXiv/PDF não contam)."""
    clf = state["classification"]
    if getattr(clf, "code_repo", None):
        return clf.code_repo
    url = state.get("github_url")
    if url and _GITHUB_RE.match(url.strip()):
        return url
    return None
```

(d) No fim de `route_to_reviewers`, antes do `return sends`, adicionar o Send condicional:
```python
    if os.getenv("REDINK_REPRO") and "reproducibility" in clf.dimensions:
        repo_url = _repro_url(state)
        if repo_url:
            sends.append(Send("repro_check", {"code_repo": repo_url}))
    return sends
```

(e) Registrar o node e o edge. Depois de `builder.add_node("synthesize", synthesize)` (~linha 75):
```python
builder.add_node("repro_check", repro_check)
```
Na lista de destinos do conditional edge (linha 79), incluir `"repro_check"`:
```python
builder.add_conditional_edges("classify", route_to_reviewers, ["reviewer", "figure_reviewer", "repro_check"])
```
E o edge de junção, junto aos outros que apontam pra `debate` (~linha 81):
```python
builder.add_edge("repro_check", "debate")
```

- [ ] **Step 4: Rodar e ver passar**

Run: `python -m pytest core/tests/test_route_repro.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Sanidade — grafo compila e importa limpo**

Run: `python -c "from redink_core.graph import graph, graph_runner; print('ok')"`
Expected: imprime `ok` sem exceção.

- [ ] **Step 6: Commit**

```bash
git add core/redink_core/graph.py core/tests/test_route_repro.py
git commit -m "$(cat <<'EOF'
feat: wiring do repro_check no grafo

Send extra no fan-out (paralelo aos reviewers) quando REDINK_REPRO está
ligado, há code_repo (ou input github) e reproducibility é dimensão.
Edge repro_check -> debate; state ganha repro_result.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_012HuCydr63ssfhGkTBzPMyF
EOF
)"
```

---

## Task 6: surfacing do repro_result no veredito

**Files:**
- Modify: `core/redink_core/nodes_synthesis.py:194-290`
- Test: `core/tests/test_repro_surfacing.py`

**Interfaces:**
- Consumes: `state["repro_result"]` (dict com `status`).
- Produces: `_repro_summary_line(repro: dict) -> str` — linha determinística PT-BR pro veredito; `""` quando não deve aparecer.

- [ ] **Step 1: Escrever o teste que falha**

Criar `core/tests/test_repro_surfacing.py`:
```python
from redink_core.nodes_synthesis import _repro_summary_line


def test_ok_line():
    line = _repro_summary_line({"status": "ok", "package": "bar"})
    assert "instala" in line and "importa" in line
    assert line.startswith("✅")


def test_install_fail_line():
    line = _repro_summary_line({"status": "install_fail"})
    assert line.startswith("❌")
    assert "não instala" in line.lower() or "nao instala" in line.lower()


def test_import_fail_line():
    line = _repro_summary_line({"status": "import_fail"})
    assert line.startswith("❌")


def test_repo_missing_line():
    line = _repro_summary_line({"status": "repo_missing"})
    assert line.startswith("❌")


def test_timeout_line():
    line = _repro_summary_line({"status": "timeout"})
    assert line.startswith("⚠️")


def test_no_docker_is_silent():
    assert _repro_summary_line({"status": "no_docker"}) == ""


def test_none_is_silent():
    assert _repro_summary_line(None) == ""
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `python -m pytest core/tests/test_repro_surfacing.py -v`
Expected: FAIL — `cannot import name '_repro_summary_line'`.

- [ ] **Step 3: Implementar a função e ligar no synthesize**

Em `core/redink_core/nodes_synthesis.py`, adicionar a função pura (antes de `def synthesize`, ~linha 194):
```python
_REPRO_LINE = {
    "ok": "✅ Reprodutibilidade (executado): o código do repositório foi baixado, instala e importa.",
    "install_fail": "❌ Reprodutibilidade (executado): o código foi baixado mas NÃO instala.",
    "import_fail": "❌ Reprodutibilidade (executado): o código instala mas o import quebra.",
    "repo_missing": "❌ Reprodutibilidade (executado): o repositório linkado não existe ou está vazio.",
    "timeout": "⚠️ Reprodutibilidade (executado): a instalação/import passou do tempo limite.",
}


def _repro_summary_line(repro) -> str:
    """Linha factual pro veredito quando o repro_check rodou. '' se não deve
    aparecer (Docker indisponível, sem repo, ou repro_check não rodou)."""
    if not repro:
        return ""
    return _REPRO_LINE.get(repro.get("status", ""), "")
```

E no fim de `synthesize`, trocar o `return {"verdict": Verdict(...)}` por uma versão que prepara o verdict e injeta a linha:
```python
    verdict = Verdict(
        status=status, summary=summary.content,
        critical_count=len(critical), major_count=len(major), minor_count=len(minor),
        findings=sorted(findings, key=lambda f: _SEV_RANK[f.severity]),
        contradiction_map=c_map, blind_spots=b_spots, judge_panel=panel,
        high_confidence_issues=high_confidence,
    )
    repro_line = _repro_summary_line(state.get("repro_result"))
    if repro_line:
        verdict.summary = repro_line + "\n\n" + verdict.summary
    return {"verdict": verdict}
```

- [ ] **Step 4: Rodar e ver passar**

Run: `python -m pytest core/tests/test_repro_surfacing.py -v`
Expected: PASS (7 passed).

- [ ] **Step 5: Rodar a suíte inteira**

Run: `python -m pytest core/tests/ -v`
Expected: PASS (todos os testes das Tasks 1-6, ~34 passed).

- [ ] **Step 6: Commit**

```bash
git add core/redink_core/nodes_synthesis.py core/tests/test_repro_surfacing.py
git commit -m "$(cat <<'EOF'
feat: veredito cita o resultado do repro_check

Quando o repro_check rodou, prefixa uma linha factual no summary do
veredito (código instala/importa, ou a falha). Silencioso quando Docker
está indisponível. Metade do valor do moat é o sinal positivo.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_012HuCydr63ssfhGkTBzPMyF
EOF
)"
```

---

## Verificação end-to-end (manual, opcional — precisa Docker)

Com Docker rodando e uma OPENROUTER_API_KEY válida:

```bash
# repo pequeno que instala e importa → veredito deve citar "instala e importa"
REDINK_REPRO=1 redink https://github.com/psf/requests

# arquivo/arXiv sem repo → repro_check nem dispara (nenhum finding de execução)
REDINK_REPRO=1 redink automem.md
```

Sem `REDINK_REPRO`, o comportamento é idêntico ao de hoje (regressão zero).

---

## Self-Review (preenchido pelo autor do plano)

**Cobertura do spec:**
- Escopo clone/install/import determinístico → Task 2. ✓
- Isolamento Docker + limites → Task 2 (`_run_args`). ✓
- `code_repo` no classify → **campo no schema** (Task 1) + uso no route (Task 5). Nota: o *prompt* do classify não foi alterado; o campo tem `description` que instrui o LLM structured-output, suficiente pra v1. ✓
- `Finding.grounded` → Task 1. ✓
- Placement paralelo via Send → Task 5. ✓
- Debate bypass + dedup bypass → Task 4. ✓
- Mapa de severity → Task 3. ✓
- `repro_result` no state + surfacing → Tasks 3, 5, 6. ✓
- Opt-in REDINK_REPRO → Task 5. ✓
- Testes (unit + bypass + route + surfacing) → Tasks 1-6. Integração real gated em Docker é a seção de verificação manual. ✓

**Consistência de tipos:** `ReproResult(status, repo_url, package, log)` idêntico em Tasks 2/3. `repro_check → {"findings", "repro_result"}` consistente em Tasks 3/5/6. `_repro_url`, `_repro_summary_line`, `_docker_run`, `_to_finding` nomeados uma vez e usados coerentemente.
