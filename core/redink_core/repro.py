"""repro.py — verificação por execução: clona o repo do paper num sandbox
Docker efêmero e testa se instala e importa. Zero deps novas: chama o CLI do
Docker por subprocess. Todo código do repo roda dentro do container, com rede
cortada na fase de import."""
import re
import subprocess
from dataclasses import dataclass
from typing import Optional

import httpx

# Imagem full (não -slim): traz git (pro clone) e build tools (gcc etc.) que
# pacotes com extensões C precisam pra instalar. slim não tem git.
_IMAGE = "python:3.11"
_LOG_TAIL = 2000

# Fase 1 (rede ligada): clona, instala, e grava os pacotes top-level do repo.
# Exit codes distintos deixam run_repro_check mapear a causa da falha.
_PHASE1 = r"""
set -o pipefail
cd /work
rm -rf repo site
git clone --depth 1 "$REPO_URL" repo 2>&1 || exit 10
cd repo
# Instala no VOLUME (/work/site), não no site-packages do container — a fase 2
# roda num container novo e só enxerga o volume compartilhado.
if [ -f pyproject.toml ] || [ -f setup.py ]; then
  pip install --quiet --target /work/site . 2>&1 || exit 20
elif [ -f requirements.txt ]; then
  pip install --quiet --target /work/site -r requirements.txt 2>&1 || exit 20
else
  exit 30
fi
python - <<'PY'
import pathlib
root = pathlib.Path('/work/repo')
# Dirs que têm __init__.py mas não são o pacote do paper.
DENY = {'tests', 'test', 'testing', 'docs', 'doc', 'examples', 'example',
        'benchmarks', 'benchmark', 'scripts', 'script', 'tools', 'notebooks',
        'samples', 'sample'}
cands = []
# src-layout primeiro (localização autoritativa do pacote), depois a raiz.
for base in (root / 'src', root):
    if base.is_dir():
        for d in sorted(base.iterdir()):
            if d.name.lower() in DENY:
                continue
            if (d / '__init__.py').exists():
                cands.append(d.name)
seen = set()
cands = [c for c in cands if not (c in seen or seen.add(c))]
pathlib.Path('/work/candidates.txt').write_text("\n".join(cands))
PY
"""

# Fase 2 (rede cortada): importa o pacote instalado. NÃO faz cd em /work/repo —
# importa o que o pip instalou no site-packages, não a árvore de fonte.
_PHASE2 = r"""
cd /work
# /work/site = deps+pacote instalados no volume; /work/repo = fonte (caso
# requirements.txt, onde o pacote é a própria árvore do repo).
export PYTHONPATH=/work/site:/work/repo
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


_REPO_PATH_RE = re.compile(r"https://github\.com/([\w.-]+)/(.+)")


def _repo_variants(url: str, paper: str = "") -> list[str]:
    """Candidatos ordenados pra uma URL de repo possivelmente mangada.

    URLs de paper vêm quebradas do PDF: esquema http, truncadas na quebra de
    linha (`.../Adversarial_`), ou com hífen virado espaço (`feature
    intertwiner`). Gera variantes plausíveis pra validação por rede escolher a
    que existe."""
    out: list[str] = []

    def add(u: str) -> None:
        u = u.rstrip("/")
        if u and u not in out:
            out.append(u)

    url = url.strip().rstrip(".,);:")
    if url.startswith("http://"):
        url = "https://" + url[len("http://"):]
    add(url)

    m = _REPO_PATH_RE.match(url)
    if not m:
        return out
    owner, repo = m.group(1), m.group(2)

    # hífen/underscore virou espaço na extração
    if " " in repo:
        add(f"https://github.com/{owner}/{repo.replace(' ', '-')}")
        add(f"https://github.com/{owner}/{repo.replace(' ', '_')}")

    # nome truncado na quebra do PDF — reconstrói do texto do paper. O corte
    # vira \n OU espaço (`Adversarial_ Video_Generation`), então permitimos UMA
    # quebra no meio do nome. Candidatos errados só dão 404 na validação.
    if paper:
        prefix = re.sub(r"[ _-]+$", "", repo.replace(" ", "")).lower()
        if len(prefix) >= 4:
            # nome já contíguo no texto
            for mm in re.finditer(re.escape(owner) + r"/([\w.-]+)", paper):
                name = mm.group(1).rstrip(".,);:")
                if name.lower().startswith(prefix[:4]) and len(name) > len(prefix):
                    add(f"https://github.com/{owner}/{name}")
            # nome cortado por UMA quebra (\n ou espaço) — junta os dois pedaços
            for mm in re.finditer(re.escape(owner) + r"/([\w.-]+[ \t\n]+[\w.-]+)", paper):
                name = re.sub(r"\s+", "", mm.group(1)).rstrip(".,);:")
                if name.lower().startswith(prefix[:4]):
                    add(f"https://github.com/{owner}/{name}")
    return out


def _repo_exists(url: str) -> bool:
    """True se a URL do repo responde 200 (segue redirect http->https)."""
    try:
        r = httpx.head(url, follow_redirects=True, timeout=10)
        if r.status_code == 405:  # GitHub às vezes recusa HEAD
            r = httpx.get(url, follow_redirects=True, timeout=10)
        return r.status_code == 200
    except Exception:
        return False


def resolve_repo_url(url: str, paper: str = "") -> Optional[str]:
    """Normaliza + valida a URL do repo contra o GitHub, testando variantes de
    URLs quebradas. Retorna a primeira que existe, ou None se nenhuma existe
    (repo genuinamente ausente, não bug de extração)."""
    if not url:
        return None
    for cand in _repo_variants(url, paper):
        if _repo_exists(cand):
            return cand
    return None


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
        # exit 10 = git clone falhou; 20 = pip install falhou; 30 = sem manifesto
        if rc1 == 10:
            return ReproResult("repo_missing", repo_url, log=out1)
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
