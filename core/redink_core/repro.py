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
