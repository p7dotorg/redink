"""Integração real do sandbox Docker — roda containers de verdade.

Gated em docker_available(): pulado em CI/máquinas sem Docker. Lento
(clona+instala repos reais), então marcado 'slow'. Roda com:
    .venv/bin/python -m pytest core/tests/test_repro_integration.py -v
"""
import pytest

from redink_core.repro import docker_available, run_repro_check

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not docker_available(), reason="Docker não disponível"),
]


def test_real_repo_installs_and_imports():
    # requests: pura-Python, src-layout, instala e importa rápido.
    r = run_repro_check("https://github.com/psf/requests", timeout=300)
    assert r.status == "ok", f"esperava ok, veio {r.status}: {r.log[-300:]}"
    # o pacote detectado deve ser 'requests', não o dir 'tests' de scaffold.
    assert r.package == "requests"


def test_nonexistent_repo_is_repo_missing():
    r = run_repro_check("https://github.com/foo/this-does-not-exist-xyz-42", timeout=120)
    assert r.status == "repo_missing"
