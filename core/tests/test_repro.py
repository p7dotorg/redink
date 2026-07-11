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
