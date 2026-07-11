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
