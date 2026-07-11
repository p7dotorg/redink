from unittest.mock import patch

from redink_core.repro import _repo_variants, resolve_repo_url


def test_variant_normalizes_http_to_https():
    vs = _repo_variants("http://github.com/isosnovik/sesn")
    assert vs[0] == "https://github.com/isosnovik/sesn"


def test_variant_strips_trailing_punctuation():
    vs = _repo_variants("https://github.com/foo/bar).")
    assert "https://github.com/foo/bar" in vs


def test_variant_space_becomes_hyphen_and_underscore():
    vs = _repo_variants("https://github.com/hli2020/feature intertwiner")
    assert "https://github.com/hli2020/feature-intertwiner" in vs
    assert "https://github.com/hli2020/feature_intertwiner" in vs


def test_variant_reconstructs_truncated_from_paper():
    # a URL veio cortada na quebra de linha; o nome completo está no paper
    paper = "code at https://github.com/dyelax/Adversarial_Video_Generation for details"
    vs = _repo_variants("https://github.com/dyelax/Adversarial_", paper)
    assert "https://github.com/dyelax/Adversarial_Video_Generation" in vs


def test_variant_reconstructs_across_linebreak():
    # o PDF quebrou a URL no meio
    paper = "see https://github.com/dyelax/Adversarial_\nVideo_Generation here"
    vs = _repo_variants("https://github.com/dyelax/Adversarial_", paper)
    assert "https://github.com/dyelax/Adversarial_Video_Generation" in vs


def test_resolve_returns_first_existing_variant():
    def exists(u):
        return u == "https://github.com/hli2020/feature-intertwiner"
    with patch("redink_core.repro._repo_exists", side_effect=exists):
        got = resolve_repo_url("https://github.com/hli2020/feature intertwiner")
    assert got == "https://github.com/hli2020/feature-intertwiner"


def test_resolve_none_when_nothing_exists():
    with patch("redink_core.repro._repo_exists", return_value=False):
        assert resolve_repo_url("https://github.com/foo/ghost") is None


def test_resolve_none_on_empty():
    assert resolve_repo_url("") is None
