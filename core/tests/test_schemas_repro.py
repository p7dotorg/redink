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
