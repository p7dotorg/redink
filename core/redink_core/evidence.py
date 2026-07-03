"""Evidence verification — findings must quote text that actually exists in the paper.

A finding whose evidence can't be located in the source loses its critical
status and most of its confidence: unverifiable evidence is the signature of
a hallucinated finding (invented quotes, tool output passed off as paper text,
claims about sections the reviewer never saw).
"""
import re
import unicodedata

_WS = re.compile(r"\s+")
_ELLIPSIS = re.compile(r"\.{3}|…|\[\.\.\.\]|\[…\]")


def _normalize(s: str) -> str:
    """Lowercase, straighten quotes/dashes, drop punctuation, collapse whitespace."""
    s = unicodedata.normalize("NFKD", s)
    s = (s.replace("“", '"').replace("”", '"')
          .replace("‘", "'").replace("’", "'")
          .replace("–", "-").replace("—", "-"))
    s = re.sub(r"[^\w\s]", " ", s.lower())
    return _WS.sub(" ", s).strip()


def _ngram_coverage(fragment_norm: str, paper_norm: str, n: int = 8) -> float:
    """Fraction of the fragment's word n-grams present in the paper."""
    words = fragment_norm.split()
    if len(words) < n:
        return 1.0 if fragment_norm in paper_norm else 0.0
    ngrams = [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]
    hits = sum(1 for g in ngrams if g in paper_norm)
    return hits / len(ngrams)


def evidence_in_paper(evidence: str, paper_norm: str) -> bool:
    """True if the evidence quote (possibly multi-fragment with ellipses)
    can be located in the normalized paper text."""
    if not evidence:
        return False
    fragments = [f for f in _ELLIPSIS.split(evidence) if f.strip()]
    checked = 0
    matched = 0
    for frag in fragments:
        frag_norm = _normalize(frag)
        if len(frag_norm.split()) < 4:
            continue  # too short to verify meaningfully
        checked += 1
        if frag_norm in paper_norm or _ngram_coverage(frag_norm, paper_norm) >= 0.5:
            matched += 1
    if checked == 0:
        return False
    return matched / checked >= 0.5


def verify_findings(findings: list, paper: str) -> list:
    """Mark each finding's evidence as verified/unverified against the paper.

    Unverified evidence drops the finding to minor with confidence <= 3 —
    an invented quote is the signature of a hallucinated finding, and letting
    it survive as major pollutes the report and the judges' input.
    The figures dimension is exempt — its evidence describes images, not text.
    """
    paper_norm = _normalize(paper)
    for f in findings:
        if f.dimension == "figures":
            continue
        if evidence_in_paper(getattr(f, "evidence", "") or "", paper_norm):
            f.evidence_verified = True
        else:
            f.evidence_verified = False
            f.severity = "minor"
            f.confidence = min(f.confidence, 3)
    return findings
