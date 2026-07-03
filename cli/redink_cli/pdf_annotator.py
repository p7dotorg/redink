"""Annotate arXiv PDFs with review findings using red ink markers."""
import io
from pathlib import Path

import fitz  # PyMuPDF
import httpx

# Brand red + severity variants
_COLOR = {
    "critical": (0.910, 0.145, 0.165),   # #E8252A
    "major":    (0.910, 0.145, 0.165),
    "minor":    (0.910, 0.145, 0.165),
}
_FILL = {
    "critical": (1.0, 0.82, 0.82),
    "major":    (1.0, 0.90, 0.80),
    "minor":    (1.0, 0.96, 0.88),
}
_SEVERITY_ORDER = {"critical": 0, "major": 1, "minor": 2}


def annotate(arxiv_id: str, findings: list, output_path: str | Path) -> bool:
    """Download arXiv PDF, annotate with findings, save to output_path.

    Returns True on success, False if download fails or no PDF found.
    """
    pdf_bytes = _fetch_pdf(arxiv_id)
    if not pdf_bytes:
        return False

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    sorted_findings = sorted(findings, key=lambda f: _SEVERITY_ORDER.get(f.severity, 9))

    for finding in sorted_findings:
        evidence = finding.evidence or ""
        if not evidence:
            continue

        # Try progressively shorter evidence snippets to find the text
        for length in (80, 50, 30):
            snippet = evidence[:length].strip()
            if len(snippet) < 10:
                continue
            hit = _annotate_snippet(doc, snippet, finding)
            if hit:
                break

    doc.save(str(output_path), garbage=4, deflate=True)
    doc.close()
    return True


def _fetch_pdf(arxiv_id: str) -> bytes | None:
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    try:
        r = httpx.get(url, follow_redirects=True, timeout=30)
        if r.status_code == 200 and r.headers.get("content-type", "").startswith("application/pdf"):
            return r.content
    except Exception:
        pass
    return None


def _annotate_snippet(doc: fitz.Document, snippet: str, finding) -> bool:
    """Search for snippet in doc, add annotations. Returns True if found."""
    color = _COLOR.get(finding.severity, _COLOR["minor"])
    fill  = _FILL.get(finding.severity, _FILL["minor"])
    label = f"[{finding.severity.upper()}] {finding.issue}"

    for page in doc:
        hits = page.search_for(snippet, quads=True)
        if not hits:
            continue

        for quad in hits[:1]:  # only first occurrence per page
            # Highlight the text
            hl = page.add_highlight_annot(quad)
            hl.set_colors(stroke=color)
            hl.update()

            # Sticky note in the right margin
            margin_pt = fitz.Point(page.rect.width - 20, quad.ul.y)
            note = page.add_text_annot(margin_pt, label[:200])
            note.set_colors(stroke=color, fill=fill)
            note.update()

        return True

    return False


def summary_page(doc_path: str | Path, verdict) -> None:
    """Prepend a summary page with verdict + finding counts to an annotated PDF."""
    doc = fitz.open(str(doc_path))
    doc.insert_page(0)
    page = doc[0]

    RED   = (0.910, 0.145, 0.165)
    BLACK = (0.1, 0.1, 0.1)
    GRAY  = (0.5, 0.5, 0.5)

    y = 60
    # Title
    page.insert_text((60, y), "redink review", fontsize=22, color=RED, fontname="helv")
    y += 35
    status_color = {"PASS": (0.1, 0.7, 0.1), "REVISE": (0.9, 0.6, 0.0), "FAIL": RED}
    color = status_color.get(getattr(verdict, "status", ""), BLACK)
    page.insert_text((60, y), getattr(verdict, "status", ""), fontsize=18, color=color, fontname="helv")
    y += 30
    page.insert_text((60, y), getattr(verdict, "summary", "")[:300], fontsize=10, color=BLACK, fontname="helv")
    y += 30

    counts = (
        f"critical: {getattr(verdict, 'critical_count', 0)}   "
        f"major: {getattr(verdict, 'major_count', 0)}   "
        f"minor: {getattr(verdict, 'minor_count', 0)}"
    )
    page.insert_text((60, y), counts, fontsize=11, color=GRAY, fontname="helv")
    y += 40

    page.insert_text((60, y), "Annotated findings follow in the paper →", fontsize=10, color=GRAY, fontname="helv")

    # PyMuPDF can't save non-incrementally onto the file it was opened from,
    # and inserting a page rules out incremental save — write a sibling and swap.
    tmp_path = Path(doc_path).with_suffix(".tmp.pdf")
    doc.save(str(tmp_path), garbage=4, deflate=True)
    doc.close()
    tmp_path.replace(doc_path)
