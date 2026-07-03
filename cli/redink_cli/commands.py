"""Slash command handlers for the redink chat REPL."""
from pathlib import Path

from rich.console import Console
from rich.rule import Rule

console = Console()

PERSONAS = ["skeptic", "practitioner", "academic"]


def _try_annotate(state: dict, output_stem: str) -> None:
    """Generate interactive HTML report; also PDF if arXiv ID found."""
    from redink_core.nodes_helpers import extract_arxiv_id
    from redink_cli.html_annotator import generate as html_generate

    paper    = state.get("paper", "")
    findings = state.get("findings", [])
    verdict  = state.get("verdict")
    if not findings or not verdict:
        return

    # Always generate HTML
    html_out = Path(f"{output_stem}.annotated.html")
    title = Path(output_stem).name
    html_generate(paper, findings, verdict, html_out, title=title)
    console.print(f"  [#E8252A]●[/] interactive report  [dim]{html_out}[/dim]")

    # PDF only for arXiv papers
    arxiv_id = extract_arxiv_id(paper)
    if arxiv_id:
        from redink_cli.pdf_annotator import annotate, summary_page
        pdf_out = Path(f"{output_stem}.annotated.pdf")
        console.print(f"  [dim]annotating PDF {arxiv_id} ...[/dim]")
        if annotate(arxiv_id, findings, pdf_out):
            if verdict:
                summary_page(pdf_out, verdict)
            console.print(f"  [#E8252A]●[/] annotated PDF     [dim]{pdf_out}[/dim]")


def cmd_review(arg: str, stream_review, render_report) -> dict:
    """Run /review <path|url> and return final state."""
    if not arg:
        console.print("  [dim]usage: /review <path|url>[/dim]")
        return {}
    if arg.startswith("https://"):
        input_state = {"github_url": arg}
        stem = arg.rsplit("/", 1)[-1]
    else:
        p = Path(arg)
        if not p.exists():
            console.print(f"  [red]file not found: {p}[/red]")
            return {}
        input_state = {"paper": p.read_text(encoding="utf-8")}
        stem = p.stem

    console.print()
    state = stream_review(input_state)
    render_report(state)
    _try_annotate(state, stem)
    return state


def cmd_rerun(dim: str, state: dict) -> None:
    """Re-run a single dimension and merge findings back into state."""
    from redink_core.nodes import reviewer, figure_reviewer

    clf = state.get("classification")
    if not clf:
        console.print("  [dim]Run /review first.[/dim]")
        return

    if dim not in clf.dimensions:
        console.print(
            f"  [red]{dim!r} not in this paper's dimensions.[/red]\n"
            f"  available: {', '.join(clf.dimensions)}"
        )
        return

    console.print()
    console.print(Rule(f"rerun  {dim}", style="dim red", align="left"))
    paper        = state.get("paper", "")
    new_findings = []

    if dim == "figures":
        result       = figure_reviewer({"paper": paper, "classification": clf})
        new_findings = result.get("findings", [])
    else:
        for persona in PERSONAS:
            result = reviewer({
                "paper": paper, "classification": clf,
                "dimension": dim, "persona": persona,
            })
            new_findings.extend(result.get("findings", []))

    old               = [f for f in state.get("findings", []) if f.dimension != dim]
    state["findings"] = old + new_findings

    console.print()
    for f in new_findings:
        sty = {"critical": "red", "major": "yellow", "minor": "blue"}.get(f.severity, "dim")
        console.print(
            f"  [bold {sty}]{f.severity.upper()}[/bold {sty}]  "
            f"[dim]{f.persona}[/dim]  {f.issue}",
            soft_wrap=True,
        )
    console.print()
