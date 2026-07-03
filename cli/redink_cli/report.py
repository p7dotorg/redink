"""Render a Verdict as a Rich-formatted report."""
from rich.console import Console
from rich.rule import Rule
from rich.text import Text
from rich.panel import Panel
from rich.padding import Padding

console = Console()

SEVERITY_ORDER = {"critical": 0, "major": 1, "minor": 2}
PERSONA_LABEL  = {"skeptic": "skeptic", "practitioner": "practitioner", "academic": "academic"}

SEVERITY_STYLE = {
    "critical": ("red",    "CRITICAL"),
    "major":    ("yellow", "MAJOR"),
    "minor":    ("blue",   "MINOR"),
}


def _verdict_color(status: str) -> str:
    return {"PASS": "green", "REVISE": "yellow", "FAIL": "red"}.get(status, "white")


def format_report(verdict) -> str:
    """Return plain-text report (for file save / pipe output)."""
    lines = []
    lines.append("=" * 60)
    lines.append(f"  VERDICT: {verdict.status}")
    lines.append("=" * 60)
    lines.append(f"\n{verdict.summary}\n")
    lines.append(
        f"critical: {verdict.critical_count}  "
        f"major: {verdict.major_count}  "
        f"minor: {verdict.minor_count}"
    )

    if verdict.high_confidence_issues:
        lines.append("\nCONSENSUS (all personas agreed):")
        for issue in verdict.high_confidence_issues[:4]:
            lines.append(f"  · {issue}")

    sorted_findings = sorted(verdict.findings, key=lambda f: SEVERITY_ORDER.get(f.severity, 9))
    current_dim = None
    for f in sorted_findings:
        if f.dimension != current_dim:
            current_dim = f.dimension
            lines.append(f"\n── {f.dimension.upper()} " + "─" * (40 - len(f.dimension)))
        lines.append(f"\n  {f.severity.upper()} / {f.dimension} / {getattr(f, 'persona', '')}")
        lines.append(f"  {f.issue}")
        lines.append(f"  Evidence: {f.evidence}")
        lines.append(f"  Fix:      {f.suggestion}")

    c_map = getattr(verdict, "contradiction_map", None)
    if c_map and c_map.contradictions:
        lines.append("\n── CONTRADICTIONS " + "─" * 41)
        for c in c_map.contradictions[:4]:
            lines.append(f"\n  [{c.significance.upper()}] {c.dimension}  {c.persona_a} vs {c.persona_b}")
            lines.append(f"    A: {c.claim_a}")
            lines.append(f"    B: {c.claim_b}")

    b = getattr(verdict, "blind_spots", None)
    if b and b.topics_not_covered:
        lines.append("\n── BLIND SPOTS " + "─" * 44)
        for t in b.topics_not_covered[:3]:
            lines.append(f"  · {t}")
        if b.highest_priority:
            lines.append(f"\n  most critical: {b.highest_priority}")

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


def print_report(verdict) -> None:
    """Pretty-print the verdict to the terminal using Rich."""
    color = _verdict_color(verdict.status)

    # Verdict header
    console.print()
    console.print(Rule(
        Text(f"  {verdict.status}  ", style=f"bold {color} on {color}"),
        style=color,
    ))
    console.print()
    console.print(Text(verdict.summary, style="dim"), soft_wrap=True)
    console.print()

    counts = Text()
    counts.append(f"  {verdict.critical_count} critical", style="red")
    counts.append("  ·  ", style="dim")
    counts.append(f"{verdict.major_count} major", style="yellow")
    counts.append("  ·  ", style="dim")
    counts.append(f"{verdict.minor_count} minor", style="blue")
    console.print(counts)

    if verdict.high_confidence_issues:
        console.print()
        console.print("  Consensus", style="bold")
        for issue in verdict.high_confidence_issues[:4]:
            console.print(f"  · {issue}", style="dim")

    # Findings grouped by dimension
    sorted_findings = sorted(verdict.findings, key=lambda f: SEVERITY_ORDER.get(f.severity, 9))
    current_dim = None
    for f in sorted_findings:
        if f.dimension != current_dim:
            current_dim = f.dimension
            console.print()
            console.print(Rule(f.dimension.upper(), style="dim", align="left"))

        style, label = SEVERITY_STYLE.get(f.severity, ("white", f.severity.upper()))
        persona = getattr(f, "persona", "")
        conf    = getattr(f, "confidence", 5)

        header = Text()
        header.append(f"  {label}", style=f"bold {style}")
        header.append(f"  {persona}", style="dim")
        if conf != 5:
            header.append(f"  {conf}/10", style="dim")
        console.print(header)

        console.print(f"  {f.issue}", soft_wrap=True)
        console.print(f"  Evidence  {f.evidence}", style="dim", soft_wrap=True)
        console.print(f"  Fix       {f.suggestion}", style="dim", soft_wrap=True)
        console.print()

    # Contradictions
    c_map = getattr(verdict, "contradiction_map", None)
    if c_map and c_map.contradictions:
        console.print(Rule("CONTRADICTIONS", style="dim", align="left"))
        for c in c_map.contradictions[:4]:
            sig_style = {"high": "red", "medium": "yellow", "low": "blue"}.get(c.significance, "dim")
            console.print(f"  {c.dimension}  [{c.significance}]", style=sig_style)
            console.print(f"    {c.persona_a}: {c.claim_a}", style="dim", soft_wrap=True)
            console.print(f"    {c.persona_b}: {c.claim_b}", style="dim", soft_wrap=True)
            console.print()

    # Blind spots
    b = getattr(verdict, "blind_spots", None)
    if b and b.topics_not_covered:
        console.print(Rule("BLIND SPOTS", style="dim", align="left"))
        for t in b.topics_not_covered[:3]:
            console.print(f"  · {t}", style="dim", soft_wrap=True)
        if b.highest_priority:
            console.print()
            console.print(f"  most critical  {b.highest_priority}", style="yellow", soft_wrap=True)
        console.print()

    console.print(Rule(style="dim"))
