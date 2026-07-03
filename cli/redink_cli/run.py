"""Stream a review run and emit clean Rich progress output."""
from rich.console import Console
from rich.text import Text

from redink_core.graph import graph_runner
from redink_cli.report import format_report, print_report as _print_report

console = Console()


def stream_review(input_state: dict) -> dict:
    """Run the graph with streaming. Returns final merged state."""
    final_state: dict = {}

    for chunk in graph_runner.stream(
        {**input_state, "findings": [], "classification": None,
         "contradiction_map": None, "blind_spots": None, "verdict": None},
        stream_mode="updates",
    ):
        for node, update in chunk.items():
            if node == "fetch_paper":
                _step("fetch")

            elif node == "classify":
                clf = update.get("classification")
                if clf:
                    detail = (
                        f"{clf.area}  ·  {clf.paper_type}  ·  "
                        f"{len(clf.claims)} claims  ·  "
                        + ", ".join(clf.dimensions)
                    )
                    _step("classify", detail)

            elif node == "reviewer":
                findings = update.get("findings", [])
                if findings:
                    dim     = findings[0].dimension
                    persona = findings[0].persona
                    _step(f"reviewer", f"{dim} / {persona}  {len(findings)} finding(s)", dim_style="dim")

            elif node == "figure_reviewer":
                findings = update.get("findings", [])
                _step("figures", f"{len(findings)} finding(s)")

            elif node == "contradiction_map":
                _step("contradictions")

            elif node == "blind_spot":
                _step("blind spots")

            elif node == "synthesize":
                _step("synthesize")

            # Merge state
            for k, v in update.items():
                if k == "findings" and isinstance(v, list):
                    final_state.setdefault("findings", [])
                    final_state["findings"].extend(v)
                else:
                    final_state[k] = v

    return final_state


def _step(label: str, detail: str = "", dim_style: str = "") -> None:
    line = Text()
    line.append("  ● ", style="green")
    line.append(f"{label:<18}", style="bold")
    if detail:
        line.append(detail, style="dim")
    console.print(line)


def render_report(state: dict) -> str:
    """Pretty-print verdict and return plain-text string for file save."""
    verdict = state.get("verdict")
    if not verdict:
        console.print("  no verdict returned", style="red dim")
        return ""
    _print_report(verdict)
    return format_report(verdict)
