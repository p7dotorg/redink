"""Stream a review run and emit clean Rich progress output."""
from rich.console import Console, Group
from rich.live import Live
from rich.table import Table
from rich.text import Text

from redink_core.graph import graph_runner
from redink_cli.report import format_report, print_report as _print_report

console = Console()

_PERSONAS = ["skeptic", "practitioner", "academic"]

_ICON_WAITING  = ("○", "dim")
_ICON_RUNNING  = ("◌", "yellow")
_ICON_DONE     = ("✓", "green")


def _icon(status: str) -> tuple[str, str]:
    if status == "done":
        return _ICON_DONE
    if status == "running":
        return _ICON_RUNNING
    return _ICON_WAITING


def _build_display(
    step_status: dict,
    classify_detail: str,
    reviewer_grid: dict,
    figures_count: int | None,
    debate_detail: str = "",
    judge_detail: str = "",
) -> Group:
    lines: list = []

    # ── fetch ──────────────────────────────────────────────────────────────
    sym, sty = _icon(step_status.get("fetch", "waiting"))
    t = Text()
    t.append(f"  {sym} ", style=sty)
    t.append(f"{'fetch':<18}", style="bold")
    if step_status.get("fetch") == "done":
        t.append("done", style="green")
    lines.append(t)

    # ── classify ───────────────────────────────────────────────────────────
    sym, sty = _icon(step_status.get("classify", "waiting"))
    t = Text()
    t.append(f"  {sym} ", style=sty)
    t.append(f"{'classify':<18}", style="bold")
    if step_status.get("classify") == "done":
        t.append("done", style="green")
        if classify_detail:
            t.append("  ·  ", style="dim")
            t.append(classify_detail, style="dim")
    lines.append(t)

    # ── reviewer grid (only when we have dims) ─────────────────────────────
    if reviewer_grid:
        lines.append(Text(""))

        tbl = Table(
            show_header=True,
            show_edge=False,
            box=None,
            padding=(0, 2),
        )
        tbl.add_column("dimension", style="", width=16)
        tbl.add_column("skeptic",      justify="left")
        tbl.add_column("practitioner", justify="left")
        tbl.add_column("academic",     justify="left")

        dims = list(dict.fromkeys(d for (d, _) in reviewer_grid))

        for dim in dims:
            cells: list[Text] = [Text(dim, style="dim")]
            for persona in _PERSONAS:
                val = reviewer_grid.get((dim, persona))
                if val is None:
                    # check if running
                    if reviewer_grid.get((dim, persona), "sentinel") == "running":
                        cells.append(Text("◌ ...", style="yellow"))
                    else:
                        cells.append(Text("○", style="dim"))
                elif val == "running":
                    cells.append(Text("◌ ...", style="yellow"))
                else:
                    cells.append(Text(f"✓ {val} finding(s)", style="green"))
            tbl.add_row(*cells)

        # figures row
        if figures_count is not None:
            fig_sym, fig_sty = _icon("done")
            fig_cell = Text(f"{fig_sym} {figures_count} finding(s)", style=fig_sty)
            tbl.add_row(
                Text("figures / vision", style="dim"),
                fig_cell,
                Text(""),
                Text(""),
            )
        elif step_status.get("figures") == "running":
            tbl.add_row(
                Text("figures / vision", style="dim"),
                Text("◌ ...", style="yellow"),
                Text(""),
                Text(""),
            )

        lines.append(tbl)
        lines.append(Text(""))

    # ── debate ─────────────────────────────────────────────────────────────
    sym, sty = _icon(step_status.get("debate", "waiting"))
    t = Text()
    t.append(f"  {sym} ", style=sty)
    t.append(f"{'debate':<18}", style="bold")
    if step_status.get("debate") == "done":
        t.append("done", style="green")
        if debate_detail:
            t.append("  ·  ", style="dim")
            t.append(debate_detail, style="dim")
    lines.append(t)

    # ── contradiction_map ──────────────────────────────────────────────────
    sym, sty = _icon(step_status.get("contradictions", "waiting"))
    t = Text()
    t.append(f"  {sym} ", style=sty)
    t.append(f"{'contradictions':<18}", style="bold")
    if step_status.get("contradictions") == "done":
        t.append("done", style="green")
    lines.append(t)

    # ── blind_spot ─────────────────────────────────────────────────────────
    sym, sty = _icon(step_status.get("blind_spots", "waiting"))
    t = Text()
    t.append(f"  {sym} ", style=sty)
    t.append(f"{'blind spots':<18}", style="bold")
    if step_status.get("blind_spots") == "done":
        t.append("done", style="green")
    lines.append(t)

    # ── judge panel ────────────────────────────────────────────────────────
    sym, sty = _icon(step_status.get("judges", "waiting"))
    t = Text()
    t.append(f"  {sym} ", style=sty)
    t.append(f"{'judge panel':<18}", style="bold")
    if step_status.get("judges") == "done":
        t.append("done", style="green")
        if judge_detail:
            t.append("  ·  ", style="dim")
            t.append(judge_detail, style="dim")
    lines.append(t)

    # ── synthesize ─────────────────────────────────────────────────────────
    sym, sty = _icon(step_status.get("synthesize", "waiting"))
    t = Text()
    t.append(f"  {sym} ", style=sty)
    t.append(f"{'synthesize':<18}", style="bold")
    if step_status.get("synthesize") == "done":
        t.append("done", style="green")
    else:
        t.append("waiting", style="dim")
    lines.append(t)

    return Group(*lines)


def stream_review(input_state: dict) -> dict:
    """Run the graph with streaming. Returns final merged state."""
    final_state: dict = {}

    step_status: dict[str, str] = {
        "fetch":         "waiting",
        "classify":      "waiting",
        "debate":        "waiting",
        "contradictions":"waiting",
        "blind_spots":   "waiting",
        "judges":        "waiting",
        "synthesize":    "waiting",
    }
    classify_detail: str = ""
    reviewer_grid: dict[tuple[str, str], str | int | None] = {}
    figures_count: int | None = None
    debate_detail: str = ""
    judge_detail: str = ""

    with Live(
        _build_display(step_status, classify_detail, reviewer_grid, figures_count),
        console=console,
        refresh_per_second=12,
        transient=False,
    ) as live:

        def _refresh() -> None:
            live.update(
                _build_display(step_status, classify_detail, reviewer_grid,
                               figures_count, debate_detail, judge_detail)
            )

        for chunk in graph_runner.stream(
            {**input_state, "findings": [], "classification": None,
             "contradiction_map": None, "blind_spots": None, "verdict": None},
            stream_mode="updates",
        ):
            for node, update in chunk.items():

                if node == "fetch_paper":
                    step_status["fetch"] = "done"
                    _refresh()

                elif node == "classify":
                    clf = update.get("classification")
                    if clf:
                        classify_detail = (
                            f"{clf.area}  ·  {clf.paper_type}  ·  "
                            f"{len(clf.claims)} claims  ·  "
                            + ", ".join(clf.dimensions)
                        )
                        # initialise grid – exclude "figures" dimension
                        for dim in clf.dimensions:
                            if dim == "figures":
                                continue
                            for persona in _PERSONAS:
                                reviewer_grid.setdefault((dim, persona), None)
                    step_status["classify"] = "done"
                    _refresh()

                elif node == "reviewer":
                    findings = update.get("findings", [])
                    if findings:
                        dim     = findings[0].dimension
                        persona = findings[0].persona
                        # mark any previously-running cell for this dim/persona as done
                        reviewer_grid[(dim, persona)] = len(findings)
                    _refresh()

                elif node == "figure_reviewer":
                    findings = update.get("findings", [])
                    figures_count = len(findings)
                    step_status["figures"] = "done"
                    _refresh()

                elif node == "debate":
                    dd = update.get("deduped_findings") or []
                    sustained  = sum(1 for f in dd if f.debate_outcome == "sustained")
                    downgraded = sum(1 for f in dd if f.debate_outcome == "downgraded")
                    debate_detail = (
                        f"{len(dd)} findings após dedup  ·  "
                        f"{sustained} sustained  ·  {downgraded} downgraded"
                    )
                    step_status["debate"] = "done"
                    _refresh()

                elif node == "contradiction_map":
                    step_status["contradictions"] = "done"
                    _refresh()

                elif node == "blind_spot":
                    step_status["blind_spots"] = "done"
                    _refresh()

                elif node == "judge_panel":
                    jp = update.get("judge_votes")
                    if jp:
                        judge_detail = "  ".join(f"{v.lens}: {v.vote}" for v in jp.votes)
                        judge_detail += f"  →  {jp.verdict}"
                    step_status["judges"] = "done"
                    _refresh()

                elif node == "synthesize":
                    step_status["synthesize"] = "done"
                    _refresh()

                # Merge state
                if update:
                    for k, v in update.items():
                        if k == "findings" and isinstance(v, list):
                            final_state.setdefault("findings", [])
                            final_state["findings"].extend(v)
                        else:
                            final_state[k] = v

    console.print("")
    return final_state


def render_report(state: dict) -> str:
    """Pretty-print verdict and return plain-text string for file save."""
    verdict = state.get("verdict")
    if not verdict:
        console.print("  no verdict returned", style="red dim")
        return ""
    _print_report(verdict)
    return format_report(verdict)
