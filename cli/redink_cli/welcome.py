"""Welcome screen for the redink chat REPL."""
import os
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()

VERSION = "0.1.0"

_R = "#E8252A"

# Half-block pixel art mirroring the SVG (14-col × 5-row character grid).
# Each char row represents 2 SVG pixel rows via ▀ (upper half) / ▄ (lower half).
#
# SVG grid (unit=40px, canvas=560px → 14×14 units):
#   body  x=2..11  y=2..9
#   arms  x=0..1 and x=12..13  y=7..8
#   eyes  x=4 (white) and x=8 (white)  y=4..5
#   legs  x=5 and x=8  y=10..11
_MASCOT = (
    f"  [{_R}]██████████[/]  \n"                                              # rows 2-3: body
    f"  [{_R}]██[/][#ffffff]█[/][{_R}]███[/][#ffffff]█[/][{_R}]███[/]  \n"  # rows 4-5: eyes
    f"[{_R}]▄▄██████████▄▄[/]\n"                                             # rows 6-7: arms (lower)
    f"[{_R}]▀▀██████████▀▀[/]\n"                                             # rows 8-9: arms (upper)
    f"     [{_R}]█[/]  [{_R}]█[/]     "                                      # rows 10-11: legs
)

TIPS = (
    "[bold red]Getting started[/bold red]\n"
    "  /review [dim]<paper.md>[/dim]\n"
    "  /review [dim]<github-url>[/dim]\n"
    "\n"
    "[bold red]In a review[/bold red]\n"
    "  /report  /rerun [dim]<dim>[/dim]  /exit\n"
    "\n"
    "[dim]Ask anything after a review.[/dim]"
)


def show_welcome() -> None:
    model = os.getenv("REVIEWER_MODEL", "deepseek/deepseek-v4-flash")
    cwd   = str(Path.cwd()).replace(str(Path.home()), "~")

    mascot = Text.from_markup(
        f"{_MASCOT}\n\n[dim]{model}[/dim]\n[dim]{cwd}[/dim]"
    )
    grid = Table.grid(expand=True)
    grid.add_column(ratio=1)
    grid.add_column(width=2)  # spacer
    grid.add_column(ratio=1)
    grid.add_row(mascot, Text(""), Text.from_markup(TIPS))

    console.print(Panel(
        grid,
        title=f"[bold red]redink[/bold red] [dim]v{VERSION}[/dim]",
        border_style="red dim",
        padding=(1, 2),
    ))
    console.print()
