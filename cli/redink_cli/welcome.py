"""Welcome screen for the redink chat REPL."""
import os
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

console = Console()

VERSION = "0.1.0"

_MASCOT = (
    "[red]  ██████████  [/red]\n"
    "[red]  ██[/red][white]  [/white][red]██[/red][white]  [/white][red]██  [/red]\n"
    "[red]████████████[/red]\n"
    "[red]████[/red][dim]●[/dim][red]███████[/red]\n"
    "[red]  ██████████  [/red]\n"
    "[red]    ██  ██    [/red]"
)

TIPS = (
    "[bold red]Getting started[/bold red]\n"
    "  /review [dim]<paper.md>[/dim]\n"
    "  /review [dim]<github-url>[/dim]\n"
    "\n"
    "[bold red]In a review[/bold red]\n"
    "  /report\n"
    "  /rerun [dim]<dimension>[/dim]\n"
    "  /exit\n"
    "\n"
    "[dim]Type freely after a review\n"
    "to ask questions about findings.[/dim]"
)


def show_welcome() -> None:
    model = os.getenv("REVIEWER_MODEL", "deepseek/deepseek-v4-flash")
    cwd   = str(Path.cwd()).replace(str(Path.home()), "~")

    left  = Text.from_markup(
        f"[bold]Welcome![/bold]\n\n{_MASCOT}\n\n"
        f"[dim]{model}[/dim]\n[dim]{cwd}[/dim]"
    )
    right = Text.from_markup(TIPS)

    grid = Table.grid(expand=True)
    grid.add_column(ratio=1)
    grid.add_column(ratio=1)
    grid.add_row(left, right)

    console.print(Panel(
        grid,
        title=f"[bold red]redink[/bold red] [dim]v{VERSION}[/dim]",
        border_style="red dim",
        padding=(1, 2),
    ))
    console.print()
