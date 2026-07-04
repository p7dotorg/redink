"""drl — dataset research loop CLI.

  drl setup                 configure API keys → .env
  drl scan [query]          scan sources, score, write OKF concepts
      --sources hf          which sources (default: hf)
      --limit N             per-source cap (default: 50)
  drl rank [N]              top-N datasets by opportunity PageRank
  drl gaps [N]              least-covered task categories
  drl spikes [N]            recently-active datasets (velocity proxy)
  drl wiki <slug>           print an OKF concept
"""
import sys
from dotenv import load_dotenv
load_dotenv()

from rich.console import Console
from rich.rule import Rule
from rich.markdown import Markdown

console = Console()


def _int_arg(args, default):
    for a in args:
        if a.isdigit():
            return int(a)
    return default


def _cmd_scan(args) -> None:
    from redink_core.drl.graph import graph_runner
    from redink_core.drl import okf

    sources, limit, query_parts = ["hf"], 50, []
    i = 0
    while i < len(args):
        if args[i] == "--sources":
            sources = args[i + 1].split(","); i += 2
        elif args[i] == "--limit":
            limit = int(args[i + 1]); i += 2
        else:
            query_parts.append(args[i]); i += 1
    query = " ".join(query_parts)

    console.print()
    console.print(Rule(f"  drl scan  ·  {query or 'all'}  ·  {', '.join(sources)}  ", style="red dim"))
    counts = {}
    with console.status("  scanning + scoring...") as st:
        for chunk in graph_runner.stream(
            {"query": query, "sources": sources, "limit": limit}, stream_mode="updates"):
            for node, upd in (chunk or {}).items():
                counts[node] = counts.get(node, 0) + 1
                st.update(f"  {node}...")
    console.print(f"  [green]✓[/] wrote to bundle  [dim]{okf.bundle_dir()}/[/dim]")
    console.print("  [dim]drl rank · drl gaps · drl spikes[/dim]\n")


def _cmd_rank(args) -> None:
    from redink_core.drl import analysis
    top = _int_arg(args, 10)
    rows = analysis.rank(top=top)
    if not rows:
        return console.print("  [dim]empty bundle — run `drl scan <query>` first[/dim]")
    console.print(f"\n  [bold]top {len(rows)} by opportunity PageRank[/]\n")
    for title, score, d in rows:
        opp = d.get("opportunity", "?")
        console.print(f"  [#e82529]{score:.3f}[/]  {title}  [dim]· opp {opp}/3 · {d.get('downloads','?')} dl[/dim]")
    console.print()


def _cmd_gaps(args) -> None:
    from redink_core.drl import analysis
    top = _int_arg(args, 10)
    rows = analysis.task_gaps(top=top)
    if not rows:
        return console.print("  [dim]no task tags in bundle yet[/dim]")
    console.print(f"\n  [bold]least-covered tasks[/]  [dim](relative under-coverage in the catalog)[/dim]\n")
    for task, cov in rows:
        console.print(f"  [yellow]{cov:>3}[/] dataset(s)  {task}")
    console.print()


def _cmd_spikes(args) -> None:
    from redink_core.drl import analysis
    top = _int_arg(args, 10)
    rows = analysis.spikes(top=top)
    console.print(f"\n  [bold]recently active[/]  [dim](modified ≤7d · velocity proxy until scan history accrues)[/dim]\n")
    if not rows:
        return console.print("  [dim]nothing modified in the last 7 days[/dim]\n")
    for d in rows:
        console.print(f"  {d.get('title')}  [dim]· {d.get('downloads','?')} dl · opp {d.get('opportunity','?')}/3[/dim]")
    console.print()


def _cmd_wiki(args) -> None:
    from redink_core.drl import analysis
    if not args:
        return console.print("  [dim]usage: drl wiki <slug>[/dim]")
    md = analysis.read_concept(args[0])
    if md is None:
        return console.print(f"  [red]not found:[/] {args[0]}")

    # split OKF frontmatter (styled header) from body (rendered markdown)
    front, body = "", md
    if md.startswith("---"):
        end = md.find("\n---", 3)
        if end != -1:
            front, body = md[3:end].strip(), md[end + 4:].strip()

    console.print()
    if front:
        for line in front.splitlines():
            if ":" in line:
                k, _, v = line.partition(":")
                console.print(f"  [dim]{k.strip()}[/dim]  {v.strip()}")
        console.print()
    console.print(Markdown(body))
    console.print()


def _usage() -> None:
    console.print(Markdown(__doc__))


def main() -> None:
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        return _usage()
    cmd, rest = args[0], args[1:]
    if cmd == "setup":
        from redink_cli.config import wizard
        return wizard(["shared", "datasets"], header="drl setup")
    dispatch = {"scan": _cmd_scan, "rank": _cmd_rank, "gaps": _cmd_gaps,
                "spikes": _cmd_spikes, "wiki": _cmd_wiki}
    handler = dispatch.get(cmd)
    if handler:
        return handler(rest)
    console.print(f"  [dim]unknown command: {cmd}  —  drl --help[/dim]")
