"""redink — entry point.

  redink <paper.md>   one-shot review (CI / pipe)
  redink -            read from stdin
  redink              interactive chat REPL
"""
import sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from rich.console import Console
from rich.markdown import Markdown
from rich.rule import Rule
from rich.text import Text
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.styles import Style
from prompt_toolkit.formatted_text import HTML

from redink_cli.welcome import show_welcome, VERSION, TIPS
from redink_cli.run import stream_review, render_report
from redink_cli.report import format_report
from redink_cli.commands import cmd_review, cmd_rerun
from redink_cli.completer import RedinkCompleter
from redink_cli import drl_app

# dataset research loop — same slash-command surface, one chat
_DRL_COMMANDS = {
    "/scan":   drl_app._cmd_scan,
    "/rank":   drl_app._cmd_rank,
    "/gaps":   drl_app._cmd_gaps,
    "/spikes": drl_app._cmd_spikes,
    "/wiki":   drl_app._cmd_wiki,
}

console = Console()
_PROMPT_STYLE = Style.from_dict({
    "prompt": "#e82529 bold",
    "bottom-toolbar": "#888888 bg:#1c1c1c",
})
_HISTORY_FILE = Path.home() / ".redink" / "history"


# ── one-shot ──────────────────────────────────────────────────────────────────

def _run_oneshot(target: str) -> None:
    github_url, paper_text = None, None
    if target == "-":
        paper_text   = sys.stdin.read()
        display_name = "stdin"
    elif target.startswith("https://"):
        github_url   = target
        display_name = target.rsplit("/", 1)[-1]
    else:
        path = Path(target)
        if not path.exists():
            console.print(f"  [red]error: file not found: {path}[/red]")
            sys.exit(1)
        paper_text   = path.read_text(encoding="utf-8")
        display_name = path.name

    console.print()
    console.print(Rule(
        Text(f"  redink  ·  {display_name}  ", style="bold red"),
        style="red dim",
    ))
    console.print()

    input_state = {"github_url": github_url} if github_url else {"paper": paper_text}
    final_state = stream_review(input_state)
    report      = render_report(final_state)

    if target != "-":
        stem = display_name.rsplit(".", 1)[0] if "." in display_name else display_name
        out_path = (
            Path(f"{display_name}.review.md")
            if target.startswith("https://")
            else Path(target).with_suffix(".review.md")
        )
        out_path.write_text(f"# redink: {display_name}\n\n{report}", encoding="utf-8")
        console.print(f"\n  saved  [dim]{out_path}[/dim]")

        from redink_cli.commands import _try_annotate
        _try_annotate(final_state, stem)


# ── chat REPL ─────────────────────────────────────────────────────────────────

def _chat_loop() -> None:
    from redink_core.chat import answer as core_answer

    show_welcome()
    last_state: dict   = {}
    chat_history: list = []

    def _dimensions() -> list:
        clf = last_state.get("classification")
        return list(getattr(clf, "dimensions", [])) if clf else []

    def _toolbar():
        v = last_state.get("verdict")
        if not v:
            return HTML(" <b>redink</b>  ·  no review yet  —  /review &lt;paper&gt;")
        color = {"PASS": "ansigreen", "REVISE": "ansiyellow", "FAIL": "ansired"}.get(v.status, "")
        return HTML(
            f" <b>redink</b>  ·  <{color}>{v.status}</{color}>  ·  "
            f"{v.critical_count}C {v.major_count}M {v.minor_count}m  "
            f"·  ask anything, or /report /rerun /exit"
        )

    _HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    session: PromptSession = PromptSession(
        history=FileHistory(str(_HISTORY_FILE)),
        style=_PROMPT_STYLE,
        completer=RedinkCompleter(_dimensions),
        complete_while_typing=True,
        auto_suggest=AutoSuggestFromHistory(),
        bottom_toolbar=_toolbar,
    )

    while True:
        try:
            raw = session.prompt([("class:prompt", "› ")]).strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n  [dim]bye[/dim]")
            break

        if not raw:
            continue

        if raw.startswith("/"):
            parts = raw.split(maxsplit=1)
            cmd   = parts[0].lower()
            arg   = parts[1].strip() if len(parts) > 1 else ""

            if cmd in ("/exit", "/quit"):
                console.print("  [dim]bye[/dim]")
                break
            elif cmd == "/help":
                console.print(TIPS)
            elif cmd == "/clear":
                console.clear()
                show_welcome()
            elif cmd == "/report":
                if last_state.get("verdict"):
                    from redink_cli.report import print_report
                    print_report(last_state["verdict"])
                else:
                    console.print("  [dim]No review yet. Use /review <paper>[/dim]")
            elif cmd == "/review":
                state = cmd_review(arg, stream_review, render_report)
                if state:
                    last_state   = state
                    chat_history = []
            elif cmd == "/rerun":
                cmd_rerun(arg, last_state)
            elif cmd in _DRL_COMMANDS:
                _DRL_COMMANDS[cmd](arg.split())
            else:
                console.print(f"  [dim]unknown: {cmd}  —  /help[/dim]")

        else:
            if not last_state.get("verdict"):
                console.print("  [dim]Run /review <paper> first.[/dim]")
                continue
            with console.status("  [dim]thinking...[/dim]", spinner="dots"):
                reply = core_answer(raw, last_state, chat_history)
            chat_history.append({"role": "user",      "content": raw})
            chat_history.append({"role": "assistant",  "content": reply})
            console.print()
            console.print(Markdown(reply))
            console.print()


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    args = sys.argv[1:]

    if args and args[0] in ("-h", "--help"):
        console.print(TIPS)
        sys.exit(0)

    if args and args[0] in ("-v", "--version"):
        print(f"redink {VERSION}")
        sys.exit(0)

    if args or not sys.stdin.isatty():
        _run_oneshot(args[0] if args else "-")
    else:
        _chat_loop()
