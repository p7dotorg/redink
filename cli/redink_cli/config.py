"""Central configuration: the registry of every env var redink/drl read, the
.env read/write, and a reusable wizard. One source of truth behind `redink
setup`, `drl setup`, and the in-chat `/config`.
"""
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.table import Table
from prompt_toolkit import prompt

console = Console()
_ENV = Path(".env")

GROUPS = ("shared", "papers", "datasets")


@dataclass
class Field:
    env: str
    label: str
    group: str
    default: str = ""
    secret: bool = False
    kind: str = "model"   # model | key | flag | path


FIELDS = [
    # ── shared ──────────────────────────────────────────────────────────────
    Field("OPENROUTER_API_KEY", "OpenRouter API key — LLM gateway (required)", "shared", secret=True, kind="key"),
    Field("LANGSMITH_API_KEY",  "LangSmith API key — tracing (optional)", "shared", secret=True, kind="key"),
    Field("LANGSMITH_TRACING",  "Enable LangSmith tracing", "shared", default="false", kind="flag"),
    # ── papers (reviewer pipeline) ──────────────────────────────────────────
    Field("CLASSIFY_MODEL",   "Classify model", "papers", default="openai/gpt-4o-mini"),
    Field("REVIEWER_MODEL",   "Reviewer / defender model", "papers", default="deepseek/deepseek-v4-flash"),
    Field("STRUCTURED_MODEL", "Structured-output model (findings, dedup)", "papers", default="openai/gpt-4o-mini"),
    Field("TOOL_MODEL",       "Tool-calling model (citations, novelty)", "papers", default="openai/gpt-4o-mini"),
    Field("FIGURE_MODEL",     "Figure / vision model", "papers", default="google/gemini-2.5-flash"),
    Field("SYNTHESIZE_MODEL", "Synthesis model (verdict prose)", "papers", default="deepseek/deepseek-v4-flash"),
    Field("JUDGE_MODEL",      "Judge panel + rebuttal model", "papers", default="openai/gpt-4o"),
    # ── datasets (DRL) ──────────────────────────────────────────────────────
    Field("DRL_SCORE_MODEL",  "Dataset opportunity scorer", "datasets", default="openai/gpt-4o-mini"),
    Field("DRL_BUNDLE",       "OKF bundle directory", "datasets", default="bundle", kind="path"),
    Field("KAGGLE_USERNAME",  "Kaggle username (optional — higher scan limits)", "datasets"),
    Field("KAGGLE_KEY",       "Kaggle API key (optional)", "datasets", secret=True, kind="key"),
]

_BY_GROUP = {g: [f for f in FIELDS if f.group == g] for g in GROUPS}


# ── .env I/O (line-preserving) ────────────────────────────────────────────────

def read_env() -> dict:
    env = {}
    if _ENV.exists():
        for line in _ENV.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, _, v = line.partition("=")
                env[k.strip()] = v.strip()
    return env


def write_env(updates: dict) -> None:
    """Update/append the given keys, preserving existing lines, comments, order."""
    lines = _ENV.read_text(encoding="utf-8").splitlines() if _ENV.exists() else []
    seen = set()
    for i, line in enumerate(lines):
        if "=" in line and not line.lstrip().startswith("#"):
            k = line.split("=", 1)[0].strip()
            if k in updates:
                lines[i] = f"{k}={updates[k]}"
                seen.add(k)
    for k, v in updates.items():
        if k not in seen and v:
            lines.append(f"{k}={v}")
    _ENV.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _apply_to_env() -> None:
    """Reload .env into the running process so mid-session changes take effect."""
    try:
        from dotenv import load_dotenv
        load_dotenv(str(_ENV), override=True)
    except Exception:
        pass


def _mask(v: str, secret: bool) -> str:
    if not v:
        return "[dim](unset)[/dim]"
    if secret:
        return f"{v[:6]}…{v[-4:]}" if len(v) > 12 else "set"
    return v


def _effective(env: dict, f: Field) -> str:
    return env.get(f.env) or f.default


# ── views + wizard ────────────────────────────────────────────────────────────

def show(groups=GROUPS) -> None:
    env = read_env()
    for g in groups:
        console.print(f"\n  [bold #e82529]{g}[/]")
        t = Table(show_header=False, box=None, padding=(0, 2))
        t.add_column(style="dim", width=18)
        t.add_column()
        for f in _BY_GROUP[g]:
            val = _effective(env, f)
            origin = "" if f.env in env else " [dim](default)[/dim]"
            t.add_row(f.env, _mask(val, f.secret) + origin)
        console.print(t)
    console.print()


def wizard(groups, header: str = "configuration") -> None:
    env = read_env()
    updates = {}
    console.print(f"\n  [bold #e82529]{header}[/]  [dim]· enter keeps current[/dim]\n")
    for g in groups:
        console.print(f"  [bold]{g}[/]")
        for f in _BY_GROUP[g]:
            cur = _effective(env, f)
            hint = _mask(cur, f.secret)
            tag = {"key": "🔑", "model": "◆", "flag": "⚑", "path": "📁"}.get(f.kind, "·")
            console.print(f"  {tag} {f.label}  [dim]({hint})[/dim]")
            try:
                val = prompt("    → ", is_password=f.secret).strip()
            except (KeyboardInterrupt, EOFError):
                console.print("\n  [dim]cancelled — nothing written[/dim]\n")
                return
            if val:
                updates[f.env] = val
        console.print()
    if updates:
        write_env(updates)
        _apply_to_env()
        console.print(f"  [green]✓[/] wrote {len(updates)} setting(s) to {_ENV}")
        if "OPENROUTER_API_KEY" in updates:
            _test_openrouter(updates["OPENROUTER_API_KEY"])
    else:
        console.print("  [dim]no changes[/dim]")
    console.print()


def _test_openrouter(key: str) -> None:
    try:
        import httpx
        with console.status("  testing OpenRouter key..."):
            r = httpx.get("https://openrouter.ai/api/v1/key",
                          headers={"Authorization": f"Bearer {key}"}, timeout=15)
        console.print("  [green]✓ OpenRouter key valid[/]" if r.status_code == 200
                      else "  [red]✗ OpenRouter key rejected[/]")
    except Exception:
        console.print("  [dim]could not verify key (offline?)[/dim]")


def full_setup() -> None:
    """`redink setup` — the whole config in sequence."""
    console.print("\n  [bold #e82529]redink setup[/]  ·  full configuration wizard")
    wizard(GROUPS, header="setup")


def config_command(arg: str) -> None:
    """In-chat `/config [papers|datasets]`."""
    arg = (arg or "").strip().lower()
    if not arg:
        show()
        console.print("  [dim]/config papers · /config datasets to edit[/dim]\n")
    elif arg in ("papers", "datasets"):
        wizard(["shared", arg] if arg == "papers" else [arg], header=f"config · {arg}")
    else:
        console.print(f"  [dim]usage: /config [papers|datasets][/dim]")
