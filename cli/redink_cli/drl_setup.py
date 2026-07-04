"""`drl setup` — interactive wizard that writes/updates .env and tests keys."""
from pathlib import Path

from rich.console import Console
from prompt_toolkit import prompt

console = Console()
_ENV = Path(".env")

# (key, prompt, required, secret)
_FIELDS = [
    ("OPENROUTER_API_KEY", "OpenRouter API key (required — LLM scoring)", True, True),
    ("KAGGLE_USERNAME", "Kaggle username (optional — Kaggle scanning)", False, False),
    ("KAGGLE_KEY", "Kaggle API key (optional)", False, True),
]


def _read_env() -> dict:
    env = {}
    if _ENV.exists():
        for line in _ENV.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, _, v = line.partition("=")
                env[k.strip()] = v.strip()
    return env


def _write_env(env: dict) -> None:
    lines = [f"{k}={v}" for k, v in env.items() if v]
    _ENV.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _mask(v: str) -> str:
    return f"{v[:6]}…{v[-4:]}" if v and len(v) > 12 else ("set" if v else "")


def _test_openrouter(key: str) -> bool:
    try:
        import httpx
        r = httpx.get("https://openrouter.ai/api/v1/key",
                      headers={"Authorization": f"Bearer {key}"}, timeout=15)
        return r.status_code == 200
    except Exception:
        return False


def run_setup() -> None:
    console.print("\n  [bold #e82529]drl setup[/]  ·  configure keys → .env\n")
    env = _read_env()

    for key, label, required, secret in _FIELDS:
        current = env.get(key, "")
        shown = f"  [dim](current: {_mask(current)})[/dim]" if current else ""
        req = "[red]*[/red]" if required else "[dim](optional)[/dim]"
        console.print(f"  {req} {label}{shown}")
        try:
            val = prompt("    → ", is_password=secret).strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n  [dim]cancelled[/dim]")
            return
        if val:
            env[key] = val
        elif required and not current:
            console.print("    [yellow]skipped — required, config incomplete[/yellow]")
        console.print()

    _write_env(env)
    console.print(f"  [green]✓[/] wrote {_ENV}")

    orkey = env.get("OPENROUTER_API_KEY", "")
    if orkey:
        with console.status("  testing OpenRouter key..."):
            ok = _test_openrouter(orkey)
        console.print("  [green]✓ OpenRouter key valid[/]" if ok
                      else "  [red]✗ OpenRouter key rejected[/] — double-check it")
    console.print()
