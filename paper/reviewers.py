"""CLI-based reviewers using Claude and Kimi subscriptions (zero API cost)."""
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

from langsmith import traceable


@traceable(name="claude-cli", run_type="llm", metadata={"provider": "anthropic", "via": "cli"})
def _call_claude_cli(prompt: str, timeout: int = 180) -> str:
    try:
        r = subprocess.run(
            ["claude", "-p"],
            input=prompt, capture_output=True, text=True,
            timeout=timeout, errors="replace",
        )
        return r.stdout.strip()
    except Exception as e:
        return f"[claude cli error: {e}]"


@traceable(name="kimi-cli", run_type="llm", metadata={"provider": "moonshotai", "via": "kimi-cli"})
def _call_kimi_cli(prompt: str, timeout: int = 180) -> str:
    model = os.getenv("KIMI_MODEL", "kimi-code/kimi-for-coding")
    try:
        r = subprocess.run(
            ["kimi", "-p", prompt, "-m", model, "--output-format", "text"],
            capture_output=True, text=True, timeout=timeout, errors="replace",
        )
        return r.stdout.strip()
    except Exception as e:
        return f"[kimi cli error: {e}]"


def run_cli_reviewers(prompt: str) -> str:
    """Run Claude CLI and Kimi in parallel, combine outputs."""
    use_claude = os.getenv("USE_CLAUDE_CLI", "true").lower() == "true"
    use_kimi = os.getenv("USE_KIMI_CLI", "false").lower() == "true"

    fns = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        if use_claude:
            fns["Claude"] = pool.submit(_call_claude_cli, prompt)
        if use_kimi:
            fns["Kimi"] = pool.submit(_call_kimi_cli, prompt)

    parts = []
    for name, fut in fns.items():
        try:
            out = fut.result()
            if out and not out.startswith("["):
                parts.append(f"=== Revisor: {name} ===\n{out}")
        except Exception:
            pass
    return "\n\n".join(parts) if parts else ""
