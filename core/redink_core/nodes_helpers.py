"""Shared helpers for all nodes — model factory, tool loop, arXiv ID extraction."""
import os
import re

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

from redink_core.tools import REVIEWER_TOOLS, NOVELTY_TOOLS

_ARXIV_ID_RE = re.compile(r"(?:arXiv[:/])?(\d{4}\.\d{4,5})(?:v\d+)?", re.IGNORECASE)
_ALL_TOOLS = {t.name: t for t in REVIEWER_TOOLS + NOVELTY_TOOLS}


def extract_arxiv_id(text: str) -> str | None:
    m = _ARXIV_ID_RE.search(text[:2000])
    return m.group(1) if m else None


def get_api_key(config: RunnableConfig | None) -> str:
    """BYOK: config['configurable']['openrouter_api_key'] overrides env var."""
    if config:
        key = config.get("configurable", {}).get("openrouter_api_key")
        if key:
            return key
    return os.getenv("OPENROUTER_API_KEY", "")


def make_model(
    model_env_key: str,
    default: str,
    structured_schema=None,
    max_tokens: int = 2000,
    config: RunnableConfig | None = None,
):
    kwargs = dict(
        model=os.getenv(model_env_key, default),
        base_url="https://openrouter.ai/api/v1",
        api_key=get_api_key(config),
        default_headers={"HTTP-Referer": "http://localhost:2024", "X-Title": "redink"},
        temperature=0,
        max_retries=4,  # OpenRouter providers throw transient 429s — back off instead of crashing the run
        # ALWAYS cap: an uncapped call lets the model default to its full context
        # (65536 for deepseek), and OpenRouter reserves credit for that ceiling
        # even when actual usage is a few hundred tokens.
        max_tokens=max_tokens,
    )
    m = ChatOpenAI(**kwargs)
    return m.with_structured_output(structured_schema) if structured_schema else m


def tool_loop(model_with_tools, messages: list, max_rounds: int = 5) -> str:
    """Run tool-calling loop until model stops or max_rounds hit."""
    response = None
    for _ in range(max_rounds):
        response = model_with_tools.invoke(messages)
        messages.append(response)
        if not getattr(response, "tool_calls", None):
            break
        for tc in response.tool_calls:
            tool = _ALL_TOOLS.get(tc["name"])
            result = tool.invoke(tc["args"]) if tool else f"Unknown tool: {tc['name']}"
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
    return response.content if response else ""


_EXCERPT_LIMIT = 60000

_TRUNCATION_NOTICE = (
    "[AVISO AO REVISOR: o texto abaixo é um EXCERTO — o paper completo tem "
    "{total} caracteres e você vê apenas {shown}. Seções finais (experimentos "
    "tardios, hiperparâmetros, apêndices, referências) podem estar na parte "
    "omitida. NUNCA reporte como problema a AUSÊNCIA de algo que pode estar "
    "no trecho não mostrado — reporte apenas problemas visíveis neste texto.]\n\n"
)


def reviewer_excerpt(paper: str, dim: str) -> str:
    """Truncate long papers: citations gets front+tail, others get first 60k.
    Truncated excerpts carry an explicit notice so reviewers never treat
    absence-from-excerpt as absence-from-paper."""
    if len(paper) <= _EXCERPT_LIMIT:
        return paper
    if dim == "citations":
        excerpt = paper[:8000] + "\n\n[... corpo do paper omitido ...]\n\n" + paper[-8000:]
        shown = "o início e o fim (16k caracteres)"
    else:
        excerpt = paper[:_EXCERPT_LIMIT] + "\n\n[... restante do paper omitido ...]"
        shown = f"os primeiros {_EXCERPT_LIMIT} caracteres"
    return _TRUNCATION_NOTICE.format(total=len(paper), shown=shown) + excerpt
