# p7-reviewer

Adversarial pre-submission paper reviewer. Uses a STORM-style multi-persona LangGraph workflow to red-team academic papers before publication — finding citation hallucinations, statistical weaknesses, novelty gaps, and writing problems before a real reviewer does.

## Architecture

```
paper.md / GitHub URL
       │
       ▼
  [fetch_paper]   → fetches README from GitHub URL if no paper text provided
       │
       ▼
  [classify]      → LLM extracts area, type, claims, citations, active dimensions
       │
       ▼ (parallel fan-out via Send)
  [reviewer] × N  → skeptic × practitioner × academic per dimension
  [figure_reviewer] → Gemini Vision for charts/plots (ar5iv figures)
       │
       ▼
  [contradiction_map] → where do personas disagree?
  [blind_spot]        → what did all reviewers miss?
       │
       ▼
  [synthesize]    → PASS / REVISE / FAIL verdict + ranked findings
```

**Dimensions reviewed:** citations · methodology · novelty · writing · statistics · reproducibility · ethics · figures

**Models (via OpenRouter):**
| Role | Default |
|---|---|
| Classify | `qwen/qwen3-8b` |
| Reviewer (analysis) | `deepseek/deepseek-v4-flash` |
| Reviewer (tool calls) | `openai/gpt-4o-mini` |
| Figure review | `google/gemini-2.5-flash` |
| Structured output | `openai/gpt-4o-mini` |
| Synthesis | `deepseek/deepseek-v4-flash` |

All models configurable via env vars. Estimated cost per review: ~$0.10–0.20.

## Setup

```bash
git clone https://github.com/p7dotorg/paper-reviewer
cd paper-reviewer
pip install -e .

cp .env.example .env
# edit .env — add your OPENROUTER_API_KEY at minimum
```

## Usage

```bash
# Review a local paper
p7review my-paper.md

# Review from a GitHub repo (fetches README.md)
p7review https://github.com/user/paper-repo

# Pipe from stdin
cat paper.pdf | pdftotext - - | p7review -
```

The report is printed to stdout and saved as `<paper>.review.md`.

## LangGraph Studio

The graph is registered in `langgraph.json` as `paper_reviewer`.

```bash
pip install langgraph-cli
langgraph dev
```

Open Studio at `http://localhost:2024`. Pass input as:
```json
{ "paper": "paste paper text here" }
```
or:
```json
{ "github_url": "https://github.com/user/repo" }
```

**BYOK:** Set your OpenRouter key in Studio's Default Configuration panel under `openrouter_api_key` — no server restart needed.

**Interrupts:** The graph pauses before `reviewer`, `figure_reviewer`, and `synthesize` nodes so you can inspect intermediate state in Studio before continuing.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OPENROUTER_API_KEY` | required | OpenRouter API key |
| `CLASSIFY_MODEL` | `qwen/qwen3-8b` | Classification model |
| `REVIEWER_MODEL` | `deepseek/deepseek-v4-flash` | Analysis reviewer |
| `TOOL_MODEL` | `openai/gpt-4o-mini` | Tool-calling reviewer (citations/novelty) |
| `FIGURE_MODEL` | `google/gemini-2.5-flash` | Vision model for figures |
| `STRUCTURED_MODEL` | `openai/gpt-4o-mini` | Structured output extraction |
| `SYNTHESIZE_MODEL` | `deepseek/deepseek-v4-flash` | Synthesis + verdict |
| `LANGSMITH_TRACING` | `false` | Enable LangSmith traces |
| `LANGSMITH_PROJECT` | — | LangSmith project name |

## Citation Verification

The `citations` and `novelty` reviewers use live tool calls to verify references:

- **`search_papers`** — Semantic Scholar API fallback when paper7 CLI unavailable
- **`get_paper`** — fetches arXiv abstract by ID
- **`verify_doi`** — Crossref lookup for non-arXiv papers

Part of [p7dotorg](https://github.com/p7dotorg).
