# redink

<p align="center">
  <img src="assets/logo.svg" width="120" alt="redink logo"/>
</p>

Adversarial pre-submission paper red-teamer. Uses a STORM-style multi-persona LangGraph workflow to find citation hallucinations, statistical weaknesses, novelty gaps, and writing problems before a real reviewer does.

## Architecture

```
paper.md / GitHub URL / arXiv URL
         │
         ▼
  [fetch_paper]     → fetches README from GitHub or PDF text from arXiv URL
         │
         ▼
  [classify]        → extracts: area, paper_type, dimensions, citations (up to 20),
                      claims (5–8 technical subject+verb+object)
         │
         ▼ (parallel fan-out via Send — 3 personas × N dimensions)
  [reviewer] × N    → skeptic · practitioner · academic, each with different priors
  [figure_reviewer] → Gemini Vision on ar5iv figures (cherry-picking, truncated axes)
         │
         ▼
  [contradiction_map] → where do personas disagree?
  [blind_spot]        → what did all reviewers miss?
         │
         ▼
  [synthesize]      → PASS / REVISE / FAIL verdict + findings ranked by severity
```

**Dimensions reviewed:** citations · methodology · novelty · writing · statistics · reproducibility · ethics · figures

Each dimension runs three independent reviewer personas (skeptic, practitioner, academic) in parallel. Findings are de-duplicated and weighted by cross-persona confidence.

## Models (via OpenRouter)

| Role | Env var | Default |
|---|---|---|
| Classify | `CLASSIFY_MODEL` | `qwen/qwen3-8b` |
| Reviewer (analysis) | `REVIEWER_MODEL` | `deepseek/deepseek-v4-flash` |
| Reviewer (tool calls) | `TOOL_MODEL` | `openai/gpt-4o-mini` |
| Figure review | `FIGURE_MODEL` | `google/gemini-2.5-flash` |
| Structured output | `STRUCTURED_MODEL` | `openai/gpt-4o-mini` |
| Synthesis + verdict | `SYNTHESIZE_MODEL` | `deepseek/deepseek-v4-flash` |

Estimated cost per review: **~$0.10–0.20**.

> **Note on model choices:** DeepSeek V4 Flash is a reasoning model — it returns plain text instead of JSON for `with_structured_output()` calls, and returns empty `.content` in tool loops. `STRUCTURED_MODEL` and `TOOL_MODEL` default to `gpt-4o-mini` to avoid both issues.

## Setup

**Requires Python ≥ 3.11 and [uv](https://docs.astral.sh/uv/getting-started/installation/).**

```bash
git clone https://github.com/p7dotorg/redink
cd redink
uv sync

cp .env.example .env
# edit .env — set OPENROUTER_API_KEY at minimum
```

## Usage

### Interactive chat (default)

```bash
uv run redink
```

Opens a REPL. Use slash commands:

| Command | Description |
|---|---|
| `/review <path\|url>` | Run a full review (streams progress) |
| `/report` | Re-display the last verdict |
| `/rerun <dimension>` | Re-run one dimension (e.g. `novelty`) |
| `/clear` | Clear screen and restart |
| `/exit` | Quit |

After a review, type freely to ask questions about findings.

### One-shot (CI / pipe)

```bash
# Local file
uv run redink my-paper.md

# GitHub repo (fetches README.md)
uv run redink https://github.com/user/paper-repo

# arXiv
uv run redink https://arxiv.org/abs/2607.01224

# stdin
cat paper.md | uv run redink -
```

The report is printed to stdout and saved as `<paper>.review.md`.

## LangGraph Studio

The graph is registered in `langgraph.json` as `redink` and visible in Studio as **redink**.

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

**Interrupts:** The graph pauses before `reviewer`, `figure_reviewer`, and `synthesize` nodes so you can inspect intermediate state before continuing.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OPENROUTER_API_KEY` | required | OpenRouter API key |
| `CLASSIFY_MODEL` | `qwen/qwen3-8b` | Classification + claims extraction |
| `REVIEWER_MODEL` | `deepseek/deepseek-v4-flash` | Analysis reviewer (no tools) |
| `TOOL_MODEL` | `openai/gpt-4o-mini` | Tool-calling reviewer (citations/novelty) |
| `FIGURE_MODEL` | `google/gemini-2.5-flash` | Vision model for figures |
| `STRUCTURED_MODEL` | `openai/gpt-4o-mini` | Structured output extraction |
| `SYNTHESIZE_MODEL` | `deepseek/deepseek-v4-flash` | Synthesis + verdict |
| `LANGSMITH_TRACING` | `false` | Enable LangSmith traces |
| `LANGSMITH_PROJECT` | — | LangSmith project name |

## How Citation Verification Works

The `citations` reviewer uses live tool calls — but only the **skeptic** persona makes web requests. Running all three personas in parallel against the Semantic Scholar API (1 req/sec rate limit) causes all searches to fail. Practitioner and academic instead do contextual analysis (are claims well-supported by the listed references?) without hitting the network.

Skeptic tools for citations:
- **`search_papers`** — Semantic Scholar (broad cross-disciplinary coverage: CS, psychology, philosophy, medicine)
- **`get_paper`** — fetches arXiv abstract by ID
- **`verify_doi`** — Crossref lookup for non-arXiv papers (journals, books, ACM, IEEE)

**Important:** papers published before arXiv existed (pre-2000 classics, psychology journals, philosophy papers) will not appear on Crossref if they lack a DOI. A failed DOI lookup is not evidence of hallucination when the reference has a complete journal+volume+page entry in the References section.

## How Novelty Search Works

The `novelty` reviewer uses **`search_arxiv`** (paper7 CLI → arXiv API fallback), which is faster than Semantic Scholar and has no rate limit for CS/AI/ML coverage.

The classify node extracts **5–8 technical claims** in `subject+verb+object` format (e.g., "meta-LLM revises complete episode trajectories to iteratively rewrite the agent scaffold") instead of high-level assertions. These become search queries, so the novelty reviewer finds specific prior papers rather than returning generic observations.

Novelty tools:
- **`search_arxiv`** — arXiv via paper7 CLI (CS/AI/ML, no rate limit)
- **`get_paper`** — read abstract to compare methods/results against the paper under review

## Paper Truncation Strategy

Long papers are truncated before being sent to models:

| Node | Strategy | Rationale |
|---|---|---|
| `classify` | first 12k + last 3k chars | abstract + intro + start of methods + references |
| `reviewer` (citations) | first 6k + last 6k chars | intro (context) + references section |
| `reviewer` (others) | first 20k chars | covers abstract + methods + results for most papers |

Part of [p7dotorg](https://github.com/p7dotorg). · [redink.sh](https://redink.sh)
