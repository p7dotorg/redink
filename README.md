# redink

<p align="center">
  <img src="assets/logo.svg" width="120" alt="redink logo"/>
</p>

Adversarial pre-submission paper red-teamer **and** dataset opportunity scout — one chat, two LangGraph flows. It finds citation hallucinations, statistical weaknesses, novelty gaps, and writing problems in a paper before a real reviewer does; and it scans dataset sources to score and catalog opportunities into a portable knowledge bundle.

<p align="center">
  <img src="assets/demo.gif" width="820" alt="redink demo — a calibrated paper review"/>
</p>

## Two flows, one REPL

```bash
uv run redink
```

| | Review papers | Research datasets |
|---|---|---|
| **do** | `/review <path\|arxiv-url>` | `/scan <query>` |
| **read** | `/report` · `/rerun <dim>` | `/rank` · `/gaps` · `/spikes` · `/wiki <slug>` |
| **graph** | `redink` | `drl` |

Configure everything from inside the chat with `/config` (or `redink setup` for a full wizard). After a review, type freely to ask questions about the findings.

See [`examples/`](examples/) for a sample review and a sample OKF concept.

---

## Paper review

### Architecture

```
paper.md / GitHub URL / arXiv URL
   │
   ▼
 [fetch_paper]      arXiv via ar5iv (tables preserved as pipe rows; abstract-only
                    fetches are flagged so reviewers don't fault missing sections)
   │
   ▼
 [classify]         area · type · dimensions · citations · 5–8 technical claims
   │
   ▼  fan-out via Send — 3 personas × N dimensions
 [reviewer] × N     skeptic · practitioner · academic, different priors each
 [figure_reviewer]  Gemini Vision on ar5iv figures (cherry-picking, truncated axes)
   │
   ▼
 [debate]           dedup, then every CRITICAL faces a defender (argues the
                    author's side from the text) + a judge → sustained /
                    downgraded / dismissed. Kills criticals nobody can defend
                    against — and, more importantly, that nobody can uphold.
   │
   ▼
 [contradiction_map] · [blind_spot]
   │
   ▼
 [judge_panel]      3 lenses — rigor · contribution · era-appropriate standards —
                    calibrated against reference papers → PASS / REVISE / FAIL
   │
   ▼
 [synthesize]       verdict + findings, plus a self-contained interactive HTML report
```

**Dimensions:** citations · methodology · novelty · writing · statistics · reproducibility · ethics · figures. Each runs three independent personas in parallel; findings are semantically de-duplicated (two-pass: per-dimension then global) and weighted by cross-persona agreement.

### The verdict is calibrated, not vibes

Most "AI reviewer" prompts collapse to *reject everything* — every real paper has flaws, so a judge evaluating against an implicit ideal fails them all. redink is measured against **300 ICLR papers with their real reviews and decisions** (built from [ASAP-Review](https://github.com/neulab/ReviewAdvisor)):

- **Findings recall ≈ 0.73** — it surfaces ~73% of the weaknesses human reviewers actually raised.
- **Verdict calibration** — the judge panel is anchored to reference papers (real finding-profiles → the verdict their rating implies). This dropped over-FAIL from **82% → 16%** and made **REVISE** the dominant verdict, matching how peer review actually behaves. FAIL is now reserved for a central conclusion that doesn't survive the debate.

The measurement harness lives in [`eval/`](eval/) — the labeled-set collector, the overlap metric, and the cheap re-judge A/B that produced the calibration.

---

## Dataset research loop (`drl`)

A second graph that scans dataset sources, scores opportunity, and writes an [Open Knowledge Format](https://github.com/GoogleCloudPlatform/knowledge-catalog) (OKF) bundle — a portable directory of markdown concepts you can `git clone` or open in Obsidian.

```
 [scan] × sources     fan-out: HuggingFace · Kaggle · OpenML
    │
    ▼
 [merge]              dedupe across sources
    │
    ▼
 [prescore]           rule-based quality gate (source-aware)
    │
    ▼
 [score] × datasets   LLM opportunity score 0–3, one per dataset
    │
    ▼
 [catalog]            write OKF concepts + rebuild index / log
    │
    ▼
 [digest]             run summary concept
```

Analysis reads the bundle's frontmatter at query time (no DB), exactly as the OKF spec intends:

| Command | What it does |
|---|---|
| `/scan <query> [--sources hf,kaggle,openml] [--limit N]` | scan → score → write OKF concepts |
| `/rank [N]` | top datasets by opportunity **PageRank** over the tag-similarity graph |
| `/gaps [N]` | least-covered task categories |
| `/spikes [N]` | recently-active datasets (velocity proxy) |
| `/wiki <slug>` | print an OKF concept |

> Papers With Code was dropped — its API now redirects to Hugging Face. OpenML replaced it. Kaggle's list endpoint works anonymously (`KAGGLE_*` only raises limits).

Also usable one-shot / in cron: `drl scan "..."`, `drl rank`, `drl gaps`, `drl setup`.

---

## Setup

**Requires Python ≥ 3.11 and [uv](https://docs.astral.sh/uv/getting-started/installation/).**

```bash
git clone https://github.com/p7dotorg/redink
cd redink
uv sync

uv run redink setup      # interactive wizard: keys + models → .env
# or: cp .env.example .env  and set OPENROUTER_API_KEY
```

## Usage

```bash
uv run redink                                   # interactive chat (both flows)
uv run redink my-paper.md                       # one-shot review (CI / pipe)
uv run redink https://arxiv.org/abs/1706.03762  # arXiv
cat paper.md | uv run redink -                  # stdin
```

One-shot prints the report to stdout, saves `<paper>.review.md`, and writes an interactive `<paper>.annotated.html`.

## Models & config

Every model and key is configurable via `/config papers|datasets`, `redink setup`, or `.env`. All calls route through OpenRouter and are capped (`max_tokens`) to avoid runaway credit reservations.

| Role | Env var | Default |
|---|---|---|
| Classify | `CLASSIFY_MODEL` | `openai/gpt-4o-mini` |
| Reviewer / defender | `REVIEWER_MODEL` | `deepseek/deepseek-v4-flash` |
| Tool calls (citations/novelty) | `TOOL_MODEL` | `openai/gpt-4o-mini` |
| Figure review | `FIGURE_MODEL` | `google/gemini-2.5-flash` |
| Structured output / dedup | `STRUCTURED_MODEL` | `openai/gpt-4o-mini` |
| Synthesis prose | `SYNTHESIZE_MODEL` | `deepseek/deepseek-v4-flash` |
| **Judge panel + rebuttal** | `JUDGE_MODEL` | `openai/gpt-4o` |
| Dataset scorer | `DRL_SCORE_MODEL` | `openai/gpt-4o-mini` |

Estimated cost per review: **~$0.10** (dominated by the gpt-4o judge panel — drop `JUDGE_MODEL` to `gpt-4o-mini` for ~$0.03, measurable quality tradeoff via `eval/`).

> **Why the model split:** DeepSeek V4 Flash is a reasoning model — it returns plain text instead of JSON for structured calls and empty content in tool loops. So `STRUCTURED_MODEL`/`TOOL_MODEL` are `gpt-4o-mini`, and the verdict-deciding `JUDGE_MODEL` is `gpt-4o`.

## LangGraph Studio

Both graphs are registered in `langgraph.json` (`redink` and `drl`).

```bash
pip install langgraph-cli
langgraph dev   # Studio at http://localhost:2024
```

Input for `redink`: `{ "paper": "…" }` or `{ "github_url": "https://arxiv.org/abs/…" }`. **BYOK:** set `openrouter_api_key` in Studio's config panel. The `redink` graph interrupts before `reviewer`/`figure_reviewer`/`synthesize` for inspection.

## How it works — details

**Citation verification.** Only the *skeptic* persona makes web requests (all three in parallel would exhaust the Semantic Scholar rate limit). Tools: `search_papers` (Semantic Scholar, cross-disciplinary), `get_paper` (arXiv abstract), `verify_doi` (Crossref). A finding's `evidence` quote is checked against the paper text — an unverifiable quote drops the finding to `minor`, killing hallucinations.

**Novelty search.** The classify node extracts 5–8 `subject+verb+object` claims that become specific arXiv queries. Results published **after** the paper under review are filtered out in code — no more 2024 papers cited as prior work for a 2017 paper.

**Fetch & truncation.** arXiv is fetched via ar5iv with `<table>` preserved as pipe rows and `<math>` as LaTeX. Reviewers get up to 60k chars with an explicit excerpt notice, so "missing" sections in the omitted tail are never reported as flaws. Abstract-only renders (ar5iv failures) are detected and flagged.

## Data & privacy

redink runs locally, but reviewing a paper sends parts of it to third parties. **If your paper is unpublished or confidential, know what leaves your machine:**

| Goes out | To | What |
|---|---|---|
| LLM calls | OpenRouter → the chosen provider (OpenAI, DeepSeek, Google, …) | paper excerpts, findings, prompts |
| Citation / novelty tools | Semantic Scholar · arXiv · Crossref | search queries derived from your claims and references |
| Figures | ar5iv (fetch) + the vision model via OpenRouter | figure images + captions |
| Dataset scans (`drl`) | HuggingFace · Kaggle · OpenML | your search query only — no paper text |
| Tracing (**only if** `LANGSMITH_TRACING=true`) | LangSmith | full run traces, including paper text |

redink itself stores nothing beyond local files — the `*.review.md` / `*.annotated.html` report and the OKF `bundle/`. It does not phone home. For sensitive work, point the models at a self-hosted / private OpenRouter setup and keep `LANGSMITH_TRACING=false` (the default).

See [SECURITY.md](SECURITY.md) to report a vulnerability.

---

Part of [p7dotorg](https://github.com/p7dotorg). · [redink.sh](https://redink.sh)
