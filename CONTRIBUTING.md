# Contributing to redink

Thanks for looking. redink is small and opinionated; this doc gets you oriented
fast and names the one rule that matters (measure verdict changes).

## Setup

Requires Python ≥ 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run redink setup      # keys + models → .env  (or edit .env by hand)
uv run redink            # the chat
```

## Layout

A uv workspace with two packages:

```
core/redink_core/     the LangGraph graphs + nodes (domain logic)
  graph.py            the paper-reviewer graph
  nodes_*.py          fetch · classify · reviewer · debate · synthesis
  drl/                the dataset-research-loop graph (scan → score → OKF)
  prompts.py schemas.py tools.py
cli/redink_cli/       the REPL, one-shot entry, HTML report, config
eval/                 the measurement harness (see below)
langgraph.json        registers both graphs: `redink` and `drl`
```

Domain logic lives in `core`; presentation in `cli`. A new capability is
usually a node (core) plus a command (cli).

## Testing

There is no formal test suite. Validation is done with small inline scripts
(`uv run python - <<'EOF' … EOF`) and, for anything touching the verdict, the
`eval/` harness. If you add a pure function with real edge cases, a script that
asserts them in the PR description is welcome.

## The one rule: measure verdict changes

Calibrating a reviewer by eyeballing two papers gives false confidence — we
learned this the hard way (a change that looked like it discriminated was
failing 82% of papers, invisible until measured). So:

**Any change to the judge panel, the debate, or severity must be measured, not
eyeballed.** It's cheap — the pipeline output is cached per paper, so you
re-run only the judge:

```bash
# one-time (paid): build the labeled set + baseline findings
uv run python eval/collect_asap.py --n 50 --balance
uv run python eval/overlap_metric.py --n 50        # recall / noise / verdict

# every iteration after (cents, cached findings):
uv run python eval/rejudge.py --n 50 --anchors 0           # A/B a judge change
uv run python eval/confirm_calibration.py --n 50           # the production judge
```

Report the before/after distribution in your PR. Findings-quality changes
(recall/noise) and verdict changes (distribution) are separate axes — say which
you moved.

## Extending

**A new reviewer dimension.** Add a prompt to `DIMENSION_PROMPTS` in
`prompts.py`, add the name to the `dimensions` Literal in `schemas.py`
(`Classification`), and update the classify prompt if it should auto-trigger.
The fan-out (`route_to_reviewers`) and per-persona review are automatic.

**A new dataset source.** Add a `scan_<x>(query, limit) -> list[dict]` to
`drl/scanners.py` returning the normalized record shape, register it in
`SCANNERS`, and make `prescore` source-aware in `drl/nodes.py` if its metadata
is thinner than HuggingFace's. `--sources <x>` then works.

**A new model.** Everything routes through `make_model(env_key, default, …)`
(always cap `max_tokens`). Add the field to the `config.py` registry so it's
configurable via `/config` and `redink setup`.

## Conventions

- **Commits:** `type: summary` (`feat` / `fix` / `docs` / `refactor` / `chore`),
  and the body should say *why* or the trigger, not just what — the way the
  [CHANGELOG](CHANGELOG.md) reads.
- **Cost:** don't re-run the full pipeline when you only changed the verdict —
  use the cached findings. Cap `max_tokens`. State the rough cost of any eval
  run in the PR.
- **Honesty over polish:** if something is a proxy (e.g. `/spikes` until scan
  history accrues) or a known ceiling (figure-only evidence isn't read), say so
  in the output and the docs rather than hide it.

## Reporting bugs

Open an issue with the command, the paper/query, and what you expected.
Security issues go to [SECURITY.md](SECURITY.md), not a public issue.
