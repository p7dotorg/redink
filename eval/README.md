# redink eval

Turning "afinado no olho contra 2 papers" into "medido contra centenas".

## Why

The harness was calibrated by inspection against `1706.03762` (Attention) and
`automem.md`. That discriminates well on what we tested, but gives no objective
number for whether a change helps. This directory builds a labeled set so every
prompt/model change becomes measurable.

## The label choice (important)

We do **not** optimize redink to predict the conference accept/reject decision.
That label is confounded by novelty, timing and committee politics — fitting it
would teach the harness to mimic conference outcomes, not to detect rigor.

The real target is **overlap between redink's findings and the weaknesses human
reviewers actually raised**. The accept/reject decision and average rating are
kept only as coarse signals (e.g. does redink's verdict correlate with rating).

## Pipeline

1. **collect_asap.py** — build the labeled set from ASAP-Review (Yuan et al.
   2021, Apache-2.0: ICLR 2017-2020 + NIPS 2016-2019 papers + real reviews +
   decisions). Each record: `full_text` (for redink to review) plus the human
   `reviews[].text`, `meta_review`, `avg_rating`, `decision` (the label side).

   ```bash
   uv run python eval/collect_asap.py --n 300 --balance \
       --venues ICLR_2018,ICLR_2019,ICLR_2020 \
       --out eval/data/asap_300.jsonl
   ```
   First run downloads the ~235MB dataset zip to `eval/data/` (via gdown).
   Rejects only exist for ICLR (NIPS publishes reviews for accepts only).

2. **overlap_metric.py** — run redink over each `full_text`, then an LLM judge
   compares its findings to the human reviewers' weaknesses.

   ```bash
   uv run python eval/overlap_metric.py --in eval/data/asap_300.jsonl --n 50
   ```
   Reports:
   - `recall` — fraction of human-raised weaknesses redink also caught
   - `noise_rate` — fraction of redink findings judged wrong / pedantic
   - `novel_rate` — fraction that are legit concerns humans did not raise
   - `verdict × rating` — does PASS/REVISE/FAIL trend with the human avg rating?

   Every stage is cached per paper under `eval/data/cache/`, so re-running after
   a matcher tweak does not re-run the expensive redink pipeline. Each paper
   costs ~50-70 pipeline calls + 2 judge calls — start with `--n 10`.

   A non-matching redink finding is NOT auto-noise (humans miss things), so the
   judge splits unmatched findings into `plausible_unraised` vs `noise`.

## Notes

- `weaknesses` in each record is a best-effort regex pull from the meta-review's
  Cons block; ICLR meta-reviews are rarely bulleted, so it is often empty. The
  authoritative human signal is the full `reviews[].text` — the metric parses
  weaknesses from that at scoring time.
- `eval/data/` is gitignored (large zip + generated jsonl are not source).
