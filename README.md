# p7-reviewer

Adversarial pre-submission paper reviewer. Red-teams academic papers before publication using Claude CLI + Kimi CLI as free reviewers (via subscription), with OpenRouter as fallback.

## Install

```bash
pip install p7-reviewer
# or without installing:
uvx p7-reviewer my-paper.md
```

## Setup

```bash
export OPENROUTER_API_KEY=sk-or-...
export USE_CLAUDE_CLI=true    # uses your Claude subscription
export USE_KIMI_CLI=true      # uses your Kimi subscription
```

## Usage

```bash
p7review my-paper.md
```

Part of [p7dotorg](https://github.com/p7dotorg).
