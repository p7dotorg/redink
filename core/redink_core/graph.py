"""LangGraph graph — STORM-enhanced adversarial paper reviewer."""
import operator
import os
from typing import Annotated, Optional
from typing_extensions import TypedDict, NotRequired

from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from redink_core.schemas import Classification, Finding, Verdict, ContradictionMap, BlindSpot, JudgePanel
from redink_core.nodes import (
    fetch_paper, classify, reviewer, figure_reviewer,
    debate, contradiction_map, blind_spot, judge_panel, synthesize,
    repro_check,
)
from redink_core.nodes_fetch import _GITHUB_RE

PERSONAS = ["skeptic", "practitioner", "academic"]


class ReviewState(TypedDict, total=False):
    paper: str                                        # paper text — set this OR github_url
    github_url: str                                   # GitHub repo URL — fetch_paper will pull README
    classification: Optional[Classification]
    findings: Annotated[list[Finding], operator.add]
    deduped_findings: list[Finding]                   # post-dedup, post-debate — what downstream nodes use
    contradiction_map: Optional[ContradictionMap]
    blind_spots: Optional[BlindSpot]
    judge_votes: Optional[JudgePanel]
    verdict: Optional[Verdict]
    repro_result: Optional[dict]


class ReviewConfig(TypedDict, total=False):
    openrouter_api_key: str   # BYOK — overrides OPENROUTER_API_KEY env var
    paper7_api_url: str       # optional: p7web API for find_related_papers


class ReviewerInput(TypedDict):
    paper: str
    classification: Classification
    dimension: str
    persona: str


def _repro_url(state: ReviewState) -> Optional[str]:
    """Repo oficial do paper: code_repo do classify, ou o input se já for um
    repo GitHub (arXiv/PDF não contam)."""
    clf = state["classification"]
    if getattr(clf, "code_repo", None):
        return clf.code_repo
    url = state.get("github_url")
    if url and _GITHUB_RE.match(url.strip()):
        return url
    return None


def route_to_reviewers(state: ReviewState) -> list[Send]:
    """Fan-out: text dimensions × 3 personas; figures → figure_reviewer (once)."""
    clf = state["classification"]
    sends = []
    for dim in clf.dimensions:
        if dim == "figures":
            sends.append(Send("figure_reviewer", {
                "paper": state["paper"],
                "classification": clf,
            }))
        else:
            for persona in PERSONAS:
                sends.append(Send("reviewer", {
                    "paper": state["paper"],
                    "classification": clf,
                    "dimension": dim,
                    "persona": persona,
                }))
    if os.getenv("REDINK_REPRO") and "reproducibility" in clf.dimensions:
        repo_url = _repro_url(state)
        if repo_url:
            sends.append(Send("repro_check", {
                "code_repo": repo_url,
                "paper": state["paper"],  # p/ reconstruir URLs quebradas no PDF
            }))
    return sends


builder = StateGraph(ReviewState, config_schema=ReviewConfig)
builder.add_node("fetch_paper", fetch_paper)
builder.add_node("classify", classify)
builder.add_node("reviewer", reviewer)
builder.add_node("figure_reviewer", figure_reviewer)
builder.add_node("debate", debate)
builder.add_node("contradiction_map", contradiction_map)
builder.add_node("blind_spot", blind_spot)
builder.add_node("judge_panel", judge_panel)
builder.add_node("synthesize", synthesize)
builder.add_node("repro_check", repro_check)

builder.add_edge(START, "fetch_paper")
builder.add_edge("fetch_paper", "classify")
builder.add_conditional_edges("classify", route_to_reviewers, ["reviewer", "figure_reviewer", "repro_check"])
builder.add_edge("reviewer", "debate")
builder.add_edge("figure_reviewer", "debate")
builder.add_edge("repro_check", "debate")
builder.add_edge("debate", "contradiction_map")
builder.add_edge("contradiction_map", "blind_spot")
builder.add_edge("blind_spot", "judge_panel")
builder.add_edge("judge_panel", "synthesize")
builder.add_edge("synthesize", END)

graph = builder.compile(
    interrupt_before=["reviewer", "figure_reviewer", "synthesize"],
)
graph_runner = builder.compile()  # no interrupts — for CLI and testing
