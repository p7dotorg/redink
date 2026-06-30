"""LangGraph graph — STORM-enhanced adversarial paper reviewer."""
import operator
from typing import Annotated, Optional
from typing_extensions import TypedDict

from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from paper.schemas import Classification, Finding, Verdict, ContradictionMap, BlindSpot
from paper.nodes import classify, reviewer, contradiction_map, blind_spot, synthesize

PERSONAS = ["skeptic", "practitioner", "academic"]


class ReviewState(TypedDict):
    paper: str
    classification: Optional[Classification]
    findings: Annotated[list[Finding], operator.add]
    contradiction_map: Optional[ContradictionMap]
    blind_spots: Optional[BlindSpot]
    verdict: Optional[Verdict]


class ReviewerInput(TypedDict):
    paper: str
    classification: Classification
    dimension: str
    persona: str


def route_to_reviewers(state: ReviewState) -> list[Send]:
    """Fan-out: each dimension × each persona — STORM multi-perspective."""
    clf = state["classification"]
    return [
        Send("reviewer", {
            "paper": state["paper"],
            "classification": clf,
            "dimension": dim,
            "persona": persona,
        })
        for dim in clf.dimensions
        for persona in PERSONAS
    ]


builder = StateGraph(ReviewState)
builder.add_node("classify", classify)
builder.add_node("reviewer", reviewer)
builder.add_node("contradiction_map", contradiction_map)
builder.add_node("blind_spot", blind_spot)
builder.add_node("synthesize", synthesize)

builder.add_edge(START, "classify")
builder.add_conditional_edges("classify", route_to_reviewers, ["reviewer"])
builder.add_edge("reviewer", "contradiction_map")
builder.add_edge("contradiction_map", "blind_spot")
builder.add_edge("blind_spot", "synthesize")
builder.add_edge("synthesize", END)

graph = builder.compile()
