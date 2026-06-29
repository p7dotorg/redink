"""LangGraph graph definition for the adversarial paper reviewer."""
import operator
from typing import Annotated, Optional
from typing_extensions import TypedDict

from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from paper.schemas import Classification, Finding, Verdict
from paper.nodes import classify, reviewer, synthesize


class ReviewState(TypedDict):
    paper: str
    classification: Optional[Classification]
    findings: Annotated[list[Finding], operator.add]
    verdict: Optional[Verdict]


class ReviewerInput(TypedDict):
    paper: str
    classification: Classification
    dimension: str


def route_to_reviewers(state: ReviewState) -> list[Send]:
    clf = state["classification"]
    return [
        Send("reviewer", {"paper": state["paper"], "classification": clf, "dimension": dim})
        for dim in clf.dimensions
    ]


builder = StateGraph(ReviewState)
builder.add_node("classify", classify)
builder.add_node("reviewer", reviewer)
builder.add_node("synthesize", synthesize)
builder.add_edge(START, "classify")
builder.add_conditional_edges("classify", route_to_reviewers, ["reviewer"])
builder.add_edge("reviewer", "synthesize")
builder.add_edge("synthesize", END)

graph = builder.compile()
