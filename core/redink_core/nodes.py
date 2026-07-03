"""Public interface — re-exports all nodes for graph.py."""
from redink_core.nodes_classify import classify
from redink_core.nodes_debate import debate
from redink_core.nodes_fetch import fetch_paper
from redink_core.nodes_reviewer import reviewer, figure_reviewer
from redink_core.nodes_synthesis import contradiction_map, blind_spot, judge_panel, synthesize

__all__ = [
    "fetch_paper", "classify", "reviewer", "figure_reviewer",
    "debate", "contradiction_map", "blind_spot", "judge_panel", "synthesize",
]
