"""Public interface — re-exports all nodes for graph.py."""
from paper.nodes_classify import classify
from paper.nodes_fetch import fetch_paper
from paper.nodes_reviewer import reviewer, figure_reviewer
from paper.nodes_synthesis import contradiction_map, blind_spot, synthesize

__all__ = ["fetch_paper", "classify", "reviewer", "figure_reviewer", "contradiction_map", "blind_spot", "synthesize"]
