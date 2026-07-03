"""Conversational Q&A over a completed review — runs in the chat CLI."""
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from redink_core.nodes_helpers import make_model
from redink_core.schemas import Verdict


def _build_context(state: dict) -> str:
    verdict: Verdict = state.get("verdict")
    clf = state.get("classification")
    paper_excerpt = (state.get("paper") or "")[:4000]

    findings_text = ""
    if verdict:
        for i, f in enumerate(verdict.findings, 1):
            findings_text += (
                f"\n[{i}] {f.severity.upper()} / {f.dimension} / {f.persona}\n"
                f"  Issue: {f.issue}\n"
                f"  Evidence: {f.evidence}\n"
                f"  Suggestion: {f.suggestion}\n"
            )

    return (
        f"Paper area: {clf.area if clf else 'unknown'}\n"
        f"Verdict: {verdict.status if verdict else 'none'}\n\n"
        f"FINDINGS:\n{findings_text}\n\n"
        f"PAPER EXCERPT (first 4k):\n{paper_excerpt}"
    )


def answer(question: str, state: dict, history: list[dict]) -> str:
    """Answer a question about the review. Returns the reply as a string."""
    model = make_model("REVIEWER_MODEL", "deepseek/deepseek-v4-flash", max_tokens=1500)

    context = _build_context(state)
    messages = [
        SystemMessage(content=(
            "You are a research assistant helping an author understand their paper review.\n"
            "You have access to the full review findings and a paper excerpt.\n"
            "Be direct and specific. Reference finding numbers when relevant. "
            "Respond in the same language as the user's question."
        )),
        HumanMessage(content=f"REVIEW CONTEXT:\n{context}"),
    ]

    for turn in history[-6:]:  # last 3 exchanges
        cls = HumanMessage if turn["role"] == "user" else AIMessage
        messages.append(cls(content=turn["content"]))

    messages.append(HumanMessage(content=question))

    response = model.invoke(messages)
    return response.content
