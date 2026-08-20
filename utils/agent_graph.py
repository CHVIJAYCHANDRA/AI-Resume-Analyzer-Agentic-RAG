from __future__ import annotations

from typing import List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from .evaluator import evaluate_once
from .prompts import RETRIEVAL_K
from .rag_engine import build_vector_index, query_vectorstore

# One repair attempt. Higher values mostly burn tokens on models that cannot
# follow the schema at all.
MAX_ATTEMPTS = 2

# Fraction of evidence quotes that must appear verbatim in the resume for the
# result to be accepted without a repair pass.
GROUNDING_THRESHOLD = 0.5

# Prefix length used when checking a quote against the resume. Long enough to
# be specific, short enough to survive whitespace and punctuation drift.
QUOTE_MATCH_CHARS = 60


class AnalysisState(TypedDict, total=False):
    # inputs
    job_desc: str
    resume_text: str
    use_local: bool
    model_name: str
    openai_key: Optional[str]
    # working state
    context_chunks: List[str]
    result: dict
    attempts: int
    errors: List[str]
    # verification output
    grounded_ratio: float
    unsupported_quotes: List[str]
    verified: bool


def _normalise(text: str) -> str:
    """Collapse whitespace and lowercase so quote matching is not defeated by
    PDF line wrapping."""
    return " ".join((text or "").split()).lower()


# ------------------------------------------------------------------- nodes
def retrieve_node(state: AnalysisState) -> AnalysisState:
    """Fetch the resume passages most similar to the job description.

    Failure is non-fatal: the graph continues with no excerpts and the UI warns
    that the score is less reliable.
    """
    try:
        index = build_vector_index(
            state["resume_text"],
            openai_key=state.get("openai_key"),
            use_local=state.get("use_local", False),
        )
        state["context_chunks"] = query_vectorstore(
            index, state["job_desc"], k=RETRIEVAL_K
        )
    except Exception as e:
        state["context_chunks"] = []
        state.setdefault("errors", []).append(f"retrieve: {e}")
    return state


def evaluate_node(state: AnalysisState) -> AnalysisState:
    """One structured evaluation call. Attempt 2+ appends repair instructions."""
    state["attempts"] = state.get("attempts", 0) + 1
    try:
        evaluation = evaluate_once(
            state["job_desc"],
            state["resume_text"],
            state.get("context_chunks"),
            use_local=state.get("use_local", False),
            model_name=state["model_name"],
            openai_key=state.get("openai_key"),
            repair=state["attempts"] > 1,
        )
        state["result"] = evaluation.model_dump()
    except Exception as e:
        state.setdefault("errors", []).append(
            f"evaluate (attempt {state['attempts']}): {e}"
        )
        state["result"] = {}
    return state


def verify_node(state: AnalysisState) -> AnalysisState:
    """Check the result against the schema and against the resume itself.

    Grounding is measured, not assumed: each evidence quote must actually occur
    in the resume text. This is the step that catches a fluent, invented answer.
    """
    result = state.get("result") or {}
    resume = _normalise(state.get("resume_text", ""))
    quotes = [q for q in (result.get("evidence") or []) if q and q.strip()]

    unsupported: List[str] = []
    for q in quotes:
        needle = _normalise(q)[:QUOTE_MATCH_CHARS]
        if needle and needle not in resume:
            unsupported.append(q)

    state["unsupported_quotes"] = unsupported
    state["grounded_ratio"] = (
        (len(quotes) - len(unsupported)) / len(quotes) if quotes else 0.0
    )

    schema_ok = isinstance(result.get("fit_score"), int) and bool(
        result.get("matching_skills") or result.get("missing_skills")
    )
    state["verified"] = bool(
        schema_ok and quotes and state["grounded_ratio"] >= GROUNDING_THRESHOLD
    )
    return state


def route_after_verify(state: AnalysisState) -> str:
    """Conditional edge: accept, or repair if budget remains."""
    if state.get("verified"):
        return "done"
    if state.get("attempts", 0) >= MAX_ATTEMPTS:
        state.setdefault("errors", []).append(
            "verification failed after max attempts; returning best effort"
        )
        return "done"
    return "repair"


# ------------------------------------------------------------------- graph
def build_graph():
    graph = StateGraph(AnalysisState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("verify", verify_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "evaluate")
    graph.add_edge("evaluate", "verify")
    graph.add_conditional_edges(
        "verify",
        route_after_verify,
        {"done": END, "repair": "evaluate"},
    )
    return graph.compile()


# Compiled once; the graph is stateless between invocations.
_COMPILED = None


def run_analysis(
    *,
    job_desc: str,
    resume_text: str,
    use_local: bool = False,
    model_name: str = "llama3",
    openai_key: Optional[str] = None,
) -> AnalysisState:
    """Run the full pipeline and return the final state."""
    global _COMPILED
    if _COMPILED is None:
        _COMPILED = build_graph()

    return _COMPILED.invoke(
        {
            "job_desc": job_desc,
            "resume_text": resume_text,
            "use_local": use_local,
            "model_name": model_name,
            "openai_key": openai_key,
            "context_chunks": [],
            "result": {},
            "attempts": 0,
            "errors": [],
        }
    )
