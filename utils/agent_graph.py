from __future__ import annotations

from typing import List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from .evaluator import evaluate_once
from .rag_engine import build_vector_index, query_vectorstore

MAX_ATTEMPTS = 2


class AnalysisState(TypedDict, total=False):
    job_desc: str
    resume_text: str
    use_local: bool
    model_name: str
    openai_key: Optional[str]
    context_chunks: List[str]
    result: dict
    attempts: int
    errors: List[str]
    grounded_ratio: float


def retrieve_node(state: AnalysisState) -> AnalysisState:
    try:
        index = build_vector_index(
            state["resume_text"],
            openai_key=state.get("openai_key"),
            use_local=state.get("use_local", False),
        )
        state["context_chunks"] = query_vectorstore(index, state["job_desc"], k=5)
    except Exception as e:                       # degrade, never crash
        state["context_chunks"] = []
        state.setdefault("errors", []).append(f"retrieve: {e}")
    return state


def evaluate_node(state: AnalysisState) -> AnalysisState:
    state["attempts"] = state.get("attempts", 0) + 1
    try:
        ev = evaluate_once(
            state["job_desc"], state["resume_text"],
            state.get("context_chunks"),
            use_local=state.get("use_local", False),
            model_name=state["model_name"],
            openai_key=state.get("openai_key"),
            repair=state["attempts"] > 1,
        )
        state["result"] = ev.model_dump()
    except Exception as e:
        state.setdefault("errors", []).append(f"evaluate#{state['attempts']}: {e}")
        state["result"] = {}
    return state


def verify_node(state: AnalysisState) -> AnalysisState:
    """Grounding check: what fraction of evidence quotes exist in the resume?"""
    res, resume = state.get("result") or {}, state["resume_text"].lower()
    quotes = res.get("evidence") or []
    hits = sum(1 for q in quotes if q and q.lower()[:60] in resume)
    state["grounded_ratio"] = (hits / len(quotes)) if quotes else 0.0
    return state


def route(state: AnalysisState) -> str:
    res = state.get("result") or {}
    ok = (
        isinstance(res.get("fit_score"), int)
        and res.get("matching_skills")
        and state.get("grounded_ratio", 0.0) >= 0.5
    )
    if ok or state.get("attempts", 0) >= MAX_ATTEMPTS:
        return "done"
    return "repair"


def build_graph():
    g = StateGraph(AnalysisState)
    g.add_node("retrieve", retrieve_node)
    g.add_node("evaluate", evaluate_node)
    g.add_node("verify", verify_node)
    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "evaluate")
    g.add_edge("evaluate", "verify")
    g.add_conditional_edges("verify", route, {"done": END, "repair": "evaluate"})
    return g.compile()


def run_analysis(**kwargs) -> AnalysisState:
    return build_graph().invoke({"attempts": 0, "errors": [], **kwargs})
