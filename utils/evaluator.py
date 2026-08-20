from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import List, Optional, Tuple

from .prompts import (
    JD_CHAR_LIMIT,
    PROMPT_TEMPLATE,
    REPAIR_SUFFIX,
    RESUME_LIMIT_CLOUD,
    RESUME_LIMIT_LOCAL,
    Evaluation,
)

DEFAULT_OLLAMA_URL = "http://localhost:11434"


# --------------------------------------------------------------- backends
def _call_ollama(prompt: str, model: str, base_url: str) -> str:
    import ollama

    client = ollama.Client(host=base_url)
    resp = client.generate(
        model=model,
        prompt=prompt,
        format="json",  # constrained decoding: valid JSON or nothing
        options={
            "temperature": 0.0,
            "num_predict": 1500,
            "num_ctx": 8192,
        },
    )
    return resp.get("response", "") or ""


def _call_openai(prompt: str, model: str, api_key: str) -> str:
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model=model,
        temperature=0.0,
        api_key=api_key,
        model_kwargs={"response_format": {"type": "json_object"}},
    )
    resp = llm.invoke(prompt)
    return getattr(resp, "content", None) or str(resp)


def call_llm(
    prompt: str,
    *,
    use_local: bool,
    model_name: str,
    openai_key: Optional[str] = None,
    local_model_url: Optional[str] = None,
) -> str:
    """Dispatch to Ollama or OpenAI, normalising failures into RuntimeError."""
    if use_local:
        url = local_model_url or os.getenv("OLLAMA_URL", DEFAULT_OLLAMA_URL)
        try:
            return _call_ollama(prompt, model_name, url)
        except ImportError as e:
            raise RuntimeError(
                "The `ollama` package is missing. Run: pip install ollama"
            ) from e
        except Exception as e:
            msg = str(e).lower()
            if "not found" in msg or "no such model" in msg:
                raise RuntimeError(
                    f"Model '{model_name}' is not pulled. "
                    f"Run: ollama pull {model_name}"
                ) from e
            if "connection" in msg or "refused" in msg or "timed out" in msg:
                raise RuntimeError(
                    f"Cannot reach Ollama at {url}. Run: ollama serve"
                ) from e
            raise RuntimeError(f"Ollama error: {e}") from e

    if not openai_key:
        raise RuntimeError("OPENAI_API_KEY is not set in .env")
    try:
        return _call_openai(prompt, model_name, openai_key)
    except Exception as e:
        msg = str(e).lower()
        if "429" in msg or "quota" in msg or "insufficient_quota" in msg:
            raise RuntimeError(
                "OpenAI quota exceeded. Check "
                "https://platform.openai.com/account/billing"
            ) from e
        if "401" in msg or "invalid_api_key" in msg:
            raise RuntimeError("Invalid OpenAI API key - check your .env") from e
        if "does not exist" in msg or "model_not_found" in msg:
            raise RuntimeError(
                f"Model '{model_name}' is not available on this account."
            ) from e
        raise RuntimeError(f"OpenAI error: {e}") from e


# ---------------------------------------------------------------- parsing
# Restores missing backticks to match opening/closing code fences
_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def parse_evaluation(raw: str) -> Evaluation:
    """Extract the first JSON object from `raw` and validate it.

    Tolerates markdown fences and leading prose because local models ignore
    formatting instructions more often than hosted ones.
    """
    if not raw or not raw.strip():
        raise ValueError("Model returned an empty response")

    text = _FENCE_RE.sub("", raw.strip())
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object found in output: {text[:200]!r}")

    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError as e:
        raise ValueError(f"Malformed JSON: {e}") from e

    if not isinstance(payload, dict):
        raise ValueError("Top-level JSON value is not an object")

    return Evaluation(**payload)


# ------------------------------------------------------------ prompt build
def build_prompt(
    job_desc: str,
    resume_text: str,
    context_chunks: Optional[List[str]] = None,
    *,
    use_local: bool = False,
) -> Tuple[str, bool]:
    """Assemble the prompt.

    Returns (prompt, was_truncated) so the caller can tell the user the resume
    did not fit, instead of silently dropping the tail.
    """
    limit = RESUME_LIMIT_LOCAL if use_local else RESUME_LIMIT_CLOUD
    truncated = len(resume_text) > limit

    chunks = [c for c in (context_chunks or []) if c and c.strip()]
    context = (
        "\n---\n".join(chunks)
        if chunks
        else "(no excerpts retrieved - judge from the full resume below)"
    )

    prompt = PROMPT_TEMPLATE.format(
        job=job_desc[:JD_CHAR_LIMIT],
        resume=resume_text[:limit],
        context=context,
        k=len(chunks),
    )
    return prompt, truncated


def evaluate_once(
    job_desc: str,
    resume_text: str,
    context_chunks: Optional[List[str]] = None,
    *,
    use_local: bool = False,
    model_name: str = "llama3",
    openai_key: Optional[str] = None,
    repair: bool = False,
) -> Evaluation:
    """One retrieve-grounded evaluation call. Raises on failure."""
    prompt, _ = build_prompt(
        job_desc, resume_text, context_chunks, use_local=use_local
    )
    if repair:
        prompt += REPAIR_SUFFIX

    raw = call_llm(
        prompt,
        use_local=use_local,
        model_name=model_name,
        openai_key=openai_key,
    )
    return parse_evaluation(raw)


# ------------------------------------------------------------ persistence
def save_evaluation_report(
    evaluation: dict,
    job_desc: str,
    resume_text: str,
    output_dir: str = "output",
    trace: Optional[dict] = None,
) -> Optional[str]:
    """Persist the structured result plus the run trace as JSON.

    Storing the trace (attempts, retrieved chunk count, grounding ratio) is what
    makes a later offline eval possible - prose reports cannot be aggregated.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(output_dir, f"evaluation_{ts}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "timestamp": ts,
                    "evaluation": evaluation,
                    "trace": trace or {},
                    "job_description": job_desc[:500],
                    "resume_preview": resume_text[:500],
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        return path
    except Exception:
        return None
