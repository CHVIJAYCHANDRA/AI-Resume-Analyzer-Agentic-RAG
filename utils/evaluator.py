from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import List, Optional, Tuple

from pydantic import ValidationError

from .prompts import (
    JD_CHAR_LIMIT, PROMPT_TEMPLATE, REPAIR_SUFFIX,
    RESUME_LIMIT_CLOUD, RESUME_LIMIT_LOCAL, Evaluation,
)


# ---------- backends ----------
def _call_ollama(prompt: str, model: str, base_url: str) -> str:
    import ollama
    client = ollama.Client(host=base_url)
    resp = client.generate(
        model=model,
        prompt=prompt,
        format="json",                       # forces valid JSON from Ollama
        options={"temperature": 0.0, "num_predict": 1500, "num_ctx": 8192},
    )
    return resp.get("response", "")


def _call_openai(prompt: str, model: str, api_key: str) -> str:
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(
        model=model, temperature=0.0, api_key=api_key,
        model_kwargs={"response_format": {"type": "json_object"}},
    )
    r = llm.invoke(prompt)
    return getattr(r, "content", str(r))


def call_llm(prompt: str, *, use_local: bool, model_name: str,
             openai_key: Optional[str] = None,
             local_model_url: str = "http://localhost:11434") -> str:
    """Dispatch to Ollama or OpenAI. Raises with an actionable message."""
    if use_local:
        try:
            return _call_ollama(prompt, model_name, local_model_url)
        except ImportError as e:
            raise RuntimeError("pip install ollama  (and: ollama serve)") from e
        except Exception as e:
            m = str(e).lower()
            if "not found" in m:
                raise RuntimeError(f"Model missing. Run: ollama pull {model_name}") from e
            if "connection" in m or "refused" in m:
                raise RuntimeError("Ollama not reachable. Run: ollama serve") from e
            raise RuntimeError(f"Ollama error: {e}") from e

    if not openai_key:
        raise RuntimeError("OPENAI_API_KEY missing in .env")
    try:
        return _call_openai(prompt, model_name, openai_key)
    except Exception as e:
        m = str(e).lower()
        if "quota" in m or "429" in m:
            raise RuntimeError("OpenAI quota exceeded - check billing.") from e
        if "401" in m or "invalid_api_key" in m:
            raise RuntimeError("Invalid OpenAI API key.") from e
        raise RuntimeError(f"OpenAI error: {e}") from e


# ---------- parsing ----------
def parse_evaluation(raw: str) -> Evaluation:
    """Tolerant JSON extraction -> validated Evaluation."""
    text = raw.strip()
    text = re.sub(r"$", "", text, flags=re.M).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object in model output")
    return Evaluation(**json.loads(text[start:end + 1]))


def build_prompt(job_desc: str, resume_text: str,
                 context_chunks: Optional[List[str]] = None,
                 *, use_local: bool = False) -> Tuple[str, bool]:
    """Returns (prompt, was_truncated) so the UI can warn honestly."""
    limit = RESUME_LIMIT_LOCAL if use_local else RESUME_LIMIT_CLOUD
    truncated = len(resume_text) > limit
    chunks = context_chunks or []
    context = "\n---\n".join(chunks) if chunks else "(retrieval unavailable)"
    prompt = PROMPT_TEMPLATE.format(
        job=job_desc[:JD_CHAR_LIMIT],
        resume=resume_text[:limit],
        context=context,
        k=len(chunks),
    )
    return prompt, truncated


def evaluate_once(job_desc: str, resume_text: str,
                  context_chunks: Optional[List[str]] = None,
                  *, use_local: bool = False, model_name: str = "llama3",
                  openai_key: Optional[str] = None,
                  repair: bool = False) -> Evaluation:
    prompt, _ = build_prompt(job_desc, resume_text, context_chunks,
                             use_local=use_local)
    if repair:
        prompt += REPAIR_SUFFIX
    raw = call_llm(prompt, use_local=use_local, model_name=model_name,
                   openai_key=openai_key)
    return parse_evaluation(raw)


# ---------- persistence ----------
def save_evaluation_report(evaluation: dict, job_desc: str, resume_text: str,
                           output_dir: str = "output",
                           trace: Optional[dict] = None) -> Optional[str]:
    try:
        os.makedirs(output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(output_dir, f"evaluation_{ts}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"timestamp": ts, "evaluation": evaluation,
                       "trace": trace or {},
                       "job_description": job_desc[:500],
                       "resume_preview": resume_text[:500]},
                      f, indent=2, ensure_ascii=False)
        return path
    except Exception:
        return None
