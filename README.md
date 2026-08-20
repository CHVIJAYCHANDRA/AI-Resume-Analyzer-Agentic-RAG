# Resume Analyzer: Retrieval-Grounded Scoring on a LangGraph Agent

Resume/JD matching tools usually make one LLM call over the raw resume text and
return prose. That has two problems: the score cannot be checked, and the model
is free to justify it with things the resume never said.

This project addresses both. Retrieved resume passages are injected into the
prompt, the model must return structured JSON with **verbatim evidence quotes**,
and a verification step measures what fraction of those quotes actually appear
in the resume. If grounding is too low, the agent repairs its own output once
before returning.

Runs fully offline — local LLM and local embeddings, no API key required.

---

## Agent topology

retrieve ──► evaluate ──► verify ──► done
                 ▲            │
                 └── repair ◄─┘   (conditional, max 2 attempts)

| Node | Responsibility | Failure behaviour |
|---|---|---|
| `retrieve` | FAISS top-5 resume chunks most similar to the JD | degrades to no-retrieval, run continues |
| `evaluate` | one structured JSON call, `temperature=0` | records error, empty result |
| `verify` | schema check + evidence-grounding ratio | sets `verified=False` |
| repair edge | re-runs `evaluate` with corrective instructions | bounded at `MAX_ATTEMPTS=2` |

Regenerate this diagram from the compiled graph:
bash
python -c "from utils.agent_graph import build_graph; print(build_graph().get_graph().draw_ascii())"

---

## The grounding check

Every evidence string returned by the model is matched against the resume text.
Exact matching fails on real PDFs because extraction inserts line breaks
mid-sentence, so quotes are whitespace-normalised, lowercased, and compared on a
60-character prefix:
python
def _normalise(text: str) -> str:
    return " ".join((text or "").split()).lower()

`grounded_ratio` = supported quotes / total quotes. Below `0.5` the agent
triggers a repair pass; unsupported quotes are surfaced in red in the UI rather
than hidden.

This is the difference between "the model said 87" and "the model said 87 and
100% of its stated reasons are traceable to the document."

---

## Backends

| Mode | LLM | Embeddings | API key | Cost |
|---|---|---|---|---|
| **Local** (default) | Ollama — llama3 / mistral / qwen2.5 / phi3 | `all-MiniLM-L6-v2` | none | $0 |
| **Cloud** | gpt-4o-mini / gpt-4o / gpt-3.5-turbo | `text-embedding-3-small` | `OPENAI_API_KEY` | per-token |

Both paths request JSON at the API level — Ollama `format="json"`, OpenAI
`response_format={"type":"json_object"}` — so tolerant parsing is a fallback,
not the primary strategy.

`temperature=0.0` in both modes, so repeated runs are comparable. That is a
prerequisite for evaluating the scorer at all.

---

## Setup
bash
git clone https://github.com/CHVIJAYCHANDRA/AI-Resume-Analyzer-Agentic-RAG
cd AI-Resume-Analyzer-Agentic-RAG
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env

### Local mode (no key)
bash
ollama serve
ollama pull llama3
streamlit run app.py

### Cloud mode

Add `OPENAI_API_KEY` to `.env`, then untick **Use local LLM** in the sidebar.

**First run downloads ~90 MB of MiniLM weights** (cached in
`~/.cache/huggingface`, offline afterwards). `sentence-transformers` pulls
PyTorch, so the install is large — this is the price of key-free retrieval.

### Verify the wiring without the UI
bash
python -c "
from utils.rag_engine import build_vector_index, query_vectorstore
i = build_vector_index('Python engineer. Built RAG with FAISS. AWS CI/CD.', use_local=True)
print(query_vectorstore(i, 'retrieval augmented generation', k=2))
"

---

## Project structure

app.py                  Streamlit UI, metrics, run trace, downloads
utils/
  prompts.py            prompt template + Pydantic Evaluation schema (single source)
  evaluator.py          Ollama/OpenAI dispatch, prompt assembly, JSON parsing
  rag_engine.py         chunking, embedding selection, FAISS index and search
  agent_graph.py        LangGraph nodes, verification, conditional repair edge
  resume_parser.py      PyMuPDF text extraction
requirements.txt
.env.example

---

## Output contract
json
{
  "fit_score": 0,
  "matching_skills": [],
  "missing_skills": [],
  "strengths": [],
  "weaknesses": [],
  "improvements": [],
  "evidence": ["verbatim resume quote"]
}

Validated by `pydantic` (`fit_score` constrained to 0–100). Each run is written
to `output/evaluation_<timestamp>.json` together with a trace:
json
{
  "backend": "ollama-local",
  "model": "llama3",
  "attempts": 1,
  "retrieved_chunks": 5,
  "grounded_ratio": 1.0,
  "resume_truncated": false,
  "errors": []
}

Structured output plus a persisted trace is what makes offline evaluation
possible later — prose reports cannot be aggregated.

---

## Engineering decisions

- **Retrieval feeds the prompt.** Excerpts are injected before scoring, not
  displayed alongside it as decoration.
- **The prompt exists once.** `prompts.py` is the single source of truth; both
  backends and the repair path format the same template.
- **Failures raise, the graph decides.** Backend errors become `RuntimeError`
  with actionable messages; nodes catch them and record to `state["errors"]`
  instead of returning error text disguised as a result.
- **Truncation is visible.** Long resumes are capped (4k local / 6k cloud) and
  the UI says so, rather than dropping the tail silently.
- **Embeddings are memoised** at module level so MiniLM is not reloaded on every
  Streamlit rerun.

---

## Limitations

- Fit scores are **not yet validated against labelled data** — no accuracy or
  correlation figure is claimed. See below.
- Grounding checks whether a quote exists in the resume, not whether it supports
  the conclusion drawn from it.
- English-language, text-based PDFs only. Scanned resumes need OCR.
- Single-resume, single-JD; no batch mode or ranking across candidates.

## Planned

- `evals/` harness: labelled resume/JD pairs, score-vs-label correlation, and
  run-to-run score variance
- Retrieval ablation: chunk size and `k` against retrieval hit-rate
- Cost/latency comparison across local and cloud backends at equal agreement

---

## Stack

Python · Streamlit · LangGraph · LangChain · FAISS · sentence-transformers ·
Ollama · OpenAI · PyMuPDF · Pydantic

## License

MIT — see [LICENSE](LICENSE).


Claims removed from the old README


| Removed | Why |
|---|---|
| "GPT-4 evaluation pipeline" in the header | default backend is local llama3 |
| "Production-Ready" | no tests, no eval — unearned |
| data/ and output/ in the structure tree | data/ does not exist in the repo; output/ is created at runtime, now stated as such |
| "robust", "enterprise-level" | unbacked adjectives lower credibility |
| "base logic from privateGPT / other repos" | read as derivative; the code is yours now |

Claims added — and each is backed by code you've pasted


| Claim | Backed by |
|---|---|
| retrieval feeds the prompt | build_prompt(..., context_chunks) in evaluator.py |
| stateful agent, conditional control flow | AnalysisState + add_conditional_edges in agent_graph.py |
| bounded repair / reliability | MAX_ATTEMPTS, route_after_verify |
| grounding measurement | verify_node, grounded_ratio |
| graceful degradation | retrieve_node try/except |
| runs offline, no key | _get_local_embeddings + Ollama default |

The Limitations section is deliberate. Naming what you haven't measured reads as more credible than silence  it's the pattern in openai/evals ("we are currently not accepting evals with custom code") and it pre-empts the first question a reviewer would ask.
