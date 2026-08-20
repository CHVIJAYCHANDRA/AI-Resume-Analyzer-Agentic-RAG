import json
import os

import streamlit as st
from dotenv import load_dotenv

from utils.agent_graph import MAX_ATTEMPTS, run_analysis
from utils.evaluator import build_prompt, save_evaluation_report
from utils.resume_parser import extract_text

# ---------------------------------------------------------------- config
env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=env_path, override=True)
openai_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(
    page_title="Resume Analyzer - Agentic RAG",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        padding: 0.8rem 0 0.2rem 0;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 1.2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<p class="main-header">Resume Analyzer</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Retrieval-grounded resume scoring on a LangGraph '
    "agent: retrieve &rarr; evaluate &rarr; verify &rarr; bounded repair. "
    "Runs fully local (Ollama + MiniLM) or on OpenAI.</p>",
    unsafe_allow_html=True,
)
st.markdown("---")


# ---------------------------------------------------------------- helpers
def to_markdown(result: dict, state: dict) -> str:
    """Render the structured result as a downloadable markdown report."""

    def bullets(items):
        items = items or []
        return "\n".join(f"- {i}" for i in items) if items else "- (none reported)"

    return f"""# Resume Evaluation Report

**Fit score:** {result.get('fit_score', 'n/a')}/100

**Run trace:** attempts {state.get('attempts', 0)}/{MAX_ATTEMPTS} ·
retrieved chunks {len(state.get('context_chunks') or [])} ·
evidence grounded {state.get('grounded_ratio', 0.0):.0%}

## Matching skills
{bullets(result.get('matching_skills'))}

## Missing skills
{bullets(result.get('missing_skills'))}

## Strengths
{bullets(result.get('strengths'))}

## Weaknesses
{bullets(result.get('weaknesses'))}

## Suggested improvements
{bullets(result.get('improvements'))}

## Evidence (verbatim from resume)
{bullets(result.get('evidence'))}
"""


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Configuration")

    st.markdown("### Backend")
    use_local_llm = st.checkbox(
        "Use local LLM (Ollama)",
        value=True,
        help="Local mode needs no API key. Embeddings fall back to a local "
        "MiniLM model, so retrieval works offline too.",
    )

    if use_local_llm:
        model_name = st.selectbox(
            "Local model",
            ["llama3", "llama3.1", "mistral", "qwen2.5", "phi3"],
            index=0,
        )
        st.caption("Retrieval: local MiniLM embeddings + FAISS")
    else:
        model_name = st.selectbox(
            "OpenAI model",
            ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
            index=0,
        )
        st.caption("Retrieval: OpenAI embeddings + FAISS")
        if not openai_key:
            st.error("OPENAI_API_KEY not found in .env")

    st.markdown("---")
    st.markdown("### Agent behaviour")
    st.markdown(
        f"""
- **retrieve** - top-5 resume chunks most similar to the JD
- **evaluate** - single structured-JSON call, temperature 0
- **verify** - schema check + evidence-grounding check
- **repair** - re-runs once if verification fails (max {MAX_ATTEMPTS} attempts)
"""
    )

    st.markdown("---")
    st.markdown("### How to use")
    st.markdown(
        """
1. Upload a resume PDF
2. Paste the job description
3. Click **Analyze resume**
4. Review the score, evidence and run trace
"""
    )

# ---------------------------------------------------------------- inputs
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Resume")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    if uploaded_file:
        size_kb = len(uploaded_file.read()) / 1024
        uploaded_file.seek(0)
        st.success(f"{uploaded_file.name} · {size_kb:.1f} KB")

with col2:
    st.subheader("Job description")
    job_description = st.text_area(
        "Paste the job description",
        placeholder="Enter job description...",
        height=220,
    )
    if job_description:
        st.caption(
            f"{len(job_description.split())} words · "
            f"{len(job_description)} characters"
        )

st.markdown("---")
analyze = st.button("Analyze resume", type="primary", use_container_width=True)

# ---------------------------------------------------------------- run
if analyze:
    if not uploaded_file:
        st.error("Upload a PDF resume first.")
        st.stop()
    if not job_description.strip():
        st.error("Paste the job description.")
        st.stop()
    if not use_local_llm and not openai_key:
        st.error("OpenAI selected but OPENAI_API_KEY is missing in .env")
        st.stop()

    progress = st.progress(0)
    status = st.empty()

    try:
        # 1. parse ------------------------------------------------------
        status.text("Extracting text from PDF...")
        progress.progress(20)
        resume_text = extract_text(uploaded_file)
        st.success(f"Extracted {len(resume_text):,} characters")

        # honest truncation notice, before the model ever sees the text
        _, truncated = build_prompt(
            job_description, resume_text, use_local=use_local_llm
        )
        if truncated:
            st.warning(
                f"Resume is {len(resume_text):,} characters and will be "
                "truncated to fit the model context. Retrieved excerpts still "
                "cover the whole document."
            )

        # 2. run the agent ---------------------------------------------
        status.text("Running agent: retrieve -> evaluate -> verify...")
        progress.progress(55)
        if use_local_llm:
            st.info(
                "Local inference can take 30-90s. The first run also loads "
                "the model and the embedding model into memory."
            )

        with st.spinner("Analyzing..."):
            state = run_analysis(
                job_desc=job_description,
                resume_text=resume_text,
                use_local=use_local_llm,
                model_name=model_name,
                openai_key=None if use_local_llm else openai_key,
            )

        progress.progress(100)
        status.text("Done")

        result = state.get("result") or {}
        chunks = state.get("context_chunks") or []
        attempts = state.get("attempts", 0)
        grounded = state.get("grounded_ratio", 0.0)
        errors = state.get("errors") or []

        st.markdown("---")

        if not result:
            st.error(
                "The agent could not produce a valid evaluation. "
                "See the run details below."
            )
        else:
            # 3. headline metrics --------------------------------------
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Fit score", f"{result.get('fit_score', 0)}/100")
            m2.metric("Retrieved chunks", len(chunks))
            m3.metric("Evidence grounded", f"{grounded:.0%}")
            m4.metric("Attempts", f"{attempts}/{MAX_ATTEMPTS}")

            if not chunks:
                st.warning(
                    "Retrieval returned nothing, so the score is based on the "
                    "raw resume text only. Treat it as less reliable."
                )
            if grounded < 0.5:
                st.warning(
                    "Under half the evidence quotes were found verbatim in the "
                    "resume. The model may be paraphrasing or inventing support."
                )

            # 4. skills -----------------------------------------------
            s1, s2 = st.columns(2)
            with s1:
                st.subheader("Matching skills")
                for s in result.get("matching_skills") or ["(none reported)"]:
                    st.markdown(f"- {s}")
            with s2:
                st.subheader("Missing skills")
                for s in result.get("missing_skills") or ["(none reported)"]:
                    st.markdown(f"- {s}")

            # 5. qualitative ------------------------------------------
            q1, q2 = st.columns(2)
            with q1:
                st.subheader("Strengths")
                for s in result.get("strengths") or ["(none reported)"]:
                    st.markdown(f"- {s}")
            with q2:
                st.subheader("Weaknesses")
                for s in result.get("weaknesses") or ["(none reported)"]:
                    st.markdown(f"- {s}")

            st.subheader("Suggested improvements")
            for s in result.get("improvements") or ["(none reported)"]:
                st.markdown(f"- {s}")

            # 6. grounding evidence -----------------------------------
            st.subheader("Evidence")
            st.caption(
                "Quotes the model used to justify the score. Each is checked "
                "against the resume text; unmatched quotes are flagged."
            )
            resume_lower = resume_text.lower()
            for q in result.get("evidence") or []:
                if q and q.lower()[:60] in resume_lower:
                    st.success(q)
                else:
                    st.error(f"not found in resume: {q}")

            # 7. retrieved context ------------------------------------
            with st.expander(f"Retrieved resume excerpts ({len(chunks)})"):
                for i, c in enumerate(chunks, 1):
                    st.markdown(f"**Chunk {i}**")
                    st.info(c[:600] + ("..." if len(c) > 600 else ""))

        # 8. run details ----------------------------------------------
        with st.expander("Run details"):
            st.json(
                {
                    "backend": "ollama-local" if use_local_llm else "openai",
                    "model": model_name,
                    "attempts": attempts,
                    "max_attempts": MAX_ATTEMPTS,
                    "retrieved_chunks": len(chunks),
                    "grounded_ratio": round(grounded, 3),
                    "resume_chars": len(resume_text),
                    "resume_truncated": truncated,
                    "errors": errors,
                }
            )
        for e in errors:
            st.warning(e)

        # 9. persist + download ---------------------------------------
        if result:
            trace = {
                "backend": "ollama-local" if use_local_llm else "openai",
                "model": model_name,
                "attempts": attempts,
                "retrieved_chunks": len(chunks),
                "grounded_ratio": round(grounded, 3),
                "resume_truncated": truncated,
                "errors": errors,
            }
            path = save_evaluation_report(
                result, job_description, resume_text,
                output_dir="output", trace=trace,
            )
            if path:
                st.caption(f"Saved: {path}")

            base = uploaded_file.name.rsplit(".", 1)[0]
            d1, d2 = st.columns(2)
            d1.download_button(
                "Download report (Markdown)",
                data=to_markdown(result, state),
                file_name=f"{base}_evaluation.md",
                mime="text/markdown",
                use_container_width=True,
            )
            d2.download_button(
                "Download result (JSON)",
                data=json.dumps({"evaluation": result, "trace": trace}, indent=2),
                file_name=f"{base}_evaluation.json",
                mime="application/json",
                use_container_width=True,
            )

    except Exception as e:
        st.error(f"Analysis failed: {e}")

    finally:
        progress.empty()
        status.empty()

# ---------------------------------------------------------------- footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align:center; color:#888; padding:16px;'>
      Resume Analyzer · LangGraph agent · FAISS retrieval · Ollama / OpenAI
    </div>
    """,
    unsafe_allow_html=True,
)
