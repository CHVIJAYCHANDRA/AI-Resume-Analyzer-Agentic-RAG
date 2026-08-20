from __future__ import annotations

from typing import List, Optional

try:  # langchain >= 0.2
    from langchain_community.vectorstores import FAISS
except ImportError:  # pragma: no cover - legacy layout
    from langchain.vectorstores import FAISS

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:  # pragma: no cover
    from langchain.text_splitter import RecursiveCharacterTextSplitter

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
LOCAL_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OPENAI_EMBED_MODEL = "text-embedding-3-small"

# MiniLM loads from disk on first use; cache it across Streamlit reruns.
_local_embeddings = None


def _get_local_embeddings():
    """Load and memoise the MiniLM embedding model."""
    global _local_embeddings
    if _local_embeddings is not None:
        return _local_embeddings

    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
        except ImportError as e:
            raise RuntimeError(
                "Local embeddings need: "
                "pip install langchain-huggingface sentence-transformers"
            ) from e

    _local_embeddings = HuggingFaceEmbeddings(
        model_name=LOCAL_EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )
    return _local_embeddings


def _get_embeddings(openai_key: Optional[str], use_local: bool):
    """Local model when running locally or when no key is available."""
    if use_local or not openai_key:
        return _get_local_embeddings()

    try:
        from langchain_openai import OpenAIEmbeddings
    except ImportError:  # pragma: no cover
        from langchain.embeddings import OpenAIEmbeddings

    try:
        return OpenAIEmbeddings(model=OPENAI_EMBED_MODEL, api_key=openai_key)
    except TypeError:  # older signature
        return OpenAIEmbeddings(openai_api_key=openai_key)


def chunk_text(text: str) -> List[str]:
    """Split resume text into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return [c for c in splitter.split_text(text or "") if c.strip()]


def build_vector_index(
    text: str,
    openai_key: Optional[str] = None,
    use_local: bool = False,
):
    """Build a FAISS index over the resume.

    Raises RuntimeError with an actionable message; the agent's retrieve node
    catches it and degrades to no-retrieval rather than failing the run.
    """
    chunks = chunk_text(text)
    if not chunks:
        raise RuntimeError("Resume text produced no chunks to index")

    embeddings = _get_embeddings(openai_key, use_local)

    try:
        return FAISS.from_texts(chunks, embeddings)
    except Exception as e:
        msg = str(e).lower()
        if "429" in msg or "quota" in msg or "insufficient_quota" in msg:
            raise RuntimeError(
                "OpenAI embedding quota exceeded. Switch to local mode or "
                "check billing."
            ) from e
        if "401" in msg or "invalid_api_key" in msg:
            raise RuntimeError("Invalid OpenAI API key for embeddings.") from e
        raise RuntimeError(f"Failed to build vector index: {e}") from e


def query_vectorstore(store, query: str, k: int = 5) -> List[str]:
    """Return the top-k chunk texts most similar to `query`."""
    if store is None or not (query or "").strip():
        return []
    try:
        docs = store.similarity_search(query, k=k)
        return [d.page_content for d in docs if d.page_content]
    except Exception as e:
        raise RuntimeError(f"Vector search failed: {e}") from e


def query_with_scores(store, query: str, k: int = 5):
    """Return (chunk_text, distance) pairs.

    Useful for an eval harness: retrieval quality can be inspected numerically
    instead of eyeballed.
    """
    if store is None or not (query or "").strip():
        return []
    try:
        return [
            (d.page_content, float(score))
            for d, score in store.similarity_search_with_score(query, k=k)
        ]
    except Exception as e:
        raise RuntimeError(f"Scored vector search failed: {e}") from e
