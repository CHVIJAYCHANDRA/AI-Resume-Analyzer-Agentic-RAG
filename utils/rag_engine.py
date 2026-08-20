try:
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS
except ImportError:
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.vectorstores import FAISS

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

def _get_embeddings(openai_key: str | None, use_local: bool):
    if use_local or not openai_key:
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(api_key=openai_key)


def build_vector_index(text, openai_key=None, use_local=False):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=100, length_function=len)
    chunks = splitter.split_text(text)
    return FAISS.from_texts(chunks, _get_embeddings(openai_key, use_local))
