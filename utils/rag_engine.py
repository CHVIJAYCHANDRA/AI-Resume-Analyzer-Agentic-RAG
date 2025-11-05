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


def build_vector_index(text, openai_key):
    """
    Build a FAISS vector index from resume text.
    
    Args:
        text: Resume text content
        openai_key: OpenAI API key for embeddings
        
    Returns:
        FAISS: Vector store index
    """
    try:
        # Split text into chunks for better embedding
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            length_function=len,
        )
        
        # Create text chunks
        chunks = splitter.split_text(text)
        
        # Initialize embeddings
        # Try new API first, fallback to old API
        try:
            embeddings = OpenAIEmbeddings(api_key=openai_key)
        except TypeError:
            embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
        
        # Build FAISS vector store
        vector_store = FAISS.from_texts(chunks, embeddings)
        
        return vector_store
    except Exception as e:
        error_msg = str(e)
        # Check for specific OpenAI API errors
        if "429" in error_msg or "insufficient_quota" in error_msg or "quota" in error_msg.lower():
            raise Exception(
                "OpenAI API Quota Exceeded. Check billing at https://platform.openai.com/account/billing"
            )
        elif "401" in error_msg or "invalid_api_key" in error_msg.lower():
            raise Exception(
                "Invalid OpenAI API Key. Check your .env file."
            )
        else:
            raise Exception(f"Error building vector index: {error_msg}")


def query_vectorstore(store, query, k=3):
    try:
        # Perform similarity search
        results = store.similarity_search(query, k=k)
        
        # Extract page content from results
        suggestions = [r.page_content for r in results]
        
        return suggestions
    except Exception as e:
        raise Exception(f"Error querying vector store: {str(e)}")

