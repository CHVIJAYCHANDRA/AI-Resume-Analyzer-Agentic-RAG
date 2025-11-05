import streamlit as st
import os
from dotenv import load_dotenv
from utils.evaluator import evaluate_resume, save_evaluation_report
from utils.resume_parser import extract_text
from utils.rag_engine import build_vector_index, query_vectorstore

env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=env_path, override=True)
openai_key = os.getenv("OPENAI_API_KEY")
st.set_page_config(
    page_title="AI Resume Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">AI Resume Analyzer (Agentic RAG)</p>', unsafe_allow_html=True)
st.markdown("**A complete AI-powered resume analysis system combining FAISS + LangChain ingestion, Streamlit UI, and GPT-4 evaluation pipeline**")
st.markdown("---")

with st.sidebar:
    st.header("Configuration")
    
    st.markdown("### Model Selection")
    use_local_llm = st.checkbox("Use Local LLM (Ollama)", value=True)
    
    if use_local_llm:
        local_model = st.selectbox(
            "Local Model",
            ["llama3", "llama2", "mistral", "neural-chat", "codellama"],
            index=0
        )
        model_name = local_model
        openai_model = None
    else:
        openai_model = st.selectbox(
            "OpenAI Model",
            ["gpt-3.5-turbo", "gpt-4-turbo-preview", "gpt-4"],
            index=0
        )
        model_name = openai_model
        if not openai_key:
            st.error("OPENAI_API_KEY not found in .env file")
    
    st.markdown("---")
    st.markdown("### How to Use")
    st.markdown("""
    1. Upload your resume PDF
    2. Paste the job description
    3. Click 'Analyze Resume'
    4. Review the evaluation report
    """)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Upload Resume")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf"
    )
    
    if uploaded_file:
        st.success(f"File uploaded: {uploaded_file.name}")
        file_size = len(uploaded_file.read()) / 1024
        uploaded_file.seek(0)
        st.info(f"File size: {file_size:.2f} KB")

with col2:
    st.subheader("Job Description")
    job_description = st.text_area(
        "Paste the job description",
        placeholder="Enter job description...",
        height=200
    )
    
    if job_description:
        word_count = len(job_description.split())
        st.info(f"{word_count} words")

st.markdown("---")

analyze_button = st.button("Analyze Resume", type="primary", use_container_width=True)

if analyze_button:
    if not uploaded_file:
        st.error("Please upload a PDF resume first")
    elif not job_description.strip():
        st.error("Please paste the job description")
    elif not use_local_llm and not openai_key:
        st.error("OpenAI API key not configured")
    else:
        # Initialize progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Extract text from PDF
            status_text.text("Step 1/4: Extracting text from PDF...")
            progress_bar.progress(25)
            resume_text = extract_text(uploaded_file)
            st.success(f"Extracted {len(resume_text)} characters from resume")
            
            # Step 2: Build vector index (optional - only if using OpenAI)
            vector_index = None
            rag_error = None
            if not use_local_llm and openai_key:
                try:
                    status_text.text("Step 2/4: Building vector embeddings...")
                    progress_bar.progress(50)
                    vector_index = build_vector_index(resume_text, openai_key)
                    st.success("Vector index created")
                except Exception as e:
                    rag_error = str(e)
                    st.warning(f"RAG Vector Index: {rag_error}")
            else:
                progress_bar.progress(50)
            
            model_display = "Local LLM" if use_local_llm else model_name
            status_text.text(f"Step 3/4: Evaluating resume with {model_display}...")
            progress_bar.progress(75)
            
            if use_local_llm:
                st.info("Processing may take 30-90 seconds. First run loads model into memory.")
            
            try:
                with st.spinner("Processing with local AI..."):
                    
                    evaluation_report = evaluate_resume(
                        job_description, 
                        resume_text, 
                        openai_key=openai_key if not use_local_llm else None,
                        model_name=model_name,
                        use_local=use_local_llm
                    )
                
            except Exception as eval_error:
                error_msg = str(eval_error)
                if "timeout" in error_msg.lower() or "connection" in error_msg.lower():
                    evaluation_report = f"## Error: Connection Timeout\n\nOllama took too long to respond. Wait longer or try a smaller model.\n\nError: {error_msg}"
                elif "not found" in error_msg.lower():
                    evaluation_report = f"## Error: Model Not Found\n\nRun: `ollama pull {model_name}`"
                else:
                    evaluation_report = f"## Error during evaluation\n\n{error_msg}"
            
            rag_suggestions = []
            if vector_index:
                try:
                    status_text.text("Step 4/4: Generating RAG-based suggestions...")
                    rag_suggestions = query_vectorstore(vector_index, job_description, k=5)
                except Exception as e:
                    st.warning(f"Could not generate RAG suggestions: {str(e)}")
            
            progress_bar.progress(100)
            status_text.text("Analysis complete")
            
            # Display results
            st.markdown("---")
            
            # Resume Evaluation Report Section
            st.subheader("Resume Evaluation Report")
            with st.expander("View Full Evaluation Report", expanded=True):
                st.markdown(evaluation_report)
            
            # RAG-Based Suggestions Section
            st.subheader("RAG-Based Suggestions")
            if rag_suggestions:
                st.markdown("**Top relevant sections from your resume based on job description:**")
                for idx, suggestion in enumerate(rag_suggestions, 1):
                    with st.container():
                        st.markdown(f"**Suggestion {idx}:**")
                        st.info(suggestion[:500] + "..." if len(suggestion) > 500 else suggestion)
            else:
                if rag_error:
                    st.warning("RAG suggestions unavailable. Check OpenAI API quota.")
                else:
                    st.info("No RAG suggestions available.")
            
            # Save results
            try:
                output_path = save_evaluation_report(
                    evaluation_report,
                    job_description,
                    resume_text,
                    output_dir="output"
                )
                if output_path:
                    st.success(f"Report saved to: {output_path}")
            except Exception as e:
                st.warning(f"Could not save report: {str(e)}")
            
            # Download button for report
            st.download_button(
                label="Download Evaluation Report",
                data=evaluation_report,
                file_name=f"resume_evaluation_{uploaded_file.name.replace('.pdf', '.txt')}",
                mime="text/plain"
            )
            
        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")
        
        finally:
            progress_bar.empty()
            status_text.empty()

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>AI Resume Analyzer | Built with Local AI, LangChain, and FAISS</p>
    </div>
    """,
    unsafe_allow_html=True
)

