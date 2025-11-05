# AI Resume Analyzer (Agentic RAG)

**A complete AI-powered resume analysis system that combines FAISS + LangChain ingestion, Streamlit UI, and GPT-4 evaluation pipeline to score resumes against job descriptions with zero API costs using local AI.**

##  What Makes This Unique?

1. 100% FREE - Uses local Ollama models (no OpenAI API costs)
2. Privacy-First - All processing happens locally, your data never leaves your machine
3. Agentic RAG Architecture - Combines semantic search (FAISS) with LLM evaluation for intelligent suggestions
4. Production-Ready - Optimized for local inference with streaming support
5. Comprehensive Analysis - Fit score, skills matching, missing skills, improvement suggestions

## Architecture

```
AI Resume Analyzer (Agentic RAG)
│
├─ User Interface (Streamlit)
│  │
│  ├─ PDF Upload Component
│  │  │
│  └─ Job Description Input
│     │
│     │
│     
│
├─ Processing Layer
│  │
│  ├─ PDF Parser (PyMuPDF)
│  │  └─ Extract text from PDF
│  │     │
│  └─ Text Chunker (LangChain)
│     └─ Split into chunks for embedding
│        │
│        │
│        
│
├─ AI Engine Layer
│  │
│  ├─ Local LLM (Ollama)
│  │  ├─ llama3
│  │  ├─ llama2
│  │  └─ mistral
│  │     │
│  ├─ RAG Engine (FAISS + LangChain)
│  │  ├─ Build embeddings
│  │  ├─ Vector search
│  │  └─ Semantic matching
│  │     │
│  └─ Evaluation Pipeline
│     ├─ GPT-4 style prompt
│     ├─ Fit Score (0-100)
│     ├─ Skills Analysis
│     └─ Improvement Tips
│        │
│        │
│        
│
└─ Output Layer
   │
   ├─ Evaluation Report
   │  ├─ Overall Fit Score
   │  ├─ Matching Skills
   │  ├─ Missing Skills
   │  ├─ Improvements
   │  └─ Strengths/Weaknesses
   │
   ├─ RAG Suggestions
   │  ├─ Semantic matches
   │  ├─ Relevant sections
   │  └─ Improvement hints
   │
   └─ Export
      ├─ JSON (output/)
      └─ Download TXT
```

## Requirements

- **Folder**: `/AI_Portfolio/1_AI_Resume_Analyzer`
- **Subfolders**: `/utils` (evaluator.py, resume_parser.py, rag_engine.py), `/data`, `/output`
- **Language**: Python 3.10+
- **UI**: Streamlit
- **Core Libraries**: langchain, openai, faiss-cpu, streamlit, PyMuPDF, python-dotenv, ollama

## Quick Start

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.ai) installed

### Installation

```bash

cd AI-Resume-Analyzer-Agentic-RAG

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Pull Ollama model (choose one)
ollama pull llama3      
# OR
ollama pull llama2 
# OR
ollama pull mistral    
```

### Run the Application

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

## How to Use

1. **Upload Resume**: Click "Browse files" and select your PDF resume
2. **Enter Job Description**: Paste or type the complete job description
3. **Select Model**: Choose local LLM (default: llama3) or OpenAI (requires API key)
4. **Analyze**: Click "Analyze Resume" button
5. **Review Results**: Get comprehensive evaluation report with:
   - **Overall Fit Score** (0-100)
   - **Key Matching Skills**
   - **Missing Skills**
   - **Suggested Improvements** (5-7 actionable points)
   - **Strengths** (3-5 key strengths)
   - **Weaknesses** (2-3 areas for improvement)
6. **RAG-Based Suggestions**: Semantic search results from your resume

## Functional Specification

### 1. PDF Resume Upload
- Supports PDF format
- Extracts text using PyMuPDF (fitz)
- Handles multi-page resumes

### 2. Job Description Input
- Text area for pasting job descriptions
- Real-time word count
- No character limits

### 3. Resume Parsing
- Automatic text extraction from PDF
- Multi-page support
- Error handling for corrupted/invalid PDFs

### 4. GPT-4 Evaluation Pipeline
- **Overall Fit Score**: Numerical score (0-100)
- **Matching Skills**: Top skills from resume matching job requirements
- **Missing Skills**: Critical skills not evident in resume
- **Improvement Suggestions**: 5-7 actionable bullet points
- **Strengths**: 3-5 key strengths relevant to role
- **Weaknesses**: 2-3 areas where candidate may fall short

### 5. Output Sections
- **Resume Evaluation Report**: Comprehensive GPT-4 analysis
- **RAG-Based Suggestions**: Semantic search using FAISS vector embeddings

### 6. Results Export
- Saves as JSON in `/output` folder with timestamp
- Format: `evaluation_YYYYMMDD_HHMMSS.json`
- Download as text file option

##  Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **LLM** | Ollama (Local) / OpenAI (Optional) | Language understanding & evaluation |
| **Framework** | LangChain | Text processing, prompt chaining |
| **Vector DB** | FAISS | Semantic search & embedding store |
| **UI** | Streamlit | Interactive frontend |
| **Parser** | PyMuPDF | PDF text extraction |
| **Language** | Python 3.10+ | Runtime environment |

##  Project Structure

```
AI-Resume-Analyzer-Agentic-RAG/
├── app.py                 # Main Streamlit application
├── utils/
│   ├── __init__.py
│   ├── resume_parser.py   # PDF text extraction (PyMuPDF)
│   ├── evaluator.py       # GPT-4/Ollama evaluation logic
│   └── rag_engine.py      # FAISS vector store (LangChain)
├── data/                  # Sample resumes (optional)
├── output/                # Saved evaluation reports (JSON)
├── requirements.txt       # Python dependencies
├── .env                   # API keys (not in git)
├── .gitignore            # Git ignore rules
└── README.md             # This file
```

##  Configuration

### Using Local AI (Recommended - FREE)

The app defaults to local Ollama. Just ensure:
- Ollama is installed: https://ollama.ai
- Model is pulled: `ollama pull llama3`
- Ollama service is running (starts automatically)

### Using OpenAI (Optional - Costs Money)

1. Create `.env` file:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```
2. Uncheck "Use Local LLM" in the sidebar
3. Select OpenAI model

##  References

- **Base ingestion logic** → https://github.com/imartinez/privateGPT
- **UI inspiration** → https://github.com/yashk2810/Resume-Analyzer-using-LangChain

##  Troubleshooting

**Model not found?**
```bash
ollama pull llama3
ollama list  # Verify installation
```

**Ollama connection error?**
```bash
ollama serve  # Start Ollama service
```

**Slow first run?**
- First run loads model into memory (30-90 seconds)
- Subsequent runs are faster (model stays in memory)

**PDF extraction failed?**
- Ensure PDF is text-based (not scanned image)
- Try a different PDF file


