try:
    from langchain_core.prompts import PromptTemplate
except ImportError:
    from langchain.prompts import PromptTemplate

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    from langchain.chat_models import ChatOpenAI
import json
import os
from datetime import datetime


def evaluate_resume(job_desc, resume_text, openai_key=None, model_name="gpt-3.5-turbo", use_local=False, local_model_url=None):
    """
    Evaluate resume against job description using OpenAI or local LLM.
    
    Args:
        job_desc: Job description text
        resume_text: Resume text content
        openai_key: OpenAI API key (required if use_local=False)
        model_name: Model to use (default: gpt-3.5-turbo, options: gpt-3.5-turbo, gpt-4, gpt-4-turbo)
        use_local: Whether to use local LLM via Ollama
        local_model_url: URL for local Ollama API (default: http://localhost:11434)
        
    Returns:
        str: Formatted evaluation report
    """
    try:
        if use_local:
            # Use local Ollama LLM - Try multiple methods
            try:
                # Method 1: Try direct ollama package (most reliable)
                import ollama
                
                # Create prompt template first
                template = PromptTemplate(
                    input_variables=["job", "resume"],
                    template="""
You are an expert HR AI analyst with years of experience in recruitment and talent assessment.

Compare the following resume with the job description and provide a comprehensive evaluation:

1. **Overall Fit Score**: A numerical score from 0-100 indicating how well the candidate matches the job requirements.

2. **Key Matching Skills**: List the top skills from the resume that directly match the job description requirements.

3. **Missing Skills**: Identify critical skills mentioned in the job description that are not evident in the resume.

4. **Suggested Improvements**: Provide 5-7 actionable bullet points on how the candidate can improve their resume to better match this job.

5. **Strengths**: Highlight 3-5 key strengths of the candidate relevant to this role.

6. **Weaknesses**: Identify 2-3 areas where the candidate may fall short.

Format your response clearly with headers and bullet points for easy reading.

JOB DESCRIPTION:
{job}

RESUME:
{resume}

Provide your evaluation:
"""
                )
                
                resume_limit = 4000 if use_local else 6000
                prompt = template.format(
                    job=job_desc[:3000],
                    resume=resume_text[:resume_limit]
                )
                
                try:
                    response = ollama.generate(
                        model=model_name,
                        prompt=prompt,
                        options={
                            'temperature': 0.3,
                            'num_predict': 1500,
                            'num_ctx': 4096
                        }
                    )
                    
                    result = response.get('response', '')
                    if not result:
                        return "## Error: Empty response from Ollama\n\nTry again or use a different model."
                    
                    return result
                    
                except Exception as e:
                    error_msg = str(e)
                    if "model" in error_msg.lower() and "not found" in error_msg.lower():
                        raise Exception(f"Model '{model_name}' not found. Run: ollama pull {model_name}")
                    elif "connection" in error_msg.lower() or "refused" in error_msg.lower():
                        raise Exception("Cannot connect to Ollama. Make sure Ollama is running: `ollama serve`")
                    else:
                        raise Exception(f"Ollama error: {error_msg}")
                
            except ImportError:
                try:
                    try:
                        from langchain_community.llms import Ollama
                    except ImportError:
                        try:
                            from langchain_ollama import OllamaLLM as Ollama
                        except ImportError:
                            from langchain.llms import Ollama
                    
                    local_url = local_model_url or "http://localhost:11434"
                    llm = Ollama(
                        model=model_name,
                        base_url=local_url,
                        temperature=0.3
                    )
                    
                    # Create prompt and invoke
                    template = PromptTemplate(
                        input_variables=["job", "resume"],
                        template="""
You are an expert HR AI analyst with years of experience in recruitment and talent assessment.

Compare the following resume with the job description and provide a comprehensive evaluation:

1. **Overall Fit Score**: A numerical score from 0-100 indicating how well the candidate matches the job requirements.

2. **Key Matching Skills**: List the top skills from the resume that directly match the job description requirements.

3. **Missing Skills**: Identify critical skills mentioned in the job description that are not evident in the resume.

4. **Suggested Improvements**: Provide 5-7 actionable bullet points on how the candidate can improve their resume to better match this job.

5. **Strengths**: Highlight 3-5 key strengths of the candidate relevant to this role.

6. **Weaknesses**: Identify 2-3 areas where the candidate may fall short.

Format your response clearly with headers and bullet points for easy reading.

JOB DESCRIPTION:
{job}

RESUME:
{resume}

Provide your evaluation:
"""
                    )
                    
                    prompt = template.format(
                        job=job_desc,
                        resume=resume_text[:6000]
                    )
                    
                    response = llm.invoke(prompt)
                    if hasattr(response, 'content'):
                        return response.content
                    return str(response)
                    
                except Exception as e:
                    return f"## Error: Ollama not available\n\nPlease install:\n1. Ollama: https://ollama.ai\n2. Python package: pip install ollama\n3. Pull model: ollama pull {model_name}\n\nError: {str(e)}"
            except Exception as e:
                error_msg = str(e)
                if "model" in error_msg.lower() and "not found" in error_msg.lower():
                    return f"## Error: Model '{model_name}' not found\n\n**Solution:**\n1. Pull the model: `ollama pull {model_name}`\n2. Verify: `ollama list`\n3. Make sure Ollama is running: `ollama serve`"
                elif "connection" in error_msg.lower() or "refused" in error_msg.lower():
                    return f"## Error: Cannot connect to Ollama\n\n**Solution:**\n1. Make sure Ollama is running: `ollama serve`\n2. Or start Ollama service in background\n3. Check if port 11434 is available\n\n**Error:** {str(e)}"
                else:
                    return f"## Error connecting to local LLM\n\n**Steps to fix:**\n1. Make sure Ollama is installed: https://ollama.ai\n2. Start Ollama: `ollama serve`\n3. Pull model: `ollama pull {model_name}`\n4. Verify: `ollama list`\n\n**Error:** {str(e)}"
        else:
            # Use OpenAI
            if not openai_key:
                return "## Error: OpenAI API Key Required\n\nPlease provide an OpenAI API key in the .env file."
            
            # Try gpt-3.5-turbo first (most accessible), then try the requested model
            models_to_try = [model_name] if model_name != "gpt-4" else ["gpt-3.5-turbo", "gpt-4-turbo-preview", "gpt-4"]
            
            llm = None
            last_error = None
            
            for model in models_to_try:
                try:
                    # Try new API first, fallback to old API
                    try:
                        llm = ChatOpenAI(
                            model=model,
                            temperature=0.3,
                            api_key=openai_key
                        )
                        # Test if model works by checking attributes
                        break
                    except TypeError:
                        # Fallback for older LangChain versions
                        llm = ChatOpenAI(
                            model_name=model,
                            temperature=0.3,
                            openai_api_key=openai_key
                        )
                        break
                except Exception as e:
                    last_error = str(e)
                    continue
            
            if llm is None:
                return f"## Error: Model not available\n\nCould not access requested model. Tried: {', '.join(models_to_try)}\n\nError: {last_error}\n\n**Suggestion:** Try using 'gpt-3.5-turbo' which is more widely available."
        
        # Create prompt template
        template = PromptTemplate(
            input_variables=["job", "resume"],
            template="""
You are an expert HR AI analyst with years of experience in recruitment and talent assessment.

Compare the following resume with the job description and provide a comprehensive evaluation:

1. **Overall Fit Score**: A numerical score from 0-100 indicating how well the candidate matches the job requirements.

2. **Key Matching Skills**: List the top skills from the resume that directly match the job description requirements.

3. **Missing Skills**: Identify critical skills mentioned in the job description that are not evident in the resume.

4. **Suggested Improvements**: Provide 5-7 actionable bullet points on how the candidate can improve their resume to better match this job.

5. **Strengths**: Highlight 3-5 key strengths of the candidate relevant to this role.

6. **Weaknesses**: Identify 2-3 areas where the candidate may fall short.

Format your response clearly with headers and bullet points for easy reading.

JOB DESCRIPTION:
{job}

RESUME:
{resume}

Provide your evaluation:
"""
        )
        
        prompt = template.format(
            job=job_desc,
            resume=resume_text[:6000]
        )
        
        response = llm.invoke(prompt)
        
        if hasattr(response, 'content'):
            evaluation = response.content
        else:
            evaluation = str(response)
        
        return evaluation
    
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "insufficient_quota" in error_msg or "quota" in error_msg.lower():
            return (
                "## Error: OpenAI API Quota Exceeded\n\n"
                "Check your billing at: https://platform.openai.com/account/billing\n"
                "Add credits or upgrade your plan."
            )
        elif "401" in error_msg or "invalid_api_key" in error_msg.lower():
            return (
                "## Error: Invalid API Key\n\n"
                "Please check your OpenAI API key in the .env file."
            )
        else:
            return f"## Error during evaluation\n\n{error_msg}"


def save_evaluation_report(evaluation, job_desc, resume_text, output_dir="output"):
    """
    Save evaluation report as JSON with timestamp.
    
    Args:
        evaluation: Evaluation text report
        job_desc: Job description
        resume_text: Resume text
        output_dir: Output directory path
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Prepare report data
        report_data = {
            "timestamp": timestamp,
            "evaluation": evaluation,
            "job_description": job_desc[:500],  # Store first 500 chars
            "resume_preview": resume_text[:500],  # Store first 500 chars
        }
        
        # Save to JSON
        output_path = os.path.join(output_dir, f"evaluation_{timestamp}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        return output_path
    except Exception as e:
        return None

