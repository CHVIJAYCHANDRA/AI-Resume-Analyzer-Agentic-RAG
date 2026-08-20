from typing import List

from pydantic import BaseModel, Field

# Context budgets. Local models get less room than hosted ones.
JD_CHAR_LIMIT = 3000
RESUME_LIMIT_LOCAL = 4000
RESUME_LIMIT_CLOUD = 6000

# Retrieval depth used by the agent's retrieve node.
RETRIEVAL_K = 5


class Evaluation(BaseModel):
    """Validated evaluation result.

    `evidence` is the reliability lever: every entry must be a verbatim resume
    quote, which lets the verify node measure grounding instead of trusting
    the model's self-report.
    """

    fit_score: int = Field(ge=0, le=100, description="0-100 match score")
    matching_skills: List[str] = Field(default_factory=list)
    missing_skills: List[str] = Field(default_factory=list)
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    improvements: List[str] = Field(default_factory=list)
    evidence: List[str] = Field(
        default_factory=list,
        description="Verbatim resume quotes supporting the score",
    )


PROMPT_TEMPLATE = """You are an expert technical recruiter.

Score the candidate against the job description. Ground your judgement in the
RETRIEVED RESUME EXCERPTS first - those are the passages most similar to this
job description. Use the FULL RESUME only for context the excerpts miss.

Rules:
- Every string in "evidence" must be copied verbatim from the resume.
- If a skill is not present in the resume, it belongs in "missing_skills",
  never in "matching_skills".
- Do not invent employers, dates, degrees or numbers.
- "improvements" must be concrete and actionable, 5-7 items.

Return ONLY a valid JSON object, with no prose and no markdown fences:
{{"fit_score": <int 0-100>,
  "matching_skills": ["..."],
  "missing_skills": ["..."],
  "strengths": ["..."],
  "weaknesses": ["..."],
  "improvements": ["..."],
  "evidence": ["verbatim resume quote", "..."]}}

JOB DESCRIPTION:
{job}

RETRIEVED RESUME EXCERPTS (top-{k} by similarity to the job description):
{context}

FULL RESUME:
{resume}

JSON:"""

REPAIR_SUFFIX = """

Your previous response failed validation: it was either not valid JSON, was
missing required fields, or contained evidence quotes that do not appear in the
resume. Return ONLY the JSON object, and copy evidence quotes exactly from the
resume text."""
