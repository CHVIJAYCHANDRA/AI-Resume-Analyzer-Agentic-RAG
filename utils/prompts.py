from typing import List
from pydantic import BaseModel, Field

JD_CHAR_LIMIT = 3000
RESUME_LIMIT_LOCAL = 4000
RESUME_LIMIT_CLOUD = 6000


class Evaluation(BaseModel):
    fit_score: int = Field(ge=0, le=100)
    matching_skills: List[str] = []
    missing_skills: List[str] = []
    strengths: List[str] = []
    weaknesses: List[str] = []
    improvements: List[str] = []
    evidence: List[str] = Field(
        default=[], description="Verbatim resume quotes supporting the score"
    )


PROMPT_TEMPLATE = """You are an expert technical recruiter.

Score the candidate against the job description. Base your judgement on the
RETRIEVED RESUME EXCERPTS first - they are the passages most relevant to this
job. Every claim in "evidence" must be a verbatim quote from the resume.

Return ONLY valid JSON, no prose, no markdown fences:
{{"fit_score": <0-100 int>,
  "matching_skills": [], "missing_skills": [],
  "strengths": [], "weaknesses": [], "improvements": [],
  "evidence": []}}

JOB DESCRIPTION:
{job}

RETRIEVED RESUME EXCERPTS (top-{k} by similarity to the JD):
{context}

FULL RESUME:
{resume}

JSON:"""

REPAIR_SUFFIX = (
    "\n\nYour previous reply was not valid JSON matching the schema. "
    "Return ONLY the JSON object."
)
