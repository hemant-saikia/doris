from pydantic import BaseModel
from typing import List

from llm import LLMProvider


class JobAnalysis(BaseModel):
    role: str
    company: str
    required_skills: List[str]
    preferred_skills: List[str]
    experience_years: str
    key_responsibilities: List[str]
    company_values: List[str]
    ats_keywords: List[str]


_SYSTEM_PROMPT = """You are an expert job posting analyzer. Extract structured information from job postings.

Return ONLY valid JSON matching this schema:
{{
    "role": "job title",
    "company": "company name",
    "required_skills": ["skill1", "skill2"],
    "preferred_skills": ["skill1", "skill2"],
    "experience_years": "X-Y years or 'Not specified'",
    "key_responsibilities": ["resp1", "resp2"],
    "company_values": ["value1", "value2"],
    "ats_keywords": ["keyword1", "keyword2"]
}}

Be thorough. Extract all skills, even if implied."""


class JobAnalyzerAgent:
    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    def analyze(self, job_posting: str) -> JobAnalysis:
        import json

        response = self._llm.generate(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=f"Analyze this job posting:\n\n{job_posting}",
        )

        analysis_data = json.loads(response)
        return JobAnalysis(**analysis_data)
