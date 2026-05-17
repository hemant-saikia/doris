from llm import LLMProvider
from agents.job_analyzer import JobAnalysis


_SYSTEM_PROMPT = """You are an expert cover letter writer. Write compelling, personalized cover letters.

Structure:
- Opening: Why this specific role excites you
- Body: How your experience matches their needs (2-3 specific examples)
- Closing: Value you'll bring, enthusiasm for next steps

Tone: {{tone}} but authentic
Length: 250-350 words

Avoid:
- Generic phrases ("I am writing to apply...")
- Repetition of resume
- Desperation or over-eagerness
- Clichés"""


class CoverLetterAgent:
    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    def generate(
        self,
        job_analysis: JobAnalysis,
        resume: str,
        tone: str = "professional",
    ) -> str:
        system_prompt = _SYSTEM_PROMPT.replace("{{tone}}", tone)

        user_prompt = f"""Job: {job_analysis.role} at {job_analysis.company}

Company Values: {', '.join(job_analysis.company_values)}
Required Skills: {', '.join(job_analysis.required_skills[:5])}
Key Responsibilities: {', '.join(job_analysis.key_responsibilities[:3])}

My Background (from resume):
{resume}

Write a cover letter that shows genuine interest and clear fit."""

        return self._llm.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
