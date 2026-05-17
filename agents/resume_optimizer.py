from llm import LLMProvider
from agents.job_analyzer import JobAnalysis


_SYSTEM_PROMPT = """You are an expert resume optimizer specializing in ATS systems.
PRIORITIES:
1. DONOT FABRICATE information
2. DONOT lie
3. ONly try to structure existing resume data to fit job description only if applicable. If you find no way to spin existing experience to relevant experience, that's fine. DONOT LIE
4. DONOT ADD extra text to your reply. The entire text should be the content of the resume. No extra notes or information. 

Your task:
1. Reorganize skills section to prioritize matching skills
2. Emphasize relevant experience and projects
3. Quantify achievements where possible
4. Integrate ATS keywords naturally (NO keyword stuffing)
5. Keep format clean and ATS-friendly
6. Maintain truthfulness - only highlight, don't fabricate
7. Donot explicitly add the ATS keywords, just incorporate them wherever possible naturally. 

Return the complete optimized resume in markdown format.

"""


class ResumeOptimizerAgent:
    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    def optimize(self, base_resume: str, job_analysis: JobAnalysis) -> str:
        user_prompt = f"""Base Resume: {base_resume}

Job Requirements:
- Role: {job_analysis.role}
- Required Skills: {', '.join(job_analysis.required_skills)}
- Preferred Skills: {', '.join(job_analysis.preferred_skills)}
- Key Responsibilities: {', '.join(job_analysis.key_responsibilities)}
- ATS Keywords: {', '.join(job_analysis.ats_keywords)}

Create an optimized version that highlights my relevant experience."""

        return self._llm.generate(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )
