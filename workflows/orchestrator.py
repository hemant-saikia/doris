from langgraph.graph import StateGraph, END
from typing import TypedDict

from agents.job_analyzer import JobAnalyzerAgent, JobAnalysis
from agents.resume_optimizer import ResumeOptimizerAgent
from agents.cover_letter import CoverLetterAgent
from llm import LLMProvider, LLMConfig, create_llm


class ApplicationState(TypedDict):
    job_posting: str
    base_resume: str
    job_analysis: JobAnalysis
    optimized_resume: str
    cover_letter: str
    interview_prep: dict
    application_id: str


def create_workflow(
    api_key: str,
    provider: str = "claude",
    model: str = "",
) -> StateGraph:
    config = LLMConfig(api_key=api_key, model=model)
    llm = create_llm(provider, config)

    job_analyzer = JobAnalyzerAgent(llm)
    resume_optimizer = ResumeOptimizerAgent(llm)
    cover_letter_agent = CoverLetterAgent(llm)

    def analyze_job(state: ApplicationState):
        analysis = job_analyzer.analyze(state["job_posting"])
        return {"job_analysis": analysis}

    def optimize_resume(state: ApplicationState):
        optimized = resume_optimizer.optimize(
            state["base_resume"],
            state["job_analysis"],
        )
        return {"optimized_resume": optimized}

    def write_cover_letter(state: ApplicationState):
        letter = cover_letter_agent.generate(
            state["job_analysis"],
            state["optimized_resume"],
        )
        return {"cover_letter": letter}

    workflow = StateGraph(ApplicationState)

    workflow.add_node("analyze", analyze_job)
    workflow.add_node("optimize", optimize_resume)
    workflow.add_node("cover_letter", write_cover_letter)

    workflow.set_entry_point("analyze")
    workflow.add_edge("analyze", "optimize")
    workflow.add_edge("optimize", "cover_letter")
    workflow.add_edge("cover_letter", END)

    return workflow.compile()
