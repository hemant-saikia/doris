# Doris — AI Job Application Assistant

Doris is a Streamlit app that analyzes job postings, optimizes your resume for ATS systems, and generates tailored cover letters using LLMs.

## Architecture

```
app.py                 # Streamlit UI (entry point)
├── workflows/
│   └── orchestrator.py  # LangGraph pipeline (builds the DAG)
│       ├── agents/
│       │   ├── job_analyzer.py     # Extracts structured data from job posts
│       │   ├── resume_optimizer.py # ATS-optimizes your resume
│       │   └── cover_letter.py     # Generates tailored cover letters
│       └── llm/
│           ├── interface.py        # LLMConfig + LLMProvider ABC
│           ├── claude.py           # Anthropic Claude
│           ├── groq.py            # Groq (OpenAI-compat)
│           ├── openrouter.py      # OpenRouter (OpenAI-compat)
│           └── mistral.py         # Mistral AI
```

The pipeline runs through three stages:
1. **Analyze** — LLM extracts role, skills, keywords, etc. from the job posting
2. **Optimize** — Your base resume is rewritten to highlight matching skills
3. **Cover Letter** — A personalized cover letter is generated

## Setup

```bash
git clone <repo> && cd doris
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file with at least one API key:

```env
CLAUDE_API_KEY=sk-ant-...
GROQ_API_KEY=gsk-...
OPENROUTER_API_KEY=sk-or-v1-...
MISTRAL_API_KEY=...
```

## Usage

```bash
streamlit run app.py
```

Select your provider and model from the sidebar, paste a job posting, upload your resume, and click **Generate**.

## Adding a Provider

Drop a new file in `llm/` that implements `LLMProvider`:

```python
from llm.interface import LLMConfig, LLMProvider

class MyProvider(LLMProvider):
    def __init__(self, config: LLMConfig) -> None:
        super().__init__(config)
        # set up your client

    def generate(self, system_prompt, user_prompt, temperature=None, max_tokens=None) -> str:
        # call the API and return response text
```

Then register it in `llm/__init__.py`:

```python
from llm.myprovider import MyProvider

registry = {
    ...
    "myprovider": MyProvider,
}
```

## Programmatic Use

```python
from llm import LLMConfig, create_llm

config = LLMConfig(api_key="sk-...", model="claude-sonnet-4-20250514")
llm = create_llm("claude", config)

response = llm.generate(
    system_prompt="You are a helpful assistant.",
    user_prompt="Tell me a joke",
)
```

Or run the full pipeline:

```python
from workflows.orchestrator import create_workflow

app = create_workflow(api_key="sk-...", provider="claude")
result = app.invoke({
    "job_posting": "...",
    "base_resume": "...",
})
```
