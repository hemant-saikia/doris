import streamlit as st
from workflows.orchestrator import create_workflow
import os
from dotenv import load_dotenv

load_dotenv()

PROVIDER_MAP = {
    "claude": {
        "label": "Claude (Anthropic)",
        "env_key": "CLAUDE_API_KEY",
        "models": ["claude-sonnet-4-20250514", "claude-sonnet-4-20250514", "claude-3-haiku-20240307"],
    },
    "groq": {
        "label": "Groq",
        "env_key": "GROQ_API_KEY",
        "models": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
    },
    "openrouter": {
        "label": "OpenRouter",
        "env_key": "OPENROUTER_API_KEY",
        "models": [
            "openrouter/free",
            "meta-llama/llama-3.3-70b-instruct:free",
            "qwen/qwen3-coder:free",
            "deepseek/deepseek-v4-flash:free",
            "google/gemma-4-31b-it:free",
            "nvidia/nemotron-3-super-120b-a12b:free",
            "openai/gpt-oss-120b:free",
            "minimax/minimax-m2.5:free",
            "arcee-ai/trinity-large-thinking:free",
            "microsoft/phi-4:free",
            "anthropic/claude-sonnet-4-20250514",
            "openai/gpt-4o",
            "google/gemini-2.0-flash-001",
        ],
    },
    "mistral": {
        "label": "Mistral",
        "env_key": "MISTRAL_API_KEY",
        "models": ["mistral-large-latest", "mistral-medium-latest", "mistral-small-latest"],
    },
}

st.set_page_config(page_title="Job Application Assistant", page_icon="💼")

st.title("💼 Smart Job Application Assistant")

with st.sidebar:
    provider_name = st.selectbox(
        "LLM Provider",
        options=list(PROVIDER_MAP.keys()),
        format_func=lambda x: PROVIDER_MAP[x]["label"],
    )

    provider_info = PROVIDER_MAP[provider_name]

    model = st.selectbox(
        "Model",
        options=provider_info["models"],
        index=0,
    )

tab1, tab2, tab3 = st.tabs(["📝 Create Application", "📊 Track Applications", "⚙️ Settings"])

with tab1:
    st.header("Create New Application Package")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Job Posting")
        job_posting = st.text_area(
            "Paste the job posting here",
            height=300,
            placeholder="Copy-paste the entire job description...",
        )

    with col2:
        st.subheader("Your Base Resume")
        resume_file = st.file_uploader("Upload resume (PDF or DOCX)", type=["pdf", "docx"])

        base_resume = None
        if resume_file:
            if resume_file.type == "application/pdf":
                import PyPDF2
                pdf = PyPDF2.PdfReader(resume_file)
                base_resume = "\n".join([page.extract_text() for page in pdf.pages])
            else:
                from docx import Document
                doc = Document(resume_file)
                base_resume = "\n".join([p.text for p in doc.paragraphs])

            st.success(f"✅ Resume loaded ({len(base_resume)} characters)")

    can_generate = bool(job_posting and base_resume)

    if st.button("🚀 Generate Application Package", type="primary", disabled=not can_generate):
        with st.spinner("🔍 Analyzing job posting..."):
            try:
                api_key = os.getenv(provider_info["env_key"], "")
                if not api_key:
                    st.error(f"❌ {provider_info['env_key']} not set in .env file")
                    st.stop()

                workflow = create_workflow(
                    api_key=api_key,
                    provider=provider_name,
                    model=model,
                )

                result = workflow.invoke({
                    "job_posting": job_posting,
                    "base_resume": base_resume,
                })

                st.success("✅ Application package ready!")

                st.subheader("📋 Job Analysis")
                analysis = result["job_analysis"]
                st.json({
                    "role": analysis.role,
                    "company": analysis.company,
                    "required_skills": analysis.required_skills,
                    "ats_keywords": analysis.ats_keywords,
                })

                st.subheader("📄 Optimized Resume")
                st.markdown(result["optimized_resume"])
                st.download_button(
                    "⬇️ Download Resume",
                    result["optimized_resume"],
                    f"resume_{analysis.company}.md",
                    mime="text/markdown",
                )

                st.subheader("✉️ Cover Letter")
                st.markdown(result["cover_letter"])
                st.download_button(
                    "⬇️ Download Cover Letter",
                    result["cover_letter"],
                    f"cover_letter_{analysis.company}.md",
                    mime="text/markdown",
                )

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("Check your API key in .env and try again")

with tab2:
    st.header("📊 Application Tracker")
    st.info("Coming soon - track all your applications in one place")

with tab3:
    st.header("⚙️ Settings")
    tone = st.selectbox("Cover letter tone", ["professional", "enthusiastic", "confident"])
    st.info("Settings will be saved in future versions")
