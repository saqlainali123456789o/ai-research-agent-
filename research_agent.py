from crewai import Agent, Crew, Process, Task
from crewai.llm import LLM

from config import get_groq_api_key, get_groq_model
from research_tools import (
    perform_searches,
    read_webpage,
    web_search,
)
from source_quality import rank
from report_generator import generate_report


# ============================================================
# LLM CONFIGURATION
# ============================================================

def build_llm():
    """
    Build CrewAI LLM using Groq's OpenAI-compatible API.

    Groq model:
        openai/gpt-oss-20b

    CrewAI provider:
        openai

    Groq endpoint:
        https://api.groq.com/openai/v1
    """

    api_key = get_groq_api_key()
    model_name = get_groq_model()

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing. "
            "Add it to Streamlit Secrets."
        )

    if not model_name:
        raise ValueError(
            "GROQ_MODEL is missing."
        )

    model_name = str(model_name).strip()

    # --------------------------------------------------------
    # Normalize provider prefixes
    # --------------------------------------------------------

    if model_name.startswith("groq/"):
        model_name = model_name[len("groq/"):]

    if model_name == "gpt-oss-20b":
        model_name = "openai/gpt-oss-20b"

    elif model_name == "gpt-oss-120b":
        model_name = "openai/gpt-oss-120b"

    elif not model_name.startswith("openai/"):
        model_name = f"openai/{model_name}"

    # --------------------------------------------------------
    # CrewAI provider prefix
    #
    # CrewAI sees:
    #
    # openai/openai/gpt-oss-20b
    #
    # First "openai/" = provider
    # Second "openai/" = Groq model namespace
    # --------------------------------------------------------

    crewai_model = f"openai/{model_name}"

    print("=" * 70)
    print("CREWAI + GROQ CONFIGURATION")
    print("=" * 70)
    print("Groq model:", model_name)
    print("CrewAI model:", crewai_model)
    print(
        "Base URL:",
        "https://api.groq.com/openai/v1"
    )
    print(
        "API key loaded:",
        bool(api_key)
    )
    print("=" * 70)

    return LLM(
        model=crewai_model,
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
        temperature=0.1,
    )


# ============================================================
# RESEARCH AGENT
# ============================================================

def build_agent():

    return Agent(
        role="Senior Research Analyst",

        goal=(
            "Find, evaluate and synthesize reliable web evidence "
            "without fabricating facts."
        ),

        backstory=(
            "You are a rigorous research analyst. "
            "You prioritize authoritative, academic, primary "
            "and reputable sources. You compare evidence, "
            "identify uncertainty and clearly report conflicts."
        ),

        llm=build_llm(),

        tools=[
            web_search,
            read_webpage,
        ],

        allow_delegation=False,

        verbose=False,
    )


# ============================================================
# MAIN RESEARCH WORKFLOW
# ============================================================

def run_research(
    topic,
    depth,
    report_type,
    source_count,
    recency,
):

    # ========================================================
    # STEP 1 — VALIDATE INPUT
    # ========================================================

    if not topic or not str(topic).strip():
        raise ValueError(
            "Research topic cannot be empty."
        )

    topic = str(topic).strip()

    # ========================================================
    # STEP 2 — SOURCE COUNT
    # ========================================================

    try:
        source_count = max(
            int(source_count),
            1
        )
    except (TypeError, ValueError):
        source_count = 5

    # ========================================================
    # IMPORTANT TOKEN CONTROL
    #
    # Do not retrieve 15 large webpages unnecessarily.
    # ========================================================

    max_sources = min(
        source_count,
        8
    )

    # ========================================================
    # STEP 3 — SEARCH
    # ========================================================

    raw = perform_searches(
        topic,
        depth,
        max_sources,
    )

    if not raw:
        raise RuntimeError(
            "No web sources were discovered for this topic."
        )

    # ========================================================
    # STEP 4 — RANK
    # ========================================================

    ranked = rank(raw)

    if not ranked:
        raise RuntimeError(
            "Sources were discovered, but none could be ranked."
        )

    # ========================================================
    # STEP 5 — SELECT SOURCES
    # ========================================================

    selected = ranked[:max_sources]

    # ========================================================
    # STEP 6 — READ SOURCES
    # ========================================================

    usable = []

    for source in selected:

        url = source.get("url")

        if not url:
            continue

        try:
            text = read_webpage.run(url)
        except Exception as exc:
            print(
                f"Unable to read source: {url}"
            )
            print(exc)
            continue

        if not text:
            continue

        if text.startswith(
            "Unable to read webpage:"
        ):
            continue

        # Limit evidence before sending it to LLM
        source["evidence"] = text[:7000]

        usable.append(source)

    # ========================================================
    # STEP 7 — BUILD AGENT
    # ========================================================

    agent = build_agent()

    # ========================================================
    # STEP 8 — BUILD TASK
    # ========================================================

    task = Task(

        description=f"""
Research the following topic:

{topic}

Research depth:
{depth}

Preferred recency:
{recency}

Number of sources:
{len(usable)}


Your task is to produce a concise evidence brief.

Requirements:

1. Identify the most important findings.

2. Compare evidence across sources.

3. Prefer authoritative and primary sources.

4. Distinguish facts from uncertainty.

5. Mention credible disagreements.

6. Never invent facts.

7. Never invent statistics.

8. Never invent quotations.

9. Never invent URLs.

10. Do not claim that a source was accessed unless
    the source evidence supports that claim.

11. Include source URLs when available.

12. Keep the response concise and evidence-focused.
""",

        expected_output=(
            "A concise structured research evidence brief "
            "with key findings, evidence, uncertainty, "
            "source comparisons and URLs."
        ),

        agent=agent,
    )

    # ========================================================
    # STEP 9 — CREW
    # ========================================================

    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
    )

    # ========================================================
    # STEP 10 — RUN
    # ========================================================

    try:

        agent_result = str(
            crew.kickoff()
        )

    except Exception as exc:

        error_text = str(exc)

        if (
            "RateLimitError" in type(exc).__name__
            or "rate_limit" in error_text.lower()
            or "429" in error_text
        ):
            raise RuntimeError(
                "Groq rate limit reached. "
                "Please wait for the limit to reset "
                "or use a model with available quota."
            ) from exc

        raise

    # ========================================================
    # STEP 11 — EVIDENCE PACKAGE
    # ========================================================

    evidence_blocks = []

    for i, source in enumerate(
        usable,
        1
    ):

        evidence_blocks.append(
            f"""
SOURCE [{i}]

Title:
{source.get("title", "")}

URL:
{source.get("url", "")}

Type:
{source.get("source_type", "")}

Quality Tier:
{source.get("quality_tier", "")}

Evidence:
{source.get("evidence", "")[:7000]}
""".strip()
        )

    package = "\n\n".join(
        evidence_blocks
    )

    if not package:
        package = (
            "No directly readable source evidence "
            "was available."
        )

    package += (
        "\n\n"
        "CREWAI RESEARCH BRIEF\n"
        "======================\n"
        f"{agent_result}"
    )

    # ========================================================
    # STEP 12 — FINAL REPORT
    # ========================================================

    report = generate_report(
        topic,
        package,
        report_type,
        depth,
    )

    # ========================================================
    # STEP 13 — QUALITY SUMMARY
    # ========================================================

    tiers = {}

    for source in usable:

        tier = source.get(
            "quality_tier",
            "Unknown"
        )

        tiers[tier] = (
            tiers.get(tier, 0) + 1
        )

    quality_summary = (
        ", ".join(
            f"{key}: {value}"
            for key, value in tiers.items()
        )
        if tiers
        else "—"
    )

    # ========================================================
    # STEP 14 — RETURN
    # ========================================================

    return {
        "report": report,

        "sources": usable,

        "metadata": {
            "results_discovered": len(raw),
            "sources_used": len(usable),
            "quality_summary": quality_summary,
            "research_depth": depth,
        },
    }
