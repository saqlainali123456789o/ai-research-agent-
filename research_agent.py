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

    CrewAI provider:
        openai

    Actual Groq model:
        openai/gpt-oss-120b

    Therefore CrewAI receives:
        openai/openai/gpt-oss-120b

    CrewAI removes the first `openai/` provider prefix,
    while Groq receives:
        openai/gpt-oss-120b
    """

    api_key = get_groq_api_key()
    groq_model = get_groq_model()

    # --------------------------------------------------------
    # API KEY VALIDATION
    # --------------------------------------------------------

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing. "
            "Please add GROQ_API_KEY to Streamlit Secrets."
        )

    # --------------------------------------------------------
    # MODEL VALIDATION
    # --------------------------------------------------------

    if not groq_model:
        raise ValueError(
            "GROQ_MODEL is missing."
        )

    groq_model = str(groq_model).strip()

    # --------------------------------------------------------
    # NORMALIZE MODEL NAME
    # --------------------------------------------------------

    if groq_model.startswith("groq/"):
        groq_model = groq_model[len("groq/"):]

    if groq_model == "gpt-oss-120b":
        groq_model = "openai/gpt-oss-120b"

    elif not groq_model.startswith("openai/"):
        groq_model = f"openai/{groq_model}"

    # --------------------------------------------------------
    # CREWAI MODEL
    # --------------------------------------------------------

    crewai_model = f"openai/{groq_model}"

    # --------------------------------------------------------
    # DEBUG INFORMATION
    # --------------------------------------------------------

    print("=" * 70)
    print("CREWAI + GROQ CONFIGURATION")
    print("=" * 70)
    print("Configured Groq model:", groq_model)
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

    # --------------------------------------------------------
    # CREATE LLM
    # --------------------------------------------------------

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
    """
    Create the main research analyst agent.
    """

    return Agent(
        role="Senior Research Analyst",

        goal=(
            "Find, evaluate and synthesize reliable web evidence "
            "without fabricating facts."
        ),

        backstory=(
            "You are a rigorous research analyst. "
            "You prioritize original, authoritative, academic "
            "and reputable evidence. You compare sources carefully, "
            "identify uncertainty, and clearly communicate "
            "conflicting findings."
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
# MAIN RESEARCH FUNCTION
# ============================================================

def run_research(
    topic,
    depth,
    report_type,
    source_count,
    recency,
):
    """
    Run the complete research workflow.
    """

    # ========================================================
    # STEP 1 — VALIDATE TOPIC
    # ========================================================

    if not topic or not str(topic).strip():
        raise ValueError(
            "Research topic cannot be empty."
        )

    topic = str(topic).strip()

    # ========================================================
    # STEP 2 — VALIDATE SOURCE COUNT
    # ========================================================

    try:
        source_count = max(
            int(source_count),
            1
        )

    except (TypeError, ValueError):
        source_count = 5

    # ========================================================
    # STEP 3 — DISCOVER SOURCES
    # ========================================================

    raw = perform_searches(
        topic,
        depth,
        source_count,
    )

    if not raw:
        raise RuntimeError(
            "No web sources were discovered for this topic."
        )

    # ========================================================
    # STEP 4 — RANK SOURCES
    # ========================================================

    ranked = rank(raw)

    if not ranked:
        raise RuntimeError(
            "Sources were discovered, but none could be ranked."
        )

    selected = ranked[
        :max(source_count, 5)
    ]

    # ========================================================
    # STEP 5 — READ SOURCE CONTENT
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
                f"Unable to read source {url}: {exc}"
            )
            continue

        if not text:
            continue

        if text.startswith(
            "Unable to read webpage:"
        ):
            continue

        source["evidence"] = text

        usable.append(source)

    # ========================================================
    # STEP 6 — CREATE AGENT
    # ========================================================

    agent = build_agent()

    # ========================================================
    # STEP 7 — CREATE TASK
    # ========================================================

    task = Task(
        description=f"""
Research this topic carefully:

{topic}

Research depth:
{depth}

Preferred recency:
{recency}

Target number of sources:
{source_count}

Your job is to produce a reliable research evidence brief.

REQUIREMENTS:

1. Identify the most important facts and findings.

2. Compare information across multiple sources.

3. Prefer authoritative, primary, academic and reputable
   sources whenever possible.

4. Clearly distinguish established facts from uncertain
   or incomplete findings.

5. Mention conflicting evidence when credible sources disagree.

6. Do not invent facts.

7. Do not invent statistics.

8. Do not invent quotations.

9. Do not fabricate sources.

10. Do not fabricate URLs.

11. Do not claim that you visited a source unless evidence
    supports that claim.

12. Include source URLs whenever possible.

13. Keep the evidence brief structured and useful.

14. Use clear headings.

15. Make the final research useful for a human reader.
""",

        expected_output=(
            "A structured research evidence brief containing "
            "key findings, supporting evidence, uncertainty, "
            "conflicting findings where relevant, and source URLs."
        ),

        agent=agent,
    )

    # ========================================================
    # STEP 8 — CREATE CREW
    # ========================================================

    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
    )

    # ========================================================
    # STEP 9 — RUN CREWAI
    # ========================================================

    agent_result = str(
        crew.kickoff()
    )

    # ========================================================
    # STEP 10 — BUILD EVIDENCE PACKAGE
    # ========================================================

    evidence_blocks = []

    for i, source in enumerate(
        usable,
        1,
    ):

        title = source.get(
            "title",
            "",
        )

        url = source.get(
            "url",
            "",
        )

        source_type = source.get(
            "source_type",
            "",
        )

        quality_tier = source.get(
            "quality_tier",
            "",
        )

        evidence = source.get(
            "evidence",
            "",
        )

        evidence_blocks.append(
            f"""
SOURCE [{i}]

Title:
{title}

URL:
{url}

Type:
{source_type}

Quality Tier:
{quality_tier}

Evidence:
{evidence[:10000]}
""".strip()
        )

    # ========================================================
    # STEP 11 — COMBINE EVIDENCE
    # ========================================================

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
    # STEP 12 — GENERATE FINAL REPORT
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
            "Unknown",
        )

        tiers[tier] = (
            tiers.get(tier, 0) + 1
        )

    if tiers:
        quality_summary = ", ".join(
            f"{key}: {value}"
            for key, value in tiers.items()
        )
    else:
        quality_summary = "—"

    # ========================================================
    # STEP 14 — RETURN RESULT
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
