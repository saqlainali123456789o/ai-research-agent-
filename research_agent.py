from crewai import Agent, Crew, Process, Task
from crewai.llm import LLM

from config import get_groq_api_key, get_groq_model
from research_tools import perform_searches, read_webpage, web_search
from source_quality import rank
from report_generator import generate_report


# ============================================================
# LLM CONFIGURATION
# ============================================================

def build_llm():
    """
    Configure CrewAI to use Groq through its OpenAI-compatible API.

    Important:
    CrewAI receives the provider as "openai".
    Groq is used through the OpenAI-compatible base URL.
    """

    api_key = get_groq_api_key()
    model_name = get_groq_model()

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing. "
            "Add GROQ_API_KEY to Streamlit Secrets."
        )

    if not model_name:
        raise ValueError(
            "GROQ_MODEL is missing. "
            "Set GROQ_MODEL to gpt-oss-120b in Streamlit Secrets."
        )

    # --------------------------------------------------------
    # Clean accidental provider prefixes.
    #
    # Examples:
    #   gpt-oss-120b
    #   openai/gpt-oss-120b
    #   groq/gpt-oss-120b
    #
    # All become:
    #   gpt-oss-120b
    # --------------------------------------------------------

    model_name = model_name.strip()

    if model_name.startswith("openai/"):
        model_name = model_name[len("openai/"):]

    if model_name.startswith("groq/"):
        model_name = model_name[len("groq/"):]

    # CrewAI should see:
    # openai/gpt-oss-120b
    #
    # NOT:
    # groq/gpt-oss-120b
    #
    # because we are using Groq's OpenAI-compatible endpoint.

    return LLM(
        model=f"openai/{model_name}",
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
            "You are a rigorous research analyst. You prioritize "
            "original, authoritative and academic evidence. "
            "You compare sources carefully, identify uncertainty, "
            "and clearly communicate conflicting findings."
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
    """
    Run the complete research workflow.

    Steps:
    1. Search for sources.
    2. Rank sources by quality.
    3. Read the strongest sources.
    4. Ask CrewAI to synthesize the evidence.
    5. Generate the final report.
    6. Return report + sources + metadata.
    """

    # --------------------------------------------------------
    # Validate basic inputs
    # --------------------------------------------------------

    if not topic or not str(topic).strip():
        raise ValueError("Research topic cannot be empty.")

    topic = str(topic).strip()

    try:
        source_count = max(int(source_count), 1)
    except (TypeError, ValueError):
        source_count = 5

    # --------------------------------------------------------
    # STEP 1 — Search
    # --------------------------------------------------------

    raw = perform_searches(
        topic,
        depth,
        source_count,
    )

    if not raw:
        raise RuntimeError(
            "No web sources were discovered for this research topic."
        )

    # --------------------------------------------------------
    # STEP 2 — Rank sources
    # --------------------------------------------------------

    ranked = rank(raw)

    if not ranked:
        raise RuntimeError(
            "Sources were discovered, but none could be ranked."
        )

    # --------------------------------------------------------
    # STEP 3 — Select strongest sources
    # --------------------------------------------------------

    selected = ranked[:max(source_count, 5)]

    usable = []

    for source in selected:

        url = source.get("url")

        if not url:
            continue

        try:
            text = read_webpage.run(url)
        except Exception:
            continue

        if not text:
            continue

        if text.startswith("Unable to read webpage:"):
            continue

        source["evidence"] = text

        usable.append(source)

    # --------------------------------------------------------
    # STEP 4 — Build CrewAI agent
    # --------------------------------------------------------

    agent = build_agent()

    # --------------------------------------------------------
    # STEP 5 — Create research task
    # --------------------------------------------------------

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

Requirements:

1. Identify the most important facts and findings.
2. Compare information across sources.
3. Prefer authoritative, primary, academic and reputable sources.
4. Clearly distinguish established facts from uncertain findings.
5. Mention conflicting evidence when sources disagree.
6. Do not invent facts, statistics, quotations or sources.
7. Do not claim that you visited a source unless evidence supports it.
8. Include source URLs whenever possible.
9. Keep the final evidence brief concise but useful.
""",

        expected_output=(
            "A structured research evidence brief containing "
            "key findings, supporting evidence, uncertainty, "
            "conflicting findings where relevant, and source URLs."
        ),

        agent=agent,
    )

    # --------------------------------------------------------
    # STEP 6 — Run CrewAI
    # --------------------------------------------------------

    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
    )

    agent_result = str(crew.kickoff())

    # --------------------------------------------------------
    # STEP 7 — Prepare evidence package
    # --------------------------------------------------------

    evidence_blocks = []

    for i, source in enumerate(usable, 1):

        title = source.get("title", "")
        url = source.get("url", "")
        source_type = source.get("source_type", "")
        quality_tier = source.get("quality_tier", "")
        evidence = source.get("evidence", "")

        evidence_blocks.append(
            f"""
SOURCE [{i}]
Title: {title}
URL: {url}
Type: {source_type}
Quality Tier: {quality_tier}

Evidence:
{evidence[:10000]}
""".strip()
        )

    package = "\n\n".join(evidence_blocks)

    if not package:
        package = "No directly readable source evidence was available."

    package += (
        "\n\n"
        "CREWAI RESEARCH BRIEF\n"
        "======================\n"
        f"{agent_result}"
    )

    # --------------------------------------------------------
    # STEP 8 — Generate final report
    # --------------------------------------------------------

    report = generate_report(
        topic,
        package,
        report_type,
        depth,
    )

    # --------------------------------------------------------
    # STEP 9 — Calculate source quality summary
    # --------------------------------------------------------

    tiers = {}

    for source in usable:

        tier = source.get(
            "quality_tier",
            "Unknown",
        )

        tiers[tier] = tiers.get(tier, 0) + 1

    quality_summary = (
        ", ".join(
            f"{key}: {value}"
            for key, value in tiers.items()
        )
        if tiers
        else "—"
    )

    # --------------------------------------------------------
    # STEP 10 — Return final result
    # --------------------------------------------------------

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
