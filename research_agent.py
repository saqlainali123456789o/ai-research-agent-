from crewai import Agent, Crew, Process, Task
from crewai.llm import LLM

from config import (
    get_groq_api_key,
    get_groq_model,
)

from research_tools import (
    perform_searches,
    read_webpage,
)

from source_quality import rank


GROQ_BASE_URL = (
    "https://api.groq.com/openai/v1"
)


# ============================================================
# BUILD LLM
# ============================================================

def build_llm():

    api_key = get_groq_api_key()

    model = get_groq_model()

    if not api_key:

        raise ValueError(
            "GROQ_API_KEY is missing."
        )

    if model != "openai/gpt-oss-120b":

        raise ValueError(
            "Invalid GROQ_MODEL. "
            "Expected: openai/gpt-oss-120b"
        )

    return LLM(

        # IMPORTANT:
        # Do NOT use:
        # openai/openai/gpt-oss-120b

        model="openai/gpt-oss-120b",

        # Important for Groq's
        # OpenAI-compatible endpoint
        custom_openai=True,

        base_url=GROQ_BASE_URL,

        api_key=api_key,

        temperature=0.1,
    )


# ============================================================
# BUILD AGENT
# ============================================================

def build_agent():

    return Agent(

        role="Senior Web Research Analyst",

        goal=(
            "Produce accurate, evidence-grounded research "
            "from the supplied web sources without inventing "
            "facts, statistics, quotations or citations."
        ),

        backstory=(
            "You are a professional research analyst. "
            "You carefully compare multiple sources, "
            "prioritize authoritative evidence, identify "
            "uncertainty, distinguish facts from claims, "
            "and never fabricate information."
        ),

        llm=build_llm(),

        allow_delegation=False,

        verbose=False,

        max_iter=1,

        max_retry_limit=0,
    )


# ============================================================
# RUN RESEARCH
# ============================================================

def run_research(
    topic,
    depth,
    report_type,
    source_count,
    recency
):

    topic = str(topic).strip()

    if len(topic) < 5:

        raise ValueError(
            "Research topic is too short."
        )

    try:

        requested_sources = int(
            source_count
        )

    except (
        TypeError,
        ValueError
    ):

        requested_sources = 5

    # Keep the evidence package controlled.
    max_sources = min(
        max(
            requested_sources,
            5
        ),
        8
    )

    # ========================================================
    # SEARCH
    # ========================================================

    raw_sources = perform_searches(
        topic,
        depth,
        max_sources
    )

    if not raw_sources:

        raise RuntimeError(
            "DuckDuckGo did not return any "
            "web sources. Please try again."
        )

    # ========================================================
    # RANK
    # ========================================================

    ranked_sources = rank(
        raw_sources
    )

    if not ranked_sources:

        raise RuntimeError(
            "Sources were found but could "
            "not be ranked."
        )

    selected_sources = ranked_sources[
        :max_sources
    ]

    # ========================================================
    # READ WEBPAGES
    # ========================================================

    usable_sources = []

    for source in selected_sources:

        url = source.get("url")

        if not url:
            continue

        try:

            text = read_webpage.run(
                url
            )

        except Exception as exc:

            print(
                "Page extraction failed:",
                url,
                exc
            )

            continue

        if not text:
            continue

        if text.startswith(
            "Unable to read webpage:"
        ):

            continue

        # Keep evidence small.
        source["evidence"] = text[
            :6000
        ]

        usable_sources.append(
            source
        )

    if not usable_sources:

        raise RuntimeError(
            "Search results were found, "
            "but their webpages could not "
            "be read."
        )

    # ========================================================
    # BUILD EVIDENCE PACKAGE
    # ========================================================

    evidence_blocks = []

    for index, source in enumerate(
        usable_sources,
        1
    ):

        evidence_blocks.append(
            f"""
SOURCE {index}

Title:
{source.get("title", "Untitled")}

URL:
{source.get("url", "")}

Domain:
{source.get("domain", "")}

Source type:
{source.get("source_type", "Unknown")}

Quality tier:
{source.get("quality_tier", "Unknown")}

Retrieved evidence:
{source.get("evidence", "")[:6000]}
""".strip()
        )

    evidence_package = (
        "\n\n"
        .join(evidence_blocks)
    )

    # ========================================================
    # BUILD ONE AGENT
    # ========================================================

    agent = build_agent()

    # ========================================================
    # ONE TASK
    # ========================================================

    task = Task(

        description=f"""
You are researching:

{topic}

Research depth:
{depth}

Desired report type:
{report_type}

Preferred recency:
{recency}

You have been provided with retrieved web evidence below.

IMPORTANT:
Use ONLY the supplied evidence for factual claims.

Do NOT invent:
- facts
- statistics
- quotations
- studies
- organizations
- URLs
- publication dates

If evidence is insufficient, explicitly say:
"Insufficient evidence in the retrieved sources."

RESEARCH REQUIREMENTS:

1. Start with a clear executive summary.

2. Explain the major findings.

3. Compare findings across sources.

4. Identify important agreements.

5. Identify credible disagreements.

6. Distinguish evidence from interpretation.

7. Mention important limitations.

8. Give practical implications where supported.

9. Include source references using the supplied URLs.

10. Do not claim certainty when the evidence is uncertain.

11. Do not add information from your own memory.

12. Do not create citations that are not present in the evidence.

SOURCE EVIDENCE:

{evidence_package}
""",

        expected_output=(
            "A professional evidence-grounded research "
            "report with executive summary, key findings, "
            "source comparison, limitations, implications "
            "and source URLs."
        ),

        agent=agent,
    )

    # ========================================================
    # CREW
    # ========================================================

    crew = Crew(

        agents=[
            agent
        ],

        tasks=[
            task
        ],

        process=Process.sequential,

        verbose=False,
    )

    # ========================================================
    # ONE LLM CALL
    # ========================================================

    try:

        result = crew.kickoff()

        report = str(
            result
        )

    except Exception as exc:

        message = str(exc)

        if (
            "429" in message
            or "rate_limit" in message.lower()
            or "RateLimitError" in type(exc).__name__
        ):

            raise RuntimeError(
                "Groq rate limit reached for "
                "openai/gpt-oss-120b. "
                "Your code reached Groq successfully, "
                "but the model quota is currently "
                "exhausted. Wait for the quota reset "
                "before running another research request."
            ) from exc

        raise

    # ========================================================
    # QUALITY SUMMARY
    # ========================================================

    quality_counts = {}

    for source in usable_sources:

        tier = source.get(
            "quality_tier",
            "Unknown"
        )

        quality_counts[tier] = (
            quality_counts.get(
                tier,
                0
            ) + 1
        )

    quality_summary = ", ".join(
        f"{key}: {value}"
        for key, value in quality_counts.items()
    )

    if not quality_summary:

        quality_summary = "Unknown"

    # ========================================================
    # RETURN
    # ========================================================

    return {

        "report": report,

        "sources": usable_sources,

        "metadata": {

            "results_discovered": len(
                raw_sources
            ),

            "sources_used": len(
                usable_sources
            ),

            "quality_summary":
                quality_summary,

            "research_depth":
                depth,
        },
    }
