from crewai import Agent, Crew, Process, Task
from crewai.llm import LLM

from config import get_groq_api_key, get_groq_model
from research_tools import perform_searches, read_webpage, web_search
from source_quality import rank
from report_generator import generate_report

def build_llm():
    return LLM(
        model=f"groq/{get_groq_model()}",
        api_key=get_groq_api_key(),
        temperature=0.1,
    )

def build_agent():
    return Agent(
        role="Senior Research Analyst",
        goal="Find, evaluate and synthesize reliable web evidence without fabricating facts.",
        backstory=(
            "You are a rigorous research analyst. You prioritize original, "
            "authoritative and academic evidence and clearly communicate uncertainty."
        ),
        llm=build_llm(),
        tools=[web_search, read_webpage],
        allow_delegation=False,
        verbose=False,
    )

def run_research(topic, depth, report_type, source_count, recency):
    raw = perform_searches(topic, depth, source_count)
    ranked = rank(raw)

    # Retrieve the strongest candidates first.
    selected = ranked[:max(source_count, 5)]
    usable = []

    for source in selected:
        text = read_webpage.run(source["url"])
        if text and not text.startswith("Unable to read webpage:"):
            source["evidence"] = text
            usable.append(source)

    agent = build_agent()
    task = Task(
        description=f"""
Research this topic: {topic}

Depth: {depth}
Preferred recency: {recency}
Target sources: {source_count}

Use the available web tools to identify important evidence, compare sources,
and flag uncertainty or conflicting findings. Do not invent evidence.
Return a concise research evidence brief with source URLs.
""",
        expected_output="A structured evidence brief with claims and supporting sources.",
        agent=agent,
    )

    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
    )
    agent_result = str(crew.kickoff())

    evidence_blocks = []
    for i, s in enumerate(usable, 1):
        evidence_blocks.append(
            f"""SOURCE [{i}]
Title: {s.get('title','')}
URL: {s.get('url','')}
Type: {s.get('source_type','')}
Tier: {s.get('quality_tier','')}
Evidence:
{s.get('evidence','')[:10000]}"""
        )

    package = "\n\n".join(evidence_blocks)
    package += "\n\nCREWAI RESEARCH BRIEF\n" + agent_result

    report = generate_report(topic, package, report_type, depth)

    tiers = {}
    for s in usable:
        tiers[s["quality_tier"]] = tiers.get(s["quality_tier"],0) + 1
    quality_summary = ", ".join(f"{k}: {v}" for k,v in tiers.items()) or "—"

    return {
        "report": report,
        "sources": usable,
        "metadata": {
            "results_discovered": len(raw),
            "sources_used": len(usable),
            "quality_summary": quality_summary,
            "research_depth": depth,
        }
    }
