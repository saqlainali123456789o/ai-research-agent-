import re
from urllib.parse import urlparse

import requests
import trafilatura
from bs4 import BeautifulSoup
from ddgs import DDGS
from crewai.tools import tool


USER_AGENT = (
    "Mozilla/5.0 "
    "(compatible; AIResearchAgent/1.0)"
)


# ============================================================
# DOMAIN
# ============================================================

def get_domain(url):

    try:

        return (
            urlparse(url)
            .netloc
            .lower()
            .removeprefix("www.")
        )

    except Exception:

        return ""


# ============================================================
# CLEAN TEXT
# ============================================================

def clean_text(text, limit=7000):

    text = re.sub(
        r"\s+",
        " ",
        text or ""
    ).strip()

    return text[:limit]


# ============================================================
# DUCKDUCKGO SEARCH
# ============================================================

def search_web(
    query,
    max_results=5
):

    results = []

    try:

        with DDGS(
            timeout=10
        ) as ddgs:

            items = ddgs.text(
                query=query,
                region="us-en",
                safesearch="moderate",
                max_results=max_results,
                backend="auto",
            )

            for item in items:

                url = (
                    item.get("href")
                    or item.get("url")
                )

                if not url:
                    continue

                results.append(
                    {
                        "title": item.get(
                            "title",
                            "Untitled"
                        ),

                        "url": url,

                        "domain": get_domain(
                            url
                        ),

                        "snippet": item.get(
                            "body",
                            ""
                        ),
                    }
                )

    except Exception as exc:

        print(
            "DDGS search error:",
            type(exc).__name__,
            str(exc)
        )

    return results


# ============================================================
# CREWAI SEARCH TOOL
#
# This remains available to CrewAI, but our main workflow
# performs the search before the LLM call.
# ============================================================

@tool("web_search")
def web_search(query: str) -> str:
    """
    Search the public web using DuckDuckGo.
    """

    results = search_web(
        query,
        max_results=5
    )

    if not results:

        return (
            "No web results were returned."
        )

    output = []

    for i, result in enumerate(
        results,
        1
    ):

        output.append(
            f"""
RESULT {i}

Title:
{result["title"]}

URL:
{result["url"]}

Domain:
{result["domain"]}

Snippet:
{result["snippet"]}
""".strip()
        )

    return "\n\n".join(output)


# ============================================================
# WEBPAGE EXTRACTION
# ============================================================

@tool("read_webpage")
def read_webpage(url: str) -> str:
    """
    Fetch and extract readable webpage text.
    """

    try:

        response = requests.get(
            url,
            headers={
                "User-Agent": USER_AGENT
            },
            timeout=15,
        )

        response.raise_for_status()

        extracted = trafilatura.extract(
            response.text,
            include_links=True,
            include_tables=True,
            favor_precision=True,
        )

        if extracted:

            return clean_text(
                extracted,
                7000
            )

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        for tag in soup(
            [
                "script",
                "style",
                "noscript",
                "svg",
            ]
        ):

            tag.decompose()

        return clean_text(
            soup.get_text(" "),
            7000
        )

    except Exception as exc:

        return (
            "Unable to read webpage: "
            f"{type(exc).__name__}"
        )


# ============================================================
# SEARCH QUERY BUILDER
# ============================================================

def build_queries(
    topic,
    depth
):

    queries = [
        f'"{topic}" research',
        f'"{topic}" evidence',
        f'"{topic}" report',
    ]

    if depth in (
        "Deep",
        "Comprehensive"
    ):

        queries.extend(
            [
                f'"{topic}" academic study',
                f'"{topic}" systematic review',
            ]
        )

    if depth == "Comprehensive":

        queries.extend(
            [
                f'"{topic}" government report',
                f'"{topic}" official data',
            ]
        )

    return queries


# ============================================================
# MULTI-QUERY RESEARCH SEARCH
# ============================================================

def perform_searches(
    topic,
    depth,
    max_sources
):

    max_sources = max(
        1,
        min(
            int(max_sources),
            8
        )
    )

    queries = build_queries(
        topic,
        depth
    )

    seen = set()

    results = []

    for query in queries:

        print(
            f"Research search: {query}"
        )

        found = search_web(
            query,
            max_results=5
        )

        for item in found:

            url = item.get(
                "url",
                ""
            ).strip()

            if not url:
                continue

            key = (
                url
                .rstrip("/")
                .lower()
            )

            if key in seen:
                continue

            seen.add(key)

            results.append(item)

            if len(results) >= max_sources:

                return results

    return results
