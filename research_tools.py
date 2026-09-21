
import re
from urllib.parse import urlparse

import requests
import trafilatura
from bs4 import BeautifulSoup
from ddgs import DDGS
from crewai.tools import tool


UA = "Mozilla/5.0 (compatible; AIResearchAgent/1.0)"


# ---------------------------------------------------------
# URL / TEXT HELPERS
# ---------------------------------------------------------

def domain(url):
    try:
        return urlparse(url).netloc.lower().removeprefix("www.")
    except Exception:
        return ""


def clean(text, limit=12000):
    text = re.sub(r"\s+", " ", text or "").strip()
    return text[:limit]


# ---------------------------------------------------------
# WEB SEARCH
# ---------------------------------------------------------

def search_web(query, max_results=8):

    results = []

    try:

        with DDGS() as ddgs:

            rows = ddgs.text(
                query,
                region="us-en",
                safesearch="moderate",
                max_results=max_results,
                backend="auto",
            )

        for item in rows:

            url = item.get("href") or item.get("url")

            if not url:
                continue

            results.append(
                {
                    "title": item.get(
                        "title",
                        "Untitled",
                    ),
                    "url": url,
                    "domain": domain(url),
                    "snippet": item.get(
                        "body",
                        "",
                    ),
                }
            )

        return results

    except Exception as exc:

        # IMPORTANT:
        # Do not silently hide the real search error.
        print(
            "WEB SEARCH ERROR:",
            type(exc).__name__,
            str(exc),
        )

        return []


# ---------------------------------------------------------
# CREWAI WEB SEARCH TOOL
# ---------------------------------------------------------

@tool("web_search")
def web_search(query: str) -> str:
    """Search the public web for research sources using DDGS."""

    results = search_web(
        query,
        max_results=8,
    )

    if not results:
        return (
            "No search results were returned. "
            "The search provider may be temporarily unavailable."
        )

    blocks = []

    for i, result in enumerate(
        results,
        1,
    ):

        blocks.append(
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

    return "\n\n".join(blocks)


# ---------------------------------------------------------
# WEBPAGE READER
# ---------------------------------------------------------

@tool("read_webpage")
def read_webpage(url: str) -> str:
    """Fetch and extract readable text from a public webpage."""

    try:

        response = requests.get(
            url,
            headers={
                "User-Agent": UA,
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

            return clean(
                extracted,
                14000,
            )

        soup = BeautifulSoup(
            response.text,
            "html.parser",
        )

        for tag in soup(
            [
                "script",
                "style",
                "noscript",
            ]
        ):
            tag.decompose()

        return clean(
            soup.get_text(" "),
            14000,
        )

    except Exception as exc:

        return (
            "Unable to read webpage: "
            f"{type(exc).__name__}"
        )


# ---------------------------------------------------------
# MULTI-QUERY RESEARCH SEARCH
# ---------------------------------------------------------

def perform_searches(
    topic,
    depth,
    max_sources,
):

    topic = str(topic).strip()

    try:
        max_sources = max(
            int(max_sources),
            1,
        )
    except (
        TypeError,
        ValueError,
    ):
        max_sources = 8

    # Basic queries
    queries = [
        f"{topic} research",
        f"{topic} study",
        f"{topic} evidence",
        f"{topic} report",
        f"{topic} statistics",
    ]

    # Deep research
    if depth in (
        "Deep",
        "Comprehensive",
    ):

        queries.extend(
            [
                f"{topic} academic research",
                f"{topic} government report",
                f"{topic} systematic review",
            ]
        )

    # Comprehensive research
    if depth == "Comprehensive":

        queries.extend(
            [
                f"{topic} official data",
                f"{topic} policy report",
                f"{topic} meta analysis",
            ]
        )

    seen = set()
    results = []

    for query in queries:

        print(
            f"Searching web for: {query}"
        )

        search_results = search_web(
            query,
            max_results=max_sources,
        )

        for result in search_results:

            url = result.get(
                "url",
                "",
            ).strip()

            if not url:
                continue

            key = url.rstrip(
                "/"
            ).lower()

            if key in seen:
                continue

            seen.add(key)
            results.append(result)

            if len(results) >= max_sources:

                return results

    return results

