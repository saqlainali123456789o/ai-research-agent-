import re
from urllib.parse import urlparse

import requests
import trafilatura
from bs4 import BeautifulSoup
from ddgs import DDGS
from crewai.tools import tool


UA = "Mozilla/5.0 (compatible; AIResearchAgent/1.0)"


# ============================================================
# DOMAIN
# ============================================================

def domain(url):
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

def clean(text, limit=9000):

    text = re.sub(
        r"\s+",
        " ",
        text or ""
    ).strip()

    return text[:limit]


# ============================================================
# WEB SEARCH
# ============================================================

def search_web(
    query,
    max_results=5
):

    rows = []

    try:

        with DDGS() as ddgs:

            for item in ddgs.text(
                query,
                max_results=max_results
            ):

                url = (
                    item.get("href")
                    or item.get("url")
                )

                if not url:
                    continue

                rows.append({
                    "title": item.get(
                        "title",
                        "Untitled"
                    ),

                    "url": url,

                    "domain": domain(url),

                    "snippet": item.get(
                        "body",
                        ""
                    ),
                })

    except Exception as exc:

        print(
            "Web search failed:",
            exc
        )

        return []

    return rows


# ============================================================
# CREWAI WEB SEARCH TOOL
# ============================================================

@tool("web_search")
def web_search(
    query: str
) -> str:
    """
    Search the public web for research sources.
    """

    results = search_web(
        query,
        max_results=5
    )

    if not results:
        return "No results found."

    return "\n\n".join(
        f"""
RESULT {i}

Title:
{r['title']}

URL:
{r['url']}

Domain:
{r['domain']}

Snippet:
{r['snippet']}
""".strip()
        for i, r in enumerate(
            results,
            1
        )
    )


# ============================================================
# READ WEBPAGE
# ============================================================

@tool("read_webpage")
def read_webpage(
    url: str
) -> str:
    """
    Fetch and extract readable text
    from a public webpage.
    """

    try:

        response = requests.get(
            url,
            headers={
                "User-Agent": UA
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
                9000
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
            ]
        ):
            tag.decompose()

        return clean(
            soup.get_text(" "),
            9000
        )

    except Exception as exc:

        return (
            "Unable to read webpage: "
            f"{type(exc).__name__}"
        )


# ============================================================
# RESEARCH SEARCHES
# ============================================================

def perform_searches(
    topic,
    depth,
    max_sources
):

    max_sources = min(
        int(max_sources),
        8
    )

    queries = [
        f'"{topic}" research',
        f'"{topic}" evidence',
        f'"{topic}" report',
    ]

    if depth in (
        "Deep",
        "Comprehensive"
    ):

        queries.extend([
            f'"{topic}" academic research',
            f'"{topic}" systematic review',
        ])

    if depth == "Comprehensive":

        queries.extend([
            f'"{topic}" government report',
            f'"{topic}" official data',
        ])

    seen = set()
    results = []

    for query in queries:

        for result in search_web(
            query,
            max_results=4
        ):

            key = (
                result["url"]
                .rstrip("/")
                .lower()
            )

            if key in seen:
                continue

            seen.add(key)
            results.append(result)

            if len(results) >= max_sources:
                return results

    return results
