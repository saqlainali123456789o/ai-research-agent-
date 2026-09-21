import re
from urllib.parse import urlparse
import requests
import trafilatura
from bs4 import BeautifulSoup
from ddgs import DDGS
from crewai.tools import tool

UA = "Mozilla/5.0 (compatible; AIResearchAgent/1.0)"

def domain(url):
    try:
        return urlparse(url).netloc.lower().removeprefix("www.")
    except Exception:
        return ""

def clean(text, limit=12000):
    text = re.sub(r"\s+", " ", text or "").strip()
    return text[:limit]

def search_web(query, max_results=8):
    rows = []
    try:
        with DDGS() as ddgs:
            for item in ddgs.text(query, max_results=max_results):
                url = item.get("href") or item.get("url")
                if url:
                    rows.append({
                        "title": item.get("title","Untitled"),
                        "url": url,
                        "domain": domain(url),
                        "snippet": item.get("body","")
                    })
    except Exception:
        return []
    return rows

@tool("web_search")
def web_search(query: str) -> str:
    """Search the public web for research sources using DuckDuckGo."""
    results = search_web(query, 8)
    if not results:
        return "No results found."
    return "\n\n".join(
        f"RESULT {i}\nTitle: {r['title']}\nURL: {r['url']}\nDomain: {r['domain']}\nSnippet: {r['snippet']}"
        for i,r in enumerate(results,1)
    )

@tool("read_webpage")
def read_webpage(url: str) -> str:
    """Fetch and extract readable text from a public webpage."""
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=15)
        r.raise_for_status()
        extracted = trafilatura.extract(
            r.text, include_links=True, include_tables=True, favor_precision=True
        )
        if extracted:
            return clean(extracted, 14000)
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script","style","noscript"]):
            tag.decompose()
        return clean(soup.get_text(" "), 14000)
    except Exception as e:
        return f"Unable to read webpage: {type(e).__name__}"

def perform_searches(topic, depth, max_sources):
    queries = [
        f'"{topic}" research',
        f'"{topic}" study',
        f'"{topic}" evidence',
        f'"{topic}" report',
        f'"{topic}" statistics',
    ]
    if depth in ("Deep","Comprehensive"):
        queries += [
            f'"{topic}" academic research',
            f'"{topic}" government report',
            f'"{topic}" systematic review',
        ]
    if depth == "Comprehensive":
        queries += [
            f'"{topic}" official data',
            f'"{topic}" policy report',
        ]

    seen, results = set(), []
    for q in queries:
        for r in search_web(q, max_sources):
            key = r["url"].rstrip("/").lower()
            if key not in seen:
                seen.add(key)
                results.append(r)
    return results
