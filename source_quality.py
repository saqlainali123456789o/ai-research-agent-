from urllib.parse import urlparse

PRIMARY = (
    ".gov", ".gov.uk", ".gov.au", ".gov.pk", ".edu", ".ac.uk",
    "who.int", "worldbank.org", "imf.org", "oecd.org", "un.org",
    "unicef.org", "nih.gov", "ncbi.nlm.nih.gov", "nasa.gov"
)
ACADEMIC = (
    "nature.com", "sciencedirect.com", "springer.com", "ieee.org",
    "pubmed.ncbi.nlm.nih.gov", "jstor.org", "ssrn.com", "arxiv.org"
)

def classify(url):
    try:
        d = urlparse(url).netloc.lower().removeprefix("www.")
    except Exception:
        d = ""
    if any(d.endswith(x) or d == x for x in PRIMARY):
        return "Primary / authoritative", "Tier 1", 5
    if any(d.endswith(x) or d == x for x in ACADEMIC):
        return "Academic / research", "Tier 1", 5
    if d.endswith(".org"):
        return "Organization", "Tier 2", 3
    if d.endswith(".com") or d.endswith(".net"):
        return "General publication", "Tier 3", 2
    return "General web", "Tier 4", 1

def enrich(source):
    kind,tier,score = classify(source["url"])
    source.update(source_type=kind, quality_tier=tier, quality_score=score)
    return source

def rank(sources):
    return sorted((enrich(s) for s in sources), key=lambda x:x["quality_score"], reverse=True)
