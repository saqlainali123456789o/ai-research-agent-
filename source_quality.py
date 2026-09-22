from urllib.parse import urlparse


HIGH_QUALITY_DOMAINS = {

    "gov": 5,
    "edu": 5,

    "who.int": 5,
    "worldbank.org": 5,
    "imf.org": 5,
    "oecd.org": 5,
    "un.org": 5,

    "nature.com": 5,
    "sciencedirect.com": 5,
    "springer.com": 5,
    "pubmed.ncbi.nlm.nih.gov": 5,

    "reuters.com": 4,
    "apnews.com": 4,
    "bbc.com": 4,
}


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


def score_source(source):

    url = source.get(
        "url",
        ""
    )

    domain = get_domain(
        url
    )

    score = 1

    tier = "Standard"

    for trusted_domain, value in HIGH_QUALITY_DOMAINS.items():

        if (
            domain == trusted_domain
            or domain.endswith(
                "." + trusted_domain
            )
        ):

            score = value

            if value >= 5:
                tier = "High"

            elif value >= 4:
                tier = "Good"

            break

    source["quality_score"] = score

    source["quality_tier"] = tier

    if not source.get(
        "source_type"
    ):

        if (
            ".edu" in domain
            or ".gov" in domain
        ):

            source["source_type"] = (
                "Academic/Government"
            )

        else:

            source["source_type"] = (
                "Web source"
            )

    return source


def rank(sources):

    scored = []

    for source in sources:

        scored.append(
            score_source(
                source
            )
        )

    scored.sort(
        key=lambda item:
        item.get(
            "quality_score",
            0
        ),
        reverse=True,
    )

    return scored
