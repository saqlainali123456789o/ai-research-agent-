# 🔬 AI Research Agent

Evidence-grounded research application using Streamlit, CrewAI, Groq and DuckDuckGo search.

## What it does

Research topic → multi-query web discovery → source quality ranking → webpage retrieval → single CrewAI research agent → Groq synthesis → cited report → transparent sources.

## Deployment

This project is designed for GitHub + Streamlit Community Cloud. No local execution is required.

### GitHub files

Upload all repository files except secrets:

- app.py
- config.py
- research_agent.py
- research_tools.py
- source_quality.py
- report_generator.py
- prompts.py
- utils.py
- requirements.txt
- README.md
- .gitignore
- .python-version
- .streamlit/config.toml

### Streamlit Secrets

In Streamlit Cloud → App settings → Secrets:

```toml
GROQ_API_KEY = "your_groq_api_key"
GROQ_MODEL = "openai/gpt-oss-120b"
```

Never upload `secrets.toml` or an API key to GitHub.

## Default model

`openai/gpt-oss-120b`

The model is configurable through `GROQ_MODEL`.

## Research integrity

The system is designed to:

- prefer authoritative and academic sources
- retrieve original webpages rather than relying only on snippets
- rank sources by a transparent quality tier
- avoid fabricated citations
- report uncertainty and conflicting evidence
- preserve source URLs
- warn users to verify high-stakes findings

## Important limitation

No web research system can guarantee that every source or generated statement is correct. Always verify important findings against the original source, especially for legal, medical, financial, regulatory and academic submission use.

## License

Educational/research project. Review the licenses and terms of all dependencies and external services before commercial deployment.
