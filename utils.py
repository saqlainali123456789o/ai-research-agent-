import re

def clean_text(text):
    return re.sub(r"\n{3,}", "\n\n", text or "").strip()

def safe_filename(text):
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", text or "").strip("_")[:80]
