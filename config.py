import os
import streamlit as st


# ============================================================
# DEFAULT MODEL
# ============================================================

DEFAULT_MODEL = "openai/gpt-oss-120b"


# ============================================================
# SECRET HELPER
# ============================================================

def get_secret(name: str, default=None):
    """
    Read a value from Streamlit Secrets first,
    then environment variables.
    """

    try:
        value = st.secrets.get(name)
    except Exception:
        value = None

    if value:
        return str(value).strip()

    value = os.getenv(name)

    if value:
        return str(value).strip()

    return default


# ============================================================
# GROQ API KEY
# ============================================================

def get_groq_api_key():
    return get_secret("GROQ_API_KEY")


# ============================================================
# GROQ MODEL
# ============================================================

def get_groq_model():
    """
    Always return the actual Groq model ID.

    Expected:
        openai/gpt-oss-120b
    """

    model = get_secret(
        "GROQ_MODEL",
        DEFAULT_MODEL
    )

    if not model:
        return DEFAULT_MODEL

    model = str(model).strip()

    # Remove accidental groq/ prefix
    if model.startswith("groq/"):
        model = model[len("groq/"):]

    model = model.strip()

    # Bare model name
    if model == "gpt-oss-120b":
        return "openai/gpt-oss-120b"

    # Correct model
    if model == "openai/gpt-oss-120b":
        return "openai/gpt-oss-120b"

    # Preserve other openai-prefixed models
    if model.startswith("openai/"):
        return model

    # Fallback
    return f"openai/{model}"


# ============================================================
# VALIDATE CONFIGURATION
# ============================================================

def validate_configuration():

    api_key = get_groq_api_key()
    model = get_groq_model()

    if not api_key:
        return (
            "GROQ_API_KEY is missing from Streamlit Secrets."
        )

    if not model:
        return (
            "GROQ_MODEL is missing from Streamlit Secrets."
        )

    return None
