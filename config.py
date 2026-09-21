import os
import streamlit as st


# ============================================================
# DEFAULT CONFIGURATION
# ============================================================

DEFAULT_MODEL = "openai/gpt-oss-120b"


# ============================================================
# SECRET / ENVIRONMENT VARIABLE HELPER
# ============================================================

def get_secret(name: str, default=None):
    """
    Read configuration from Streamlit Secrets first,
    then fall back to environment variables.
    """

    # Try Streamlit Cloud Secrets
    try:
        value = st.secrets.get(name)
    except Exception:
        value = None

    if value:
        return str(value).strip()

    # Fall back to environment variables
    value = os.getenv(name)

    if value:
        return str(value).strip()

    return default


# ============================================================
# GROQ API KEY
# ============================================================

def get_groq_api_key():
    """
    Return Groq API key.
    """

    return get_secret("GROQ_API_KEY")


# ============================================================
# GROQ MODEL
# ============================================================

def get_groq_model():
    """
    Return the correct Groq model ID.

    The application internally expects the exact Groq model ID:

        openai/gpt-oss-120b

    This function also cleans common incorrect formats.
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

    # If user entered the bare model name
    if model == "gpt-oss-120b":
        return "openai/gpt-oss-120b"

    # If already correct
    if model == "openai/gpt-oss-120b":
        return "openai/gpt-oss-120b"

    # If another openai-prefixed model is supplied,
    # preserve it.
    if model.startswith("openai/"):
        return model

    # For other bare model IDs, use OpenAI-compatible
    # provider format.
    return f"openai/{model}"


# ============================================================
# CONFIGURATION VALIDATION
# ============================================================

def validate_configuration():
    """
    Validate required application configuration.
    """

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
