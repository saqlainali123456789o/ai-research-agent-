import os
import streamlit as st


# ============================================================
# DEFAULT GROQ MODEL
# ============================================================

DEFAULT_MODEL = "openai/gpt-oss-20b"


# ============================================================
# SECRET HELPER
# ============================================================

def get_secret(name: str, default=None):
    """
    Read a value from Streamlit Secrets first.
    Fall back to environment variables.
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
    Return a normalized Groq model ID.

    Accepted examples:

        openai/gpt-oss-20b
        gpt-oss-20b
        groq/openai/gpt-oss-20b
    """

    model = get_secret(
        "GROQ_MODEL",
        DEFAULT_MODEL
    )

    if not model:
        return DEFAULT_MODEL

    model = str(model).strip()

    # Remove accidental Groq provider prefix
    if model.startswith("groq/"):
        model = model[len("groq/"):]

    # GPT-OSS models need the OpenAI model namespace
    if model == "gpt-oss-20b":
        return "openai/gpt-oss-20b"

    if model == "gpt-oss-120b":
        return "openai/gpt-oss-120b"

    # Already correctly formatted
    if model.startswith("openai/"):
        return model

    return model


# ============================================================
# CONFIGURATION VALIDATION
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
