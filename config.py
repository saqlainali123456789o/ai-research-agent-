import os
import streamlit as st


# ============================================================
# DEFAULT MODEL
# ============================================================

DEFAULT_MODEL = "gpt-oss-120b"


# ============================================================
# SECRET HELPER
# ============================================================

def get_secret(name: str, default=None):
    """
    Get configuration value from Streamlit Secrets.
    Falls back to environment variables.
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
    Return only the model name.

    Expected:
        gpt-oss-120b

    Remove accidental provider prefixes if they exist.
    """

    model = get_secret(
        "GROQ_MODEL",
        DEFAULT_MODEL
    )

    if not model:
        return DEFAULT_MODEL

    model = model.strip()

    # Remove accidental prefixes
    if model.startswith("groq/"):
        model = model[len("groq/"):]

    if model.startswith("openai/"):
        model = model[len("openai/"):]

    return model


# ============================================================
# CONFIGURATION VALIDATION
# ============================================================

def validate_configuration():

    if not get_groq_api_key():
        return "GROQ_API_KEY is missing from Streamlit Secrets."

    if not get_groq_model():
        return "GROQ_MODEL is missing from Streamlit Secrets."

    return None
