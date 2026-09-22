import os
import streamlit as st


DEFAULT_MODEL = "openai/gpt-oss-120b"


def get_secret(name: str, default=None):

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


def get_groq_api_key():
    return get_secret("GROQ_API_KEY")


def get_groq_model():

    model = get_secret(
        "GROQ_MODEL",
        DEFAULT_MODEL
    )

    if not model:
        return DEFAULT_MODEL

    model = str(model).strip()

    # Remove accidental provider prefix
    if model.startswith("groq/"):
        model = model[len("groq/"):]

    # Normalize GPT-OSS 120B
    if model == "gpt-oss-120b":
        return "openai/gpt-oss-120b"

    if model == "openai/gpt-oss-120b":
        return "openai/gpt-oss-120b"

    return model


def validate_configuration():

    api_key = get_groq_api_key()

    model = get_groq_model()

    if not api_key:
        return (
            "GROQ_API_KEY is missing from Streamlit Secrets."
        )

    if model != "openai/gpt-oss-120b":
        return (
            "GROQ_MODEL must be "
            "openai/gpt-oss-120b"
        )

    return None
