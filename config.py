import os
import streamlit as st

DEFAULT_MODEL = "openai/gpt-oss-120b"

def get_secret(name: str, default=None):
    try:
        value = st.secrets.get(name)
    except Exception:
        value = None
    return str(value) if value else os.getenv(name, default)

def get_groq_api_key():
    return get_secret("GROQ_API_KEY")

def get_groq_model():
    return get_secret("GROQ_MODEL", DEFAULT_MODEL)

def validate_configuration():
    if not get_groq_api_key():
        return "GROQ_API_KEY is missing from Streamlit Secrets."
    if not get_groq_model():
        return "GROQ_MODEL is missing from Streamlit Secrets."
    return None
