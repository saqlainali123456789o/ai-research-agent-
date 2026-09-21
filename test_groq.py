import streamlit as st
from openai import OpenAI


# Get API key from Streamlit Secrets
api_key = st.secrets.get("GROQ_API_KEY")


# Check API key
if not api_key:
    st.error("GROQ_API_KEY is missing from Streamlit Secrets.")
    st.stop()


# Create Groq OpenAI-compatible client
client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1",
)


st.title("Groq Connection Test")

st.write("Testing model: openai/gpt-oss-120b")


try:

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "user",
                "content": "Say hello in one short sentence."
            }
        ],
    )

    st.success("Groq connection successful!")

    st.write(
        response.choices[0].message.content
    )


except Exception as e:

    st.error("Groq test failed.")

    st.code(str(e))
