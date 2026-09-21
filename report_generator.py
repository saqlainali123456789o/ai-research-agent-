from groq import Groq
from config import get_groq_api_key, get_groq_model
from prompts import SYSTEM_PROMPT, REPORT_PROMPT

def generate_report(topic, evidence, report_type, depth):
    client = Groq(api_key=get_groq_api_key())
    prompt = REPORT_PROMPT.format(
        topic=topic,
        evidence=evidence,
        report_type=report_type,
        depth=depth,
    )
    response = client.chat.completions.create(
        model=get_groq_model(),
        messages=[
            {"role":"system","content":SYSTEM_PROMPT},
            {"role":"user","content":prompt},
        ],
        temperature=0.1,
        max_tokens=12000,
    )
    text = response.choices[0].message.content
    if not text:
        raise RuntimeError("Groq returned an empty response.")
    return text
