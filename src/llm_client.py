import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = "openai/gpt-oss-20b"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

_client = None


def get_client():
    global _client

    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY not found. Make sure you have a .env file "
                "in the project root with:\n"
                "GROQ_API_KEY=your_key_here"
            )

        _client = OpenAI(
            api_key=api_key,
            base_url=GROQ_BASE_URL,
        )

    return _client


def ask_llm(prompt: str, max_tokens: int = 500) -> str:
    client = get_client()

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=max_tokens,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        extra_body={"reasoning_effort": "low"},
    )

    return response.choices[0].message.content


def ask_llm_json(
    prompt: str,
    max_tokens: int = 500,
) -> dict:
    raw = ask_llm(prompt, max_tokens=max_tokens)
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json", "", 1).strip()
    return json.loads(cleaned)    

    