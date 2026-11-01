import requests

MODEL = "llama3.1:8b"

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"


def generate_answer(query, context):

    prompt = f"""
Answer ONLY using the provided context.

Context:
{context}

Question:
{query}

Answer:
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0,
                "num_predict": 120
            }
        },
        timeout=300
    )

    response.raise_for_status()

    return response.json()["response"]