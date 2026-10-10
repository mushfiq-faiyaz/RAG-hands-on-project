import config

import json
from groq import Groq

groq_client = Groq(api_key=config.GROQ_API_KEY)

SYSTEM_PROMPT = (
    "You rewrite a user's question into 1 to 3 standalone search queries "
    "for a document retrieval system. "
    "Resolve pronouns using the conversation history. "
    "Include synonyms or related phrases only if they help retrieval. "
    "Return ONLY a JSON array of strings. No prose, no markdown."
)


def expand(question, history, max_queries=3):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history:
        messages.append(turn)
    messages.append({
        "role": "user",
        "content": f"Question: {question}\n\nReturn the JSON array now.",
    })

    response = groq_client.chat.completions.create(
        model=config.GROQ_MODEL,
        messages=messages,
        temperature=0,
    )

    raw = response.choices[0].message.content.strip()
    try:
        queries = json.loads(raw)
    except json.JSONDecodeError:
        return [question]

    if not isinstance(queries, list):
        return [question]

    queries = [q for q in queries if isinstance(q, str) and q.strip()]
    queries = queries[:max_queries]
    return queries if queries else [question]