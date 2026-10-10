import config

from groq import Groq
from rerank import hybrid_search, rerank, meta_by_id, text_by_id
from expand import expand

groq_client = Groq(api_key=config.GROQ_API_KEY)

SYSTEM_PROMPT = (
    "You are a helpful assistant for RanFy Inc. "
    "Answer the user's question using ONLY the context below. "
    "If the answer is not in the context, say you don't know. "
    "When sources disagree, prefer the one with the more recent date. "
    "Cite the source of each fact using ASCII square brackets "
    "like [1] or [2] — plain ASCII, not fullwidth or CJK brackets. "
    "Put the marker immediately after the claim. "
    "Be concise."
)


def normalize_citations(text):
    for n in range(10):
        text = text.replace(f"【{n}】", f"[{n}]")
    return text


def dedup_by_source(doc_ids):
    seen = set()
    result = []
    for doc_id in doc_ids:
        meta = meta_by_id[doc_id]
        key = (meta["source"], meta["page"])
        if key not in seen:
            seen.add(key)
            result.append(doc_id)
    return result


def build_context(doc_ids):
    parts = []
    for i, doc_id in enumerate(doc_ids, start=1):
        meta = meta_by_id[doc_id]
        text = text_by_id[doc_id]
        parts.append(f"[{i}] Source: {meta['source']} page {meta['page']}\n{text}")
    return "\n\n".join(parts)


def build_sources(top_ids):
    return [
        {"index": i, "source": meta_by_id[doc_id]["source"], "page": meta_by_id[doc_id]["page"]}
        for i, doc_id in enumerate(top_ids, start=1)
    ]


def ask(question, history, k_retrieve=10, k_final=3):
    queries = expand(question, history)
    merged = []
    seen = set()
    for q in queries:
        for doc_id in hybrid_search(q, k=k_retrieve):
            if doc_id not in seen:
                merged.append(doc_id)
                seen.add(doc_id)

    top_ids = rerank(question, merged, top_n=k_final)
    top_ids = dedup_by_source(top_ids)
    context = build_context(top_ids)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history:
        messages.append(turn)
    messages.append({
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion: {question}",
    })

    response = groq_client.chat.completions.create(
        model=config.GROQ_MODEL,
        messages=messages,
        temperature=0,
    )

    clean = response.choices[0].message.content
    clean = clean.replace("\u202f", " ").replace("\u00a0", " ")
    clean = normalize_citations(clean)
    return {
        "answer": clean,
        "sources": build_sources(top_ids),
    }


def ask_stream(question, history, k_retrieve=10, k_final=3):
    queries = expand(question, history)
    merged = []
    seen = set()
    for q in queries:
        for doc_id in hybrid_search(q, k=k_retrieve):
            if doc_id not in seen:
                merged.append(doc_id)
                seen.add(doc_id)

    top_ids = rerank(question, merged, top_n=k_final)
    top_ids = dedup_by_source(top_ids)
    context = build_context(top_ids)

    yield {"type": "sources", "data": build_sources(top_ids)}

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history:
        messages.append(turn)
    messages.append({
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion: {question}",
    })

    stream = groq_client.chat.completions.create(
        model=config.GROQ_MODEL,
        messages=messages,
        temperature=0,
        stream=True,
    )

    for event in stream:
        delta = event.choices[0].delta.content
        if not delta:
            continue
        delta = delta.replace("\u202f", " ").replace("\u00a0", " ")
        yield {"type": "token", "data": delta}

    yield {"type": "done"}