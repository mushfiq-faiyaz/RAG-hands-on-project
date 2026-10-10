import config

import re
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

model = SentenceTransformer(config.EMBED_MODEL)
reranker = CrossEncoder(config.RERANK_MODEL)
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection(config.COLLECTION_NAME)

data = collection.get()
ids = data["ids"]
texts = data["documents"]
meta_by_id = {i: m for i, m in zip(ids, data["metadatas"])}
text_by_id = {i: t for i, t in zip(ids, texts)}


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


bm25 = BM25Okapi([tokenize(t) for t in texts])


def vector_search(question, k=20):
    q_vec = model.encode(question).tolist()
    results = collection.query(query_embeddings=[q_vec], n_results=k)
    return results["ids"][0]


def bm25_search(question, k=20):
    scores = bm25.get_scores(tokenize(question))
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [ids[i] for i in ranked[:k]]


def rrf(vector_ids, bm25_ids, k=60):
    scores = {}
    for rank, doc_id in enumerate(vector_ids):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    for rank, doc_id in enumerate(bm25_ids):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)


def hybrid_search(question, k=20):
    return rrf(vector_search(question, k), bm25_search(question, k))[:k]


RECENCY_KEYWORDS = ("latest", "newest", "current", "recent", "now", "today")


def rerank(question, doc_ids, top_n=3, recency_boost=2.0):
    pairs = [(question, text_by_id[doc_id]) for doc_id in doc_ids]
    scores = reranker.predict(pairs)

    q_lower = question.lower()
    has_recency = any(kw in q_lower for kw in RECENCY_KEYWORDS)

    if has_recency:
        dates = [meta_by_id[doc_id].get("date", "0000-00-00") for doc_id in doc_ids]
        max_date = max(dates)
        scores = [
            float(s) + (recency_boost if d == max_date else 0.0)
            for s, d in zip(scores, dates)
        ]

    ranked = sorted(zip(doc_ids, scores), key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, _ in ranked[:top_n]]