import os
os.environ["HF_HUB_OFFLINE"] = "1"

import re
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection("ranfy")

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


def rerank(question, doc_ids, top_n=3):
    pairs = [(question, text_by_id[doc_id]) for doc_id in doc_ids]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(doc_ids, scores), key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, _ in ranked[:top_n]]









# QUESTIONS = [
#     "How much does Fresh Texfy Pro cost?",
#     "Can I get my money back?",
#     "When is support available?",
#     "What is the latest version of PidiFie?",
#     "Who founded RanFy and when?",
# ]

# for q in QUESTIONS:
#     print("=" * 70)
#     print("Q:", q)

#     candidates = hybrid_search(q, k=20)
#     reranked = rerank(q, candidates, top_n=3)

#     print("  [HYBRID top 3]")
#     for doc_id in candidates[:3]:
#         m = meta_by_id[doc_id]
#         print(f"    - {m['source']} p.{m['page']}")

#     print("  [HYBRID + RERANK top 3]")
#     for doc_id in reranked:
#         m = meta_by_id[doc_id]
#         print(f"    - {m['source']} p.{m['page']}")