import os
os.environ["HF_HUB_OFFLINE"] = "1"

import re
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection("ranfy")

data = collection.get()
ids = data["ids"]
texts = data["documents"]
metadatas = data["metadatas"]
meta_by_id = {i: m for i, m in zip(ids, metadatas)}



def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())

tokenized = [tokenize(t) for t in texts]
bm25 = BM25Okapi(tokenized)




def vector_search(question, k=10):
    q_vec = model.encode(question).tolist()
    results = collection.query(query_embeddings=[q_vec], n_results=k)
    return results["ids"][0]


def bm25_search(question, k=10):
    q_tokens = tokenize(question)
    scores = bm25.get_scores(q_tokens)
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [ids[i] for i in ranked[:k]]


def rrf(vector_ids, bm25_ids, k=60):
    scores = {}
    for rank, doc_id in enumerate(vector_ids):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    for rank, doc_id in enumerate(bm25_ids):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)











QUESTIONS = [
    "How much does Fresh Texfy Pro cost?",
    "Can I get my money back?",
    "When is support available?",
    "What is the latest version of PidiFie?",
    "Who founded RanFy and when?",
]

for q in QUESTIONS:
    print("=" * 70)
    print("Q:", q)
    v = vector_search(q, k=3)
    b = bm25_search(q, k=3)
    h = rrf(vector_search(q, k=10), bm25_search(q, k=10))[:3]

    for label, doc_ids in [("VECTOR", v), ("BM25", b), ("HYBRID", h)]:
        print(f"  [{label}]")
        for doc_id in doc_ids:
            m = meta_by_id[doc_id]
            print(f"    - {m['source']} p.{m['page']}")