import os
os.environ["HF_HUB_OFFLINE"] = "1"

from sentence_transformers import SentenceTransformer
import chromadb

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="chroma_db")

collections = {
    "400_80": client.get_collection("ranfy"),
    "700_150": client.get_collection("ranfy_700_150"),
}

QUESTIONS = [
    "How much does Fresh Texfy Pro cost?",
    "Can I get my money back?",
    "When is support available?",
    "What is the latest version of PidiFie?",
    "Who founded RanFy and when?",
]









def top_sources(collection, question, k=3):
    q_vec = model.encode(question).tolist()
    results = collection.query(query_embeddings=[q_vec], n_results=k)
    metas = results["metadatas"][0]
    return [(m["source"], m["page"]) for m in metas]


for q in QUESTIONS:
    print("=" * 70)
    print("Q:", q)
    for label, coll in collections.items():
        sources = top_sources(coll, q)
        print(f"  [{label}]")
        for src, page in sources:
            print(f"    - {src} p.{page}")