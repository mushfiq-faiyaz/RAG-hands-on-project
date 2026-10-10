import os
os.environ["HF_HUB_OFFLINE"] = "1"

from sentence_transformers import SentenceTransformer
import chromadb

client = chromadb.PersistentClient(path="chroma_db")

SETUPS = [
    ("minilm_400_80", "ranfy", "all-MiniLM-L6-v2"),
    ("gte_400_80", "ranfy_gte", "thenlper/gte-small"),
]

QUESTIONS = [
    "How much does Fresh Texfy Pro cost?",
    "Can I get my money back?",
    "When is support available?",
    "What is the latest version of PidiFie?",
    "Who founded RanFy and when?",
]

models = {name: SentenceTransformer(model_name) for name, _, model_name in SETUPS}
collections = {name: client.get_collection(coll_name) for name, coll_name, _ in SETUPS}


def top_sources(setup_name, question, k=3):
    model = models[setup_name]
    coll = collections[setup_name]
    q_vec = model.encode(question).tolist()
    results = coll.query(query_embeddings=[q_vec], n_results=k)
    return [(m["source"], m["page"]) for m in results["metadatas"][0]]


for q in QUESTIONS:
    print("=" * 70)
    print("Q:", q)
    for setup_name, _, _ in SETUPS:
        sources = top_sources(setup_name, q)
        print(f"  [{setup_name}]")
        for src, page in sources:
            print(f"    - {src} p.{page}")














# import os
# os.environ["HF_HUB_OFFLINE"] = "1"

# from sentence_transformers import SentenceTransformer
# import chromadb

# model = SentenceTransformer("all-MiniLM-L6-v2")
# client = chromadb.PersistentClient(path="chroma_db")

# # collections = {
# #     "400_80": client.get_collection("ranfy"),
# #     "700_150": client.get_collection("ranfy_700_150"),
# # }

# collections = {
#     "400_80": client.get_collection("ranfy"),
#     "lines": client.get_collection("ranfy_lines"),
# }

# QUESTIONS = [
#     "How much does Fresh Texfy Pro cost?",
#     "Can I get my money back?",
#     "When is support available?",
#     "What is the latest version of PidiFie?",
#     "Who founded RanFy and when?",
# ]









# def top_sources(collection, question, k=3):
#     q_vec = model.encode(question).tolist()
#     results = collection.query(query_embeddings=[q_vec], n_results=k)
#     metas = results["metadatas"][0]
#     return [(m["source"], m["page"]) for m in metas]


# for q in QUESTIONS:
#     print("=" * 70)
#     print("Q:", q)
#     for label, coll in collections.items():
#         sources = top_sources(coll, q)
#         print(f"  [{label}]")
#         for src, page in sources:
#             print(f"    - {src} p.{page}")