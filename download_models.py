from sentence_transformers import SentenceTransformer, CrossEncoder

print("Downloading embedding model...")
SentenceTransformer("thenlper/gte-small")

print("Downloading reranker...")
CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

print("Models cached.")