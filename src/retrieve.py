from sentence_transformers import SentenceTransformer
import chromadb

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection("ranfy")

question = "Can I get my money back?"

q_vec = model.encode(question).tolist()

results = collection.query(
    query_embeddings=[q_vec],
    n_results=3,
)

for i, doc in enumerate(results["documents"][0]):
    meta = results["metadatas"][0][i]
    print(f"--- Result {i+1} | {meta['source']} p.{meta['page']} ---")
    print(doc)
    print()