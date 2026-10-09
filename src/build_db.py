import os
os.environ["HF_HUB_OFFLINE"] = "1"

from pathlib import Path
from pypdf import PdfReader
from chunk import chunk_text
from sentence_transformers import SentenceTransformer
import chromadb

# CHUNK_SIZE = 700
# OVERLAP = 150

CHUNK_SIZE = 400
OVERLAP = 80


COLLECTION_NAME = f"ranfy_{CHUNK_SIZE}_{OVERLAP}"

pdf_dir = Path("Docs")
pdf_files = sorted(pdf_dir.glob("*.pdf"))
for p in pdf_files:
    print("Found:", p.name)





all_chunks = []
for p in pdf_files:
    reader = PdfReader(p)
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        # for piece in chunk_text(text, chunk_size=400, overlap=80):
        for piece in chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
            all_chunks.append({
                "text": piece,
                "source": p.name,
                "page": page_num,
            })

print("Total chunks:", len(all_chunks))




model = SentenceTransformer("all-MiniLM-L6-v2")

texts = [c["text"] for c in all_chunks]
embeddings = model.encode(texts, show_progress_bar=True)

# client = chromadb.PersistentClient(path="chroma_db")
# collection = client.get_or_create_collection("ranfy")

client = chromadb.PersistentClient(path="chroma_db")
try:
    # collection = client.get_collection("ranfy")
    collection = client.get_or_create_collection(COLLECTION_NAME)
except Exception:
    print("ERROR: Chroma collection 'ranfy' not found. Run build_db.py first.")
    sys.exit(1)





ids = [f"chunk_{i}" for i in range(len(all_chunks))]

collection.add(
    ids=ids,
    documents=texts,
    embeddings=embeddings.tolist(),
    metadatas=[{"source": c["source"], "page": c["page"]} for c in all_chunks],
)

print("Stored in Chroma:", collection.count())