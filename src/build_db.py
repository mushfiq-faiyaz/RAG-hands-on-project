from pathlib import Path
from pypdf import PdfReader
from chunk import chunk_text
from sentence_transformers import SentenceTransformer
import chromadb

pdf_dir = Path("Docs")
pdf_files = sorted(pdf_dir.glob("*.pdf"))
for p in pdf_files:
    print("Found:", p.name)





all_chunks = []
for p in pdf_files:
    reader = PdfReader(p)
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        for piece in chunk_text(text, chunk_size=400, overlap=80):
            all_chunks.append({
                "text": piece,
                "source": p.name,
                "page": page_num,
            })

print("Total chunks:", len(all_chunks))




model = SentenceTransformer("all-MiniLM-L6-v2")

texts = [c["text"] for c in all_chunks]
embeddings = model.encode(texts, show_progress_bar=True)

client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection("ranfy")

ids = [f"chunk_{i}" for i in range(len(all_chunks))]

collection.add(
    ids=ids,
    documents=texts,
    embeddings=embeddings.tolist(),
    metadatas=[{"source": c["source"], "page": c["page"]} for c in all_chunks],
)

print("Stored in Chroma:", collection.count())