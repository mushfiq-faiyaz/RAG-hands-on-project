import os
os.environ["HF_HUB_OFFLINE"] = "1"

from pathlib import Path
from pypdf import PdfReader
from chunk import chunk_text
from dates import extract_date
from sentence_transformers import SentenceTransformer
import chromadb



# from pathlib import Path
# from pypdf import PdfReader
# from chunk import chunk_text
# from sentence_transformers import SentenceTransformer
# import chromadb

CHUNK_SIZE = 400
OVERLAP = 80
COLLECTION_NAME = "ranfy_gte"
MODEL_NAME = "thenlper/gte-small"

pdf_dir = Path("Docs")
pdf_files = sorted(pdf_dir.glob("*.pdf"))

# all_chunks = []
# for p in pdf_files:
#     reader = PdfReader(p)
#     for page_num, page in enumerate(reader.pages, start=1):
#         text = page.extract_text()
#         for piece in chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
#             all_chunks.append({"text": piece, "source": p.name, "page": page_num})



all_chunks = []
for p in pdf_files:
    reader = PdfReader(p)
    pages_text = [page.extract_text() for page in reader.pages]
    full_text = "\n".join(pages_text)
    doc_date = extract_date(full_text) or "0000-00-00"
    print(f"{p.name}: date={doc_date}")

    for page_num, text in enumerate(pages_text, start=1):
        for piece in chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
            all_chunks.append({
                "text": piece,
                "source": p.name,
                "page": page_num,
                "date": doc_date,
            })

print("Total chunks:", len(all_chunks))

model = SentenceTransformer(MODEL_NAME)
texts = [c["text"] for c in all_chunks]
embeddings = model.encode(texts, show_progress_bar=True)

client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection(COLLECTION_NAME)
ids = [f"chunk_{i}" for i in range(len(all_chunks))]
collection.add(
    ids=ids,
    documents=texts,
    embeddings=embeddings.tolist(),
    metadatas=[{"source": c["source"], "page": c["page"], "date": c["date"]} for c in all_chunks],
    # metadatas=[{"source": c["source"], "page": c["page"]} for c in all_chunks],
)
print("Stored in Chroma:", collection.count())