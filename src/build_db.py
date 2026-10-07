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