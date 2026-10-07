from pypdf import PdfReader
from chunk import chunk_text

reader = PdfReader("Docs/RanFy_Products.pdf")
full_text = ""
for page in reader.pages:
    full_text += page.extract_text()

chunks = chunk_text(full_text, chunk_size=400, overlap=80)
print("Number of chunks:", len(chunks))
print("--- First chunk ---")
print(chunks[0])