def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks




def chunk_by_lines(text, max_size=600, min_size=100):
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    chunks = []
    buffer = []
    size = 0

    for line in lines:
        is_heading = bool(line) and line[0].isdigit() and "." in line[:4]

        if (size + len(line) + 1 > max_size) or (is_heading and size >= min_size):
            if buffer:
                chunks.append("\n".join(buffer))
            buffer = []
            size = 0

        buffer.append(line)
        size += len(line) + 1

    if buffer:
        chunks.append("\n".join(buffer))

    return chunks







if __name__ == "__main__":
    from pypdf import PdfReader
    reader = PdfReader("Docs/RanFy_Policies_FAQ.pdf")
    full = "\n".join(p.extract_text() for p in reader.pages)
    chunks = chunk_by_lines(full)
    print("Total chunks:", len(chunks))
    for i, c in enumerate(chunks):
        print(f"\n--- chunk {i} ({len(c)} chars) ---")
        print(c[:200])