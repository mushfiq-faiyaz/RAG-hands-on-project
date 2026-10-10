# RanFy RAG

A retrieval-augmented chatbot over a set of RanFy Inc. PDFs. Built from scratch:
no LangChain, no LlamaIndex. Every component is visible and testable.

## Architecture

    PDFs -> chunk -> embed -> ChromaDB
                                  |
    query -> expand -> hybrid (vector + BM25) -> RRF -> rerank -> top 3
                                  |
                        Groq LLM -> answer with [N] citations
                                  |
                    FastAPI /chat, /chat/stream -> HTML widget

## Setup

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    pip install -r requirements.txt

Create `.env` at the project root:

    GROQ_API_KEY=your_key_here

Build the vector DB (one time):

    python src/build_gte.py

Run the API:

    uvicorn src.api:app --reload

Open `widget.html` in a browser. Or test from the terminal:

    python src/test_api.py
    python src/test_stream.py

Run the eval:

    python src/eval.py

## Files

    src/config.py          env flags, model names, API key — imported first by everything
    src/load_pdfs.py       read one PDF and print its text (Stage 1 demo)
    src/chunk.py           chunk_text (char window) + chunk_by_lines (unused experiment)
    src/chunk_pdf.py       chunk a real PDF, print first chunk
    src/dates.py           extract "Last updated: <date>" from PDF text
    src/embed_test.py      three sentences, three vectors, cosine sim demo
    src/build_gte.py       build the ranfy_gte Chroma collection (GTE + dates)
    src/build_db.py        older build script (MiniLM, no dates) — kept for reference
    src/retrieve.py        top-k retrieval demo (MiniLM)
    src/hybrid.py          vector + BM25 + RRF comparison demo
    src/rerank.py          THE retrieval library — hybrid_search, rerank, lookups
    src/expand.py          query expansion via Groq
    src/pipeline.py        THE RAG library — ask, ask_stream, prompts, dedup
    src/session.py         SQLite-backed conversation store
    src/api.py             FastAPI app: /chat, /chat/stream
    src/eval.py            run the 5 test questions, print pass/fail
    src/compare.py         compare two collections side by side
    src/test_api.py        hit /chat, test session memory
    src/test_stream.py     hit /chat/stream, print SSE events
    src/test_persist_step1.py / step2.py   prove sessions survive restart

## Stack

    Python 3.14
    pypdf                PDF text extraction
    sentence-transformers  embeddings + cross-encoder reranker
    thenlper/gte-small   embedding model (384 dims)
    cross-encoder/ms-marco-MiniLM-L-6-v2   reranker
    chromadb             vector store (persistent, local)
    rank_bm25            keyword retrieval
    groq                 LLM API (openai/gpt-oss-120b)
    fastapi + uvicorn    HTTP server
    sqlite3              session persistence

## Key design choices

**GTE-small over MiniLM.** Same size, same speed. GTE retrieves the correct
"Monthly p.1" chunk for the PidiFie question directly; MiniLM does not.

**Hybrid search.** Vector search fails on exact strings ("1.2" vs "1.0").
BM25 fails on paraphrase ("money back" vs "refund"). RRF merges both.

**Cross-encoder rerank.** Retrieves 20 candidates, scores each query+chunk pair
jointly, keeps top 3. Cannot reason about dates — that requires the LLM.

**Recency boost.** When the question contains "latest", "newest", etc., add +2.0
to any chunk from the newest document. A heuristic, not a model. Works because
the docs have clean "Last updated" lines.

**Chunk size 400/80.** Tested 700/150 and line-based chunking. Both lost.
400/80 wins on these short dense docs.

**Dedup before numbering.** Duplicate source+page chunks are removed before the
prompt is built, so every [N] the LLM cites exists in the sources list.

## Known limitations

1. **Recency is heuristic.** Breaks on corpora without date lines, or when the
   newest document isn't the most relevant.
2. **Vague first-turns guess.** "When was that released?" with no history gives
   an answer that's right by luck on this corpus.
3. **No score fusion in RRF.** Ranks only, not scores. Adds a bit of noise.
4. **BGE reranker regression.** Swapping to `BAAI/bge-reranker-base` dropped the
   correct PidiFie chunk entirely. Reverted. Documented so we don't try again
   without a plan.
5. **Single worker.** SQLite is fine for one process; not for multiple.