# Deployment Guide

## Pre-deploy checklist

- [ ] .env is NOT committed (check `git log --all --full-history -- .env` is empty)
- [ ] .env.example exists with placeholder values
- [ ] requirements.txt is up to date (`pip freeze > requirements.txt`)
- [ ] chroma_db/ is NOT committed (rebuildable)
- [ ] sessions.db is NOT committed (runtime data)
- [ ] rag.log is NOT committed
- [ ] eval.py passes 5/5 with rank-1 PASS locally
- [ ] You have a fresh Groq API key (not the one used in dev)
- [ ] You have a long random AUTH_TOKEN (32+ chars)

## Deploy to Render (free tier)

1. Push to GitHub first. Confirm the repo has no secrets.
2. Sign up at render.com. Connect your GitHub.
3. New → Web Service → select the repo.
4. Settings:
   - Environment: Python 3
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn src.api:app --host 0.0.0.0 --port $PORT`
5. Environment variables (Render dashboard, not .env):
   - GROQ_API_KEY = <your key>
   - AUTH_TOKEN = <your token>
6. Deploy. First build takes 5-10 minutes (torch is large).
7. Once live, visit `https://your-app.onrender.com/`.

## After deploy

- [ ] Open the URL. Password prompt appears.
- [ ] Enter AUTH_TOKEN. Widget loads.
- [ ] Ask a test question. Answer comes back with citations.
- [ ] Check Render logs for errors.
- [ ] Run eval against the deployed URL (edit eval.py's ask() to use requests, or skip).

## Important: ChromaDB and the deployment

The vector DB (chroma_db/) is NOT in git. On a fresh deploy the server won't
find the `ranfy_gte` collection and will crash. Two options:

A) Commit chroma_db/ to the repo (simpler, bloats the repo).
B) Add a build step that runs `python src/build_gte.py` before start
   (better, but slow — model download + embedding each deploy).

For a demo, use option A. For production, use a persistent volume.

## Cost expectations

- Render free tier: server sleeps after 15 min idle. Cold start ~30s.
- Render paid tier ($7/mo): always on.
- Groq: pay-per-token. A typical query (~1.5k tokens in, 300 out) costs
  fractions of a cent. 1000 questions ≈ $0.20–0.50 depending on model.

## Rebuilding after doc changes

When the client adds or updates documents:

1. Drop new PDFs into `data/pdfs/` (or wherever `build_gte.py` points).
2. Delete the old collection:
   `python -c "import chromadb; c=chromadb.PersistentClient(path='chroma_db'); c.delete_collection('ranfy_gte')"`
3. Rebuild: `python src/build_gte.py`
4. Commit the updated chroma_db/ (if using option A above).
5. Redeploy.