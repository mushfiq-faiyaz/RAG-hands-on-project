import sys
sys.path.insert(0, "src")

import config

import json
from pathlib import Path
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

import session
from pipeline import ask, ask_stream

limiter = Limiter(key_func=get_remote_address)

app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@app.on_event("startup")
def on_startup():
    deleted = session.cleanup_old_sessions()
    print(f"Cleaned up {deleted} stale session rows")


@app.get("/")
def index():
    return FileResponse(PROJECT_ROOT / "widget.html")


def check_auth(x_auth_token: str | None):
    if x_auth_token != config.AUTH_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing auth token")


class ChatRequest(BaseModel):
    question: str
    session_id: str | None = None


class Source(BaseModel):
    source: str
    page: int
    index: int | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    session_id: str


@app.post("/chat", response_model=ChatResponse)
@limiter.limit("20/minute")
def chat(request: Request, req: ChatRequest, x_auth_token: str | None = Header(default=None)):
    check_auth(x_auth_token)
    session_id = req.session_id or session.new_session_id()
    history = session.get_history(session_id)
    result = ask(req.question, history)
    session.append_turn(session_id, req.question, result["answer"])
    return ChatResponse(
        answer=result["answer"],
        sources=result["sources"],
        session_id=session_id,
    )


@app.post("/chat/stream")
@limiter.limit("20/minute")
def chat_stream(request: Request, req: ChatRequest, x_auth_token: str | None = Header(default=None)):
    check_auth(x_auth_token)
    session_id = req.session_id or session.new_session_id()
    history = session.get_history(session_id)

    def event_generator():
        collected = ""
        for event in ask_stream(req.question, history):
            if event["type"] == "sources":
                payload = {"type": "sources", "data": event["data"], "session_id": session_id}
                yield f"data: {json.dumps(payload)}\n\n"
            elif event["type"] == "token":
                collected += event["data"]
                yield f"data: {json.dumps(event)}\n\n"
            elif event["type"] == "done":
                session.append_turn(session_id, req.question, collected)
                yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")