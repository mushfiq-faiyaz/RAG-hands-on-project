import sys
sys.path.insert(0, "src")

import config

import json
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import session
from pipeline import ask, ask_stream

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


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
def chat(req: ChatRequest):
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
def chat_stream(req: ChatRequest):
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