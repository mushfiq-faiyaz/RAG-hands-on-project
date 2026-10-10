import sqlite3
import uuid

DB_PATH = "sessions.db"


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT NOT NULL,
            turn_index INTEGER NOT NULL,
            role       TEXT NOT NULL,
            content    TEXT NOT NULL
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_session ON sessions(session_id, turn_index)"
    )
    return conn


def new_session_id():
    return str(uuid.uuid4())


def get_history(session_id):
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT role, content FROM sessions "
            "WHERE session_id = ? ORDER BY turn_index",
            (session_id,),
        ).fetchall()
    finally:
        conn.close()
    return [{"role": r[0], "content": r[1]} for r in rows]


def append_turn(session_id, question, answer):
    conn = _connect()
    try:
        next_idx = conn.execute(
            "SELECT COALESCE(MAX(turn_index), -1) + 1 FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()[0]
        conn.execute(
            "INSERT INTO sessions VALUES (?, ?, ?, ?)",
            (session_id, next_idx, "user", question),
        )
        conn.execute(
            "INSERT INTO sessions VALUES (?, ?, ?, ?)",
            (session_id, next_idx + 1, "assistant", answer),
        )
        conn.commit()
    finally:
        conn.close()


def clear_session(session_id):
    conn = _connect()
    try:
        conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        conn.commit()
    finally:
        conn.close()