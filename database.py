import sqlite3
import json

CHAT_DB = "chat_history.db"


def _chat_conn():
    conn = sqlite3.connect(CHAT_DB)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn


def init_chat_db():
    conn = _chat_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            model TEXT NOT NULL,
            tokens_used INTEGER DEFAULT 0,
            context_size INTEGER,
            context TEXT,
            session_id TEXT,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            is_rag INTEGER DEFAULT 0,
            sources TEXT,
            stats TEXT,
            created_at INTEGER NOT NULL,
            FOREIGN KEY (chat_id) REFERENCES chats(id) ON DELETE CASCADE
        )
    """)
    # Migrations for existing databases
    for col, defn in [("context", "TEXT"), ("session_id", "TEXT")]:
        try:
            conn.execute(f"ALTER TABLE chats ADD COLUMN {col} {defn}")
        except Exception:
            pass  # column already exists
    conn.commit()
    conn.close()


def db_list_chats() -> list:
    """Return all chats ordered by updated_at DESC."""
    conn = _chat_conn()
    rows = conn.execute(
        "SELECT id, title, model, tokens_used, context_size, session_id, created_at, updated_at "
        "FROM chats ORDER BY updated_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def db_create_chat(chat_id: str, model: str, ctx, session_id: str, now: int) -> None:
    """Insert a new chat record."""
    conn = _chat_conn()
    conn.execute(
        "INSERT INTO chats (id, title, model, tokens_used, context_size, context, session_id, created_at, updated_at) "
        "VALUES (?, ?, ?, 0, ?, NULL, ?, ?, ?)",
        (chat_id, "New Chat", model, ctx, session_id, now, now)
    )
    conn.commit()
    conn.close()


def db_get_chat(chat_id: str):
    """Return (chat_dict, messages_list) or (None, None) if not found."""
    conn = _chat_conn()
    chat = conn.execute("SELECT * FROM chats WHERE id = ?", (chat_id,)).fetchone()
    if not chat:
        conn.close()
        return None, None
    msgs = conn.execute(
        "SELECT id, role, content, is_rag, sources, stats, created_at "
        "FROM messages WHERE chat_id = ? ORDER BY created_at, id",
        (chat_id,)
    ).fetchall()
    conn.close()
    return dict(chat), [dict(m) for m in msgs]


def db_delete_chat(chat_id: str) -> None:
    conn = _chat_conn()
    conn.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
    conn.commit()
    conn.close()


def db_update_model(chat_id: str, model: str, ctx) -> None:
    """Update the model and context_size for a chat."""
    conn = _chat_conn()
    conn.execute(
        "UPDATE chats SET model = ?, context_size = ? WHERE id = ?",
        (model, ctx, chat_id)
    )
    conn.commit()
    conn.close()


def db_save_turn(
    chat_id: str,
    display_message: str,
    is_rag: bool,
    assistant_content: str,
    sources_payload: list,
    stats_payload: dict,
    tokens_used: int,
    new_title,          # str or None
    context_json,       # str (JSON-encoded KV cache) or None
    session_id: str,
    now: int,
) -> None:
    """Save user message, assistant message, KV cache, and update chat stats."""
    db = _chat_conn()

    # Save user message
    db.execute(
        "INSERT INTO messages (chat_id, role, content, is_rag, sources, stats, created_at) "
        "VALUES (?, 'user', ?, ?, NULL, NULL, ?)",
        (chat_id, display_message, 1 if is_rag else 0, now)
    )
    # Save assistant message
    db.execute(
        "INSERT INTO messages (chat_id, role, content, is_rag, sources, stats, created_at) "
        "VALUES (?, 'assistant', ?, 0, ?, ?, ?)",
        (chat_id, assistant_content, json.dumps(sources_payload), json.dumps(stats_payload), now + 1)
    )
    # Update chat — store KV cache + session_id
    if new_title:
        db.execute(
            "UPDATE chats SET updated_at = ?, tokens_used = ?, title = ?, context = ?, session_id = ? WHERE id = ?",
            (now, tokens_used, new_title, context_json, session_id, chat_id)
        )
    else:
        db.execute(
            "UPDATE chats SET updated_at = ?, tokens_used = ?, context = ?, session_id = ? WHERE id = ?",
            (now, tokens_used, context_json, session_id, chat_id)
        )
    db.commit()
    db.close()
