import time, uuid
from fastapi import APIRouter, HTTPException
from llm import get_context_size, LLM_MODEL
from database import db_list_chats, db_create_chat, db_get_chat, db_delete_chat, db_update_model
from schemas import CreateChat, PatchChat
from state import SESSION_ID

router = APIRouter()


@router.get("/chats")
def list_chats():
    return {"chats": db_list_chats()}


@router.post("/chats")
def create_chat(body: CreateChat):
    model = body.model or LLM_MODEL
    ctx   = get_context_size(model)
    chat_id = str(uuid.uuid4())
    now = int(time.time())
    db_create_chat(chat_id, model, ctx, SESSION_ID, now)
    return {"id": chat_id, "title": "New Chat", "model": model,
            "tokens_used": 0, "context_size": ctx, "session_id": SESSION_ID,
            "created_at": now, "updated_at": now}


@router.get("/chats/{chat_id}")
def get_chat(chat_id: str):
    chat, messages = db_get_chat(chat_id)
    if not chat:
        raise HTTPException(404, "Chat not found")
    return {
        "chat": chat,
        "messages": messages,
    }


@router.delete("/chats/{chat_id}")
def delete_chat(chat_id: str):
    db_delete_chat(chat_id)
    return {"ok": True}


@router.patch("/chats/{chat_id}")
def patch_chat(chat_id: str, body: PatchChat):
    ctx = get_context_size(body.model)
    db_update_model(chat_id, body.model, ctx)
    return {"ok": True, "model": body.model, "context_size": ctx}
