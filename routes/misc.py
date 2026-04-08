import requests
from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from llm import LLM_MODEL, OLLAMA_URL
from state import SESSION_ID

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
def index_page():
    with open("index.html") as f:
        return f.read()


@router.get("/models")
def list_models():
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
        models = [m["name"] for m in r.json().get("models", [])]
        return {"models": models or [LLM_MODEL]}
    except:
        return {"models": [LLM_MODEL]}


@router.get("/session")
def get_session():
    """Returns the current server session ID. Used by the frontend to detect stale chats."""
    return {"session_id": SESSION_ID}
