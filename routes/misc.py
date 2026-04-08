import requests
from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from llm import LLM_MODEL, OLLAMA_URL
from state import SESSION_ID
from retrieval import _article_chunks

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


@router.get("/titles")
def search_titles(q: str = ""):
    q = q.strip()
    if len(q) < 1:
        return {"titles": []}
    ql = q.lower()
    results = sorted(
        (t for t in _article_chunks if ql in t.lower()),
        key=lambda t: (not t.lower().startswith(ql), t.lower()),
    )
    return {"titles": results[:10]}
