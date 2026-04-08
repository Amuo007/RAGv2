import json
import requests
from typing import Optional

# ── Config ───────────────────────────────────────────────────────────────────
with open("config.json") as f:
    _cfg = json.load(f)

OLLAMA_URL = _cfg["ollama_url"]
LLM_MODEL  = _cfg["llm_model"]

# ── Model context size cache ──────────────────────────────────────────────────
_model_ctx_cache: dict = {}

def get_context_size(model: str) -> Optional[int]:
    if model in _model_ctx_cache:
        return _model_ctx_cache[model]
    try:
        info = requests.post(f"{OLLAMA_URL}/api/show", json={"name": model}, timeout=10).json()
        data = info.get("model_info", {})
        ctx = (
            data.get("llama.context_length") or
            data.get("gemma.context_length") or
            next((v for k, v in data.items() if "context_length" in k), None)
        )
        _model_ctx_cache[model] = ctx
        return ctx
    except Exception as e:
        print(f"  Could not fetch context size for {model}: {e}")
        _model_ctx_cache[model] = None
        return None


# ── Ollama generate streaming helper ─────────────────────────────────────────
def stream_generate(model: str, prompt: str, context: list = None):
    """Stream from Ollama /api/generate with optional KV cache context.
    Yields (token, done_data_or_None).
    done_data includes 'context' (new KV cache) and token counts.
    """
    payload = {"model": model, "prompt": prompt, "stream": True}
    if context:
        payload["context"] = context
    r = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json=payload,
        stream=True,
        timeout=120,
    )
    for line in r.iter_lines():
        if line:
            data = json.loads(line)
            token = data.get("response", "")
            if data.get("done"):
                yield token, data
            else:
                yield token, None


# ── Prompt helpers ────────────────────────────────────────────────────────────
def build_context_content(chunks: list) -> str:
    """Format retrieved chunks into a context block."""
    return "\n\n---\n\n".join(f"[{title}]\n{text}" for title, text, *_ in chunks)


def build_rag_prompt(rag_query: str, chunks: list) -> str:
    """Build a self-contained prompt that injects Wikipedia context inline."""
    ctx = build_context_content(chunks)
    return (
        f"Using the following Wikipedia context, answer the question concisely.\n"
        f"Do not add information not present in the context.\n\n"
        f"CONTEXT:\n{ctx}\n\n"
        f"QUESTION: {rag_query}"
    )
