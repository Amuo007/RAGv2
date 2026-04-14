import re, json, time
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from filters import generic_relevance_filter
from retrieval import embed_query, retrieve, retrieve_for_article, RELEVANCE_THRESHOLD, TOP_K
from llm import get_context_size, stream_generate, build_rag_prompt
from database import db_get_chat, db_update_model, db_save_turn
from schemas import SendMessage
from state import SESSION_ID

with open("config.json") as _f:
    _TEST_MODE = json.load(_f).get("test", False)

router = APIRouter()


@router.post("/chat")
def chat(req: SendMessage):
    t_start = time.time()

    # Load chat record
    chat, _ = db_get_chat(req.chat_id)
    if not chat:
        raise HTTPException(404, "Chat not found")

    # Reject requests for stale sessions (Ollama was restarted — KV cache is gone)
    if chat.get("session_id") != SESSION_ID:
        raise HTTPException(409, "Session expired — start a new chat to continue")

    # Update model if overridden
    model = req.model or chat["model"]
    if model != chat["model"]:
        ctx = get_context_size(model)
        db_update_model(req.chat_id, model, ctx)
        chat["model"] = model
        chat["context_size"] = ctx

    # Load stored KV cache (may be None for the first turn)
    stored_context = json.loads(chat["context"]) if chat.get("context") else None

    # Detect /search and /article commands
    message = req.message.strip()
    is_rag     = message.lower().startswith("/search")
    is_article = message.lower().startswith("/article")
    rag_query     = None
    article_title = None
    article_query = None

    if is_rag:
        rag_query = re.sub(r'^/search\s*', '', message, flags=re.IGNORECASE).strip()
        if not rag_query:
            is_rag = False
    elif is_article:
        rest = re.sub(r'^/article\s*', '', message, flags=re.IGNORECASE)
        if '|' in rest:
            article_title, article_query = [p.strip() for p in rest.split('|', 1)]
            if not article_title or not article_query:
                is_article = False
        else:
            is_article = False

    display_message = message

    # Build stats placeholders
    embed_ms = retrieve_ms = 0
    chunks = []
    is_fallback = False

    # Build the prompt for this turn
    if is_rag and rag_query:
        t0 = time.time()
        qvec = embed_query(rag_query)
        embed_ms = round((time.time() - t0) * 1000)

        t0 = time.time()
        raw_chunks = retrieve(qvec, rag_query, top_k=req.top_k or TOP_K)
        retrieve_ms = round((time.time() - t0) * 1000)

        chunks = generic_relevance_filter(rag_query, raw_chunks, RELEVANCE_THRESHOLD)
        if not chunks:
            is_fallback = True
        prompt = build_rag_prompt(rag_query, chunks, top_k=req.top_k or TOP_K) if chunks else rag_query

    elif is_article:
        t0 = time.time()
        qvec = embed_query(article_query)
        embed_ms = round((time.time() - t0) * 1000)

        t0 = time.time()
        chunks = retrieve_for_article(qvec, article_title, top_k=req.top_k or TOP_K)
        retrieve_ms = round((time.time() - t0) * 1000)

        if not chunks:
            is_fallback = True
        prompt = build_rag_prompt(article_query, chunks, top_k=req.top_k or TOP_K) if chunks else article_query

    else:
        prompt = message

    def stream():
        t_gen = time.time()
        t_first_token_ms = None
        prompt_tokens = 0
        gen_tokens    = 0
        new_kv_context = None
        full_response = []

        try:
            for token, done_data in stream_generate(model, prompt, stored_context):
                if token:
                    if t_first_token_ms is None:
                        t_first_token_ms = round((time.time() - t_gen) * 1000)
                    full_response.append(token)
                    yield token
                if done_data:
                    prompt_tokens  = done_data.get("prompt_eval_count", 0)
                    gen_tokens     = done_data.get("eval_count", 0)
                    new_kv_context = done_data.get("context")  # updated KV cache
        except Exception as e:
            err = f"\n\n[Error during generation: {e}]"
            full_response.append(err)
            yield err

        gen_ms   = round((time.time() - t_gen) * 1000)
        total_ms = round((time.time() - t_start) * 1000)

        assistant_content = "".join(full_response)
        now = int(time.time())

        # Build payloads
        stats_payload = {
            "embed_ms":      embed_ms,
            "retrieve_ms":   retrieve_ms,
            "prompt_tokens": prompt_tokens,
            "ttft_ms":       t_first_token_ms or 0,
            "gen_ms":        gen_ms,
            "gen_tokens":    gen_tokens,
            "total_ms":      total_ms,
            "context_size":  chat["context_size"],
            "is_fallback":   is_fallback,
        }
        sources_payload = [
            {"title": t, "score": round(s, 4), "text": tx,
             **( {"db_id": db_id} if _TEST_MODE else {} )}
            for t, tx, s, _, db_id in chunks
        ] if (is_rag or is_article) and not is_fallback else []

        # Determine new title
        new_title = None
        if chat["title"] == "New Chat":
            base = display_message
            if is_rag:
                base = rag_query or display_message
            elif is_article:
                base = f"{article_title}: {article_query}"
            new_title = base[:48].strip()
            if len(base) > 48:
                new_title += "…"

        tokens_used  = prompt_tokens + gen_tokens
        context_json = json.dumps(new_kv_context) if new_kv_context else chat.get("context")

        db_save_turn(
            req.chat_id,
            display_message,
            is_rag or is_article,
            assistant_content,
            sources_payload,
            stats_payload,
            tokens_used,
            new_title,
            context_json,
            SESSION_ID,
            now,
        )

        stats_payload["sources"]   = sources_payload
        stats_payload["new_title"] = new_title
        stats_payload["chat_id"]   = req.chat_id
        stats_payload["is_rag"]    = is_rag or is_article
        yield f"\n\n[STATS]{json.dumps(stats_payload)}"

    return StreamingResponse(stream(), media_type="text/plain")
