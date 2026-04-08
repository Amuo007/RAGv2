import re, json, time
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from classifier import classify_query
from filters import generic_relevance_filter, rule_based_title_filter
from retrieval import embed_query, retrieve, RELEVANCE_THRESHOLD
from llm import get_context_size, stream_generate, build_rag_prompt
from database import db_get_chat, db_update_model, db_save_turn
from schemas import SendMessage
from state import SESSION_ID

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

    # Detect /search command
    message = req.message.strip()
    is_rag  = message.lower().startswith("/search")
    if is_rag:
        rag_query = re.sub(r'^/search\s*', '', message, flags=re.IGNORECASE).strip()
        if not rag_query:
            is_rag = False
        display_message = message
    else:
        rag_query       = None
        display_message = message

    # Build stats placeholders
    embed_ms = classify_ms = retrieve_ms = 0
    query_type = "chat"
    entity_name = ""
    chunks = []
    is_fallback = False

    # Build the prompt for this turn
    if is_rag and rag_query:
        # Embed + retrieve
        t0 = time.time()
        qvec = embed_query(rag_query)
        embed_ms = round((time.time() - t0) * 1000)

        t0 = time.time()
        query_type, entity_name = classify_query(rag_query)
        classify_ms = round((time.time() - t0) * 1000)

        t0 = time.time()
        raw_chunks = retrieve(qvec, rag_query)
        retrieve_ms = round((time.time() - t0) * 1000)

        if query_type == "generic":
            chunks = generic_relevance_filter(rag_query, raw_chunks, RELEVANCE_THRESHOLD)
            if not chunks:
                is_fallback = True
        else:
            filtered = rule_based_title_filter(rag_query, raw_chunks)
            if filtered:
                chunks = filtered
            else:
                is_fallback = True

        if chunks:
            # Inject Wikipedia context inline — gets baked into Ollama's KV cache
            prompt = build_rag_prompt(rag_query, chunks)
        else:
            # Fallback: let model answer from its training data / existing KV cache
            prompt = rag_query
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
            "classify_ms":   classify_ms,
            "retrieve_ms":   retrieve_ms,
            "prompt_tokens": prompt_tokens,
            "ttft_ms":       t_first_token_ms or 0,
            "gen_ms":        gen_ms,
            "gen_tokens":    gen_tokens,
            "total_ms":      total_ms,
            "context_size":  chat["context_size"],
            "is_fallback":   is_fallback,
            "query_type":    query_type,
            "entity_name":   entity_name,
        }
        sources_payload = [
            {"title": t, "score": round(s, 4), "text": tx}
            for t, tx, s, *_ in chunks
        ] if is_rag and not is_fallback else []

        # Determine new title
        new_title = None
        if chat["title"] == "New Chat":
            base = display_message
            if is_rag:
                base = rag_query or display_message
            new_title = base[:48].strip()
            if len(base) > 48:
                new_title += "…"

        tokens_used  = prompt_tokens + gen_tokens
        context_json = json.dumps(new_kv_context) if new_kv_context else chat.get("context")

        db_save_turn(
            req.chat_id,
            display_message,
            is_rag,
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
        stats_payload["is_rag"]    = is_rag
        yield f"\n\n[STATS]{json.dumps(stats_payload)}"

    return StreamingResponse(stream(), media_type="text/plain")
