import uuid

# Generated once at server startup. Chats that have a different session_id
# in the DB were created in a previous Ollama process — their KV cache is gone.
SESSION_ID = str(uuid.uuid4())
