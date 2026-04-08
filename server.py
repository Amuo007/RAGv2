from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from database import init_chat_db
from llm import get_context_size, LLM_MODEL
from state import SESSION_ID
from routes import misc, chats, chat

app = FastAPI()

# Startup
print(f"  Context window: {get_context_size(LLM_MODEL)}")
print(f"  Session ID: {SESSION_ID}")
init_chat_db()
print("Ready.")

# Routers
app.include_router(misc.router)
app.include_router(chats.router)
app.include_router(chat.router)

# Static files (must be after routes)
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
