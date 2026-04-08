from pydantic import BaseModel
from typing import Optional

class CreateChat(BaseModel):
    model: Optional[str] = None

class PatchChat(BaseModel):
    model: str

class SendMessage(BaseModel):
    chat_id: str
    message: str
    model: Optional[str] = None
