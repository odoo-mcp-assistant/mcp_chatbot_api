"""Pydantic request/response models for the HTTP endpoints."""

from pydantic import BaseModel, Field


class MessageRequest(BaseModel):
    message: str = Field(..., min_length=1, description="The user's chat message.")


class MessageResponse(BaseModel):
    reply: str
    summarized: bool = False


class HistoryMessage(BaseModel):
    role: str
    content: str


class HistoryResponse(BaseModel):
    status: str  # 'open' | 'closed' | 'not_found'
    messages: list[HistoryMessage] = []


class CloseRequest(BaseModel):
    rating: str | None = Field(None, description="One of: bad, neutral, good")
    feedback: str | None = None


class CloseResponse(BaseModel):
    status: str = "closed"


class InfoResponse(BaseModel):
    bot_name: str
    status: str
    first_name: str = ""
    is_authenticated: bool = False

