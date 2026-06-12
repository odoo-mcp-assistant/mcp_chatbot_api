"""Pydantic request/response models for the HTTP endpoints."""

from pydantic import BaseModel, Field


class MessageRequest(BaseModel):
    message: str = Field(..., min_length=1, description="The user's chat message.")


class MessageResponse(BaseModel):
    reply: str
    summarized: bool = False
    # True when this turn pushed the caller past ~90% of their daily token
    # budget — the widget shows a one-time "approaching today's limit" notice
    # so the eventual block doesn't feel like a wall out of nowhere.
    usage_warning: bool = False


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


class ConversationSummary(BaseModel):
    """One row in the past-conversations sidebar (logged-in users only)."""
    id: int
    title: str                       # preview built from the first user message
    created_at: str = ""             # ISO-8601 UTC string ("...Z"), formatted client-side
    message_count: int = 0
    state: str = "open"              # 'open' (the live session) | 'closed'
    rating: str = "none"             # 'none' | 'bad' | 'neutral' | 'good'


class ConversationListResponse(BaseModel):
    conversations: list[ConversationSummary] = []


class ConversationDetailResponse(BaseModel):
    id: int
    status: str = "ok"               # 'ok' | 'not_found'
    state: str = "closed"
    messages: list[HistoryMessage] = []

