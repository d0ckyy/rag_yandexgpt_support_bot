from __future__ import annotations

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone
from uuid import uuid4
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.config import settings
from app.llm.client import build_llm
from app.rag.embedder import E5Embedder
from app.rag.index import FaissStore
from app.rag.retriever import Retriever
from app.rag.prompt import build_messages


app = FastAPI(title="RAG Support Bot (YandexGPT local)", version="1.0.0")

llm = build_llm()
store: Optional[FaissStore] = None
retriever: Optional[Retriever] = None

SESSIONS: Dict[str, List[Dict[str, str]]] = {}
MANAGER_CHATS: Dict[str, Dict[str, Any]] = {}


class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Идентификатор диалога (можно UUID)")
    message: str = Field(..., min_length=1, description="Сообщение пользователя")
    top_k: int = Field(default=settings.TOP_K_DEFAULT, ge=1, le=20)


class SourceItem(BaseModel):
    source: str
    chunk_id: int
    score: float
    text_preview: str


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceItem]
    manager_handoff: bool = False
    handoff_reason: Optional[str] = None
    manager_chat_id: Optional[str] = None


class ManagerMessage(BaseModel):
    role: str
    content: str
    ts: datetime


class ManagerChat(BaseModel):
    chat_id: str
    session_id: str
    status: str
    created_at: datetime
    updated_at: datetime
    messages: List[ManagerMessage]


class ManagerChatListItem(BaseModel):
    chat_id: str
    session_id: str
    status: str
    created_at: datetime
    updated_at: datetime
    last_message: Optional[str] = None


class ManagerChatListResponse(BaseModel):
    chats: List[ManagerChatListItem]


class ManagerChatRequest(BaseModel):
    session_id: str = Field(..., description="Идентификатор диалога (можно UUID)")
    message: str = Field(..., min_length=1, description="Сообщение пользователя менеджеру")
    chat_id: Optional[str] = Field(default=None, description="ID чата менеджера, если уже создан")


class ManagerReplyRequest(BaseModel):
    message: str = Field(..., min_length=1, description="Ответ менеджера")


def _needs_manager(hits: List[Tuple[Any, float]]) -> bool:
    if not hits:
        return True
    best_score = max(score for _, score in hits)
    return best_score < settings.RAG_MIN_SCORE


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _create_manager_chat(session_id: str) -> Dict[str, Any]:
    chat_id = str(uuid4())
    now = _now()
    chat = {
        "chat_id": chat_id,
        "session_id": session_id,
        "status": "open",
        "created_at": now,
        "updated_at": now,
        "messages": [],
    }
    MANAGER_CHATS[chat_id] = chat
    return chat


def _find_open_chat_by_session(session_id: str) -> Optional[Dict[str, Any]]:
    for chat in MANAGER_CHATS.values():
        if chat["session_id"] == session_id and chat["status"] == "open":
            return chat
    return None


def _append_manager_message(chat: Dict[str, Any], role: str, content: str) -> None:
    msg = {"role": role, "content": content, "ts": _now()}
    chat["messages"].append(msg)
    chat["updated_at"] = msg["ts"]


@app.on_event("startup")
def _startup() -> None:
    global store, retriever
    try:
        store = FaissStore.load(settings.INDEX_PATH, settings.CHUNKS_PATH)
    except Exception as e:
        store = None
        retriever = None
        print(f"[WARN] Could not load FAISS store: {e}")
        return

    embedder = E5Embedder(settings.EMBEDDING_MODEL)
    retriever = Retriever(store=store, embedder=embedder)


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "rag_index_loaded": store is not None,
        "llm_backend": settings.LLM_BACKEND,
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    if retriever is None or store is None:
        raise HTTPException(status_code=503, detail="FAISS индекс не загружен. Сначала запустите scripts/build_index.py")

    hits = retriever.retrieve(req.message, top_k=req.top_k)
    history = SESSIONS.get(req.session_id, [])
    if _needs_manager(hits):
        chat = _find_open_chat_by_session(req.session_id) or _create_manager_chat(req.session_id)
        _append_manager_message(chat, role="user", content=req.message)
        handoff_msg = settings.MANAGER_HANDOFF_MESSAGE
        history.append({"role": "user", "content": req.message})
        history.append({"role": "assistant", "content": handoff_msg})
        SESSIONS[req.session_id] = history[-2 * settings.MAX_HISTORY_TURNS :]
        return ChatResponse(
            answer=handoff_msg,
            sources=[],
            manager_handoff=True,
            handoff_reason="no_relevant_kb",
            manager_chat_id=chat["chat_id"],
        )

    messages = build_messages(req.message, history=history, retrieved=hits)
    llm_resp = await llm.chat(messages)

    history.append({"role": "user", "content": req.message})
    history.append({"role": "assistant", "content": llm_resp.content})
    SESSIONS[req.session_id] = history[-2 * settings.MAX_HISTORY_TURNS :]

    sources = [
        SourceItem(
            source=ch.source,
            chunk_id=ch.chunk_id,
            score=score,
            text_preview=(ch.text[:240] + ("..." if len(ch.text) > 240 else "")),
        )
        for ch, score in hits
    ]
    return ChatResponse(answer=llm_resp.content, sources=sources)


@app.post("/manager/chat", response_model=ManagerChat)
async def manager_chat(req: ManagerChatRequest) -> ManagerChat:
    if req.chat_id:
        chat = MANAGER_CHATS.get(req.chat_id)
        if chat is None:
            raise HTTPException(status_code=404, detail="Чат менеджера не найден")
        if chat["session_id"] != req.session_id:
            raise HTTPException(status_code=400, detail="session_id не совпадает с чатом")
        if chat["status"] != "open":
            raise HTTPException(status_code=409, detail="Чат менеджера закрыт")
    else:
        chat = _find_open_chat_by_session(req.session_id) or _create_manager_chat(req.session_id)

    _append_manager_message(chat, role="user", content=req.message)
    return ManagerChat(**chat)


@app.get("/manager/chat/{chat_id}", response_model=ManagerChat)
async def manager_chat_get(chat_id: str) -> ManagerChat:
    chat = MANAGER_CHATS.get(chat_id)
    if chat is None:
        raise HTTPException(status_code=404, detail="Чат менеджера не найден")
    return ManagerChat(**chat)


@app.get("/crm/chats", response_model=ManagerChatListResponse)
async def crm_list_chats(status: Optional[str] = None) -> ManagerChatListResponse:
    chats = list(MANAGER_CHATS.values())
    if status:
        chats = [c for c in chats if c["status"] == status]
    chats.sort(key=lambda c: c["updated_at"], reverse=True)
    items: List[ManagerChatListItem] = []
    for chat in chats:
        last_msg = chat["messages"][-1]["content"] if chat["messages"] else None
        items.append(ManagerChatListItem(
            chat_id=chat["chat_id"],
            session_id=chat["session_id"],
            status=chat["status"],
            created_at=chat["created_at"],
            updated_at=chat["updated_at"],
            last_message=last_msg,
        ))
    return ManagerChatListResponse(chats=items)


@app.get("/crm/chats/{chat_id}", response_model=ManagerChat)
async def crm_get_chat(chat_id: str) -> ManagerChat:
    chat = MANAGER_CHATS.get(chat_id)
    if chat is None:
        raise HTTPException(status_code=404, detail="Чат менеджера не найден")
    return ManagerChat(**chat)


@app.post("/crm/chats/{chat_id}/reply", response_model=ManagerChat)
async def crm_reply_chat(chat_id: str, req: ManagerReplyRequest) -> ManagerChat:
    chat = MANAGER_CHATS.get(chat_id)
    if chat is None:
        raise HTTPException(status_code=404, detail="Чат менеджера не найден")
    if chat["status"] != "open":
        raise HTTPException(status_code=409, detail="Чат менеджера закрыт")
    _append_manager_message(chat, role="manager", content=req.message)
    return ManagerChat(**chat)


@app.post("/crm/chats/{chat_id}/close", response_model=ManagerChat)
async def crm_close_chat(chat_id: str) -> ManagerChat:
    chat = MANAGER_CHATS.get(chat_id)
    if chat is None:
        raise HTTPException(status_code=404, detail="Чат менеджера не найден")
    if chat["status"] != "closed":
        chat["status"] = "closed"
        chat["updated_at"] = _now()
    return ManagerChat(**chat)
