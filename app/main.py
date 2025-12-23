from __future__ import annotations

from typing import Dict, List, Any, Optional
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
