from __future__ import annotations

from typing import List, Tuple
from app.rag.embedder import E5Embedder
from app.rag.index import FaissStore, Chunk


class Retriever:
    def __init__(self, store: FaissStore, embedder: E5Embedder):
        self.store = store
        self.embedder = embedder

    def retrieve(self, query: str, top_k: int) -> List[Tuple[Chunk, float]]:
        qv = self.embedder.embed_query(query)  # (1, dim)
        return self.store.search(qv, top_k=top_k)
