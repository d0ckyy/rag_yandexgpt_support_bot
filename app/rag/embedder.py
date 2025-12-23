from __future__ import annotations

from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer


class E5Embedder:
    """
    Эмбеддинги через E5 (multilingual-e5-*).
    Для корректной работы формируем префиксы: "query:" и "passage:".
    """
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_passages(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        prefixed = [f"passage: {t}" for t in texts]
        vecs = self.model.encode(
            prefixed,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return np.asarray(vecs, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        vec = self.model.encode(
            [f"query: {text}"],
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return np.asarray(vec, dtype=np.float32)
