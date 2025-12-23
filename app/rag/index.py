from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import json

import faiss
import numpy as np


@dataclass
class Chunk:
    chunk_id: int
    source: str
    text: str


class FaissStore:
    def __init__(self, index: faiss.Index, chunks: List[Chunk]):
        self.index = index
        self.chunks = chunks

    @classmethod
    def load(cls, index_path: str, chunks_path: str) -> "FaissStore":
        index = faiss.read_index(index_path)
        chunks: List[Chunk] = []
        with open(chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                chunks.append(Chunk(
                    chunk_id=int(obj["chunk_id"]),
                    source=obj["source"],
                    text=obj["text"]
                ))
        return cls(index=index, chunks=chunks)

    def search(self, query_vec: np.ndarray, top_k: int) -> List[Tuple[Chunk, float]]:
        scores, ids = self.index.search(query_vec, top_k)
        out: List[Tuple[Chunk, float]] = []
        for idx, score in zip(ids[0].tolist(), scores[0].tolist()):
            if idx < 0:
                continue
            out.append((self.chunks[idx], float(score)))
        return out
