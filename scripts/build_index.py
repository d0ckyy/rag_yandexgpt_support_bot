from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple
import re

import numpy as np
import faiss
from tqdm import tqdm
from pypdf import PdfReader
from docx import Document

from app.config import settings
from app.rag.embedder import E5Embedder

DOCS_DIR = Path("knowledge_base/docs")
OUT_INDEX = Path(settings.INDEX_PATH)
OUT_CHUNKS = Path(settings.CHUNKS_PATH)


def read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    parts = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n".join(parts)


def read_docx(path: Path) -> str:
    doc = Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs if p.text)


def normalize(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def load_documents(root: Path) -> List[Tuple[str, str]]:
    docs: List[Tuple[str, str]] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        try:
            if ext in [".txt", ".md"]:
                text = read_txt(p)
            elif ext == ".pdf":
                text = read_pdf(p)
            elif ext == ".docx":
                text = read_docx(p)
            else:
                continue
            text = normalize(text)
            if text.strip():
                docs.append((str(p).replace("\\", "/"), text))
        except Exception as e:
            print(f"[WARN] skip {p}: {e}")
    return docs


def chunk_text(text: str, chunk_size: int = 700, overlap: int = 150) -> List[str]:
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        window = text[start:end]
        cut = window.rfind("\n")
        if cut > int(chunk_size * 0.6):
            end = start + cut
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = max(end - overlap, end)
    return chunks


def main() -> None:
    docs = load_documents(DOCS_DIR)
    if not docs:
        raise SystemExit(f"Нет документов в {DOCS_DIR}. Добавьте файлы и повторите.")

    chunk_records = []
    for src, text in docs:
        for ch in chunk_text(text):
            chunk_records.append({"source": src, "text": ch})

    embedder = E5Embedder(settings.EMBEDDING_MODEL)

    vectors = []
    batch = []
    for rec in tqdm(chunk_records, desc="Embedding"):
        batch.append(rec["text"])
        if len(batch) >= 32:
            vectors.append(embedder.embed_passages(batch))
            batch = []
    if batch:
        vectors.append(embedder.embed_passages(batch))

    X = np.vstack(vectors).astype(np.float32)
    n, dim = X.shape
    print(f"Chunks: {n}, dim: {dim}")

    index = faiss.IndexFlatIP(dim)
    index.add(X)

    OUT_INDEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_CHUNKS.parent.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(OUT_INDEX))

    with open(OUT_CHUNKS, "w", encoding="utf-8") as f:
        for i, rec in enumerate(chunk_records):
            obj = {"chunk_id": i, "source": rec["source"], "text": rec["text"]}
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Saved: {OUT_INDEX} and {OUT_CHUNKS}")


if __name__ == "__main__":
    main()
