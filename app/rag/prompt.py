from __future__ import annotations

from typing import List, Dict, Tuple
from app.config import settings
from app.rag.index import Chunk


SYSTEM_PROMPT = """\
Ты — чат-бот службы поддержки клиентов компании.
Отвечай по-русски, кратко и по делу, но достаточно информативно.
Используй ТОЛЬКО информацию из контекста базы знаний.
Если в контексте нет ответа, честно скажи, что информации нет, и попроси уточнения/предложи связаться с оператором.
Не выдумывай факты и не ссылайся на "внутренние источники", если их нет в контексте.
"""


def build_messages(
    user_message: str,
    history: List[Dict[str, str]],
    retrieved: List[Tuple[Chunk, float]],
) -> List[Dict[str, str]]:
    context_parts = []
    total = 0
    for ch, score in retrieved:
        part = f"[Источник: {ch.source} | chunk={ch.chunk_id} | score={score:.3f}]\n{ch.text.strip()}\n"
        total += len(part)
        if total > settings.MAX_CONTEXT_CHARS:
            break
        context_parts.append(part)

    context = "\n---\n".join(context_parts).strip()
    user_with_context = f"""\
Контекст из базы знаний:
{context if context else "(контекст не найден)"}

Вопрос пользователя:
{user_message}
"""

    msgs: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    msgs.extend(history[-2 * settings.MAX_HISTORY_TURNS :])
    msgs.append({"role": "user", "content": user_with_context})
    return msgs
