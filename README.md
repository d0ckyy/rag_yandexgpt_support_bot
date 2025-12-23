# RAG чат-бот поддержки клиентов (YandexGPT локально + FAISS + FastAPI)

Проект для курсовой по предмету "Интеллектуальные системы".
Реализует Retrieval-Augmented Generation (RAG): поиск по базе знаний через FAISS + генерация ответа локальной LLM (YandexGPT 5 Lite 8B).

## 1) Требования
- Python 3.10+ (рекомендовано 3.11)
- Для локальной модели: Ollama** (модель запускается локально и доступна по HTTP)


## 2) Установка зависимостей
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt
```

## 3) Запуск локальной LLM

1) Установите Ollama.
2) Подтяните модель (пример из сообщества; название может отличаться):
```bash
ollama pull second_constantine/yandex-gpt-5-lite:8b
```
3) Убедитесь, что Ollama запущен (обычно сервис поднимается автоматически).
Проверка:
```bash
curl http://localhost:11434/api/tags
```

## 4) Подготовка базы знаний и построение FAISS-индекса
1) Положите документы в `knowledge_base/docs/` (поддерживаются `.txt`, `.md`, `.pdf`, `.docx`).
2) Соберите индекс:
```bash
python scripts/build_index.py
```
По умолчанию будут созданы файлы:
- `data/faiss.index`
- `data/chunks.jsonl`

## 5) Запуск FastAPI
```bash
uvicorn app.main:app --reload --port 8000
```

Swagger:
- http://localhost:8000/docs

## 6) Пример запроса
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id":"demo",
    "message":"Как вернуть товар и получить возврат?",
    "top_k":5
  }'
```

Ответ:
- `answer`: сгенерированный ответ
- `sources`: список использованных фрагментов (файлы/чанки)
