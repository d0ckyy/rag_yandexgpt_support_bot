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
- `manager_handoff`: `true`, если вопрос не найден в базе знаний и нужно подключить менеджера
- `handoff_reason`: причина передачи, например `no_relevant_kb`
- `manager_chat_id`: ID созданного чата менеджера (если был выполнен handoff)

Если нужно изменить порог релевантности или текст ответа при передаче менеджеру:
- `RAG_MIN_SCORE` — минимальная оценка совпадения (cosine similarity), ниже будет передача менеджеру
- `MANAGER_HANDOFF_MESSAGE` — текст сообщения пользователю при передаче

## 7) API менеджера и простая CRM

### 7.1) Чат пользователя с менеджером
Отправка сообщения менеджеру (создаст чат, если его ещё нет):
```bash
curl -X POST http://localhost:8000/manager/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id":"demo",
    "message":"Хочу связаться с менеджером",
    "chat_id": null
  }'
```

Получить чат и новые сообщения:
```bash
curl http://localhost:8000/manager/chat/{chat_id}
```

### 7.2) Простая CRM для менеджера
Список чатов (можно `?status=open` или `?status=closed`):
```bash
curl http://localhost:8000/crm/chats
```

Получить детали чата:
```bash
curl http://localhost:8000/crm/chats/{chat_id}
```

Ответить пользователю:
```bash
curl -X POST http://localhost:8000/crm/chats/{chat_id}/reply \
  -H "Content-Type: application/json" \
  -d '{
    "message":"Здравствуйте! Чем могу помочь?"
  }'
```

Закрыть чат:
```bash
curl -X POST http://localhost:8000/crm/chats/{chat_id}/close
```
