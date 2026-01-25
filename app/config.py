from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # --- LLM ---
    LLM_BACKEND: str = "ollama"  # ollama | openai_compat

    OLLAMA_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "second_constantine/yandex-gpt-5-lite:8b"

    # OpenAI-compatible server (vLLM, llama.cpp server with /v1, etc.)
    OPENAI_COMPAT_URL: str = "http://localhost:8001/v1"
    OPENAI_COMPAT_MODEL: str = "yandexgpt"

    # --- RAG ---
    INDEX_PATH: str = "data/faiss.index"
    CHUNKS_PATH: str = "data/chunks.jsonl"
    EMBEDDING_MODEL: str = "intfloat/multilingual-e5-base"
    TOP_K_DEFAULT: int = 5
    RAG_MIN_SCORE: float = 0.75  # если ниже, считаем, что ответа в базе знаний нет

    # Ограничения контекста, чтобы не переполнить промпт
    MAX_CONTEXT_CHARS: int = 12000
    MAX_HISTORY_TURNS: int = 8  # сколько последних реплик хранить на сессию

    MANAGER_HANDOFF_MESSAGE: str = "Извините, не нашёл ответ в базе знаний. Подключаю менеджера к чату."

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
