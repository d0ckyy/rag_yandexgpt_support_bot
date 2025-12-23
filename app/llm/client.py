from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import httpx

from app.config import settings


@dataclass
class LLMResponse:
    content: str


class BaseLLM:
    async def chat(self, messages: List[Dict[str, str]]) -> LLMResponse:
        raise NotImplementedError


class OllamaLLM(BaseLLM):
    def __init__(self, base_url: str, model: str, timeout_s: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s

    async def chat(self, messages: List[Dict[str, str]]) -> LLMResponse:
        url = f"{self.base_url}/api/chat"
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            r = await client.post(url, json=payload)
            r.raise_for_status()
            data = r.json()
        content = (data.get("message") or {}).get("content") or ""
        return LLMResponse(content=content.strip())


class OpenAICompatLLM(BaseLLM):
    def __init__(self, base_url: str, model: str, timeout_s: float = 120.0, api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.api_key = api_key

    async def chat(self, messages: List[Dict[str, str]]) -> LLMResponse:
        url = f"{self.base_url}/chat/completions"
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
        }
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            r = await client.post(url, json=payload, headers=headers)
            r.raise_for_status()
            data = r.json()
        choices = data.get("choices") or []
        if not choices:
            return LLMResponse(content="")
        content = (choices[0].get("message") or {}).get("content") or ""
        return LLMResponse(content=content.strip())


def build_llm() -> BaseLLM:
    if settings.LLM_BACKEND.lower() == "openai_compat":
        return OpenAICompatLLM(settings.OPENAI_COMPAT_URL, settings.OPENAI_COMPAT_MODEL)
    return OllamaLLM(settings.OLLAMA_URL, settings.OLLAMA_MODEL)
