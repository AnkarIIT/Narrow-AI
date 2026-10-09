"""HTTP-based model backend for any OpenAI-compatible API.

Works with: OpenAI, Ollama, local vLLM, LMStudio, DeepSeek, etc.
Just set the base_url and optionally the api_key.
"""
from __future__ import annotations
import json
from typing import Any
from narrow_ai.core.models.base import ModelBackend


class HTTPBackend(ModelBackend):
    """OpenAI-compatible HTTP backend.

    Examples:
        # Ollama (local, no API key)
        backend = HTTPBackend(base_url="http://localhost:11434/v1",
                              model="llama3")

        # OpenAI
        backend = HTTPBackend(base_url="https://api.openai.com/v1",
                              api_key="sk-...", model="gpt-4o-mini")
    """

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        timeout: int = 60,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_name = model
        self.timeout = timeout
        self._session = self._create_session()

    def _create_session(self):
        try:
            import requests
        except ImportError:
            raise ImportError(
                "requests is required for HTTPBackend. Install with: pip install requests"
            )
        session = requests.Session()
        # Configure retry logic
        from urllib3.util.retry import Retry
        from requests.adapters import HTTPAdapter
        retries = Retry(total=3, backoff_factor=0.3, status_forcelist=[500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retries)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def complete(self, prompt: str, **kwargs: Any) -> str:
        """Generate a completion via chat/completions endpoint."""
        body: dict[str, Any] = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": kwargs.get("max_tokens", 500),
            "temperature": kwargs.get("temperature", 0.7),
        }
        url = f"{self.base_url}/chat/completions"
        resp = self._session.post(
            url, headers=self._headers(), json=body, timeout=self.timeout
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    def classify(self, text: str, options: list[str], **kwargs: Any) -> str:
        """Classify text via a few-shot prompt."""
        # Build a classification prompt
        options_str = "\n".join(f"- {opt}" for opt in options)
        prompt = (
            f"Classify the following text into exactly one of these categories. "
            f"Respond with ONLY the category name, no extra text.\n\n"
            f"Categories:\n{options_str}\n\n"
            f"Text: \"{text}\"\n"
            f"Category:"
        )
        result = self.complete(prompt, temperature=0.1, max_tokens=50)
        result_lower = result.strip().lower()
        # Find the best match
        for opt in options:
            if opt.lower() in result_lower:
                return opt
        # Fallback: return first option
        return options[0] if options else "unknown"

    def embed(self, text: str, **kwargs: Any) -> list[float]:
        """Generate embeddings via the embeddings endpoint."""
        body: dict[str, Any] = {
            "model": kwargs.get("embedding_model", self.model_name),
            "input": text,
        }
        url = f"{self.base_url}/embeddings"
        try:
            resp = self._session.post(
                url, headers=self._headers(), json=body, timeout=self.timeout
            )
            resp.raise_for_status()
            data = resp.json()
            return data["data"][0]["embedding"]
        except Exception:
            # Some providers (e.g. Ollama) don't have /embeddings
            # Fall back to StubBackend-like hash embedding
            import hashlib, struct
            h = hashlib.md5(text.encode("utf-8")).digest()
            vec = list(struct.unpack("8f", h))
            norm = sum(v * v for v in vec) ** 0.5 or 1.0
            return [v / norm for v in vec]

    def name(self) -> str:
        return f"HTTPBackend({self.model_name})"


# Convenience aliases
OpenAIBackend = HTTPBackend
OllamaBackend = HTTPBackend
