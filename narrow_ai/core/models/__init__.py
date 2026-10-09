"""Model backend interface and implementations."""
from narrow_ai.core.models.base import ModelBackend, StubBackend
from narrow_ai.core.models.http import HTTPBackend, OpenAIBackend, OllamaBackend

__all__ = [
    "ModelBackend",
    "StubBackend",
    "HTTPBackend",
    "OpenAIBackend",
    "OllamaBackend",
]
