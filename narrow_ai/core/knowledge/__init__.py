"""Knowledge layer — pluggable knowledge providers.

The core AI doesn't hardcode any specific domain knowledge.
Instead it depends on a KnowledgeProvider interface.
Later, an application can plug in:
  - DentalKnowledgeProvider
  - EducationKnowledgeProvider
  - etc.

Multiple providers can be combined via KnowledgeHub.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class KnowledgeResult:
    """A single result from a knowledge lookup."""
    content: str
    source: str
    score: float = 0.0
    metadata: dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class KnowledgeProvider(ABC):
    """Abstract interface for any knowledge source."""

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> list[KnowledgeResult]:
        """Search the knowledge store and return top_k results."""

    @abstractmethod
    def get(self, key: str) -> str | None:
        """Direct lookup by key/id."""

    def name(self) -> str:
        return self.__class__.__name__


class DocumentProvider(KnowledgeProvider):
    """In-memory document store with simple keyword search."""

    def __init__(self, docs: list[tuple[str, str]] | None = None):
        # docs: list of (content, source) tuples
        self._docs: list[tuple[str, str]] = docs or []

    def add_document(self, content: str, source: str) -> None:
        self._docs.append((content, source))

    def search(self, query: str, top_k: int = 5) -> list[KnowledgeResult]:
        query_lower = query.lower()
        query_words = [w.strip(".,?!") for w in query_lower.split() if len(w) > 2]
        results: list[KnowledgeResult] = []
        for content, source in self._docs:
            content_lower = content.lower()
            # 1. Full substring match
            if query_lower in content_lower:
                score = 1.0
            else:
                # 2. Word-level overlap with substring matching
                #    (handles inflections: "refund" matches "refunds")
                doc_words = [w.strip(".,?!") for w in content_lower.split()]
                matches = 0
                for qw in query_words:
                    for dw in doc_words:
                        if qw in dw or dw in qw:
                            matches += 1
                            break
                score = matches / max(len(query_words), 1) if query_words else 0.0
            if score > 0:
                results.append(KnowledgeResult(
                    content=content[:500],  # truncate for context
                    source=source,
                    score=score,
                ))
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def get(self, key: str) -> str | None:
        for content, source in self._docs:
            if key == source or key in content:
                return content
        return None


class KnowledgeHub(KnowledgeProvider):
    """Combine multiple knowledge providers.

    Searches all providers and merges/ranks results.
    """

    def __init__(self, providers: list[KnowledgeProvider] | None = None):
        self.providers: list[KnowledgeProvider] = providers or []

    def register(self, provider: KnowledgeProvider) -> None:
        self.providers.append(provider)

    def search(self, query: str, top_k: int = 5) -> list[KnowledgeResult]:
        all_results: list[KnowledgeResult] = []
        for provider in self.providers:
            results = provider.search(query, top_k=top_k)
            for r in results:
                r.metadata["provider"] = provider.name()
            all_results.extend(results)
        all_results.sort(key=lambda r: r.score, reverse=True)
        return all_results[:top_k]

    def get(self, key: str) -> str | None:
        for provider in self.providers:
            result = provider.get(key)
            if result:
                return result
        return None


# Import vector store at the bottom to avoid circular dependency
from narrow_ai.core.knowledge.vector_store import VectorStore, HybridStore

__all__ = [
    "KnowledgeProvider",
    "KnowledgeResult",
    "DocumentProvider",
    "KnowledgeHub",
    "VectorStore",
    "HybridStore",
]
