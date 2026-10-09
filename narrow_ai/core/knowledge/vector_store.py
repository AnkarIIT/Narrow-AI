"""Vector store for semantic knowledge retrieval.

Uses the model backend's embed() method to create dense vector
representations, then performs cosine similarity search.
Falls back to the stub backend's hash-based embeddings when no
real model is available.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np

from narrow_ai.core.knowledge import KnowledgeProvider, KnowledgeResult
from narrow_ai.core.models.base import ModelBackend


@dataclass
class VectorDoc:
    content: str
    source: str
    embedding: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


class VectorStore(KnowledgeProvider):
    """In-memory vector store with cosine similarity search.

    Usage:
        store = VectorStore(backend=StubBackend())
        store.add("Your order ships in 3-5 days.", source="policy")
        results = store.search("How long does shipping take?")
    """

    def __init__(self, backend: ModelBackend, dimension: int = 8):
        self.backend = backend
        self.dimension = dimension
        self._docs: list[VectorDoc] = []

    def add(self, content: str, source: str = "unknown", **metadata: Any) -> None:
        """Add a document with its embedding."""
        emb = self.backend.embed(content)
        self._docs.append(VectorDoc(
            content=content,
            source=source,
            embedding=np.array(emb, dtype=np.float32),
            metadata=metadata,
        ))

    def add_batch(self, docs: list[tuple[str, str]], **shared_metadata: Any) -> None:
        """Add multiple documents at once."""
        for content, source in docs:
            self.add(content, source, **shared_metadata)

    def search(self, query: str, top_k: int = 5) -> list[KnowledgeResult]:
        if not self._docs:
            return []

        query_emb = np.array(self.backend.embed(query), dtype=np.float32)
        # Normalize
        q_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)

        scores: list[tuple[float, VectorDoc]] = []
        for doc in self._docs:
            # Cosine similarity
            d_norm = doc.embedding / (np.linalg.norm(doc.embedding) + 1e-8)
            sim = float(np.dot(q_norm, d_norm))
            scores.append((sim, doc))

        # Sort by similarity descending
        scores.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, doc in scores[:top_k]:
            if score > 0.0:  # only return positive matches
                results.append(KnowledgeResult(
                    content=doc.content[:500],
                    source=doc.source,
                    score=score,
                    metadata={**doc.metadata, "provider": self.name()},
                ))
        return results

    def get(self, key: str) -> str | None:
        for doc in self._docs:
            if key == doc.source or key in doc.content:
                return doc.content
        return None

    def count(self) -> int:
        return len(self._docs)

    def clear(self) -> None:
        self._docs.clear()

    def name(self) -> str:
        return "VectorStore"


class HybridStore(KnowledgeProvider):
    """Combines keyword (DocumentProvider) + semantic (VectorStore) search.

    Runs both searches and merges results by weighted score.
    """

    def __init__(
        self,
        keyword_provider: KnowledgeProvider,
        vector_provider: KnowledgeProvider,
        keyword_weight: float = 0.3,
        vector_weight: float = 0.7,
    ):
        self.keyword = keyword_provider
        self.vector = vector_provider
        self.kw_weight = keyword_weight
        self.vec_weight = vector_weight

    def search(self, query: str, top_k: int = 5) -> list[KnowledgeResult]:
        kw_results = self.keyword.search(query, top_k=top_k)
        vec_results = self.vector.search(query, top_k=top_k)

        # Merge by source to avoid duplicates
        merged: dict[str, KnowledgeResult] = {}
        for r in kw_results:
            score = r.score * self.kw_weight
            if r.source in merged:
                merged[r.source].score = max(merged[r.source].score, score)
            else:
                r.score = score
                r.metadata["provider"] = "HybridStore(keyword)"
                merged[r.source] = r

        for r in vec_results:
            score = r.score * self.vec_weight
            if r.source in merged:
                merged[r.source].score = merged[r.source].score + score
            else:
                r.score = score
                r.metadata["provider"] = "HybridStore(vector)"
                merged[r.source] = r

        results = sorted(merged.values(), key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def get(self, key: str) -> str | None:
        result = self.keyword.get(key)
        if result:
            return result
        return self.vector.get(key)

    def name(self) -> str:
        return "HybridStore"
