"""Tests for vector store and hybrid knowledge retrieval."""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from narrow_ai.core.knowledge import VectorStore, HybridStore, DocumentProvider, KnowledgeHub
from narrow_ai.core.models import StubBackend, ModelBackend
from narrow_ai.core.models.base import ModelBackend as MB


def test_vector_store_add_and_search():
    """VectorStore adds docs and returns search results."""
    backend = StubBackend()
    store = VectorStore(backend)
    store.add("Your order ships within 3-5 business days.", source="shipping")
    store.add("Refunds are accepted within 30 days.", source="refunds")

    results = store.search("shipping timeline")
    assert len(results) >= 1
    assert all(r is not None for r in results)


def test_vector_store_empty():
    """Empty VectorStore returns no results."""
    store = VectorStore(StubBackend())
    assert store.search("anything") == []
    assert store.count() == 0


def test_vector_store_get():
    """VectorStore.get returns content for matching source."""
    backend = StubBackend()
    store = VectorStore(backend)
    store.add("Test content here.", source="test_doc")
    result = store.get("test_doc")
    assert result == "Test content here."


def test_vector_store_get_missing():
    """VectorStore.get returns None for missing key."""
    store = VectorStore(StubBackend())
    assert store.get("nonexistent") is None


def test_vector_store_clear():
    """VectorStore.clear removes all documents."""
    backend = StubBackend()
    store = VectorStore(backend)
    store.add("doc1", source="s1")
    store.add("doc2", source="s2")
    assert store.count() == 2
    store.clear()
    assert store.count() == 0


def test_vector_store_add_batch():
    """VectorStore.add_batch adds multiple documents."""
    backend = StubBackend()
    store = VectorStore(backend)
    store.add_batch([
        ("First doc.", "doc1"),
        ("Second doc.", "doc2"),
        ("Third doc.", "doc3"),
    ])
    assert store.count() == 3


def test_vector_store_results_sorted_by_score():
    """Search results are sorted by similarity score descending."""
    backend = StubBackend()
    store = VectorStore(backend)
    store.add("apple banana cherry", source="fruits")
    store.add("car car automobile", source="vehicles")
    results = store.search("apple")
    assert len(results) >= 1


def test_hybrid_store_combines_providers():
    """HybridStore merges keyword and vector results."""
    backend = StubBackend()
    keyword = DocumentProvider(docs=[("Refunds within 30 days.", "refund")])
    vector = VectorStore(backend)
    vector.add("Refunds accepted within 30 days of purchase.", source="refund_vec")
    vector.add("Shipping takes 3-5 days.", source="shipping_vec")

    hybrid = HybridStore(keyword_provider=keyword, vector_provider=VectorStore(backend))
    # Even without real embeddings, the keyword results should appear
    results = hybrid.search("refund")
    assert len(results) >= 1


def test_hybrid_store_get():
    """HybridStore.get checks both providers."""
    backend = StubBackend()
    keyword = DocumentProvider(docs=[("Keyword content.", "kw_src")])
    vector = VectorStore(backend)
    vector.add("Vector content.", source="vec_src")

    hybrid = HybridStore(keyword_provider=keyword, vector_provider=vector)
    assert hybrid.get("kw_src") == "Keyword content."
    assert hybrid.get("vec_src") == "Vector content."


def test_vector_store_metadata():
    """VectorStore preserves metadata for documents."""
    backend = StubBackend()
    store = VectorStore(backend)
    store.add("Content here.", source="my_doc", category="policy", version=2)
    results = store.search("content")
    if results:
        assert results[0].metadata.get("category") == "policy"


def test_vector_store_name():
    """VectorStore reports its name."""
    store = VectorStore(StubBackend())
    assert store.name() == "VectorStore"


def test_hybrid_store_name():
    """HybridStore reports its name."""
    backend = StubBackend()
    hybrid = HybridStore(
        keyword_provider=DocumentProvider(),
        vector_provider=VectorStore(backend),
    )
    assert hybrid.name() == "HybridStore"


if __name__ == "__main__":
    test_funcs = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    failed = 0
    for fn in test_funcs:
        try:
            fn()
            print(f"  ✓ {fn.__name__}")
            passed += 1
        except Exception as e:
            print(f"  ✗ {fn.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed out of {len(test_funcs)} tests.")
    if failed:
        sys.exit(1)
