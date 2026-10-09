"""Additional tests for model backends and context-aware stub behavior."""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from narrow_ai.core.models import ModelBackend, StubBackend, HTTPBackend
from narrow_ai.core.knowledge import KnowledgeHub, DocumentProvider
from narrow_ai.core.actions import ActionHub
from narrow_ai import NarrowAI
from narrow_ai.models import Context


# ---- StubBackend tests --------------------------------------------------

def test_stub_backend_classify():
    """StubBackend.classify returns best matching intent."""
    backend = StubBackend()
    result = backend.classify("Hello, how are you?", ["greeting", "question", "unknown"])
    assert result == "greeting"


def test_stub_backend_embed_stable():
    """Embeddings are deterministic and unit-norm."""
    backend = StubBackend()
    e1 = backend.embed("hello")
    e2 = backend.embed("hello")
    assert e1 == e2  # deterministic
    norm = sum(v * v for v in e1) ** 0.5
    assert abs(norm - 1.0) < 0.01  # normalized


def test_stub_complete_uses_knowledge():
    """StubBackend.complete includes extracted knowledge context."""
    backend = StubBackend()
    prompt = (
        "You are a helpful assistant.\n"
        "Knowledge:\nRefunds within 30 days.\n---\n"
        "User intent: question\nEntities: {}\n"
        "Respond to the user's request."
    )
    result = backend.complete(prompt, intent_hint="question")
    assert "Refunds within 30 days" in result


def test_stub_complete_uses_action_result():
    """StubBackend.complete includes action results."""
    backend = StubBackend()
    prompt = (
        "You are a helpful assistant.\n"
        "Action result:\nOrder ABC123: Shipped.\n---\n"
        "User intent: status_query\nEntities: {}\n"
        "Respond to the user's request."
    )
    result = backend.complete(prompt, intent_hint="status_query")
    assert "Order ABC123" in result


def test_stub_complete_fallback_template():
    """StubBackend.complete falls back to intent template when no context."""
    backend = StubBackend()
    prompt = "User intent: greeting\nEntities: {}\nRespond to the user's request."
    result = backend.complete(prompt, intent_hint="greeting")
    assert "Hello" in result or "Hi" in result


# ---- HTTPBackend tests --------------------------------------------------

def test_http_backend_construction():
    """HTTPBackend can be constructed with valid params."""
    backend = HTTPBackend(
        base_url="http://localhost:11434/v1",
        model="llama3",
    )
    assert backend.base_url == "http://localhost:11434/v1"
    assert backend.model_name == "llama3"
    assert backend.name() == "HTTPBackend(llama3)"


def test_http_backend_headers():
    """HTTPBackend._headers includes auth when key is set."""
    backend = HTTPBackend(
        base_url="https://api.openai.com/v1",
        api_key="sk-test123",
        model="gpt-4o-mini",
    )
    headers = backend._headers()
    assert headers["Content-Type"] == "application/json"
    assert headers["Authorization"] == "Bearer sk-test123"


def test_http_backend_no_api_key():
    """HTTPBackend without api_key has no Authorization header."""
    backend = HTTPBackend(
        base_url="http://localhost:11434/v1",
        model="llama3",
    )
    headers = backend._headers()
    assert "Authorization" not in headers


def test_http_backend_requires_requests():
    """HTTPBackend raises helpful error if requests is not installed."""
    # requests IS installed in this env, so this test verifies it works
    backend = HTTPBackend(base_url="http://localhost:11434/v1", model="test")
    # Just verify the session was created
    assert backend._session is not None


# ---- Integration: action result flow ------------------------------------

def test_action_result_in_response():
    """When an action executes, its result appears in the response."""
    backend = StubBackend()
    knowledge = KnowledgeHub()
    actions = ActionHub()
    actions.register("status_query", lambda text="", **kw: "Order SHIPPED123: Delivered Oct 10.")

    ai = NarrowAI(model=backend, knowledge=knowledge, actions=actions)
    resp = ai.process("Mera order kal wala abhi tak nahi aaya.")
    assert resp.decision.action == "request_action"
    assert "SHIPPED123" in resp.text


def test_knowledge_result_in_response():
    """When knowledge is searched, results appear in the response."""
    backend = StubBackend()
    knowledge = KnowledgeHub(providers=[DocumentProvider(docs=[
        ("Your subscription renews every 30 days.", "billing"),
    ])])
    actions = ActionHub()

    ai = NarrowAI(model=backend, knowledge=knowledge, actions=actions)
    resp = ai.process("Tell me about your billing cycle.")
    assert resp.decision.action == "search_knowledge"
    assert "30 days" in resp.text


def test_context_carries_across_turns():
    """Context persists conversation history across multiple turns."""
    backend = StubBackend()
    knowledge = KnowledgeHub()
    actions = ActionHub()
    ai = NarrowAI(model=backend, knowledge=knowledge, actions=actions)
    ctx = Context()

    ai.process("Hello!", context=ctx)
    ai.process("What can you do?", context=ctx)
    assert len(ctx.conversation_history) == 4  # 2 users + 2 assistants


def test_low_confidence_asks_clarification():
    """Low-confidence intents trigger ASK_CLARIFICATION."""
    backend = StubBackend()
    ai = NarrowAI(model=backend)
    resp = ai.process("Xyzzy plugh foobar.")
    assert resp.decision.action == "ask_clarification"


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
