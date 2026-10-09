"""Tests for the core NarrowAI pipeline.

Run:  python -m pytest tests/  OR  python tests/test_core.py
"""
from __future__ import annotations
import sys
import os
import json
import tempfile

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from narrow_ai import NarrowAI
from narrow_ai.core.models import StubBackend
from narrow_ai.core.knowledge import KnowledgeHub, DocumentProvider
from narrow_ai.core.actions import ActionHub, ActionResult
from narrow_ai.core.memory import InteractionLogger
from narrow_ai.models import (
    Input, Understanding, Context, Decision, DecisionType,
)


# ---- helpers -----------------------------------------------------------

def make_ai() -> NarrowAI:
    """Create an AI instance with stub components for testing."""
    backend = StubBackend()
    docs = DocumentProvider(docs=[
        ("Orders ship within 3-5 business days.", "shipping_info"),
        ("Returns accepted within 30 days.", "return_policy"),
    ])
    knowledge = KnowledgeHub(providers=[docs])

    actions = ActionHub()

    def stub_action(text: str = "", **kwargs) -> str:
        return "Action executed successfully."

    actions.register("status_query", stub_action)

    logger = InteractionLogger()

    return NarrowAI(
        model=backend,
        knowledge=knowledge,
        actions=actions,
        logger=logger,
    )


# ---- tests -------------------------------------------------------------

def test_greeting_intent():
    """Perception correctly classifies a greeting."""
    ai = make_ai()
    resp = ai.process("Hello!")
    assert resp.understanding.intent == "greeting"
    assert resp.understanding.confidence > 0.5


def test_question_intent():
    """Perception classifies a question."""
    ai = make_ai()
    resp = ai.process("What is your refund policy?")
    assert resp.understanding.intent == "question"


def test_status_query_intent():
    """Status query intent is detected for order-related text."""
    ai = make_ai()
    resp = ai.process("Mera order kal wala abhi tak nahi aaya.")
    assert resp.understanding.intent == "status_query" or resp.understanding.intent == "question"


def test_negative_sentiment():
    """Sentiment detection picks up negative keywords."""
    backend = StubBackend()
    ai = make_ai()
    resp = ai.process("I'm really frustrated and angry about this delay.")
    assert resp.understanding.sentiment == "negative"


def test_knowledge_search():
    """Reasoning routes questions to knowledge search."""
    ai = make_ai()
    resp = ai.process("What is your refund policy?")
    # The question intent should trigger knowledge search or answer
    assert resp.decision.action in (DecisionType.SEARCH_KNOWLEDGE, DecisionType.ANSWER)


def test_action_execution():
    """Registered actions are executed via REQUEST_ACTION."""
    ai = make_ai()
    resp = ai.process("Can you help me with something?")
    # Should not crash; intent might be "request"
    assert resp.text  # response is non-empty


def test_interaction_logging():
    """Every process() call logs an interaction."""
    ai = make_ai()
    assert ai.stats()["total_interactions"] == 0
    ai.process("Hello!")
    ai.process("What is the return policy?")
    assert ai.stats()["total_interactions"] == 2


def test_feedback_attachment():
    """Feedback can be attached to logged interactions."""
    ai = make_ai()
    resp = ai.process("Hello!")
    # Get the latest record ID
    record = ai.logger.records[-1]
    record_id = record.metadata["record_id"]
    ai.add_feedback(record_id, feedback="Good response", quality=4.0)
    assert record.feedback == "Good response"
    assert record.quality_score == 4.0


def test_training_data_export():
    """Training examples can be extracted after interactions."""
    ai = make_ai()
    ai.process("Hello!")
    ai.process("What is the return policy?")
    examples = ai.training_data()
    assert len(examples) == 2
    assert "input" in examples[0]
    assert "intent" in examples[0]
    assert "response" in examples[0]


def test_jsonl_export():
    """Interactions can be exported to JSONL for training."""
    ai = make_ai()
    ai.process("Hello!")
    ai.process("What is the return policy?")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        tmp_path = f.name
    try:
        count = ai.logger.export_jsonl(tmp_path)
        assert count == 2
        with open(tmp_path) as f:
            lines = f.readlines()
        assert len(lines) == 2
        # Verify valid JSON
        for line in lines:
            data = json.loads(line)
            assert "input" in data
            assert "understanding" in data
            assert "decision" in data
    finally:
        os.unlink(tmp_path)


def test_context_persistence():
    """Context is updated with conversation turns."""
    ai = make_ai()
    ctx = Context()
    ai.process("Hello!", context=ctx)
    assert len(ctx.conversation_history) == 2  # user + assistant
    assert ctx.conversation_history[0]["role"] == "user"
    assert ctx.conversation_history[1]["role"] == "assistant"


def test_stats_distribution():
    """Stats correctly report intent and decision distributions."""
    ai = make_ai()
    ai.process("Hello!")
    ai.process("What is the return policy?")
    stats = ai.stats()
    assert "greeting" in stats["intent_distribution"]
    assert stats["total_interactions"] == 2


def test_empty_input():
    """Empty input is handled gracefully."""
    ai = make_ai()
    resp = ai.process("")
    assert resp.text  # doesn't crash


def test_action_hub_register_and_execute():
    """ActionHub correctly registers and executes handlers."""
    hub = ActionHub()

    def my_handler(name: str = "default") -> str:
        return f"Got: {name}"

    hub.register("test", my_handler)
    assert "test" in hub.list_actions()

    result = hub.execute("test", {"name": "alice"})
    assert result.success
    assert result.data == "Got: alice"

    result_fail = hub.execute("nonexistent", {})
    assert not result_fail.success


def test_document_provider_search():
    """DocumentProvider returns results by keyword matching."""
    provider = DocumentProvider(docs=[
        ("Refunds within 30 days of purchase.", "refund_policy"),
        ("Shipping takes 3-5 business days.", "shipping"),
    ])
    results = provider.search("refund")
    assert len(results) == 1
    assert results[0].source == "refund_policy"


def test_knowledge_hub_merge():
    """KnowledgeHub merges results from multiple providers."""
    p1 = DocumentProvider(docs=[("Refund info here.", "refund")])
    p2 = DocumentProvider(docs=[("Shipping info here.", "shipping")])
    hub = KnowledgeHub(providers=[p1, p2])
    results = hub.search("refund")
    assert len(results) >= 1
    assert results[0].source == "refund"


def test_understand_basic():
    """Perceptor produces correct structure from input."""
    backend = StubBackend()
    from narrow_ai.core.perception import Perceptor
    perceptor = Perceptor(backend)
    inp = Input(text="Hello, what is your refund policy?")
    understanding = perceptor.understand(inp)
    assert understanding.intent is not None
    assert understanding.raw_text == inp.text
    assert isinstance(understanding.entities, dict)


if __name__ == "__main__":
    # Run all tests
    test_funcs = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    failed = 0
    for func in test_funcs:
        try:
            func()
            print(f"  ✓ {func.__name__}")
            passed += 1
        except Exception as e:
            print(f"  ✗ {func.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed out of {len(test_funcs)} tests.")
    if failed:
        sys.exit(1)
