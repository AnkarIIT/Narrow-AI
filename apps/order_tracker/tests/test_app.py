"""Integration tests for the Order Tracker AI application."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from apps.order_tracker.narrowai.assembly import build_order_tracker_ai


def test_greeting():
    ai = build_order_tracker_ai()
    resp = ai.process("Hi there!")
    assert resp is not None
    assert resp.understanding.intent == "greeting"
    assert "help" in resp.text.lower()
    print("  test_greeting: PASS")


def test_status_query():
    ai = build_order_tracker_ai()
    resp = ai.process("Mera order kal wala abhi tak nahi aaya")
    assert resp is not None
    assert resp.understanding.intent == "status_query"
    assert resp.understanding.entities.get("object") == "order"
    assert resp.decision is not None
    print("  test_status_query: PASS")


def test_refund_policy():
    ai = build_order_tracker_ai()
    resp = ai.process("What is your refund policy?")
    assert resp is not None
    assert resp.understanding.intent == "question"
    assert "refund" in resp.text.lower()
    print("  test_refund_policy: PASS")


def test_shipped_order():
    ai = build_order_tracker_ai()
    resp = ai.process("Status of ORD-001?")
    assert resp is not None
    assert resp.decision is not None
    print("  test_shipped_order: PASS")


def test_escalate():
    ai = build_order_tracker_ai()
    resp = ai.process("I'm really frustrated and want to cancel everything")
    assert resp is not None
    # Should either escalate or ask clarification (both valid responses to frustration)
    assert resp.decision is not None
    print("  test_escalate: PASS")


def test_training_data_export():
    ai = build_order_tracker_ai()
    ai.process("Hi there!")
    ai.process("What is your refund policy?")
    examples = ai.training_data()
    assert len(examples) == 2
    assert all("input" in e and "response" in e for e in examples)
    print("  test_training_data_export: PASS")


def test_stats():
    ai = build_order_tracker_ai()
    ai.process("Hello!")
    stats = ai.logger.stats()
    assert stats["total_interactions"] >= 1
    assert "intent_distribution" in stats
    print("  test_stats: PASS")


if __name__ == "__main__":
    print("Running Order Tracker AI integration tests...")
    print()
    test_greeting()
    test_status_query()
    test_refund_policy()
    test_shipped_order()
    test_escalate()
    test_training_data_export()
    test_stats()
    print()
    print("All integration tests passed!")
