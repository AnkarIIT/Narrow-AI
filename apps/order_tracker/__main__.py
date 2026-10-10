"""Entry point for the Order Tracker AI application.

Usage:
    python -m apps.order_tracker.main

This wires domain-specific knowledge and tools into the
general-purpose NarrowAI core engine.
"""
from __future__ import annotations
import sys
from pathlib import Path

# Ensure both the core package and the app package are importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from apps.order_tracker.narrowai.assembly import build_order_tracker_ai


def main() -> None:
    ai = build_order_tracker_ai()

    print("=" * 60)
    print("  NarrowAI — Order Tracker Assistant")
    print("=" * 60)
    print("Ask me about: order status, refund policy, shipping,")
    print("returns, or item details. Type '/quit' to exit.")
    print("=" * 60)
    print()

    while True:
        try:
            text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not text:
            continue
        if text.lower() in ("/quit", "/exit", "quit", "exit"):
            print("Goodbye!")
            break
        if text.lower() in ("/stats",):
            stats = ai.logger.stats()
            print(f"Stats: {stats}")
            continue
        if text.lower() in ("/export",):
            path = input("Export path: ").strip() or "/tmp/interactions.jsonl"
            count = ai.logger.export_jsonl(path)
            print(f"Exported {count} records to {path}")
            continue

        response = ai.process(text)

        print(f"\n[Intent: {response.understanding.intent}]")
        print(f"[Decision: {response.decision.action if response.decision else 'N/A'}]")
        print(f"[Confidence: {response.understanding.confidence:.2f}]")
        print(f"\nBot: {response.text}")
        print()


if __name__ == "__main__":
    main()
