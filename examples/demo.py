"""Demo: exercise the full NarrowAI pipeline without external dependencies.

Shows all 5 layers working together using the StubBackend.
Run:  python examples/demo.py
"""
from __future__ import annotations
from narrow_ai import NarrowAI
from narrow_ai.core.models import StubBackend
from narrow_ai.core.knowledge import KnowledgeHub, DocumentProvider
from narrow_ai.core.actions import ActionHub
from narrow_ai.core.memory import InteractionLogger


def main() -> None:
    # --- Set up knowledge base (Layer 3) ---
    docs = DocumentProvider(docs=[
        ("Your order ships via courier and arrives within 3-5 business days. "
         "Track it with your order ID.", "order_policy"),
        ("Refunds are processed within 7-10 business days after returns are received.",
         "refund_policy"),
        ("Customer support is available 24/7 via chat and email.", "support_info"),
    ])
    knowledge = KnowledgeHub(providers=[docs])

    # --- Set up action handlers (Layer 4) ---
    actions = ActionHub()

    def get_order_status(order_id: str = "UNKNOWN") -> str:
        # Stub — real implementation would call an API
        return f"Order {order_id}: Shipped. Expected delivery: Oct 12, 2026. Current location: Mumbai."

    def get_user_data(user_id: str = "default") -> str:
        return f"User {user_id}: Name=Test User, Tier=Standard, Member since 2024."

    actions.register("status_query", get_order_status, "Look up order delivery status")
    actions.register("get_user_data", get_user_data, "Retrieve user profile data")

    # --- Logger (Layer 5) ---
    logger = InteractionLogger()

    # --- Build the AI core ---
    backend = StubBackend()
    ai = NarrowAI(
        model=backend,
        knowledge=knowledge,
        actions=actions,
        logger=logger,
    )

    # --- Run interactions ---
    print("=" * 60)
    print("NarrowAI Core Demo")
    print("=" * 60)

    examples = [
        "Hello! What's your name?",
        "Mera order kal wala abhi tak nahi aaya. Kya aap mujhe bata sakte hain status?",
        "Can you explain your refund policy?",
        "I'm really frustrated about the delay.",
        "What's the weather today?",
    ]

    for text in examples:
        print(f"\n>>> {text}")
        resp = ai.process(text)
        print(f"    [intent: {resp.understanding.intent}]")
        print(f"    [confidence: {resp.understanding.confidence:.2f}]")
        print(f"    [decision: {resp.decision.action}]")
        print(f"    [reason: {resp.decision.reason}]")
        print(f"    [response: {resp.text}]")

    # --- Show stats ---
    print("\n" + "=" * 60)
    stats = ai.stats()
    print(f"Total interactions: {stats['total_interactions']}")
    print(f"Intent distribution: {stats['intent_distribution']}")
    print(f"Decision distribution: {stats['decision_distribution']}")
    print(f"Feedback count: {stats['feedback_count']}")

    # --- Show a training record ---
    if logger.count() > 0:
        record = logger.records[1]  # the order status query
        print(f"\nSample interaction record (record_id={record.metadata.get('record_id')}):")
        print(f"  Input: {record.input.text}")
        print(f"  Intent: {record.understanding.intent}")
        print(f"  Entities: {record.understanding.entities}")
        print(f"  Decision: {record.decision.action}")
        print(f"  Response: {record.response.text}")

    print("\n✓ Demo complete — all 5 layers exercised.")


if __name__ == "__main__":
    main()
