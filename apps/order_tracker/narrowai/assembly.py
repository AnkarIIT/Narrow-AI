"""Application assembly for the Order Tracker AI.

This is Phase 4+ — where you plug your domain-specific knowledge
and tools into the general-purpose NarrowAI core.
"""
from __future__ import annotations
from pathlib import Path

from narrow_ai import NarrowAI
from narrow_ai.core.models import StubBackend
from narrow_ai.core.knowledge import DocumentProvider, KnowledgeHub, VectorStore
from narrow_ai.core.actions import ActionHub

# Import domain-specific components
from apps.order_tracker.narrowai.actions import ACTION_REGISTRY


# ---- Domain knowledge (static facts) ------------------------------------

KNOWLEDGE_DOCS = [
    ("Refund Policy: Full refunds available within 30 days of purchase. "
     "Refunds are processed within 7-10 business days. "
     "Items must be unused and in original packaging.",
     "policy_refunds"),

    ("Shipping Information: Standard delivery takes 3-5 business days "
     "(₹99). Express delivery takes 1-2 business days (₹299). "
     "Free shipping on orders above ₹999. "
     "All shipments include tracking.",
     "policy_shipping"),

    ("Return Process: To return an item, contact support with your order ID. "
     "A return shipping label will be emailed within 24 hours. "
     "Once the item is received and inspected, your refund will be processed.",
     "process_returns"),

    ("Order Status: Orders go through 'processing' → 'shipped' → 'delivered' stages. "
     "You can track your order using the order ID (e.g., ORG-001). "
     "If your order is delayed, contact support with your order ID.",
     "info_status"),

    ("Contact Information: Support is available Monday-Friday, 9 AM to 7 PM IST. "
     "Email: support@example.com. Phone: +91-1800-SUPPORT. "
     "Live chat available on our website.",
     "info_contact"),
]


def build_order_tracker_ai() -> NarrowAI:
    """Assemble the Order Tracker AI with all domain-specific knowledge and tools."""

    # 1. Model backend — stub for now (no API key needed)
    backend = StubBackend()

    # 2. Knowledge providers — static keyword docs + semantic vector search
    doc_provider = DocumentProvider(docs=KNOWLEDGE_DOCS)
    vector_store = VectorStore(backend)
    for content, source in KNOWLEDGE_DOCS:
        vector_store.add(content, source=source)
    knowledge_hub = KnowledgeHub(providers=[doc_provider, vector_store])

    # 3. Action hub — register domain-specific functions
    action_hub = ActionHub()
    for intent_name, (handler, desc) in ACTION_REGISTRY.items():
        action_hub.register(intent_name, handler)

    # 4. Assemble the AI
    ai = NarrowAI(
        model=backend,
        knowledge=knowledge_hub,
        actions=action_hub,
    )
    return ai
