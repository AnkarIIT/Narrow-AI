"""Domain-specific action handlers for the Order Tracker AI.

These are pure Python functions — no external APIs required.
Swap them for database calls, API clients, etc.
"""
from __future__ import annotations
from typing import Callable
import json
import os
from pathlib import Path


# ---- Data source (stub) -------------------------------------------------

_ORDERS_FILE = Path(__file__).parent / "data" / "orders.json"


def _load_orders() -> dict:
    if _ORDERS_FILE.exists():
        with open(_ORDERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f).get("orders", {})
    return {}


# ---- Action handlers ----------------------------------------------------

def get_order_status(order_id: str = "", **kwargs) -> str:
    """Look up an order's delivery status."""
    orders = _load_orders()
    if order_id.upper().startswith("ORD"):
        order = orders.get(order_id.upper())
        if order:
            return (
                f"Order {order_id.upper()}:\n"
                f"  Status: {order['status']}\n"
                f"  Shipped: {order['date']}\n"
                f"  ETA: {order['delivery_date']}\n"
                f"  Location: {order['location']}"
            )
        return f"Order {order_id} not found in our system."
    return "Please provide a valid order ID (e.g., ORD-001)."


def get_order_items(order_id: str = "", **kwargs) -> str:
    """List items in an order."""
    orders = _load_orders()
    order = orders.get(order_id.upper())
    if order:
        items = ", ".join(order["items"])
        return f"Order {order_id.upper()} contains: {items}"
    return f"Order {order_id} not found."


def get_order_price(order_id: str = "", **kwargs) -> str:
    """Get the total price of an order."""
    orders = _load_orders()
    order = orders.get(order_id.upper())
    if order:
        return f"Order {order_id.upper()} total: ₹{order['price']}"
    return f"Order {order_id} not found."


def escalate_to_human(text: str = "", **kwargs) -> str:
    """Escalate to a human agent."""
    return "I've escalated your issue to a human support agent. They'll respond within 2 hours."


def get_refund_policy(**kwargs) -> str:
    """Return the refund policy."""
    return (
        "Refund Policy:\n"
        "- Full refunds available within 30 days of purchase\n"
        "- Refunds processed within 7-10 business days\n"
        "- Items must be unused and in original packaging\n"
        "- Contact support with your order ID to initiate"
    )


def get_shipping_info(**kwargs) -> str:
    """Return shipping information."""
    return (
        "Shipping Info:\n"
        "- Standard delivery: 3-5 business days (₹99)\n"
        "- Express delivery: 1-2 business days (₹299)\n"
        "- Free shipping on orders above ₹999\n"
        "- All shipments include tracking"
    )


# ---- Registry -----------------------------------------------------------

# Intent → (handler, description)
ACTION_REGISTRY: dict[str, tuple[Callable, str]] = {
    "status_query": (get_order_status, "Look up order delivery status"),
    "get_order_items": (get_order_items, "List items in an order"),
    "get_order_price": (get_order_price, "Get order total price"),
    "escalate": (escalate_to_human, "Transfer to human support"),
    "refund_policy": (get_refund_policy, "Get refund policy information"),
    "shipping_info": (get_shipping_info, "Get shipping information"),
}
