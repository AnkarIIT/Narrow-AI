NarrowAI Order Tracker - A customer support AI built on the NarrowAI core.

## What This AI Can Do

1. **Track orders** – "Mera order kal wala abhi tak nahi aaya?" → "ORD-001: Shipped, ETA Nov 12"
2. **Answer FAQs** – "Refund policy kya hai?" → "30-day refund window, 7-10 days for processing"
3. **Explain items/pricing** – "Ye keyboard kitna hai?" → pulls from order data
4. **Escalate unresolved issues** – routes to human support
5. **Remember you** – context persists across turns in the same session

## Knowledge Sources
- **Static docs**: refund policy, shipping info, return rules (DocumentProvider)
- **Order database**: stubbed JSON file with real order IDs/statuses (ActionProvider)

## Run It
```bash
PYTHONPATH=. python apps/order_tracker/main.py
```
