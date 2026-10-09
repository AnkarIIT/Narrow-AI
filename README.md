# NarrowAI Core Engine

A general-purpose **narrow AI core** — not tied to any specific application.
Build the intelligence layer first, then specialize per use case later.

## Architecture

```
                    ┌──────────────────────┐
                    │   APPLICATION        │
                    │  (later, not now)     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   FINAL AI LAYER     │
                    │ prompts / tools / RAG│
                    │ domain rules / UX    │
                    └──────────┬───────────┘
                               │
═══════════════════════════════╪═══════════
                               │
                               ▼
                    ┌──────────────────────┐
                    │  YOUR NARROW AI      │  ← THIS REPO
                    │                      │
                    │ • Understand         │   Layer 1: Perception
                    │ • Reason             │   Layer 2: Reasoning
                    │ • Classify           │
                    │ • Decide             │
                    │ • Generate           │
                    │ • Remember           │   Layer 5: Learning
                    │ • Learn from feedback│
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  Base Model(s)       │
                    │  + Training Data     │
                    └──────────────────────┘
```

## The 5 Layers

| Layer | File | Responsibility |
|-------|------|----------------|
| **Perception** | `core/perception.py` | Input → structured Understanding (intent, entities, sentiment, urgency, confidence) |
| **Reasoning** | `core/reasoning.py` | Understanding + Context → Decision (what to do: answer, ask, search, act) |
| **Knowledge** | `core/knowledge/`, `core/knowledge/vector_store.py` | KnowledgeProvider, DocumentProvider, VectorStore, HybridStore |
| **Action** | `core/actions.py` | Action providers — app supplies concrete handlers; core only knows WHAT to request |
| **Learning** | `core/memory.py` | Interaction logging, feedback collection, training data export |

## Model Backends

| Backend | File | Notes |
|---------|------|-------|
| **StubBackend** | `core/models/base.py` | Rule-based (keyword matching + templates). Zero deps. For testing. |
| **HTTPBackend** | `core/models/http.py` | OpenAI-compatible API client (OpenAI, Ollama, vLLM, LMStudio). |

```python
from narrow_ai.core.models import HTTPBackend

# Ollama (local)
backend = HTTPBackend(base_url="http://localhost:11434/v1", model="llama3")

# OpenAI
backend = HTTPBackend(base_url="https://api.openai.com/v1", api_key="sk-...", model="gpt-4o-mini")
```

## CLI

```bash
# Interactive mode (stub backend, no API key needed)
python -m narrow_ai.cli

# With a local Ollama backend
python -m narrow_ai.cli --backend ollama --model llama3

# With OpenAI
python -m narrow_ai.cli --backend openai --api-key sk-... --model gpt-4o-mini

# Load a knowledge base from file
python -m narrow_ai.cli --backend ollama --model llama3 --knowledge data/knowledge.jsonl
```

Commands in chat: `/help`, `/knowledge add <text>`, `/knowledge search <query>`, `/action list`, `/stats`, `/export <path>`, `/feedback <id> <rating> <text>`, `/backend <stub|openai|ollama>`, `/quit`

## Design Principles

1. **Capability boundary, not use case** — The core understands intent, reasons, and decides. The application layer later adds domain knowledge and tools.
2. **No coupling to specific apps** — KnowledgeProvider, ActionProvider, and ModelBackend are all abstract interfaces.
3. **Swappable backends** — Start with `StubBackend` (rule-based, no dependencies). Swap in `OpenAI`, `Ollama`, or a custom fine-tuned model.
4. **Learning-ready from day one** — Every interaction is logged with full context for later training.

## Quick Start

```bash
# Install in development mode
pip install -e ".[dev]"

# Run the demo (stub backend, no API keys needed)
python examples/demo.py

# Run tests
python tests/test_core.py
python tests/test_backends.py
python tests/test_vector_store.py
```

## Usage

```python
from narrow_ai import NarrowAI
from narrow_ai.core.models import StubBackend
from narrow_ai.core.knowledge import DocumentProvider, KnowledgeHub
from narrow_ai.core.actions import ActionHub

# 1. Set up backends and providers
backend = StubBackend()
knowledge = KnowledgeHub(providers=[DocumentProvider(docs=[
    ("Refunds within 30 days.", "refund_policy"),
])])
actions = ActionHub()
actions.register("status_query", lambda: "Order shipped, ETA: Oct 12")

# 2. Create the AI core
ai = NarrowAI(model=backend, knowledge=knowledge, actions=actions)

# 3. Process input
response = ai.process("What is your refund policy?")
print(response.text)
print(response.understanding.intent)  # 'question'
print(response.decision.action)       # 'search_knowledge'

# 4. Add feedback later (for training)
ai.add_feedback(response.metadata.get("record_id", 1),
                feedback="Good answer", quality=4.5)

# 5. Export training data
ai.training_data()  # returns list of {input, intent, entities, response, feedback...}
```

## Project Roadmap

| Phase | Description |
|-------|-------------|
| 0 | Define capability boundary |
| 1 | Build core architecture |
| 2 | Build inference pipeline (StubBackend, HTTPBackend) |
| 3 | Build memory + knowledge interfaces (VectorStore, HybridStore) |
| 4 | Build tool/action interfaces ← **YOU ARE HERE** |
| 5 | Collect interaction + feedback data |
| 6 | Find where base models fail |
| 7 | Train/fine-tune specialized components |
| 8 | Application-specific layer |

## Roadmap to Real Models

The `StubBackend` lets you build and test the full system without any external API.
When ready, implement `ModelBackend` with a real provider:

```python
class OpenAIBackend(ModelBackend):
    def complete(self, prompt, **kwargs):
        # call OpenAI API
        ...
    def classify(self, text, options):
        # few-shot prompt to GPT
        ...
    def embed(self, text):
        # call embedding endpoint
        ...
```

Then swap it in: `ai = NarrowAI(model=OpenAIBackend(), ...)` — no other code changes needed.
