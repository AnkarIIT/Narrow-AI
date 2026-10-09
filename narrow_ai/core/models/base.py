"""Base interfaces for pluggable model backends.

Any backend — a real LLM API, a fine-tuned classifier, or a rule-based stub —
must implement these methods so the core AI works regardless of which
backend is plugged in.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class ModelBackend(ABC):
    """Abstract interface every model backend must satisfy."""

    @abstractmethod
    def complete(self, prompt: str, **kwargs: Any) -> str:
        """Generate a text completion for *prompt*."""

    @abstractmethod
    def classify(self, text: str, options: list[str], **kwargs: Any) -> str:
        """Return the best-matching option from *options* for *text*."""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Return a dense embedding vector for *text*."""

    # ---- optional hooks -------------------------------------------------

    def complete_json(self, prompt: str, **kwargs: Any) -> str:
        """Override if the backend natively supports JSON output."""
        return self.complete(prompt, **kwargs)

    def name(self) -> str:
        return self.__class__.__name__


class StubBackend(ModelBackend):
    """Rule-based and template backend for local testing.

    No network, no API keys — uses keyword matching and simple templates.
    This lets the entire pipeline run end-to-end during development.
    Swap this for a real LLM backend (OpenAI, Ollama, etc.) in production.
    """
    import re as _re

    # --- intent classification via keyword scoring -----------------------
    _INTENT_KEYWORDS: dict[str, list[str]] = {
        "status_query": ["status", "where", "track", "kal", "yesterday",
                          "abhi tak", "aaya", "aayi", "order", "shipped",
                          "deliver", "dispatch"],
        "question": ["what", "how", "why", "when", "kya", "kya", "kahan",
                      "kahan", "kis", "explain", "tell", "define"],
        "greeting": ["hi", "hello", "hey", "namaste", "good morning",
                      "good evening", "good afternoon"],
        "thank_you": ["thank", "dhanyavaad", "shukriya"],
        "request": ["please", "kindly", "mangta", "mangti", "chahiye",
                     "do", "please", "karo", "banayiye"],
    }

    _SENTIMENT_KEYWORDS: dict[str, list[str]] = {
        "negative": ["nahi", "not", "bad", "worst", "hate", "angry",
                      "frustated", "problem", "issue", "delay", "late",
                      "wrong", "burer", "galti"],
        "positive": ["great", "awesome", "thank", "shukriya", "good",
                      "nice", "perfect", "excellent", "mast", "super"],
    }

    # --- templates for response generation -------------------------------
    _RESPONSE_TEMPLATES: dict[str, list[str]] = {
        "question": [
            "That's a good question about '{entity}'. Let me think through this.",
            "Based on what I know, here's what I can tell you about '{entity}'.",
        ],
        "greeting": [
            "Hello! How can I help you today?",
            "Hi there! What can I do for you?",
        ],
        "status_query": [
            "Let me check the status of your request. I'll need to look that up.",
        ],
        "request": [
            "I understand you need help with that. Let me see what I can do.",
        ],
        "default": [
            "I'm processing your request. Could you provide more details?",
        ],
    }

    def classify(self, text: str, options: list[str], **kwargs: Any) -> str:
        """Score each option by keyword overlap; return the best match."""
        text_lower = text.lower()
        # Normalize: if the option is an intent label, use its keywords
        scores: dict[str, float] = {opt: 0.0 for opt in options}
        for opt in options:
            # Map option to known keyword set if available
            keywords = self._INTENT_KEYWORDS.get(opt)
            if keywords is None:
                # Use the option string itself as a keyword
                keywords = [opt]
            for kw in keywords:
                if kw in text_lower:
                    scores[opt] += 1.0 / len(keywords)
        best = max(scores, key=scores.get)  # type: ignore
        return best if scores[best] > 0 else "unknown"

    # --- sentiment -------------------------------------------------------
    def _detect_sentiment(self, text: str) -> str:
        text_lower = text.lower()
        neg = sum(1 for kw in self._SENTIMENT_KEYWORDS["negative"] if kw in text_lower)
        pos = sum(1 for kw in self._SENTIMENT_KEYWORDS["positive"] if kw in text_lower)
        if neg > pos:
            return "negative"
        if pos > neg:
            return "positive"
        return "neutral"

    # --- entity extraction (simple) -------------------------------------
    def _extract_entities(self, text: str) -> dict[str, str]:
        entities: dict[str, str] = {}
        # Time references
        if "kal" in text.lower() or "yesterday" in text.lower():
            entities["time_reference"] = "yesterday"
        if "aaj" in text.lower() or "today" in text.lower():
            entities["time_reference"] = "today"
        # Object extraction — look for common nouns
        for obj in ["order", "बद्ध", "oder", "report", "bill", "payment", "ticket"]:
            if obj in text.lower():
                entities["object"] = obj
                break
        return entities

    # --- urgency detection ----------------------------------------------
    def _detect_urgency(self, text: str) -> str:
        text_lower = text.lower()
        high_markers = ["urgent", "जल्दी", "immediately", "asap", "तुरंत", "abhi"]
        if any(m in text_lower for m in high_markers):
            return "high"
        return "normal"

    # --- complete --------------------------------------------------------
    def complete(self, prompt: str, **kwargs: Any) -> str:
        """Template + keyword completion for stubbing."""
        intent_hint = kwargs.get("intent_hint", "")
        templates = self._RESPONSE_TEMPLATES.get(intent_hint, self._RESPONSE_TEMPLATES["default"])
        return templates[0]

    def embed(self, text: str) -> list[float]:
        """Deterministic hash-based embedding (32-dim). Not semantic, just stable."""
        import hashlib, struct
        h = hashlib.md5(text.encode("utf-8")).digest()
        vec = list(struct.unpack("8f", h))
        # Normalise
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]

    def name(self) -> str:
        return "StubBackend"
