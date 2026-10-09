"""Perception layer — converts raw input into structured Understanding.

This layer is pluggable: you can swap in a fine-tuned intent classifier,
a different entity extractor, or the stub keyword-based approach.
"""
from __future__ import annotations
from typing import Any
from narrow_ai.core.models.base import ModelBackend
from narrow_ai.models import Input, Understanding, Sentiment, Urgency


class IntentClassifier:
    """Classify raw text into an intent label."""

    def __init__(self, backend: ModelBackend, intents: list[str] | None = None):
        self.backend = backend
        self.intents = intents or [
            "greeting", "question", "request", "status_query",
            "thank_you", "command", "unknown",
        ]

    def classify(self, text: str) -> tuple[str, float]:
        intent = self.backend.classify(text, self.intents)
        # Confidence: check if the backend returned "unknown"
        confidence = 0.85 if intent != "unknown" else 0.20
        return intent, confidence


class EntityExtractor:
    """Extract named entities from text."""

    def __init__(self, backend: ModelBackend):
        self.backend = backend

    def extract(self, text: str) -> dict[str, str]:
        # For now, delegate to backend-specific extraction
        # StubBackend has _extract_entities
        if hasattr(self.backend, "_extract_entities"):
            return self.backend._extract_entities(text)  # type: ignore
        return {}


class Perceptor:
    """Orchestrates intent classification, entity extraction, sentiment, urgency.

    This is the single entry point for the perception layer.
    """

    def __init__(self, backend: ModelBackend, intents: list[str] | None = None):
        self.backend = backend
        self.intent_classifier = IntentClassifier(backend, intents)
        self.entity_extractor = EntityExtractor(backend)

    def understand(self, inp: Input) -> Understanding:
        text = inp.text

        intent, confidence = self.intent_classifier.classify(text)
        entities = self.entity_extractor.extract(text)

        # Sentiment
        sentiment = "neutral"
        if hasattr(self.backend, "_detect_sentiment"):
            sentiment = self.backend._detect_sentiment(text)  # type: ignore

        # Urgency
        urgency = "normal"
        if hasattr(self.backend, "_detect_urgency"):
            urgency = self.backend._detect_urgency(text)  # type: ignore

        # Constraints (simple: detect explicit constraints)
        constraints: list[str] = []
        if "please" in text.lower() or "kindly" in text.lower() or "कृपया" in text:
            constraints.append("polite_request")
        if "?" in text or "क्या" in text.lower():
            constraints.append("question_form")

        return Understanding(
            intent=intent,
            entities=entities,
            sentiment=sentiment,
            urgency=urgency,
            confidence=confidence,
            constraints=constraints,
            raw_text=text,
        )
