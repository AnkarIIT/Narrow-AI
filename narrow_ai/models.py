"""Core data structures for NarrowAI.

These are pure data — no logic, no dependencies.
They define the contracts between layers.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


class Urgency(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


class DecisionType(str, Enum):
    ANSWER = "answer"
    ASK_CLARIFICATION = "ask_clarification"
    REQUEST_ACTION = "request_action"
    SEARCH_KNOWLEDGE = "search_knowledge"
    DELEGATE = "delegate"


@dataclass
class Input:
    """Raw user input coming into the system."""
    text: str
    user_id: str = "default"
    session_id: str = "default"
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Understanding:
    """Structured representation of user input."""
    intent: str
    entities: dict[str, str] = field(default_factory=dict)
    sentiment: str = Sentiment.NEUTRAL
    urgency: str = Urgency.NORMAL
    confidence: float = 0.0          # 0.0 – 1.0
    constraints: list[str] = field(default_factory=list)
    raw_text: str = ""

    def __post_init__(self):
        # Allow passing Enum values as strings
        if isinstance(self.sentiment, Sentiment):
            self.sentiment = self.sentiment.value
        if isinstance(self.urgency, Urgency):
            self.urgency = self.urgency.value


@dataclass
class Context:
    """Session-level context available to the reasoning layer."""
    conversation_history: list[dict[str, Any]] = field(default_factory=list)
    user_profile: dict[str, Any] = field(default_factory=dict)
    session_state: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_turn(self, role: str, content: str, **extra: Any) -> None:
        turn: dict[str, Any] = {"role": role, "content": content}
        turn.update(extra)
        self.conversation_history.append(turn)

    def last_turn(self) -> dict[str, Any] | None:
        return self.conversation_history[-1] if self.conversation_history else None


@dataclass
class Decision:
    """What the reasoning engine decided to do."""
    action: str  # one of DecisionType values
    confidence: float = 0.0
    reason: str = ""
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class Response:
    """Final output from the system."""
    text: str
    decision: Decision
    understanding: Understanding
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Interaction:
    """Full interaction record for the learning layer."""
    input: Input
    understanding: Understanding
    context: Context
    decision: Decision
    response: Response
    feedback: str | None = None
    quality_score: float | None = None
    logged_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "input": {"text": self.input.text, "user_id": self.input.user_id,
                       "session_id": self.input.session_id,
                       "metadata": self.input.metadata},
            "understanding": {
                "intent": self.understanding.intent,
                "entities": self.understanding.entities,
                "sentiment": self.understanding.sentiment,
                "urgency": self.understanding.urgency,
                "confidence": self.understanding.confidence,
                "constraints": self.understanding.constraints,
                "raw_text": self.understanding.raw_text,
            },
            "context": {
                "conversation_history": self.context.conversation_history,
                "user_profile": self.context.user_profile,
                "session_state": self.context.session_state,
            },
            "decision": {
                "action": self.decision.action,
                "confidence": self.decision.confidence,
                "reason": self.decision.reason,
                "payload": self.decision.payload,
            },
            "response": {"text": self.response.text, "metadata": self.response.metadata},
            "feedback": self.feedback,
            "quality_score": self.quality_score,
            "logged_at": self.logged_at.isoformat(),
        }
