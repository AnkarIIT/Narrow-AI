"""Reasoning layer — decides what to do given Understanding + Context.

This is the decision engine. It does NOT execute actions or search knowledge
directly — it returns a Decision that tells the caller what step to take next.
The caller (NarrowAI orchestrator) then carries out the decision.
"""
from __future__ import annotations
from typing import Any
from narrow_ai.models import (
    Understanding, Context, Decision, DecisionType,
)
from narrow_ai.core.knowledge import KnowledgeProvider
from narrow_ai.core.actions import ActionProvider


class Reasoner:
    """Decision engine that maps Understanding + Context → Decision.

    Logic (independent of any specific application):
      1. If confidence is low → ask for clarification.
      2. If intent is a greeting/thank_you → answer directly.
      3. If we have relevant knowledge or context → answer.
      4. If a registered action matches the intent → request_action.
      5. If intent is a question needing info → search_knowledge.
      6. Otherwise → ask for clarification (fallback).
    """

    # Confidence threshold below which we ask for clarification
    LOW_CONFIDENCE_THRESHOLD: float = 0.30

    def __init__(
        self,
        knowledge: KnowledgeProvider | None = None,
        actions: ActionProvider | None = None,
    ):
        self.knowledge = knowledge
        self.actions = actions

    def reason(
        self,
        understanding: Understanding,
        context: Context,
    ) -> Decision:
        intent = understanding.intent
        conf = understanding.confidence
        entities = understanding.entities

        # 1. Low confidence → ask
        if conf < self.LOW_CONFIDENCE_THRESHOLD:
            return Decision(
                action=DecisionType.ASK_CLARIFICATION,
                confidence=conf,
                reason=f"Low confidence ({conf:.2f}) for intent '{intent}'.",
                payload={"question": f"I'm not sure I understood. Could you rephrase or give more details about your '{entities.get('object', 'request')}'?"},
            )

        # 2. Social intents → answer directly
        if intent in ("greeting", "thank_you"):
            return Decision(
                action=DecisionType.ANSWER,
                confidence=conf,
                reason=f"Social intent '{intent}' — answer directly.",
                payload={},
            )

        # 3. Check conversation history for context
        if context.conversation_history:
            last = context.last_turn()
            if last and last.get("role") == "user" and self._topic_continues(last.get("content", ""), understanding):
                return Decision(
                    action=DecisionType.ANSWER,
                    confidence=conf,
                    reason="Follow-up on previous turn — answer from context.",
                    payload={"use_context": True},
                )

        # 4. Check registered actions
        if self.actions and intent in self.actions.list_actions():
            return Decision(
                action=DecisionType.REQUEST_ACTION,
                confidence=conf,
                reason=f"Intent '{intent}' maps to registered action.",
                payload={
                    "action_name": intent,
                    "arguments": {
                        "text": understanding.raw_text,
                        "entities": entities,
                    },
                },
            )

        # 5. Question or info-seeking → search knowledge
        if intent in ("question", "status_query", "request"):
            # Build a search query from the raw text
            search_query = understanding.raw_text
            return Decision(
                action=DecisionType.SEARCH_KNOWLEDGE,
                confidence=conf,
                reason=f"Intent '{intent}' requires information lookup.",
                payload={"query": search_query, "entities": entities},
            )

        # 6. Fallback
        return Decision(
            action=DecisionType.ASK_CLARIFICATION,
            confidence=conf,
            reason=f"No clear path for intent '{intent}'.",
            payload={"question": "Could you provide more context? I want to make sure I help you correctly."},
        )

    # --- helpers ---------------------------------------------------------

    def _topic_continues(self, prev_text: str, understanding: Understanding) -> bool:
        """Crude heuristic: same entity or keyword overlap."""
        prev_lower = prev_text.lower()
        raw = understanding.raw_text.lower()
        # If both mention the same entity, it's likely a continuation
        for entity_val in understanding.entities.values():
            if entity_val and entity_val.lower() in prev_lower:
                return True
        # Keyword overlap
        prev_words = set(w for w in prev_lower.split() if len(w) > 3)
        curr_words = set(w for w in raw.split() if len(w) > 3)
        overlap = prev_words & curr_words
        return len(overlap) >= 2
