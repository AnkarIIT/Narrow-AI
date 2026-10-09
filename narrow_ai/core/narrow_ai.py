"""Main NarrowAI orchestrator.

Wires together all five layers:
  Perception → Reasoning → Knowledge/Action → Response → Learning

The AI is configured with a model backend (for understanding + generation),
optional knowledge and action providers, and an interaction logger.
Applications plug in their own knowledge providers, action handlers, and
can later add specialized model backends.
"""
from __future__ import annotations
from typing import Any
from narrow_ai.core.models.base import ModelBackend, StubBackend
from narrow_ai.core.perception import Perceptor
from narrow_ai.core.reasoning import Reasoner
from narrow_ai.core.knowledge import KnowledgeHub, KnowledgeProvider
from narrow_ai.core.actions import ActionHub, ActionProvider
from narrow_ai.core.memory import InteractionLogger
from narrow_ai.models import (
    Input, Understanding, Context, Decision, Response,
    Interaction, DecisionType,
)


class NarrowAI:
    """General-purpose narrow AI core engine.

    This is the 'brain'. It is application-agnostic.
    To specialize it, create an Application layer that:
      - Supplies domain knowledge (KnowledgeProvider)
      - Registers tool handlers (ActionProvider)
      - Optionally swaps the model backend
    """

    def __init__(
        self,
        model: ModelBackend | None = None,
        knowledge: KnowledgeProvider | None = None,
        actions: ActionProvider | None = None,
        intents: list[str] | None = None,
        logger: InteractionLogger | None = None,
        config: dict[str, Any] | None = None,
    ):
        self.model = model or StubBackend()
        self.config = config or {}

        # --- Layer 1: Perception ---
        self.perceptor = Perceptor(self.model, intents)

        # --- Layer 2: Reasoning ---
        self.knowledge = knowledge  # type: ignore[assignment]
        self.actions = actions  # type: ignore[assignment]
        self.reasoner = Reasoner(self.knowledge, self.actions)

        # --- Layer 5: Learning ---
        self.logger = logger or InteractionLogger()

    # --- public API ------------------------------------------------------

    def process(
        self,
        text: str,
        user_id: str = "default",
        session_id: str = "default",
        context: Context | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Response:
        """Process a single user turn end-to-end.

        Returns a Response containing the AI's text output and the
        full decision/understanding trace.
        """
        inp = Input(
            text=text,
            user_id=user_id,
            session_id=session_id,
            metadata=metadata or {},
        )
        ctx = context or Context()

        # --- Layer 1: Perception ---
        understanding = self.perceptor.understand(inp)

        # --- Layer 2: Reasoning ---
        decision = self.reasoner.reason(understanding, ctx)

        # --- Layer 3+4: Execute the decision ---
        response_text = self._execute_decision(decision, understanding, ctx, inp)

        # --- Build response ---
        response = Response(
            text=response_text,
            decision=decision,
            understanding=understanding,
            metadata={"model": self.model.name()},
        )

        # --- Layer 5: Learning (log interaction) ---
        interaction = Interaction(
            input=inp,
            understanding=understanding,
            context=ctx,
            decision=decision,
            response=response,
        )
        self.logger.log(interaction)

        # Update context with this turn
        ctx.add_turn(role="user", content=text)
        ctx.add_turn(role="assistant", content=response_text,
                      intent=understanding.intent,
                      decision=decision.action)

        return response

    def add_feedback(self, record_id: int, feedback: str, quality: float | None = None) -> bool:
        """Attach human feedback to a past interaction."""
        return self.logger.add_feedback(record_id, feedback, quality)

    def training_data(self, **kwargs: Any) -> list[dict[str, Any]]:
        """Export training-ready examples from logged interactions."""
        return self.logger.get_training_examples(**kwargs)

    def stats(self) -> dict[str, Any]:
        """Return learning-layer statistics."""
        return self.logger.stats()

    # --- decision execution ---------------------------------------------

    def _execute_decision(
        self,
        decision: Decision,
        understanding: Understanding,
        context: Context,
        inp: Input,
    ) -> str:
        """Route a Decision to the appropriate handler and return text."""
        action = decision.action

        if action == DecisionType.ANSWER:
            return self._answer(understanding, decision, context)

        elif action == DecisionType.ASK_CLARIFICATION:
            payload = decision.payload
            return payload.get("question", "Could you clarify?")

        elif action == DecisionType.SEARCH_KNOWLEDGE:
            query = decision.payload.get("query", understanding.raw_text)
            results = self._search_knowledge(query)
            if results:
                return self._answer_from_knowledge(understanding, results)
            # Knowledge not found — fall back to clarification
            return self._answer(
                understanding,
                Decision(
                    action=DecisionType.ASK_CLARIFICATION,
                    reason="Knowledge search returned no results.",
                    payload={"question": "I don't have information on that yet. Could you provide more context?"},
                ),
                context,
            )

        elif action == DecisionType.REQUEST_ACTION:
            action_name = decision.payload.get("action_name", "")
            args = decision.payload.get("arguments", {})
            result = self._execute_action(action_name, args)
            if result.success:
                return self._generate_response(
                    understanding, str(result.data), context,
                    context_label="Action result:",
                )
            return self._generate_response(
                understanding,
                f"I tried to look that up but encountered an issue: {result.error}",
                context,
                context_label="Error:",
            )

        elif action == DecisionType.DELEGATE:
            return self._generate_response(
                understanding,
                "I'm passing this to a more specialized system.",
                context,
            )

        return "I'm not sure how to handle this."

    # --- knowledge / action / generation helpers ------------------------

    def _search_knowledge(self, query: str, top_k: int = 3) -> list[Any]:
        if self.knowledge is not None:
            return self.knowledge.search(query, top_k=top_k)
        return []

    def _execute_action(self, name: str, arguments: dict[str, Any]) -> Any:
        if self.actions is not None:
            return self.actions.execute(name, arguments)
        from narrow_ai.core.actions import ActionResult
        return ActionResult(success=False, error="No action provider registered.")

    def _answer(self, understanding: Understanding, decision: Decision, context: Context) -> str:
        """Generate an answer using the model backend."""
        return self._generate_response(understanding, "", context)

    def _answer_from_knowledge(
        self,
        understanding: Understanding,
        results: list[Any],
    ) -> str:
        """Generate an answer grounded in knowledge search results."""
        context_text = "\n\n".join(
            f"{r.content}" for r in results
        )
        return self._generate_response(understanding, context_text, Context(),
                                        context_label="Knowledge:")

    def _generate_response(
        self,
        understanding: Understanding,
        context_text: str,
        context: Context,
        context_label: str = "Knowledge:",
    ) -> str:
        """Use the model backend to generate a response.

        Builds a prompt from the understanding, optional context text,
        and conversation history, then asks the backend to complete.
        """
        prompt = self._build_prompt(understanding, context_text, context, context_label)
        return self.model.complete(prompt, intent_hint=understanding.intent)

    def _build_prompt(
        self,
        understanding: Understanding,
        context_text: str,
        context: Context,
        context_label: str = "Knowledge:",
    ) -> str:
        """Construct a response-generation prompt.

        This is the prompt template that maps Understanding + knowledge
        into a response. Later this can be improved per-application.
        """
        parts: list[str] = []

        # System instruction
        parts.append(
            "You are a helpful, concise AI assistant. Your job is to "
            "understand the user's intent and respond appropriately.\n"
        )

        # Context from conversation history
        if context.conversation_history:
            parts.append("Previous conversation:")
            for turn in context.conversation_history[-6:]:  # last 6 turns
                role = turn.get("role", "?")
                content = turn.get("content", "")
                parts.append(f"  {role}: {content}")
            parts.append("---")

        # Retrieved knowledge / action result
        if context_text:
            parts.append(f"{context_label}\n{context_text}\n---")

        # Current understanding
        parts.append(f"User intent: {understanding.intent}")
        parts.append(f"Entities: {understanding.entities}")
        parts.append(f"Sentiment: {understanding.sentiment}")
        parts.append(f"Urgency: {understanding.urgency}")

        # The user's actual request
        parts.append(f"\nRespond to the user's request.")

        return "\n".join(parts)
