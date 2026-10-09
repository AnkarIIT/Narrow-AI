"""Learning layer — interaction logging and feedback collection.

Records every interaction for later analysis and training.
The core AI generates training records automatically; no specific
model is trained here yet (that happens in Phase 6-7 after data exists).
"""
from __future__ import annotations
import json
from datetime import datetime
from typing import Any
from narrow_ai.models import Interaction


class InteractionLogger:
    """Collects and persists interaction records.

    By default stores in memory. Subclasses or configuration can
    add file persistence, database storage, etc.
    """

    def __init__(self):
        self._records: list[Interaction] = []
        self._next_id: int = 0

    def log(self, interaction: Interaction) -> int:
        """Store an interaction record. Returns a record ID."""
        interaction.logged_at = datetime.now()
        self._next_id += 1
        # Attach ID in metadata for reference
        interaction.metadata["record_id"] = self._next_id
        self._records.append(interaction)
        return self._next_id

    def add_feedback(self, record_id: int, feedback: str, quality_score: float | None = None) -> bool:
        """Attach feedback to a previously logged interaction."""
        for record in self._records:
            if record.metadata.get("record_id") == record_id:
                record.feedback = feedback
                record.quality_score = quality_score
                return True
        return False

    @property
    def records(self) -> list[Interaction]:
        return list(self._records)

    def count(self) -> int:
        return len(self._records)

    def export_jsonl(self, path: str) -> int:
        """Export all interactions to JSONL format for training."""
        exported = 0
        with open(path, "w", encoding="utf-8") as f:
            for record in self._records:
                f.write(json.dumps(record.to_dict(), default=str, ensure_ascii=False))
                f.write("\n")
                exported += 1
        return exported

    def get_training_examples(
        self,
        min_quality: float = 0.0,
        intent_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Extract training-ready examples from logged interactions.

        Each example has: input, intent, entities, feedback, quality.
        Used to feed future training jobs.
        """
        examples: list[dict[str, Any]] = []
        for record in self._records:
            if record.quality_score is not None and record.quality_score < min_quality:
                continue
            if intent_filter and record.understanding.intent != intent_filter:
                continue
            examples.append({
                "input": record.input.text,
                "intent": record.understanding.intent,
                "entities": record.understanding.entities,
                "sentiment": record.understanding.sentiment,
                "urgency": record.understanding.urgency,
                "decision_action": record.decision.action,
                "response": record.response.text,
                "feedback": record.feedback,
                "quality_score": record.quality_score,
            })
        return examples

    def stats(self) -> dict[str, Any]:
        """Return summary statistics."""
        intents: dict[str, int] = {}
        actions: dict[str, int] = {}
        for r in self._records:
            intents[r.understanding.intent] = intents.get(r.understanding.intent, 0) + 1
            actions[r.decision.action] = actions.get(r.decision.action, 0) + 1
        feedback_count = sum(1 for r in self._records if r.feedback is not None)
        rated_count = sum(1 for r in self._records if r.quality_score is not None)
        return {
            "total_interactions": len(self._records),
            "intent_distribution": intents,
            "decision_distribution": actions,
            "feedback_count": feedback_count,
            "rated_count": rated_count,
            "avg_quality": (
                sum(r.quality_score for r in self._records if r.quality_score is not None) / rated_count
                if rated_count > 0 else None
            ),
        }
