"""
NarrowAI - A general-purpose narrow AI core engine.

Build a general-purpose core first, then specialize per application.
Five layers: Perception, Reasoning, Knowledge, Action, Learning.
"""
from narrow_ai.core.narrow_ai import NarrowAI
from narrow_ai.core.models import ModelBackend, StubBackend, HTTPBackend

__version__ = "0.1.0"
__all__ = ["NarrowAI", "ModelBackend", "StubBackend", "HTTPBackend"]
