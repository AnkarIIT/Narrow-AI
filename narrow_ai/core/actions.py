"""Action layer — pluggable action providers.

The core AI knows WHAT to do (request an action) but never HOW to do it.
The application layer provides concrete implementations.
This keeps the core decoupled from specific databases, APIs, etc.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class ActionResult:
    """Result returned by an action handler."""
    success: bool
    data: Any = None
    error: str | None = None
    metadata: dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ActionProvider(ABC):
    """Abstract interface for action providers."""

    @abstractmethod
    def list_actions(self) -> list[str]:
        """Return list of action names this provider can handle."""

    @abstractmethod
    def execute(self, name: str, arguments: dict[str, Any]) -> ActionResult:
        """Execute action *name* with *arguments*."""

    def name(self) -> str:
        return self.__class__.__name__


class ActionHub(ActionProvider):
    """Registry-based action provider.

    Application registers action handlers as callables:

        hub = ActionHub()
        hub.register("get_user_data", my_user_data_func)
        hub.register("send_email", my_email_func)
    """

    def __init__(self):
        self._handlers: dict[str, Callable[..., Any]] = {}
        self._descriptions: dict[str, str] = {}

    def register(self, name: str, handler: Callable[..., Any], description: str = "") -> None:
        """Register a handler function for an action name."""
        self._handlers[name] = handler
        if description:
            self._descriptions[name] = description

    def list_actions(self) -> list[str]:
        return list(self._handlers.keys())

    def execute(self, name: str, arguments: dict[str, Any]) -> ActionResult:
        handler = self._handlers.get(name)
        if handler is None:
            return ActionResult(
                success=False,
                error=f"No handler registered for action '{name}'",
            )
        try:
            result = handler(**arguments) if arguments else handler()
            return ActionResult(success=True, data=result)
        except Exception as e:
            return ActionResult(success=False, error=str(e))

    def describe(self, name: str) -> str:
        return self._descriptions.get(name, "")
