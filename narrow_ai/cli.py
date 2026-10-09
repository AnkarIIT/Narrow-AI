"""Interactive CLI for NarrowAI.

Usage:
    python -m narrow_ai.cli                    # stub backend (no API key)
    python -m narrow_ai.cli --backend ollama --model llama3
    python -m narrow_ai.cli --backend openai --api-key sk-... --model gpt-4o-mini

Commands (type in the chat):
    /help              Show available commands
    /knowledge add <text>     Add a knowledge document
    /knowledge search <query>  Search knowledge base
    /action <name> <args>      Register/call a custom action
    /stats             Show interaction statistics
    /export <path>     Export interactions to JSONL
    /feedback <id> <rating> <text>  Add feedback to interaction
    /backend <stub|openai|ollama>  Switch backend
    /quit              Exit
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import shlex
from datetime import datetime
from typing import Any

from narrow_ai import NarrowAI
from narrow_ai.core.models import ModelBackend, StubBackend, HTTPBackend
from narrow_ai.core.knowledge import KnowledgeHub, DocumentProvider, KnowledgeProvider, KnowledgeResult
from narrow_ai.core.actions import ActionHub, ActionResult
from narrow_ai.core.memory import InteractionLogger
from narrow_ai.models import Context


class CLI:
    def __init__(self) -> None:
        self.ai: NarrowAI | None = None
        self.ctx: Context = Context()
        self.exit = False

    def setup(self, args: argparse.Namespace) -> None:
        """Initialize the AI with the selected backend and providers."""
        # Knowledge layer
        knowledge = KnowledgeHub()
        # Load knowledge documents from file if provided
        if args.knowledge:
            self._load_knowledge(knowledge, args.knowledge)

        # Action layer
        actions = ActionHub()
        self._register_default_actions(actions)

        # Logger
        logger = InteractionLogger()

        # Model backend
        backend = self._create_backend(args)

        self.ai = NarrowAI(
            model=backend,
            knowledge=knowledge,
            actions=actions,
            logger=logger,
        )
        self.ctx = Context()

    def _create_backend(self, args: argparse.Namespace) -> ModelBackend:
        backend_type = args.backend or "stub"
        if backend_type == "stub":
            return StubBackend()
        elif backend_type in ("openai", "ollama", "http"):
            base_url = {
                "openai": "https://api.openai.com/v1",
                "ollama": os.environ.get("OLLAMA_URL", "http://localhost:11434/v1"),
            }.get(backend_type, args.base_url or "http://localhost:11434/v1")
            api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
            model = args.model or "gpt-4o-mini"
            return HTTPBackend(base_url=base_url, api_key=api_key, model=model)
        else:
            print(f"Unknown backend: {backend_type}. Using stub.")
            return StubBackend()

    def _load_knowledge(self, hub: KnowledgeHub, path: str) -> None:
        """Load knowledge documents from a file (one per line or JSONL)."""
        if not os.path.exists(path):
            print(f"Warning: knowledge file '{path}' not found.")
            return
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    content = data.get("content", "")
                    source = data.get("source", "imported")
                    hub.register(DocumentProvider(docs=[(content, source)]))
                except json.JSONDecodeError:
                    # Plain text, one doc per paragraph
                    hub.register(DocumentProvider(docs=[(line, "imported")]))

    def _register_default_actions(self, actions: ActionHub) -> None:
        """Register useful default actions."""
        def get_time() -> str:
            return f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

        def echo(text: str = "") -> str:
            return f"You said: {text}" if text else "Echo"

        actions.register("get_time", get_time, "Get the current time")
        actions.register("echo", echo, "Echo back text")

    # --- main loop -------------------------------------------------------

    def run(self) -> None:
        if self.ai is None:
            print("AI not initialized.")
            return

        print(f"\nNarrowAI CLI — backend: {self.ai.model.name()}")
        print("Type /help for commands. Type /quit to exit.\n")

        while not self.exit:
            try:
                user_input = input("You> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not user_input:
                continue

            # Check for commands
            if user_input.startswith("/"):
                self._handle_command(user_input)
                continue

            # Process through the AI
            resp = self.ai.process(user_input, context=self.ctx)
            print(f"\nAI ({resp.understanding.intent}, conf={resp.understanding.confidence:.2f}, "
                  f"decision={resp.decision.action}): {resp.text}\n")

    def _handle_command(self, cmd_line: str) -> None:
        """Parse and execute a CLI command."""
        parts = shlex.split(cmd_line[1:])  # strip leading /
        if not parts:
            print("Empty command. Type /help for available commands.")
            return

        cmd = parts[0]
        args = parts[1:]

        handlers = {
            "help": self._cmd_help,
            "knowledge": self._cmd_knowledge,
            "action": self._cmd_action,
            "stats": self._cmd_stats,
            "export": self._cmd_export,
            "feedback": self._cmd_feedback,
            "backend": self._cmd_backend,
            "quit": self._cmd_quit,
            "exit": self._cmd_quit,
        }

        handler = handlers.get(cmd)
        if handler:
            handler(args)
        else:
            print(f"Unknown command: /{cmd}. Type /help for available commands.")

    def _cmd_help(self, args: list[str]) -> None:
        print(__doc__)

    def _cmd_knowledge(self, args: list[str]) -> None:
        if not args:
            print("Knowledge commands: add <text> | search <query>")
            return
        sub = args[0]
        text = " ".join(args[1:])

        if sub == "add" and text:
            if self.ai and isinstance(self.ai.knowledge, KnowledgeHub):
                self.ai.knowledge.register(
                    DocumentProvider(docs=[(text, f"manual_{datetime.now().strftime('%H%M%S')}")])
                )
                print(f"✓ Added knowledge document ({len(text)} chars)")
            else:
                print("No KnowledgeHub configured (use a KnowledgeHub instance).")
        elif sub == "search" and text:
            if self.ai and self.ai.knowledge:
                results = self.ai.knowledge.search(text, top_k=5)
                for r in results:
                    print(f"  [{r.score:.2f}] {r.source}: {r.content[:100]}...")
            else:
                print("No knowledge provider configured.")
        else:
            print("Knowledge commands: add <text> | search <query>")

    def _cmd_action(self, args: list[str]) -> None:
        if not args:
            print("Action commands: list | call <name> <args_json>")
            return
        sub = args[0]
        if sub == "list":
            if self.ai and self.ai.actions:
                actions = self.ai.actions.list_actions()
                print(f"Registered actions: {actions}")
            else:
                print("No action provider configured.")
        elif sub == "call":
            if len(args) < 2:
                print("Usage: /action call <name> [json_args]")
                return
            name = args[1]
            call_args: dict[str, Any] = {}
            if len(args) > 2:
                try:
                    call_args = json.loads(" ".join(args[2:]))
                except json.JSONDecodeError:
                    print("Invalid JSON args.")
                    return
            if self.ai and self.ai.actions:
                result = self.ai._execute_action(name, call_args)  # type: ignore[attr-defined]
                if result.success:
                    print(f"✓ Result: {result.data}")
                else:
                    print(f"✗ Error: {result.error}")
            else:
                print("No action provider configured.")
        else:
            print("Action commands: list | call <name> <args_json>")

    def _cmd_stats(self, args: list[str]) -> None:
        if not self.ai:
            print("AI not initialized.")
            return
        stats = self.ai.stats()
        print(json.dumps(stats, indent=2, default=str))

    def _cmd_export(self, args: list[str]) -> None:
        if not args:
            print("Usage: /export <path>")
            return
        if not self.ai:
            print("AI not initialized.")
            return
        path = args[0]
        count = self.ai.logger.export_jsonl(path)  # type: ignore[attr-defined]
        print(f"✓ Exported {count} interactions to {path}")

    def _cmd_feedback(self, args: list[str]) -> None:
        if len(args) < 3:
            print("Usage: /feedback <record_id> <rating 1-5> <feedback_text>")
            return
        if not self.ai:
            print("AI not initialized.")
            return
        record_id = int(args[0])
        rating = float(args[1])
        feedback_text = " ".join(args[2:])
        quality = rating * 2.0  # scale 1-5 to 0-10
        success = self.ai.add_feedback(record_id, feedback_text, quality)
        if success:
            print(f"✓ Feedback added to record {record_id}")
        else:
            print(f"✗ Record {record_id} not found. Use /stats to see record IDs.")

    def _cmd_backend(self, args: list[str]) -> None:
        if not args:
            print("Usage: /backend <stub|openai|ollama>")
            return
        # Re-create backend (note: this resets the AI but keeps knowledge/actions/logs)
        new_args = argparse.Namespace(
            backend=args[0],
            model=getattr(args, 'model', None) if len(args) > 1 else None,
            api_key=getattr(args, 'api_key', None) if len(args) > 2 else None,
            knowledge=None,
            base_url=None,
        )
        # Actually let me handle this more carefully
        model = args[1] if len(args) > 1 else "gpt-4o-mini"
        api_key = args[2] if len(args) > 2 else os.environ.get("OPENAI_API_KEY")
        backend_ns = argparse.Namespace(
            backend=args[0], model=model, api_key=api_key, knowledge=None, base_url=None
        )
        new_backend = self._create_backend(backend_ns)
        if self.ai:
            self.ai.model = new_backend
            print(f"✓ Backend switched to: {new_backend.name()}")

    def _cmd_quit(self, args: list[str]) -> None:
        print("Goodbye!")
        self.exit = True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NarrowAI interactive CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--backend", choices=["stub", "openai", "ollama"],
        default="stub",
        help="Model backend to use (default: stub)",
    )
    parser.add_argument("--model", default=None, help="Model name (for HTTP backends)")
    parser.add_argument("--api-key", default=None, help="API key (for OpenAI)")
    parser.add_argument("--knowledge", default=None, help="Path to knowledge file")

    args = parser.parse_args()

    cli = CLI()
    cli.setup(args)
    cli.run()


if __name__ == "__main__":
    main()
