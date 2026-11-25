"""Hermes Agent plugin & zero-dependency client for Google's Gemini Decisions API."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import cli, schemas, tools
from .client import DecisionResult, GeminiDecisionsClient
from .validation import (
    DEFAULT_ENDPOINT,
    DEFAULT_MODEL,
    DecisionsAPIError,
    DecisionsAuthError,
    DecisionsError,
    DecisionsSafetyBlockedError,
    DecisionsValidationError,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "DEFAULT_ENDPOINT",
    "DEFAULT_MODEL",
    "DecisionResult",
    "DecisionsAPIError",
    "DecisionsAuthError",
    "DecisionsError",
    "DecisionsSafetyBlockedError",
    "DecisionsValidationError",
    "GeminiDecisionsClient",
    "register",
]


def register(ctx: Any) -> None:
    """Register Gemini Decisions tools, CLI command, and bundled skill with Hermes."""
    ctx.register_tool(
        name="gemini_decide",
        toolset="gemini_decisions",
        schema=schemas.GEMINI_DECIDE,
        handler=tools.build_decide_handler(ctx),
    )
    ctx.register_tool(
        name="gemini_decide_batch",
        toolset="gemini_decisions",
        schema=schemas.GEMINI_DECIDE_BATCH,
        handler=tools.build_decide_batch_handler(ctx),
    )
    ctx.register_cli_command(
        name="gemini-decisions",
        help="Evaluate calibrated yes/no, choice, and rubric decisions via Gemini Decisions API",
        setup_fn=cli.setup_cli_parser,
        handler_fn=cli.handle_cli,
    )

    skill_path = Path(__file__).resolve().parents[1] / "skills" / "gemini-decisions" / "SKILL.md"
    if skill_path.is_file() and hasattr(ctx, "register_skill"):
        ctx.register_skill("gemini-decisions", str(skill_path))
