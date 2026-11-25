"""Hermes directory-plugin entrypoint for gemini-decisions."""

from __future__ import annotations

if __package__:
    from .gemini_decisions import (
        DEFAULT_ENDPOINT,
        DEFAULT_MODEL,
        DecisionResult,
        DecisionsAPIError,
        DecisionsAuthError,
        DecisionsError,
        DecisionsSafetyBlockedError,
        DecisionsValidationError,
        GeminiDecisionsClient,
        __version__,
        register,
    )
else:
    from gemini_decisions import (
        DEFAULT_ENDPOINT,
        DEFAULT_MODEL,
        DecisionResult,
        DecisionsAPIError,
        DecisionsAuthError,
        DecisionsError,
        DecisionsSafetyBlockedError,
        DecisionsValidationError,
        GeminiDecisionsClient,
        __version__,
        register,
    )

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
