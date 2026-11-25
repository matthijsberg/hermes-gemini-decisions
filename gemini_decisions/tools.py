"""Hermes Agent tool handlers for gemini_decide and gemini_decide_batch."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .client import GeminiDecisionsClient
from .validation import (
    DEFAULT_ENDPOINT,
    DEFAULT_MAX_FILE_MB,
    DEFAULT_MODEL,
    DecisionsAPIError,
    DecisionsAuthError,
    DecisionsSafetyBlockedError,
    DecisionsValidationError,
)


def _clamp_int(value: Any, default: int, low: int, high: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(parsed, high))


def _resolve_api_key_from_hermes() -> str:
    """Resolve Gemini API key strictly via Hermes's profile-scoped secret store."""
    try:
        from agent.secret_scope import get_secret  # type: ignore[import-not-found]
    except ImportError as exc:
        raise DecisionsAuthError("Hermes secret store (agent.secret_scope) is unavailable.") from exc

    key = (get_secret("GEMINI_API_KEY") or get_secret("GOOGLE_API_KEY") or "").strip()
    if not key:
        raise DecisionsAuthError(
            "GEMINI_API_KEY is not configured for this Hermes profile. "
            "Set it in your profile secrets (or Desktop Settings -> Plugins -> gemini-decisions)."
        )
    return key


def _resolve_session_cwd() -> Path | None:
    try:
        from agent.runtime_cwd import resolve_agent_cwd  # type: ignore[import-not-found]

        cwd = resolve_agent_cwd()
        return Path(cwd) if cwd else None
    except Exception:
        return None


def _read_runtime_settings(ctx: Any) -> dict[str, Any]:
    """Read plugin settings dynamically via ctx.get_config with safe bounds."""
    endpoint = str(ctx.get_config("endpoint", default=DEFAULT_ENDPOINT) or DEFAULT_ENDPOINT)
    model = str(ctx.get_config("decision_model", default=DEFAULT_MODEL) or DEFAULT_MODEL)
    timeout_seconds = _clamp_int(ctx.get_config("timeout_seconds", default=30), 30, 1, 300)
    max_concurrency = _clamp_int(ctx.get_config("max_concurrency", default=4), 4, 1, 10)
    max_batch_items = _clamp_int(ctx.get_config("max_batch_items", default=50), 50, 1, 200)
    max_file_mb = _clamp_int(
        ctx.get_config("max_file_mb", default=DEFAULT_MAX_FILE_MB),
        DEFAULT_MAX_FILE_MB,
        1,
        50,
    )
    return {
        "endpoint": endpoint,
        "model": model,
        "timeout_seconds": timeout_seconds,
        "max_concurrency": max_concurrency,
        "max_batch_items": max_batch_items,
        "max_file_mb": max_file_mb,
    }


def _error_json(error_type: str, message: str, **extra: Any) -> str:
    payload: dict[str, Any] = {
        "success": False,
        "error_type": error_type,
        "error": message,
    }
    payload.update({k: v for k, v in extra.items() if v is not None})
    return json.dumps(payload, ensure_ascii=False)


def build_decide_handler(ctx: Any):
    """Create the Hermes tool handler for `gemini_decide`."""

    def handle_gemini_decide(args: dict[str, Any], **kwargs: Any) -> str:
        del kwargs
        try:
            settings = _read_runtime_settings(ctx)
            api_key = _resolve_api_key_from_hermes()
            client = GeminiDecisionsClient(
                api_key=api_key,
                endpoint=settings["endpoint"],
                model=settings["model"],
                timeout_seconds=settings["timeout_seconds"],
            )
            raw_questions = args.get("questions")
            result = client.decide(
                text=args.get("text"),
                media=args.get("media"),
                questions=raw_questions if isinstance(raw_questions, dict) else {},
                max_file_mb=settings["max_file_mb"],
                base_dir=_resolve_session_cwd(),
            )
            return json.dumps({"success": True, **result.to_dict()}, ensure_ascii=False)
        except DecisionsValidationError as exc:
            return _error_json("validation_error", str(exc))
        except DecisionsAuthError as exc:
            return _error_json("auth_error", str(exc))
        except DecisionsSafetyBlockedError as exc:
            return _error_json("safety_blocked", exc.message, error_code=exc.code)
        except DecisionsAPIError as exc:
            return _error_json(
                "api_error",
                str(exc),
                status_code=exc.status_code,
                error_code=exc.code,
            )
        except Exception as exc:
            return _error_json("unexpected_error", f"Unexpected error: {exc}")

    return handle_gemini_decide


def build_decide_batch_handler(ctx: Any):
    """Create the Hermes tool handler for `gemini_decide_batch`."""

    def handle_gemini_decide_batch(args: dict[str, Any], **kwargs: Any) -> str:
        del kwargs
        try:
            settings = _read_runtime_settings(ctx)
            api_key = _resolve_api_key_from_hermes()
            client = GeminiDecisionsClient(
                api_key=api_key,
                endpoint=settings["endpoint"],
                model=settings["model"],
                timeout_seconds=settings["timeout_seconds"],
            )
            raw_items = args.get("items")
            raw_questions = args.get("questions")
            batch = client.decide_batch(
                items=raw_items if isinstance(raw_items, list) else [],
                questions=raw_questions if isinstance(raw_questions, dict) else {},
                max_concurrency=settings["max_concurrency"],
                max_items=settings["max_batch_items"],
                max_file_mb=settings["max_file_mb"],
                base_dir=_resolve_session_cwd(),
                filter_question=args.get("filter_question"),
                min_probability=args.get("min_probability"),
                min_score=args.get("min_score"),
                match_choice=args.get("match_choice"),
                min_confidence=args.get("min_confidence"),
                sort_by=args.get("sort_by"),
                descending=bool(args.get("descending", True)),
            )
            return json.dumps({"success": True, **batch}, ensure_ascii=False)
        except DecisionsValidationError as exc:
            return _error_json("validation_error", str(exc))
        except DecisionsAuthError as exc:
            return _error_json("auth_error", str(exc))
        except DecisionsAPIError as exc:
            return _error_json(
                "api_error",
                str(exc),
                status_code=exc.status_code,
                error_code=exc.code,
            )
        except Exception as exc:
            return _error_json("unexpected_error", f"Unexpected error: {exc}")

    return handle_gemini_decide_batch
