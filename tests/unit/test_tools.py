"""Unit tests for Hermes tool handlers and secret_scope isolation."""

from __future__ import annotations

import json
from unittest.mock import patch

from gemini_decisions import DecisionResult, tools


class _FakeCtx:
    def __init__(self, settings: dict | None = None) -> None:
        self._settings = settings or {}

    def get_config(self, key: str, default=None):
        return self._settings.get(key, default)


def test_tool_handler_reads_secret_and_returns_success_json():
    ctx = _FakeCtx({"timeout_seconds": 15, "decision_model": "gemini-decision-1"})
    handler = tools.build_decide_handler(ctx)

    fake_res = DecisionResult(
        answers={"q": {"type": "noul", "noul": 0.95, "confidence": 0.95}},
        usage={"total_input_tokens": 64, "total_output_tokens": 0},
        model="gemini-decision-1",
        latency_ms=120,
    )

    with (
        patch.object(tools, "_resolve_api_key_from_hermes", return_value="secret-key"),
        patch.object(tools.GeminiDecisionsClient, "decide", return_value=fake_res),
    ):
        raw = handler(
            {
                "text": "Hello",
                "questions": {"q": {"type": "noul", "instructions": "Is this a greeting?"}},
            }
        )

    data = json.loads(raw)
    assert data["success"] is True
    assert data["answers"]["q"]["noul"] == 0.95
    assert data["usage"]["total_output_tokens"] == 0


def test_tool_handler_reports_validation_error_cleanly():
    ctx = _FakeCtx()
    handler = tools.build_decide_handler(ctx)

    with patch.object(tools, "_resolve_api_key_from_hermes", return_value="secret-key"):
        raw = handler({"text": "Hello", "questions": {}})

    data = json.loads(raw)
    assert data["success"] is False
    assert data["error_type"] == "validation_error"
    assert "secret-key" not in raw


def test_tool_batch_handler_respects_max_batch_items():
    ctx = _FakeCtx({"max_batch_items": 2})
    handler = tools.build_decide_batch_handler(ctx)

    with patch.object(tools, "_resolve_api_key_from_hermes", return_value="secret-key"):
        raw = handler(
            {
                "items": [
                    {"id": "1", "text": "a"},
                    {"id": "2", "text": "b"},
                    {"id": "3", "text": "c"},
                ],
                "questions": {"q": {"type": "noul", "instructions": "Q?"}},
            }
        )

    data = json.loads(raw)
    assert data["success"] is False
    assert data["error_type"] == "validation_error"
    assert "exceeding limit of 2" in data["error"]
