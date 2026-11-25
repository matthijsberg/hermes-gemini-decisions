"""Unit tests for GeminiDecisionsClient HTTP transport, retries, errors, and batching."""

from __future__ import annotations

import io
import json
import urllib.error
from email.message import Message
from unittest.mock import MagicMock

import pytest

from gemini_decisions import (
    DecisionsAPIError,
    DecisionsAuthError,
    DecisionsSafetyBlockedError,
    DecisionsValidationError,
    GeminiDecisionsClient,
)


def _mock_response(payload: dict, status: int = 200):
    raw = json.dumps(payload).encode("utf-8")
    resp = MagicMock()
    resp.status = status
    resp.read.side_effect = lambda n=-1: raw if n < 0 else raw[:n]
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False
    return resp


def test_decide_happy_path_sends_header_and_parses_result():
    client = GeminiDecisionsClient(api_key="test-key-123")
    captured = {}

    def fake_open(req, timeout=30):
        captured["url"] = req.full_url
        captured["headers"] = dict(req.header_items())
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _mock_response(
            {
                "answers": {
                    "is_urgent": {"type": "noul", "noul": 0.92, "confidence": 0.92},
                },
                "usage": {
                    "total_input_tokens": 80,
                    "total_output_tokens": 0,
                    "total_tokens": 80,
                },
            }
        )

    client._opener.open = fake_open  # type: ignore[method-assign]
    res = client.decide(
        text="Invoice overdue!",
        questions={"is_urgent": {"type": "noul", "instructions": "Is this urgent?"}},
    )
    assert captured["url"] == "https://generativelanguage.googleapis.com/v1beta/decisions"
    assert captured["headers"]["X-goog-api-key"] == "test-key-123"
    assert "key=" not in captured["url"]
    assert res.answers["is_urgent"]["noul"] == 0.92
    assert res.usage["total_output_tokens"] == 0


def test_decide_safety_block_raises_typed_error():
    client = GeminiDecisionsClient(api_key="test-key-123", max_retries=0)
    err_body = json.dumps({"error": {"code": "safety", "message": "Request blocked due to safety violations."}}).encode(
        "utf-8"
    )

    def fake_open(req, timeout=30):
        raise urllib.error.HTTPError(req.full_url, 400, "Bad Request", Message(), io.BytesIO(err_body))

    client._opener.open = fake_open  # type: ignore[method-assign]
    with pytest.raises(DecisionsSafetyBlockedError) as exc_info:
        client.decide(
            text="blocked input",
            questions={"q": {"type": "noul", "instructions": "Check?"}},
        )
    assert exc_info.value.code == "safety"


def test_decide_auth_error_does_not_retry():
    client = GeminiDecisionsClient(api_key="bad-key", max_retries=2)
    calls = 0

    def fake_open(req, timeout=30):
        nonlocal calls
        calls += 1
        err_body = json.dumps({"error": {"message": "API key not valid"}}).encode("utf-8")
        raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", Message(), io.BytesIO(err_body))

    client._opener.open = fake_open  # type: ignore[method-assign]
    with pytest.raises(DecisionsAuthError):
        client.decide(text="hi", questions={"q": {"type": "noul", "instructions": "Ok?"}})
    assert calls == 1


def test_decide_retries_on_429_and_succeeds(monkeypatch):
    monkeypatch.setattr("gemini_decisions.client.time.sleep", lambda _s: None)
    client = GeminiDecisionsClient(api_key="test-key", max_retries=2)
    calls = 0

    def fake_open(req, timeout=30):
        nonlocal calls
        calls += 1
        if calls == 1:
            err_body = json.dumps({"error": {"message": "Rate limited"}}).encode("utf-8")
            raise urllib.error.HTTPError(req.full_url, 429, "Too Many Requests", Message(), io.BytesIO(err_body))
        return _mock_response(
            {
                "answers": {"q": {"type": "noul", "noul": 0.85, "confidence": 0.85}},
                "usage": {"total_input_tokens": 50, "total_output_tokens": 0},
            }
        )

    client._opener.open = fake_open  # type: ignore[method-assign]
    res = client.decide(text="hi", questions={"q": {"type": "noul", "instructions": "Ok?"}})
    assert calls == 2
    assert res.answers["q"]["noul"] == 0.85


def test_decide_batch_filters_and_sorts():
    client = GeminiDecisionsClient(api_key="test-key", max_retries=0)
    scores_by_text = {
        "Promo email": 0.05,
        "School deadline": 0.91,
        "Dentist appointment": 0.72,
    }

    def fake_open(req, timeout=30):
        body = json.loads(req.data.decode("utf-8"))
        p = scores_by_text[body["input"]]
        return _mock_response(
            {
                "answers": {"is_actionable": {"type": "noul", "noul": p, "confidence": max(p, 1 - p)}},
                "usage": {"total_input_tokens": 40, "total_output_tokens": 0},
            }
        )

    client._opener.open = fake_open  # type: ignore[method-assign]
    out = client.decide_batch(
        items=[
            {"id": "promo", "text": "Promo email"},
            {"id": "dentist", "text": "Dentist appointment"},
            {"id": "school", "text": "School deadline"},
        ],
        questions={"is_actionable": {"type": "noul", "instructions": "Action needed?"}},
        filter_question="is_actionable",
        min_probability=0.50,
        sort_by="is_actionable",
        descending=True,
    )
    assert out["total_items"] == 3
    assert out["succeeded"] == 3
    assert out["filtered_out"] == 1
    assert [r["id"] for r in out["results"]] == ["school", "dentist"]
    assert out["total_input_tokens"] == 120
    assert out["total_output_tokens"] == 0


def test_decide_batch_rejects_duplicate_ids_and_missing_filter_key():
    client = GeminiDecisionsClient(api_key="test-key")
    with pytest.raises(DecisionsValidationError, match="Duplicate"):
        client.decide_batch(
            items=[{"id": "dup", "text": "a"}, {"id": "dup", "text": "b"}],
            questions={"q": {"type": "noul", "instructions": "Q?"}},
        )
    with pytest.raises(DecisionsValidationError, match="filter_question"):
        client.decide_batch(
            items=[{"id": "a", "text": "a"}],
            questions={"q": {"type": "noul", "instructions": "Q?"}},
            filter_question="nonexistent",
        )


def test_empty_api_key_raises_auth_error():
    with pytest.raises(DecisionsAuthError):
        GeminiDecisionsClient(api_key="   ")


def test_malformed_json_raises_api_error():
    client = GeminiDecisionsClient(api_key="test-key", max_retries=0)
    resp = MagicMock()
    resp.status = 200
    resp.read.return_value = b"not-json"
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False
    client._opener.open = lambda req, timeout=30: resp  # type: ignore[method-assign]
    with pytest.raises(DecisionsAPIError, match="invalid JSON"):
        client.decide(text="hi", questions={"q": {"type": "noul", "instructions": "Q?"}})
