"""Pure-stdlib HTTP client for Google's Gemini Decisions API (v1beta/decisions)."""

from __future__ import annotations

import concurrent.futures
import json
import ssl
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .validation import (
    DEFAULT_ENDPOINT,
    DEFAULT_MAX_FILE_MB,
    DEFAULT_MODEL,
    DecisionsAPIError,
    DecisionsAuthError,
    DecisionsSafetyBlockedError,
    DecisionsValidationError,
    normalize_input,
    validate_endpoint,
    validate_questions,
)

USER_AGENT = "hermes-gemini-decisions/0.1.0"
SAFETY_ERROR_CODES = frozenset({"safety", "prohibited_content", "blocklist", "image_safety", "content_blocked"})
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
MAX_RESPONSE_BYTES = 10 * 1024 * 1024


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Refuse HTTP redirects so x-goog-api-key can never be forwarded to another host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        raise DecisionsAPIError(
            f"Refusing HTTP redirect ({code}) from Decisions endpoint",
            status_code=code,
            code="redirect_refused",
        )


@dataclass(frozen=True)
class DecisionResult:
    """Typed result returned by a single Gemini Decisions API call."""

    answers: dict[str, dict[str, Any]]
    usage: dict[str, Any]
    model: str
    latency_ms: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "answers": self.answers,
            "usage": self.usage,
            "latency_ms": self.latency_ms,
        }


def _parse_error_payload(raw_body: str) -> tuple[str | None, str]:
    try:
        parsed = json.loads(raw_body)
    except Exception:
        return None, raw_body.strip()[:300] or "Unknown API error"
    err = parsed.get("error") if isinstance(parsed, dict) else None
    if isinstance(err, dict):
        code = str(err.get("code") or err.get("status") or "") or None
        msg = str(err.get("message") or raw_body[:300])
        return code, msg
    return None, raw_body.strip()[:300] or "Unknown API error"


def _extract_sort_value(answers: dict[str, dict[str, Any]], question_key: str) -> float:
    ans = answers.get(question_key)
    if not isinstance(ans, dict):
        return float("-inf")
    qtype = ans.get("type")
    if qtype == "noul":
        return float(ans.get("noul", 0.0))
    if qtype == "score":
        return float(ans.get("score", 0.0))
    if qtype == "choice":
        return float(ans.get("confidence", 0.0))
    return float("-inf")


def _matches_filter(
    answers: dict[str, dict[str, Any]],
    *,
    filter_question: str,
    min_probability: float | None,
    min_score: float | None,
    match_choice: str | None,
    min_confidence: float | None,
) -> bool:
    ans = answers.get(filter_question)
    if not isinstance(ans, dict):
        return False
    if min_confidence is not None and float(ans.get("confidence", 0.0)) < min_confidence:
        return False
    qtype = ans.get("type")
    if qtype == "noul" and min_probability is not None:
        if float(ans.get("noul", 0.0)) < min_probability:
            return False
    elif qtype == "score" and min_score is not None:
        if float(ans.get("score", 0.0)) < min_score:
            return False
    elif qtype == "choice":
        if match_choice is not None and ans.get("choice") != match_choice:
            return False
        if min_probability is not None:
            probs = ans.get("probabilities") or {}
            chosen = ans.get("choice")
            if float(probs.get(chosen, 0.0)) < min_probability:
                return False
    return True


class GeminiDecisionsClient:
    """Zero-dependency client for Google's Gemini Decisions API (v1beta/decisions)."""

    def __init__(
        self,
        api_key: str,
        *,
        endpoint: str = DEFAULT_ENDPOINT,
        model: str = DEFAULT_MODEL,
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
    ) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise DecisionsAuthError(
                "Missing Gemini API key. Configure GEMINI_API_KEY via Hermes secrets or pass api_key explicitly."
            )
        self._api_key = api_key.strip()
        self.endpoint = validate_endpoint(endpoint)
        if not isinstance(model, str) or not model.strip():
            raise DecisionsValidationError("model must be a non-empty string")
        self.model = model.strip()
        self.timeout_seconds = max(1.0, min(float(timeout_seconds), 300.0))
        self.max_retries = max(0, min(int(max_retries), 5))
        self._opener = urllib.request.build_opener(
            urllib.request.HTTPSHandler(context=ssl.create_default_context()),
            _NoRedirectHandler(),
        )

    def build_request_payload(
        self,
        *,
        questions: dict[str, Any],
        text: str | None = None,
        media: list[dict[str, Any]] | None = None,
        input_data: str | dict[str, Any] | list[dict[str, Any]] | None = None,
        max_file_mb: int = DEFAULT_MAX_FILE_MB,
        base_dir: Path | None = None,
    ) -> dict[str, Any]:
        """Validate and construct the wire JSON payload without performing network I/O."""
        norm_input = normalize_input(
            text=text,
            media=media,
            input_data=input_data,
            max_file_mb=max_file_mb,
            base_dir=base_dir,
        )
        norm_questions = validate_questions(questions)
        return {
            "model": self.model,
            "input": norm_input,
            "questions": norm_questions,
        }

    def _post_json(self, payload: dict[str, Any]) -> tuple[dict[str, Any], int]:
        body_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self._api_key,
            "User-Agent": USER_AGENT,
        }

        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            req = urllib.request.Request(
                self.endpoint,
                data=body_bytes,
                headers=headers,
                method="POST",
            )
            started = time.monotonic()
            try:
                with self._opener.open(req, timeout=self.timeout_seconds) as resp:
                    raw = resp.read(MAX_RESPONSE_BYTES + 1)
                    if len(raw) > MAX_RESPONSE_BYTES:
                        raise DecisionsAPIError(
                            "Decisions API response exceeded 10 MiB safety cap",
                            status_code=resp.status,
                            code="response_too_large",
                        )
                    latency_ms = int((time.monotonic() - started) * 1000)
                    try:
                        decoded = json.loads(raw.decode("utf-8"))
                    except Exception as exc:
                        raise DecisionsAPIError(
                            "Decisions API returned invalid JSON",
                            status_code=resp.status,
                            code="invalid_json",
                        ) from exc
                    if not isinstance(decoded, dict) or not isinstance(decoded.get("answers"), dict):
                        raise DecisionsAPIError(
                            "Decisions API response missing 'answers' object",
                            status_code=resp.status,
                            code="unexpected_shape",
                        )
                    return decoded, latency_ms

            except urllib.error.HTTPError as exc:
                raw_err = ""
                try:
                    raw_err = exc.read(65536).decode("utf-8", errors="replace")
                except Exception:
                    raw_err = str(exc.reason)
                code, message = _parse_error_payload(raw_err)

                if exc.code in {401, 403}:
                    raise DecisionsAuthError(f"Gemini API authentication failed (HTTP {exc.code}): {message}") from None
                if exc.code == 400 and code and code.lower() in SAFETY_ERROR_CODES:
                    raise DecisionsSafetyBlockedError(code.lower(), message) from None
                if exc.code in RETRYABLE_STATUS_CODES and attempt < self.max_retries:
                    retry_after_hdr = exc.headers.get("Retry-After") if exc.headers else None
                    delay = 0.5 * (3**attempt)
                    if retry_after_hdr:
                        try:
                            delay = min(float(retry_after_hdr), 10.0)
                        except ValueError:
                            pass
                    time.sleep(delay)
                    last_exc = DecisionsAPIError(message, status_code=exc.code, code=code)
                    continue
                raise DecisionsAPIError(
                    f"Gemini Decisions API error (HTTP {exc.code}): {message}",
                    status_code=exc.code,
                    code=code,
                ) from None

            except DecisionsAPIError:
                raise

            except urllib.error.URLError as exc:
                if attempt < self.max_retries:
                    time.sleep(0.5 * (3**attempt))
                    last_exc = DecisionsAPIError(f"Network error contacting Decisions API: {exc.reason}")
                    continue
                raise DecisionsAPIError(
                    f"Network error contacting Decisions API: {exc.reason}",
                    code="network_error",
                ) from None

            except TimeoutError:
                if attempt < self.max_retries:
                    time.sleep(0.5 * (3**attempt))
                    last_exc = DecisionsAPIError("Decisions API request timed out", code="timeout")
                    continue
                raise DecisionsAPIError(
                    f"Decisions API request timed out after {self.timeout_seconds:.0f}s",
                    code="timeout",
                ) from None

        if last_exc is not None:
            raise last_exc
        raise DecisionsAPIError("Decisions API request failed")

    def decide(
        self,
        *,
        questions: dict[str, Any],
        text: str | None = None,
        media: list[dict[str, Any]] | None = None,
        input_data: str | dict[str, Any] | list[dict[str, Any]] | None = None,
        max_file_mb: int = DEFAULT_MAX_FILE_MB,
        base_dir: Path | None = None,
    ) -> DecisionResult:
        """Evaluate 1 to 1,000 typed questions against one shared text/multimodal input."""
        payload = self.build_request_payload(
            questions=questions,
            text=text,
            media=media,
            input_data=input_data,
            max_file_mb=max_file_mb,
            base_dir=base_dir,
        )
        decoded, latency_ms = self._post_json(payload)
        raw_usage = decoded.get("usage")
        usage: dict[str, Any] = raw_usage if isinstance(raw_usage, dict) else {}
        return DecisionResult(
            answers=decoded["answers"],
            usage=usage,
            model=self.model,
            latency_ms=latency_ms,
        )

    def decide_batch(
        self,
        *,
        items: list[dict[str, Any]],
        questions: dict[str, Any],
        max_concurrency: int = 4,
        max_items: int = 50,
        max_file_mb: int = DEFAULT_MAX_FILE_MB,
        base_dir: Path | None = None,
        filter_question: str | None = None,
        min_probability: float | None = None,
        min_score: float | None = None,
        match_choice: str | None = None,
        min_confidence: float | None = None,
        sort_by: str | None = None,
        descending: bool = True,
    ) -> dict[str, Any]:
        """Evaluate multiple items against one shared question set with bounded concurrency."""
        if not isinstance(items, list) or not items:
            raise DecisionsValidationError("items must be a non-empty list")
        if len(items) > max_items:
            raise DecisionsValidationError(f"Batch has {len(items)} items, exceeding limit of {max_items}")
        norm_questions = validate_questions(questions)

        if filter_question is not None and filter_question not in norm_questions:
            raise DecisionsValidationError(f"filter_question {filter_question!r} is not in questions")
        if sort_by is not None and sort_by not in norm_questions:
            raise DecisionsValidationError(f"sort_by {sort_by!r} is not in questions")

        # Validate every item eagerly before dispatching any network calls.
        prepared: list[tuple[str, dict[str, Any]]] = []
        seen_ids: set[str] = set()
        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                raise DecisionsValidationError(f"items[{idx}] must be an object")
            item_id = str(item.get("id") if item.get("id") is not None else idx)
            if item_id in seen_ids:
                raise DecisionsValidationError(f"Duplicate batch item id: {item_id!r}")
            seen_ids.add(item_id)
            payload = self.build_request_payload(
                questions=norm_questions,
                text=item.get("text"),
                media=item.get("media"),
                input_data=item.get("input"),
                max_file_mb=max_file_mb,
                base_dir=base_dir,
            )
            prepared.append((item_id, payload))

        workers = max(1, min(int(max_concurrency), 10, len(prepared)))

        def _run_one(entry: tuple[str, dict[str, Any]]) -> dict[str, Any]:
            item_id, payload = entry
            try:
                decoded, latency_ms = self._post_json(payload)
                return {
                    "id": item_id,
                    "ok": True,
                    "answers": decoded["answers"],
                    "usage": decoded.get("usage") if isinstance(decoded.get("usage"), dict) else {},
                    "latency_ms": latency_ms,
                }
            except DecisionsSafetyBlockedError as exc:
                return {
                    "id": item_id,
                    "ok": False,
                    "error": exc.message,
                    "error_type": "safety_blocked",
                    "error_code": exc.code,
                }
            except DecisionsAuthError as exc:
                return {
                    "id": item_id,
                    "ok": False,
                    "error": str(exc),
                    "error_type": "auth_error",
                }
            except DecisionsAPIError as exc:
                return {
                    "id": item_id,
                    "ok": False,
                    "error": str(exc),
                    "error_type": "api_error",
                    "error_code": exc.code,
                }

        results_by_id: dict[str, dict[str, Any]] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            future_map = {pool.submit(_run_one, entry): entry[0] for entry in prepared}
            for fut in concurrent.futures.as_completed(future_map):
                res = fut.result()
                results_by_id[res["id"]] = res

        ordered = [results_by_id[item_id] for item_id, _ in prepared]

        total_input_tokens = sum(
            int((r.get("usage") or {}).get("total_input_tokens", 0)) for r in ordered if r.get("ok")
        )
        succeeded = sum(1 for r in ordered if r.get("ok"))
        failed = len(ordered) - succeeded

        filtered_out = 0
        if filter_question is not None:
            kept: list[dict[str, Any]] = []
            for row in ordered:
                if not row.get("ok"):
                    kept.append(row)
                    continue
                if _matches_filter(
                    row["answers"],
                    filter_question=filter_question,
                    min_probability=min_probability,
                    min_score=min_score,
                    match_choice=match_choice,
                    min_confidence=min_confidence,
                ):
                    kept.append(row)
                else:
                    filtered_out += 1
            ordered = kept

        if sort_by is not None:
            ok_rows = [r for r in ordered if r.get("ok")]
            err_rows = [r for r in ordered if not r.get("ok")]
            ok_rows.sort(
                key=lambda r: _extract_sort_value(r["answers"], sort_by),
                reverse=descending,
            )
            ordered = ok_rows + err_rows

        return {
            "model": self.model,
            "total_items": len(prepared),
            "succeeded": succeeded,
            "failed": failed,
            "filtered_out": filtered_out,
            "returned_items": len(ordered),
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": 0,
            "results": ordered,
        }
