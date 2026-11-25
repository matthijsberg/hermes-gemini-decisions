"""Unit tests for input, question, endpoint, and media safety validation."""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

from gemini_decisions.validation import (
    DecisionsValidationError,
    normalize_input,
    validate_endpoint,
    validate_media_uri,
    validate_questions,
)


def test_validate_endpoint_accepts_https():
    url = "https://generativelanguage.googleapis.com/v1beta/decisions"
    assert validate_endpoint(url) == url


@pytest.mark.parametrize(
    "bad_url",
    [
        "",
        "http://generativelanguage.googleapis.com/v1beta/decisions",
        "https://user:pass@generativelanguage.googleapis.com/v1beta/decisions",
        "https://generativelanguage.googleapis.com/v1beta/decisions?key=leak",
        "https://generativelanguage.googleapis.com/v1beta/decisions#frag",
        "file:///etc/passwd",
    ],
)
def test_validate_endpoint_rejects_unsafe(bad_url: str):
    with pytest.raises(DecisionsValidationError):
        validate_endpoint(bad_url)


@pytest.mark.parametrize(
    "bad_uri",
    [
        "http://example.com/a.pdf",
        "gs://bucket/file.pdf",
        "file:///tmp/a.pdf",
        "https://www.youtube.com/watch?v=12345",
        "https://youtu.be/abc123",
        "https://127.0.0.1/secret.pdf",
        "https://localhost/secret.pdf",
        "https://169.254.169.254/latest/meta-data",
        "https://10.0.0.1/internal.png",
        "https://user:pass@example.com/a.png",
    ],
)
def test_validate_media_uri_rejects_disallowed(bad_uri: str):
    with pytest.raises(DecisionsValidationError):
        validate_media_uri(bad_uri)


def test_validate_questions_all_three_types():
    qs = validate_questions(
        {
            "is_urgent": {
                "type": "noul",
                "instructions": "Is this urgent?",
                "criteria": {
                    "true_description": "Needs reply today.",
                    "false_description": "Can wait.",
                },
            },
            "queue": {
                "type": "choice",
                "instructions": "Pick queue",
                "criteria": {"billing": "Invoices", "other": ""},
            },
            "severity": {
                "type": "score",
                "instructions": "Severity level",
                "criteria": ["Low", "Medium", "High"],
            },
        }
    )
    assert set(qs.keys()) == {"is_urgent", "queue", "severity"}


def test_validate_questions_rejects_bad_keys_and_limits():
    with pytest.raises(DecisionsValidationError):
        validate_questions({})
    with pytest.raises(DecisionsValidationError):
        validate_questions({"bad key!": {"type": "noul", "instructions": "Q?"}})
    with pytest.raises(DecisionsValidationError):
        validate_questions(
            {
                "q": {
                    "type": "choice",
                    "instructions": "Pick",
                    "criteria": {"only_one": "Not enough"},
                }
            }
        )
    with pytest.raises(DecisionsValidationError):
        validate_questions(
            {
                "q": {
                    "type": "choice",
                    "instructions": "Pick",
                    "criteria": {" leading": "Bad", "ok": "Ok"},
                }
            }
        )
    with pytest.raises(DecisionsValidationError):
        validate_questions(
            {
                "q": {
                    "type": "score",
                    "instructions": "Rate",
                    "criteria": ["Only one"],
                }
            }
        )
    with pytest.raises(DecisionsValidationError):
        validate_questions(
            {
                "q": {
                    "type": "noul",
                    "instructions": "Q?",
                    "criteria": {"invalid_key": "nope"},
                }
            }
        )


def test_normalize_input_local_pdf_and_blocks_secrets(tmp_path: Path):
    pdf = tmp_path / "invoice.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%fake-test-pdf\n")
    blocks = normalize_input(
        text="Check invoice",
        media=[{"file_path": "invoice.pdf"}],
        base_dir=tmp_path,
    )
    assert isinstance(blocks, list)
    assert blocks[0] == {"type": "text", "text": "Check invoice"}
    assert blocks[1]["type"] == "document"
    assert blocks[1]["mime_type"] == "application/pdf"
    assert base64.b64decode(blocks[1]["data"]) == b"%PDF-1.4\n%fake-test-pdf\n"

    # Secret filenames must be blocked even if MIME is forced.
    env_file = tmp_path / ".env"
    env_file.write_text("SECRET=123", encoding="utf-8")
    with pytest.raises(DecisionsValidationError, match="secret/credential"):
        normalize_input(
            media=[{"file_path": str(env_file), "mime_type": "application/pdf"}],
        )

    pem_file = tmp_path / "server.pem"
    pem_file.write_text("PRIVATE KEY", encoding="utf-8")
    with pytest.raises(DecisionsValidationError, match="secret/credential"):
        normalize_input(
            media=[{"file_path": str(pem_file), "mime_type": "application/pdf"}],
        )


def test_normalize_input_rejects_oversized_and_bad_resolution(tmp_path: Path):
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"\xff\xd8\xff\xe0" + b"0" * 100)
    blocks = normalize_input(
        media=[{"file_path": str(img), "resolution": "low"}],
    )
    assert isinstance(blocks, list)
    assert blocks[0]["resolution"] == "low"

    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4\n12345")
    with pytest.raises(DecisionsValidationError, match="resolution"):
        normalize_input(media=[{"file_path": str(pdf), "resolution": "low"}])

    with pytest.raises(DecisionsValidationError, match="empty"):
        normalize_input(text="   ")
