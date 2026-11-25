"""Input, question, media, and URL validation for the Gemini Decisions API."""

from __future__ import annotations

import base64
import ipaddress
import mimetypes
import re
import socket
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

DEFAULT_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/decisions"
DEFAULT_MODEL = "gemini-decision-1"

QUESTION_KEY_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
CONTROL_CHAR_RE = re.compile(r"[\x00-\x1f\x7f]")
NON_ASCII_SPACE_RE = re.compile(r"[^\x20\S]")

MAX_QUESTIONS = 1000
MIN_OPTIONS = 2
MAX_OPTIONS = 255
MAX_OPTION_KEY_CHARS = 128
MAX_OPTION_KEY_UTF8_BYTES = 512

MAX_PDF_BYTES = 50 * 1024 * 1024
MAX_TOTAL_INLINE_BYTES = 100 * 1024 * 1024
DEFAULT_MAX_FILE_MB = 20

RESOLUTION_VALUES = frozenset({"low", "medium", "high", "ultra_high"})

SUPPORTED_MIME_TYPES: dict[str, frozenset[str]] = {
    "image": frozenset(
        {
            "image/png",
            "image/jpeg",
            "image/webp",
            "image/heic",
            "image/heif",
            "image/gif",
            "image/bmp",
            "image/tiff",
        }
    ),
    "audio": frozenset(
        {
            "audio/wav",
            "audio/mp3",
            "audio/mpeg",
            "audio/aiff",
            "audio/aac",
            "audio/ogg",
            "audio/flac",
            "audio/m4a",
            "audio/opus",
            "audio/webm",
            "audio/l16",
            "audio/alaw",
            "audio/mulaw",
        }
    ),
    "video": frozenset(
        {
            "video/mp4",
            "video/mpeg",
            "video/mov",
            "video/avi",
            "video/webm",
            "video/wmv",
            "video/x-flv",
            "video/3gpp",
        }
    ),
    "document": frozenset({"application/pdf"}),
}

_EXTRA_EXTENSION_MIMES: dict[str, str] = {
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".heic": "image/heic",
    ".heif": "image/heif",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".wav": "audio/wav",
    ".mp3": "audio/mp3",
    ".ogg": "audio/ogg",
    ".opus": "audio/opus",
    ".flac": "audio/flac",
    ".aac": "audio/aac",
    ".m4a": "audio/m4a",
    ".aiff": "audio/aiff",
    ".mp4": "video/mp4",
    ".mov": "video/mov",
    ".avi": "video/avi",
    ".webm": "video/webm",
    ".wmv": "video/wmv",
    ".flv": "video/x-flv",
    ".3gp": "video/3gpp",
}

_BLOCKED_FILENAMES = frozenset(
    {
        ".env",
        "auth.json",
        "credentials.json",
        "token.json",
        "id_rsa",
        "id_ed25519",
        "id_ecdsa",
    }
)
_BLOCKED_SUFFIXES = frozenset({".pem", ".key", ".p12", ".pfx", ".jks"})
_YOUTUBE_HOSTS = frozenset(
    {
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
        "youtu.be",
        "music.youtube.com",
    }
)


class DecisionsError(Exception):
    """Base exception for Gemini Decisions operations."""


class DecisionsValidationError(DecisionsError):
    """Raised when inputs, questions, endpoints, or media fail validation."""


class DecisionsAuthError(DecisionsError):
    """Raised when credentials are missing or rejected by the API."""


class DecisionsSafetyBlockedError(DecisionsError):
    """Raised when Google's input safety filters block a Decisions request."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


class DecisionsAPIError(DecisionsError):
    """Raised on non-2xx API responses or malformed transport payloads."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code


def validate_endpoint(endpoint: str) -> str:
    """Validate that an API endpoint is clean HTTPS with no credentials, query, or fragment."""
    if not isinstance(endpoint, str) or not endpoint.strip():
        raise DecisionsValidationError("endpoint must be a non-empty HTTPS URL")
    cleaned = endpoint.strip()
    parsed = urlparse(cleaned)
    if parsed.scheme != "https":
        raise DecisionsValidationError(f"endpoint must use https:// (got {parsed.scheme!r})")
    if not parsed.hostname:
        raise DecisionsValidationError("endpoint must include a valid hostname")
    if parsed.username or parsed.password:
        raise DecisionsValidationError("endpoint must not contain embedded credentials")
    if parsed.query or parsed.fragment:
        raise DecisionsValidationError("endpoint must not contain query parameters or fragments")
    return cleaned


def _fallback_safe_https_url(uri: str) -> bool:
    """Pure-stdlib SSRF guard used when Hermes's tools.url_safety is unavailable."""
    parsed = urlparse(uri)
    if parsed.scheme != "https" or not parsed.hostname:
        return False
    if parsed.username or parsed.password:
        return False
    host = parsed.hostname.lower().rstrip(".")
    if host in {"localhost", "metadata.google.internal"} or host.endswith((".local", ".internal")):
        return False
    try:
        ip = ipaddress.ip_address(host)
        return not (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        )
    except ValueError:
        pass
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return True
    for info in infos:
        addr = info[4][0]
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError:
            continue
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            return False
    return True


def validate_media_uri(uri: str) -> str:
    """Validate a remote media URI for Gemini Decisions (HTTPS only, SSRF-safe, no YouTube/gs)."""
    if not isinstance(uri, str) or not uri.strip():
        raise DecisionsValidationError("media uri must be a non-empty string")
    cleaned = uri.strip()
    parsed = urlparse(cleaned)
    if parsed.scheme == "gs":
        raise DecisionsValidationError("Google Cloud Storage gs:// URIs are not supported by Decisions API")
    if parsed.scheme != "https":
        raise DecisionsValidationError(f"media uri must use https:// (got {parsed.scheme!r})")
    if not parsed.hostname:
        raise DecisionsValidationError("media uri must include a hostname")
    if parsed.username or parsed.password:
        raise DecisionsValidationError("media uri must not embed credentials")
    if parsed.hostname.lower() in _YOUTUBE_HOSTS:
        raise DecisionsValidationError("YouTube URLs are not supported by the Decisions API")

    try:
        from tools.url_safety import is_safe_url  # type: ignore[import-not-found]

        if not is_safe_url(cleaned):
            raise DecisionsValidationError("media uri blocked by URL safety check (private/unsafe host)")
    except ImportError:
        if not _fallback_safe_https_url(cleaned):
            raise DecisionsValidationError("media uri blocked by URL safety check (private/unsafe host)") from None

    return cleaned


def infer_mime_and_media_type(
    *,
    file_path: Path | None = None,
    uri: str | None = None,
    media_type: str | None = None,
    mime_type: str | None = None,
) -> tuple[str, str]:
    """Resolve and validate (media_type, mime_type) against Google's official allowlist."""
    resolved_mime = (mime_type or "").strip().lower()
    if not resolved_mime:
        suffix = ""
        if file_path is not None:
            suffix = file_path.suffix.lower()
        elif uri is not None:
            suffix = Path(urlparse(uri).path).suffix.lower()
        if suffix in _EXTRA_EXTENSION_MIMES:
            resolved_mime = _EXTRA_EXTENSION_MIMES[suffix]
        elif suffix:
            guessed, _ = mimetypes.guess_type(f"file{suffix}")
            resolved_mime = (guessed or "").lower()

    if not resolved_mime or resolved_mime == "application/octet-stream":
        raise DecisionsValidationError(
            "Unable to determine a supported MIME type; pass an explicit mime_type "
            "(e.g. application/pdf, image/jpeg, audio/mp3, video/mp4)."
        )

    resolved_type = (media_type or "").strip().lower()
    if not resolved_type:
        for candidate_type, allowed in SUPPORTED_MIME_TYPES.items():
            if resolved_mime in allowed:
                resolved_type = candidate_type
                break

    if resolved_type not in SUPPORTED_MIME_TYPES:
        raise DecisionsValidationError(
            f"Unsupported media type {resolved_type!r}. Expected one of: {', '.join(sorted(SUPPORTED_MIME_TYPES))}."
        )

    if resolved_mime not in SUPPORTED_MIME_TYPES[resolved_type]:
        raise DecisionsValidationError(
            f"MIME type {resolved_mime!r} is not supported for media type {resolved_type!r}."
        )
    return resolved_type, resolved_mime


def _is_blocked_secret_path(path: Path) -> bool:
    name = path.name.lower()
    if name in _BLOCKED_FILENAMES or name.startswith(".env."):
        return True
    if path.suffix.lower() in _BLOCKED_SUFFIXES:
        return True
    return False


def _build_media_block(
    spec: dict[str, Any],
    *,
    max_file_mb: int,
    base_dir: Path | None,
) -> tuple[dict[str, Any], int]:
    """Validate and convert one media attachment dict into a Decisions API Content block."""
    if not isinstance(spec, dict):
        raise DecisionsValidationError("Each media item must be an object with 'file_path' or 'uri'")

    file_path_raw = spec.get("file_path")
    uri_raw = spec.get("uri")
    data_raw = spec.get("data")
    provided = sum(1 for v in (file_path_raw, uri_raw, data_raw) if v is not None)
    if provided != 1:
        raise DecisionsValidationError("Each media item must specify exactly one of 'file_path', 'uri', or 'data'")

    media_type_raw = spec.get("type")
    mime_type_raw = spec.get("mime_type")
    resolution = spec.get("resolution")

    inline_bytes = 0

    if file_path_raw is not None:
        if not isinstance(file_path_raw, str) or not file_path_raw.strip():
            raise DecisionsValidationError("media.file_path must be a non-empty path string")
        raw_path = Path(file_path_raw).expanduser()
        if not raw_path.is_absolute() and base_dir is not None:
            raw_path = base_dir / raw_path
        try:
            resolved_path = raw_path.resolve(strict=True)
        except OSError as exc:
            raise DecisionsValidationError(f"Media file not found: {file_path_raw}") from exc
        if not resolved_path.is_file():
            raise DecisionsValidationError(f"Media path is not a regular file: {file_path_raw}")
        if _is_blocked_secret_path(resolved_path):
            raise DecisionsValidationError(
                f"Refusing to read potential secret/credential file as media: {resolved_path.name}"
            )

        media_type, mime_type = infer_mime_and_media_type(
            file_path=resolved_path,
            media_type=media_type_raw,
            mime_type=mime_type_raw,
        )
        size = resolved_path.stat().st_size
        if size == 0:
            raise DecisionsValidationError(f"Media file is empty: {resolved_path.name}")
        per_file_cap = min(max(1, max_file_mb), 50) * 1024 * 1024
        if media_type == "document":
            per_file_cap = min(per_file_cap, MAX_PDF_BYTES)
        if size > per_file_cap:
            raise DecisionsValidationError(
                f"Media file {resolved_path.name} ({size / (1024 * 1024):.1f} MB) exceeds "
                f"limit of {per_file_cap / (1024 * 1024):.0f} MB"
            )
        encoded = base64.b64encode(resolved_path.read_bytes()).decode("ascii")
        inline_bytes = size
        block: dict[str, Any] = {
            "type": media_type,
            "data": encoded,
            "mime_type": mime_type,
        }
    elif uri_raw is not None:
        safe_uri = validate_media_uri(uri_raw)
        media_type, mime_type = infer_mime_and_media_type(
            uri=safe_uri,
            media_type=media_type_raw,
            mime_type=mime_type_raw,
        )
        block = {
            "type": media_type,
            "uri": safe_uri,
            "mime_type": mime_type,
        }
    else:
        if not isinstance(data_raw, str) or not data_raw.strip():
            raise DecisionsValidationError("media.data must be a non-empty base64 string")
        try:
            decoded = base64.b64decode(data_raw, validate=True)
        except Exception as exc:
            raise DecisionsValidationError("media.data is not valid base64") from exc
        if not decoded:
            raise DecisionsValidationError("media.data decoded to 0 bytes")
        media_type, mime_type = infer_mime_and_media_type(
            media_type=media_type_raw,
            mime_type=mime_type_raw,
        )
        per_file_cap = min(max(1, max_file_mb), 50) * 1024 * 1024
        if len(decoded) > per_file_cap:
            raise DecisionsValidationError(
                f"Inline media payload exceeds limit of {per_file_cap / (1024 * 1024):.0f} MB"
            )
        inline_bytes = len(decoded)
        block = {
            "type": media_type,
            "data": data_raw.strip(),
            "mime_type": mime_type,
        }

    if resolution is not None:
        if media_type not in {"image", "video"}:
            raise DecisionsValidationError("media.resolution is only supported for image and video inputs")
        if resolution not in RESOLUTION_VALUES:
            raise DecisionsValidationError(f"media.resolution must be one of: {', '.join(sorted(RESOLUTION_VALUES))}")
        block["resolution"] = resolution

    return block, inline_bytes


def normalize_input(
    *,
    text: str | None = None,
    media: list[dict[str, Any]] | None = None,
    input_data: str | dict[str, Any] | list[dict[str, Any]] | None = None,
    max_file_mb: int = DEFAULT_MAX_FILE_MB,
    base_dir: Path | None = None,
) -> str | list[dict[str, Any]]:
    """Normalize text, media, or raw content blocks into a validated Decisions API `input` value."""
    if input_data is not None and (text is not None or media is not None):
        raise DecisionsValidationError("Pass either 'input_data' OR 'text'/'media', not both")

    if input_data is not None:
        if isinstance(input_data, str):
            if not input_data.strip():
                raise DecisionsValidationError("input text cannot be empty")
            return input_data
        raw_blocks = [input_data] if isinstance(input_data, dict) else input_data
        if not isinstance(raw_blocks, list) or not raw_blocks:
            raise DecisionsValidationError("input_data must be a non-empty string or list of content blocks")
        blocks: list[dict[str, Any]] = []
        total_inline = 0
        for item in raw_blocks:
            if not isinstance(item, dict):
                raise DecisionsValidationError("Every input content block must be an object")
            if item.get("type") == "text":
                t = item.get("text")
                if not isinstance(t, str) or not t:
                    raise DecisionsValidationError("text content block requires non-empty 'text'")
                blocks.append({"type": "text", "text": t})
            else:
                block, size = _build_media_block(item, max_file_mb=max_file_mb, base_dir=base_dir)
                total_inline += size
                blocks.append(block)
        if total_inline > MAX_TOTAL_INLINE_BYTES:
            raise DecisionsValidationError("Total inline media exceeds 100 MB request limit")
        return blocks

    clean_text = text if isinstance(text, str) and text.strip() else None
    media_list = media or []
    if media is not None and not isinstance(media, list):
        raise DecisionsValidationError("media must be a list of media objects")

    if clean_text is None and not media_list:
        raise DecisionsValidationError("Provide non-empty 'text' and/or at least one 'media' item")

    if clean_text is not None and not media_list:
        return clean_text

    blocks = []
    if clean_text is not None:
        blocks.append({"type": "text", "text": clean_text})

    total_inline = 0
    for spec in media_list:
        block, size = _build_media_block(spec, max_file_mb=max_file_mb, base_dir=base_dir)
        total_inline += size
        blocks.append(block)

    if total_inline > MAX_TOTAL_INLINE_BYTES:
        raise DecisionsValidationError("Total inline media exceeds 100 MB request limit")
    return blocks


def _validate_option_key(question_key: str, opt_key: str) -> None:
    if not isinstance(opt_key, str) or not opt_key:
        raise DecisionsValidationError(f"questions.{question_key}: choice option key must be a non-empty string")
    if len(opt_key) > MAX_OPTION_KEY_CHARS:
        raise DecisionsValidationError(
            f"questions.{question_key}: choice option key {opt_key!r} exceeds {MAX_OPTION_KEY_CHARS} chars"
        )
    if len(opt_key.encode("utf-8")) > MAX_OPTION_KEY_UTF8_BYTES:
        raise DecisionsValidationError(
            f"questions.{question_key}: choice option key {opt_key!r} exceeds {MAX_OPTION_KEY_UTF8_BYTES} UTF-8 bytes"
        )
    if opt_key != opt_key.strip(" "):
        raise DecisionsValidationError(
            f"questions.{question_key}: choice option key {opt_key!r} cannot have leading/trailing spaces"
        )
    if CONTROL_CHAR_RE.search(opt_key) or NON_ASCII_SPACE_RE.search(opt_key):
        raise DecisionsValidationError(
            f"questions.{question_key}: choice option key {opt_key!r} contains disallowed control/whitespace characters"
        )


def validate_questions(questions: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Validate and normalize a Decisions API `questions` map against all official limits."""
    if not isinstance(questions, dict) or not questions:
        raise DecisionsValidationError("questions must be a non-empty mapping of question_key -> question spec")
    if len(questions) > MAX_QUESTIONS:
        raise DecisionsValidationError(
            f"At most {MAX_QUESTIONS} questions are allowed per request (got {len(questions)})"
        )

    normalized: dict[str, dict[str, Any]] = {}
    for key, spec in questions.items():
        if not isinstance(key, str) or not QUESTION_KEY_RE.match(key):
            raise DecisionsValidationError(f"Invalid question key {key!r}: must match ^[A-Za-z0-9_-]{{1,64}}$")
        if not isinstance(spec, dict):
            raise DecisionsValidationError(f"questions.{key} must be an object")

        qtype = spec.get("type")
        if qtype not in {"noul", "choice", "score"}:
            raise DecisionsValidationError(f"questions.{key}.type must be 'noul', 'choice', or 'score' (got {qtype!r})")

        instructions = spec.get("instructions")
        if isinstance(instructions, str):
            if not instructions.strip():
                raise DecisionsValidationError(f"questions.{key}.instructions must be non-empty")
            norm_instructions: Any = instructions
        elif isinstance(instructions, list) and instructions:
            for idx, blk in enumerate(instructions):
                if (
                    not isinstance(blk, dict)
                    or blk.get("type") != "text"
                    or not isinstance(blk.get("text"), str)
                    or not blk["text"]
                ):
                    raise DecisionsValidationError(
                        f"questions.{key}.instructions[{idx}] must be a non-empty text block "
                        "(media in instructions is unsupported)"
                    )
            norm_instructions = instructions
        else:
            raise DecisionsValidationError(
                f"questions.{key}.instructions must be a non-empty string or list of text blocks"
            )

        entry: dict[str, Any] = {"type": qtype, "instructions": norm_instructions}
        criteria = spec.get("criteria")

        if qtype == "noul":
            if criteria is not None:
                if not isinstance(criteria, dict):
                    raise DecisionsValidationError(f"questions.{key}.criteria for 'noul' must be an object")
                allowed_noul_keys = {"true_description", "false_description"}
                unknown = set(criteria) - allowed_noul_keys
                if unknown:
                    raise DecisionsValidationError(
                        f"questions.{key}.criteria has unknown key(s) for 'noul': {', '.join(sorted(unknown))}"
                    )
                norm_criteria: dict[str, str] = {}
                for ck, cv in criteria.items():
                    if not isinstance(cv, str):
                        raise DecisionsValidationError(f"questions.{key}.criteria.{ck} must be a string")
                    norm_criteria[ck] = cv
                if norm_criteria:
                    entry["criteria"] = norm_criteria

        elif qtype == "choice":
            if not isinstance(criteria, dict):
                raise DecisionsValidationError(
                    f"questions.{key}.criteria for 'choice' must be an object mapping option_id -> description"
                )
            if not (MIN_OPTIONS <= len(criteria) <= MAX_OPTIONS):
                raise DecisionsValidationError(
                    f"questions.{key}.criteria for 'choice' must have {MIN_OPTIONS} to "
                    f"{MAX_OPTIONS} options (got {len(criteria)})"
                )
            norm_choice: dict[str, str] = {}
            for opt_key, opt_desc in criteria.items():
                _validate_option_key(key, opt_key)
                if not isinstance(opt_desc, str):
                    raise DecisionsValidationError(
                        f"questions.{key}.criteria.{opt_key} description must be a string (empty string allowed)"
                    )
                norm_choice[opt_key] = opt_desc
            entry["criteria"] = norm_choice

        elif qtype == "score":
            if not isinstance(criteria, list):
                raise DecisionsValidationError(
                    f"questions.{key}.criteria for 'score' must be a list of ordered level descriptions"
                )
            if not (MIN_OPTIONS <= len(criteria) <= MAX_OPTIONS):
                raise DecisionsValidationError(
                    f"questions.{key}.criteria for 'score' must have {MIN_OPTIONS} to "
                    f"{MAX_OPTIONS} levels (got {len(criteria)})"
                )
            norm_levels: list[str] = []
            for idx, level in enumerate(criteria):
                if not isinstance(level, str) or not level.strip():
                    raise DecisionsValidationError(f"questions.{key}.criteria[{idx}] must be a non-empty string")
                norm_levels.append(level)
            entry["criteria"] = norm_levels

        normalized[key] = entry
    return normalized
