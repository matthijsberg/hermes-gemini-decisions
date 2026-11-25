"""CLI subcommands for `hermes gemini-decisions` and standalone `gemini-decisions`."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from .client import GeminiDecisionsClient
from .validation import (
    DEFAULT_ENDPOINT,
    DEFAULT_MAX_FILE_MB,
    DEFAULT_MODEL,
    DecisionsError,
    validate_endpoint,
)


def _resolve_cli_api_key() -> str:
    """Prefer Hermes's profile-scoped secret store; fall back to env only standalone."""
    try:
        from agent.secret_scope import get_secret  # type: ignore[import-not-found]

        key = (get_secret("GEMINI_API_KEY") or get_secret("GOOGLE_API_KEY") or "").strip()
        if key:
            return key
    except ImportError:
        pass
    return (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip()


def _load_json_arg(raw: str) -> Any:
    path = Path(raw)
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return json.loads(raw)


def _build_questions_from_args(args: argparse.Namespace) -> dict[str, Any]:
    questions: dict[str, Any] = {}
    if getattr(args, "questions", None):
        loaded = _load_json_arg(args.questions)
        if not isinstance(loaded, dict):
            raise ValueError("--questions must be a JSON object mapping key -> question spec")
        questions.update(loaded)
    for idx, prompt in enumerate(getattr(args, "yes_no", None) or [], start=1):
        key = f"q{idx}" if len(args.yes_no) > 1 or questions else "answer"
        questions[key] = {"type": "noul", "instructions": prompt}
    if not questions:
        raise ValueError("Provide --questions <json-or-file> or at least one --yes-no <question>")
    return questions


def setup_cli_parser(parser: argparse.ArgumentParser) -> None:
    """Attach `decide`, `batch`, and `doctor` subcommands to an argparse parser."""
    sub = parser.add_subparsers(dest="gd_command", metavar="<command>")

    p_decide = sub.add_parser("decide", help="Evaluate typed questions on one input")
    p_decide.add_argument("--text", help="Input text (or omit to read from stdin if no --file)")
    p_decide.add_argument("--file", dest="files", action="append", help="Local media file (PDF/image/audio/video)")
    p_decide.add_argument("--uri", dest="uris", action="append", help="HTTPS media URL")
    p_decide.add_argument("--questions", help="JSON string or path to JSON file defining questions")
    p_decide.add_argument("--yes-no", action="append", help="Quick yes/no (noul) question")
    p_decide.add_argument("--model", default=DEFAULT_MODEL, help=f"Model ID (default: {DEFAULT_MODEL})")
    p_decide.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="Decisions API HTTPS endpoint")
    p_decide.add_argument("--timeout", type=int, default=30, help="HTTP timeout in seconds")
    p_decide.add_argument(
        "--dry-run", action="store_true", help="Validate and print request payload without calling API"
    )

    p_batch = sub.add_parser("batch", help="Evaluate multiple items from a JSON/JSONL file")
    p_batch.add_argument("--items", required=True, help="Path to JSON array or JSONL file of items")
    p_batch.add_argument("--questions", required=True, help="JSON string or path to JSON file defining questions")
    p_batch.add_argument("--filter-question", help="Question key to filter results on")
    p_batch.add_argument("--min-probability", type=float, help="Minimum noul/choice probability")
    p_batch.add_argument("--min-score", type=float, help="Minimum rubric score")
    p_batch.add_argument("--match-choice", help="Required choice option key")
    p_batch.add_argument("--sort-by", help="Question key to sort results by")
    p_batch.add_argument("--concurrency", type=int, default=4, help="Max parallel requests (1-10)")
    p_batch.add_argument("--model", default=DEFAULT_MODEL, help=f"Model ID (default: {DEFAULT_MODEL})")
    p_batch.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="Decisions API HTTPS endpoint")
    p_batch.add_argument("--timeout", type=int, default=30, help="HTTP timeout in seconds")

    p_doc = sub.add_parser("doctor", help="Check config, key availability, and optional live smoke test")
    p_doc.add_argument("--live", action="store_true", help="Send a minimal live request to Gemini Decisions API")
    p_doc.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="Decisions API HTTPS endpoint")
    p_doc.add_argument("--model", default=DEFAULT_MODEL, help=f"Model ID (default: {DEFAULT_MODEL})")


def _load_batch_items(path_str: str) -> list[dict[str, Any]]:
    raw = Path(path_str).read_text(encoding="utf-8").strip()
    if not raw:
        return []
    if raw.startswith("["):
        parsed = json.loads(raw)
        if not isinstance(parsed, list):
            raise ValueError("Batch JSON file must contain a JSON array")
        return parsed
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def handle_cli(args: argparse.Namespace) -> int:
    """Execute a `gemini-decisions` CLI subcommand and print JSON to stdout."""
    cmd = getattr(args, "gd_command", None)
    if not cmd:
        print(json.dumps({"error": "No subcommand given. Use: decide | batch | doctor"}))
        return 2

    try:
        if cmd == "doctor":
            endpoint = validate_endpoint(args.endpoint)
            key = _resolve_cli_api_key()
            status: dict[str, Any] = {
                "endpoint": endpoint,
                "model": args.model,
                "api_key_configured": bool(key),
            }
            if args.live:
                client = GeminiDecisionsClient(
                    api_key=key,
                    endpoint=endpoint,
                    model=args.model,
                    timeout_seconds=20,
                )
                res = client.decide(
                    text="The package arrived on time and intact.",
                    questions={
                        "is_damaged": {
                            "type": "noul",
                            "instructions": "Does the message report damage?",
                        }
                    },
                )
                status["live_smoke_ok"] = True
                status["live_result"] = res.to_dict()
            print(json.dumps(status, indent=2, ensure_ascii=False))
            return 0 if status["api_key_configured"] else 1

        if cmd == "decide":
            questions = _build_questions_from_args(args)
            text = args.text
            media: list[dict[str, Any]] = []
            for fp in args.files or []:
                media.append({"file_path": fp})
            for uri in args.uris or []:
                media.append({"uri": uri})
            if text is None and not media and not sys.stdin.isatty():
                text = sys.stdin.read()

            if args.dry_run:
                from .validation import normalize_input, validate_questions

                payload = {
                    "model": args.model,
                    "input": normalize_input(
                        text=text,
                        media=media or None,
                        max_file_mb=DEFAULT_MAX_FILE_MB,
                        base_dir=Path.cwd(),
                    ),
                    "questions": validate_questions(questions),
                }
                print(
                    json.dumps(
                        {"dry_run": True, "endpoint": validate_endpoint(args.endpoint), "payload": payload},
                        indent=2,
                        ensure_ascii=False,
                    )
                )
                return 0

            client = GeminiDecisionsClient(
                api_key=_resolve_cli_api_key(),
                endpoint=args.endpoint,
                model=args.model,
                timeout_seconds=args.timeout,
            )
            result = client.decide(
                questions=questions,
                text=text,
                media=media or None,
                base_dir=Path.cwd(),
            )
            print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
            return 0

        if cmd == "batch":
            items = _load_batch_items(args.items)
            questions = _build_questions_from_args(args)
            client = GeminiDecisionsClient(
                api_key=_resolve_cli_api_key(),
                endpoint=args.endpoint,
                model=args.model,
                timeout_seconds=args.timeout,
            )
            result = client.decide_batch(
                items=items,
                questions=questions,
                max_concurrency=args.concurrency,
                base_dir=Path.cwd(),
                filter_question=args.filter_question,
                min_probability=args.min_probability,
                min_score=args.min_score,
                match_choice=args.match_choice,
                sort_by=args.sort_by,
            )
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0

    except (DecisionsError, ValueError, OSError) as exc:
        print(json.dumps({"success": False, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1

    return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="gemini-decisions",
        description="CLI for Google's Gemini Decisions API (gemini-decision-1).",
    )
    setup_cli_parser(parser)
    args = parser.parse_args(argv)
    return handle_cli(args)


if __name__ == "__main__":
    raise SystemExit(main())
