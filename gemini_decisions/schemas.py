"""Tool JSON schemas exposed to Hermes Agent for the Gemini Decisions API."""

from __future__ import annotations

_MEDIA_ITEM_SCHEMA = {
    "type": "object",
    "description": (
        "One multimodal attachment (PDF, image, audio, or video). "
        "Provide exactly one of 'file_path' (local file) or 'uri' (public/signed HTTPS URL)."
    ),
    "properties": {
        "file_path": {
            "type": "string",
            "description": "Path to a local PDF, image, audio, or video file.",
        },
        "uri": {
            "type": "string",
            "description": "Public or signed HTTPS URL (no gs:// or YouTube URLs).",
        },
        "type": {
            "type": "string",
            "enum": ["document", "image", "audio", "video"],
            "description": "Optional content block type; inferred from MIME type when omitted.",
        },
        "mime_type": {
            "type": "string",
            "description": (
                "Optional MIME type (e.g. application/pdf, image/jpeg, audio/mp3, video/mp4); "
                "inferred from filename/URL extension when omitted."
            ),
        },
        "resolution": {
            "type": "string",
            "enum": ["low", "medium", "high", "ultra_high"],
            "description": "Optional visual detail level for image/video inputs.",
        },
    },
}

_QUESTIONS_SCHEMA = {
    "type": "object",
    "description": (
        "Map of 1 to 1,000 named questions (keys match ^[A-Za-z0-9_-]{1,64}$). "
        "The model never sees question keys, so put the complete question in 'instructions'. "
        "Each question has 'type' ('noul' for yes/no, 'choice' for single-select, 'score' for rubric), "
        "'instructions' (string), and 'criteria': "
        "for 'noul', optional {true_description, false_description}; "
        "for 'choice', required object mapping 2-255 option IDs to descriptions; "
        "for 'score', required array of 2-255 ordered rubric level descriptions from lowest (0) to highest."
    ),
    "additionalProperties": {
        "type": "object",
        "properties": {
            "type": {
                "type": "string",
                "enum": ["noul", "choice", "score"],
            },
            "instructions": {
                "type": "string",
            },
            "criteria": {
                "description": (
                    "For 'noul': optional {true_description, false_description}. "
                    "For 'choice': object of option_id -> description (2-255 entries). "
                    "For 'score': ordered array of 2-255 level descriptions."
                ),
            },
        },
        "required": ["type", "instructions"],
    },
}

GEMINI_DECIDE = {
    "name": "gemini_decide",
    "description": (
        "Evaluate 1 to 1,000 typed classification, routing, yes/no, or rubric-scoring questions "
        "against shared text and/or multimodal input (PDF, image, audio, video) using Google's "
        "Gemini Decisions API (gemini-decision-1). Runs in a single pass with zero output tokens "
        "and returns calibrated probabilities and confidence scores. Use this for fast triage, "
        "routing, policy checks, and document/message qualification before calling a generative model."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "Text or serialized JSON input to evaluate (optional if 'media' is provided).",
            },
            "media": {
                "type": "array",
                "items": _MEDIA_ITEM_SCHEMA,
                "description": "Optional list of multimodal attachments (PDF, image, audio, video).",
            },
            "questions": _QUESTIONS_SCHEMA,
        },
        "required": ["questions"],
    },
}

GEMINI_DECIDE_BATCH = {
    "name": "gemini_decide_batch",
    "description": (
        "Evaluate a batch of items (for example emails, chat messages, or documents) against one "
        "shared set of typed questions using Google's Gemini Decisions API (gemini-decision-1) "
        "with bounded concurrency. Supports optional threshold filtering and sorting so you "
        "receive prioritized hits directly."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "description": "List of items to evaluate (each with 'id', and 'text' and/or 'media').",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string",
                            "description": "Unique identifier echoed back in results.",
                        },
                        "text": {
                            "type": "string",
                            "description": "Text payload for this item.",
                        },
                        "media": {
                            "type": "array",
                            "items": _MEDIA_ITEM_SCHEMA,
                            "description": "Optional media attachments for this item.",
                        },
                    },
                    "required": ["id"],
                },
            },
            "questions": _QUESTIONS_SCHEMA,
            "filter_question": {
                "type": "string",
                "description": "Optional question key used to filter returned items.",
            },
            "min_probability": {
                "type": "number",
                "description": (
                    "Keep items whose 'noul' probability (or winning 'choice' probability) "
                    "on filter_question is >= this value (0.01 to 0.99)."
                ),
            },
            "min_score": {
                "type": "number",
                "description": "Keep items whose 'score' on filter_question is >= this value.",
            },
            "match_choice": {
                "type": "string",
                "description": "Keep items whose top 'choice' on filter_question equals this option key.",
            },
            "min_confidence": {
                "type": "number",
                "description": "Keep items whose confidence on filter_question is >= this value (0.0 to 1.0).",
            },
            "sort_by": {
                "type": "string",
                "description": "Optional question key to sort matching items by (probability, score, or confidence).",
            },
            "descending": {
                "type": "boolean",
                "description": "Sort highest first (default true).",
                "default": True,
            },
        },
        "required": ["items", "questions"],
    },
}
