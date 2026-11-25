# Gemini Decisions Plugin for Hermes Agent

Fast, calibrated multimodal classification, routing, policy checks, and rubric scoring via Google's [Gemini Decisions API](https://aistudio.google.com/docs/decisions-api) (`gemini-decision-1`, zero output tokens).

- **Version:** `0.1.0`
- **Hermes compatibility:** `>=0.23.0` (Python `>=3.11`)
- **Runtime dependencies:** None (pure Python standard library)
- **License:** MIT

## Why this plugin?

Generative LLMs add latency and output-token cost when your agent or automation script only needs a bounded decision (for example triaging incoming emails, WhatsApp messages, invoices, or screenshots before calling a frontier model).

`gemini-decision-1` evaluates **1 to 1,000 independent typed questions** over shared input (text, PDF, image, audio, or video up to 262,144 tokens) in a single prefill pass with **0 output tokens** and returns calibrated probabilities and confidence scores:

- **`noul` (yes/no):** Calibrated probability from `0.01` to `0.99` + `confidence`.
- **`choice` (single-select):** Pick 1 of 2–255 options with full per-option probability distribution + `confidence`.
- **`score` (ordered rubric):** Rate across 2–255 ordered levels with expected score, per-level probabilities, legend, and `confidence`.

## Features

- **Two Hermes tools (`toolset: gemini_decisions`):**
  - `gemini_decide` — evaluate 1–1,000 typed questions on one text and/or multimodal input.
  - `gemini_decide_batch` — evaluate up to 50 items (configurable up to 200) against shared questions with bounded concurrency, threshold filtering (`min_probability`, `min_score`, `match_choice`, `min_confidence`), and sorting (`sort_by`).
- **Hermes CLI subcommand:** `hermes gemini-decisions decide | batch | doctor`.
- **Reusable zero-dependency Python client:** import `GeminiDecisionsClient` directly in standalone scripts and cron jobs (no `google-genai` or third-party HTTP library required).
- **Bundled skill:** `gemini-decisions:gemini-decisions` teaches agents question-design and calibration best practices.

## Installation

### From a local checkout (development / local testing)

```bash
hermes plugins install /path/to/hermes-gemini-decisions
hermes plugins enable gemini-decisions
```

### From GitHub (exact commit pin recommended)

```bash
hermes plugins install matthijsberg/hermes-gemini-decisions --ref <40-char-commit-sha>
hermes plugins enable gemini-decisions
```

If your `config.yaml` defines explicit `platform_toolsets`, ensure `gemini_decisions` is enabled (or run `hermes plugins enable gemini-decisions`, which updates saved platform toolset lists automatically), then restart your session (`/reset` or `/restart`).

## Configuration

Set your Gemini Developer API key through Hermes's native secret management (during `hermes plugins install`, in your profile `.env`, or in Hermes Desktop under **Settings → Plugins → gemini-decisions**):

- `GEMINI_API_KEY` (required; `GOOGLE_API_KEY` is also accepted as a fallback)

Optional settings under `plugins.entries.gemini-decisions.settings` in `config.yaml`:

```yaml
plugins:
  enabled:
    - gemini-decisions
  entries:
    gemini-decisions:
      settings:
        endpoint: https://generativelanguage.googleapis.com/v1beta/decisions
        decision_model: gemini-decision-1
        timeout_seconds: 30
        max_concurrency: 4
        max_batch_items: 50
        max_file_mb: 20
```

> **Privacy note on Google AI Studio tiers:** Google's free tier permits using request data to improve Google products; use a **Paid tier** Gemini API key ($0.10 per 1M input tokens, 0 output tokens) when classifying private emails, messages, or personal documents.

## Usage Examples

### 1. Hermes tool: `gemini_decide`

```json
{
  "text": "Hi, reminder that school library book 'De Grijze Jager' is 3 days overdue.",
  "questions": {
    "needs_action": {
      "type": "noul",
      "instructions": "Does this message require a concrete action or payment from the recipient?"
    },
    "category": {
      "type": "choice",
      "instructions": "Which inbox triage category fits this message?",
      "criteria": {
        "urgent_action": "Overdue items, fines, deadlines, or urgent requests.",
        "calendar": "Confirmed or proposed appointment/meeting times.",
        "newsletter": "Marketing, digests, or FYI updates.",
        "other": "Anything outside these categories."
      }
    },
    "urgency": {
      "type": "score",
      "instructions": "Rate the urgency of this message.",
      "criteria": ["Ignore", "Low", "Medium", "High"]
    }
  }
}
```

### 2. Hermes tool: `gemini_decide_batch` (two-stage email/chat triage)

```json
{
  "items": [
    {"id": "msg_1", "text": "20% off winter coats this weekend!"},
    {"id": "msg_2", "text": "Parent-teacher conference scheduled for Thursday 19:30."}
  ],
  "questions": {
    "is_actionable": {
      "type": "noul",
      "instructions": "Does this message contain an appointment, deadline, or action item?"
    }
  },
  "filter_question": "is_actionable",
  "min_probability": 0.35,
  "sort_by": "is_actionable"
}
```

### 3. Multimodal input (PDF / Image)

```json
{
  "text": "Check this attached invoice PDF.",
  "media": [
    {"file_path": "./invoice.pdf"}
  ],
  "questions": {
    "has_due_date": {
      "type": "noul",
      "instructions": "Does this invoice specify a payment due date?"
    }
  }
}
```

### 4. CLI usage

```bash
# Check config & key (no network call)
hermes gemini-decisions doctor

# Live smoke test against gemini-decision-1
hermes gemini-decisions doctor --live

# Dry-run payload validation
hermes gemini-decisions decide \
  --text "The mug arrived cracked." \
  --yes-no "Did the item arrive damaged?" \
  --dry-run
```

### 5. Standalone Python client (zero dependencies)

```python
import os
from gemini_decisions import GeminiDecisionsClient

client = GeminiDecisionsClient(api_key=os.environ["GEMINI_API_KEY"])
result = client.decide(
    text="Tomorrow's hockey match at 09:00 is cancelled due to frost.",
    questions={
        "is_cancellation": {
            "type": "noul",
            "instructions": "Does this message announce a cancellation or schedule change?",
        }
    },
)
print(result.answers["is_cancellation"])
```

## Security & Disclosures (Hermes Catalog Rule 13)

- **Outbound network calls:** Each `gemini_decide` / `gemini_decide_batch` call sends the supplied input text, any explicitly requested media attachment (base64-encoded or HTTPS URI), and question definitions over HTTPS with TLS verification to `https://generativelanguage.googleapis.com/v1beta/decisions` (or the configured HTTPS `endpoint`). HTTP redirects are refused so API keys cannot leak across hosts. Usage is billable under your Google AI Studio / Gemini API account.
- **Credentials:** Inside Hermes, the API key is resolved strictly via `agent.secret_scope.get_secret("GEMINI_API_KEY")` (fallback `GOOGLE_API_KEY`) and transmitted only in the `x-goog-api-key` HTTP header. The plugin never reads `.env`, `auth.json`, or external credential stores, and never logs or persists API keys.
- **Local file reads:** Only files explicitly passed via `media[].file_path` are read. Reads are restricted to regular files matching Google's supported media MIME allowlist (PDF, image, audio, video), bounded by `max_file_mb` (default 20 MB, hard cap 50 MB), and sensitive filenames (`.env`, `auth.json`, `*.pem`, `*.key`, SSH keys) are explicitly rejected.
- **Remote URL validation:** Media `uri` inputs must be HTTPS, cannot be `gs://` or YouTube URLs, and are validated with Hermes's `tools.url_safety.is_safe_url` (with a stdlib SSRF fallback standalone) to block localhost/private/link-local targets.
- **No persistence or background daemons:** No local database, no background threads, no shell execution, no self-updating code, no telemetry.

## Development & Verification

```bash
# 1. Run unit & integration tests
./scripts/setup.sh
pytest

# 2. Validate against Hermes Plugin Catalog admission rules
hermes plugins validate .
hermes plugins doctor . --ci

# 3. Safe local demo (dry-run by default; pass --live for real API call)
./scripts/demo.sh
```
