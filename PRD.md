# PRD — `gemini-decisions` (Hermes Agent Plugin)

- **Version:** `0.1.0`
- **Plugin ID:** `gemini-decisions`
- **Target Repository:** `https://github.com/matthijsberg/hermes-gemini-decisions`
- **License:** MIT
- **Hermes Compatibility:** `>=0.23.0` (Python `>=3.11`)

## 1. Problem & Goal

Generative LLMs are slow and expensive when an agent or script only needs a fast, calibrated classification, routing decision, policy check, or urgency score (for example triaging incoming emails, WhatsApp messages, invoices/PDFs, or screenshots before invoking a heavier reasoning model).

Google's **Gemini Decisions API** (`v1beta/decisions`, powered by `gemini-decision-1`) evaluates 1 to 1,000 typed questions (`noul` yes/no, `choice` single-select, `score` ordered rubric) over shared multimodal input (text, PDF, image, audio, video) in a single prefill pass with **0 output tokens** and returns calibrated probabilities + confidence scores.

Hermes Agent core does not natively support the Decisions API because `model`, `delegation`, and `auxiliary` slots require generative text/chat models. This plugin exposes the Gemini Decisions API cleanly to Hermes Agent (tools, CLI, bundled skill) AND as a zero-dependency stdlib Python client for standalone automation scripts and cron jobs.

## 2. Target Users & Use Cases

1. **Hermes Agent sessions:** Fast two-stage triage (filter 20–50 emails or messages cheaply with `gemini_decide_batch`, then only read/summarize items above a probability threshold).
2. **Multimodal document/media checks:** Classify PDF invoices, school letters, receipts, screenshots, or audio clips without generating text.
3. **Standalone automation & cron scripts:** Import `GeminiDecisionsClient` directly (zero third-party runtime dependencies) or run `hermes gemini-decisions ...` from shell pipelines.
4. **Hermes Plugin Catalog community:** Clean, auditable, public-API-only plugin passing `hermes plugins validate` and `hermes plugins doctor --ci` with zero errors and zero warnings.

## 3. Functional Scope (v0.1.0)

### 3.1 Hermes Tools (`toolset: gemini_decisions`)
- **`gemini_decide`**: Evaluate 1–1,000 typed questions (`noul`, `choice`, `score`) against a single text and/or multimodal input (local PDF/image/audio/video files or safe HTTPS URIs). Returns calibrated probabilities, top choices/scores, confidence, and token usage (`total_output_tokens: 0`).
- **`gemini_decide_batch`**: Evaluate a list of items (each with an `id`, `text` and/or `media`) against one shared set of questions using bounded thread-pool concurrency (`max_concurrency`, default `4`). Supports optional deterministic post-filtering (`filter_question`, `min_probability` / `min_score` / `match_choice`) and sorting (`sort_by`) so callers receive prioritized hits directly.

### 3.2 CLI Subcommand
- Registered via `ctx.register_cli_command("gemini-decisions", ...)` as `hermes gemini-decisions`:
  - `decide`: Evaluate one input (text / `--file` / stdin) against `--questions` JSON or inline `--yes-no` / `--choice` / `--score` flags.
  - `batch`: Evaluate JSONL/JSON array of items against shared questions.
  - `doctor`: Verify local config, credential presence (without printing secrets), and optionally run a `--live` API smoke test.

### 3.3 Reusable Python Client (`client.py`)
- Pure Python 3.11+ standard library (`urllib.request`, `concurrent.futures`, `json`, `base64`, `mimetypes`, `ssl`). Zero third-party runtime dependencies.
- Explicit `GeminiDecisionsClient(api_key=..., endpoint=..., model=..., timeout_seconds=...)`.
- Automatic bounded retries with exponential backoff on transient HTTP `429` / `500` / `502` / `503` / `504` errors (never retrying `400` validation or safety blocks).
- Structured typed exceptions (`DecisionsValidationError`, `DecisionsAuthError`, `DecisionsSafetyBlockedError`, `DecisionsAPIError`).

### 3.4 Bundled Skill
- `skills/gemini-decisions/SKILL.md` registered via `ctx.register_skill("gemini-decisions", ...)` teaching agents how to design `noul`/`choice`/`score` questions, handle `0.01–0.99` calibration bounds, avoid alphabetical key traps (`other` vs `none_of_the_above`), and build two-stage triage workflows.

## 4. Non-Functional & Catalog Compliance Requirements

1. **Zero core overrides & zero private imports (Catalog Rule 9):** Import only stdlib and modules listed in Hermes `public-api.md` (`agent.secret_scope`, `agent.runtime_cwd`, `tools.url_safety`).
2. **Hermes-owned credentials only (Catalog Rule 11):** Tool handlers resolve `GEMINI_API_KEY` (fallback `GOOGLE_API_KEY`) strictly via `agent.secret_scope.get_secret(...)`. Never read `.env`, `auth.json`, or external credential stores. Send key only via `x-goog-api-key` HTTP header (never query params).
3. **Strict input & media safety:**
   - Enforce HTTPS-only endpoints (no userinfo, query, or fragment).
   - Enforce all Google Decisions API bounds locally before network calls (`1..1000` questions, key regex `^[A-Za-z0-9_-]{1,64}$`, `2..255` choice options, `2..255` non-empty score levels).
   - Strict MIME allowlist for PDF, image, audio, and video; configurable max file size (`max_file_mb`, default 20 MB, hard cap 50 MB PDF / 100 MB request).
   - Reject sensitive filenames (`.env`, `*.pem`, `*.key`, `auth.json`, SSH keys) as media attachments.
   - Validate remote media `uri` via `tools.url_safety.is_safe_url` (with pure-stdlib SSRF fallback outside Hermes); reject `http://`, `file://`, `gs://`, and YouTube URLs.
4. **Synchronous execution:** No daemon threads, no persistent background state, no self-updater, no shell execution.
5. **No fabricated data:** Dry-run/validation helpers inspect payloads without fabricating probabilities. All tests isolate mocks inside `tests/`.
6. **Transparent disclosures (Catalog Rule 13):** Document outbound network endpoint, billable usage, local file reads when media paths are supplied, and zero secret persistence in `README.md`.

## 5. Configuration Schema (`plugins.entries.gemini-decisions.settings`)

- `api_key` (`secret`, env `GEMINI_API_KEY`): Gemini Developer API key.
- `endpoint` (`str`, default `https://generativelanguage.googleapis.com/v1beta/decisions`): HTTPS Decisions API URL.
- `decision_model` (`str`, default `gemini-decision-1`): Decisions model ID.
- `timeout_seconds` (`int`, default `30`): Per-request HTTP timeout (1–300s).
- `max_concurrency` (`int`, default `4`): Max parallel requests in `gemini_decide_batch` (1–10).
- `max_batch_items` (`int`, default `50`): Max items per `gemini_decide_batch` call (1–200).
- `max_file_mb` (`int`, default `20`): Max local media attachment size in MB (1–50).

## 6. Verification Plan

1. Unit tests (`tests/unit/`): validation rules, MIME/path/SSRF guards, HTTP error/retry/safety-block handling, batch filtering/sorting, tool handler secret resolution.
2. Integration tests (`tests/integration/`): isolated `hermes plugins validate` and `hermes plugins doctor --ci`, real `PluginManager` registration check, CLI parser/execution tests.
3. Live E2E smoke test against `https://generativelanguage.googleapis.com/v1beta/decisions` covering `noul`, `choice`, `score`, batch mode, and multimodal PDF/image input.
4. Security audit: `bandit`, secret scan (`gitleaks` / regex check), `hermes plugins validate` security scan (`safe`, 0 warnings).
