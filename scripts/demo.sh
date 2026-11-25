#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"

DRY_RUN="${DRY_RUN:-1}"
if [ "${1:-}" = "--live" ]; then
  DRY_RUN=0
fi

if [ "$DRY_RUN" = "1" ]; then
  echo "=== DRY RUN (no network call; set DRY_RUN=0 or pass --live to call Gemini API) ==="
  PYTHONPATH="$ROOT_DIR" "$PYTHON_BIN" -c "
import json
from pathlib import Path
from gemini_decisions import GeminiDecisionsClient

pkg_dir = Path('$ROOT_DIR')
fixture = json.loads((pkg_dir / 'tests/fixtures/sample_decisions.json').read_text(encoding='utf-8'))
client = GeminiDecisionsClient(api_key='dry-run-placeholder')
payload = client.build_request_payload(
    text=fixture['single_text'],
    questions=fixture['questions'],
)
print(json.dumps({'dry_run': True, 'endpoint': client.endpoint, 'payload': payload}, indent=2))
"
else
  echo "=== LIVE SMOKE TEST (calling Gemini Decisions API) ==="
  PYTHONPATH="$ROOT_DIR" "$PYTHON_BIN" -c "
import json, os
from pathlib import Path
from gemini_decisions import GeminiDecisionsClient

pkg_dir = Path('$ROOT_DIR')
key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
if not key:
    raise SystemExit('GEMINI_API_KEY is not set in environment.')
fixture = json.loads((pkg_dir / 'tests/fixtures/sample_decisions.json').read_text(encoding='utf-8'))
client = GeminiDecisionsClient(api_key=key)
res = client.decide(text=fixture['single_text'], questions=fixture['questions'])
print(json.dumps(res.to_dict(), indent=2))
"
fi
