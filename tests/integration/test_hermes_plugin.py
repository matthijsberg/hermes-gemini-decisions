"""Integration tests for Hermes plugin registration, CLI, and catalog validator."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

import gemini_decisions
from gemini_decisions import cli

PLUGIN_DIR = Path(__file__).resolve().parents[2]


class _RecordingCtx:
    def __init__(self) -> None:
        self.tools: list[dict] = []
        self.cli_commands: list[dict] = []
        self.skills: list[tuple[str, str]] = []

    def get_config(self, key: str, default=None):
        return default

    def register_tool(self, *, name, toolset, schema, handler, **kwargs):
        self.tools.append(
            {
                "name": name,
                "toolset": toolset,
                "schema": schema,
                "handler": handler,
            }
        )

    def register_cli_command(self, *, name, help, setup_fn, handler_fn, **kwargs):
        self.cli_commands.append(
            {
                "name": name,
                "help": help,
                "setup_fn": setup_fn,
                "handler_fn": handler_fn,
            }
        )

    def register_skill(self, name: str, path: str):
        self.skills.append((name, path))


def test_register_wires_tools_cli_and_skill():
    ctx = _RecordingCtx()
    gemini_decisions.register(ctx)
    assert [t["name"] for t in ctx.tools] == ["gemini_decide", "gemini_decide_batch"]
    assert all(t["toolset"] == "gemini_decisions" for t in ctx.tools)
    assert [c["name"] for c in ctx.cli_commands] == ["gemini-decisions"]
    assert len(ctx.skills) == 1
    assert Path(ctx.skills[0][1]).is_file()


def test_cli_dry_run_outputs_valid_payload(capsys):
    exit_code = cli.main(
        [
            "decide",
            "--text",
            "The mug arrived cracked.",
            "--yes-no",
            "Did the item arrive damaged?",
            "--dry-run",
        ]
    )
    assert exit_code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["dry_run"] is True
    assert out["payload"]["model"] == "gemini-decision-1"
    assert out["payload"]["questions"]["answer"]["type"] == "noul"


def test_hermes_catalog_validator_passes_with_zero_warnings():
    """Run Hermes's official plugin validator CLI if `hermes` is installed."""
    hermes_bin = shutil.which("hermes")
    if not hermes_bin:
        pytest.skip("hermes CLI not installed on this host")

    proc = subprocess.run(
        [hermes_bin, "plugins", "validate", str(PLUGIN_DIR), "--json"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    raw_out = proc.stdout
    json_start = raw_out.find("{")
    assert json_start != -1, f"No JSON found in validator output: {raw_out}"
    report = json.loads(raw_out[json_start:])
    assert report["ok"] is True, f"Validation failed: {report['checks']}"
    assert report["warnings"] == [], f"Unexpected warnings: {report['warnings']}"
