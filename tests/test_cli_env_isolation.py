"""A subscription CLI must not inherit this process's Claude Code session.

When the gateway runs inside Claude Code (or the desktop, which is Claude Code),
spawning `claude -p` with the parent's CLAUDE_CODE_* env makes the child think it
is a nested session and it misbehaves. CliBackend must strip those markers before
it spawns, while keeping everything else (PATH, credentials) intact."""
import os

import harness.endpoints as eps
from harness.endpoints import CliBackend, _clean_cli_env


def test_clean_cli_env_strips_session_markers_keeps_the_rest(monkeypatch):
    monkeypatch.setenv("CLAUDECODE", "1")
    monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", "parent-abc")
    monkeypatch.setenv("CLAUDE_CODE_ENTRYPOINT", "claude-desktop")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-keep-me")  # auth-adjacent var stays
    env = _clean_cli_env()
    assert "CLAUDECODE" not in env
    assert "CLAUDE_CODE_SESSION_ID" not in env
    assert "CLAUDE_CODE_ENTRYPOINT" not in env
    assert env.get("ANTHROPIC_API_KEY") == "sk-keep-me"
    assert "PATH" in env  # the CLI still needs to resolve on PATH


def test_cli_backend_spawns_with_cleaned_env(monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", "parent-abc")
    monkeypatch.setenv("CLAUDECODE", "1")
    captured = {}

    class _P:
        returncode = 0
        stdout = b"ROUTE-OK"
        stderr = b""

    def fake_run(cmd, **kw):
        captured["env"] = kw.get("env")
        return _P()

    monkeypatch.setattr(eps.subprocess, "run", fake_run)
    backend = CliBackend(name="claude", argv=["claude", "-p", "{prompt}"])
    out = backend.chat([{"role": "user", "content": "hi"}], system="",
                       max_tokens=8, temperature=0.0, seed=0)

    assert out["text"] == "ROUTE-OK"
    assert captured["env"] is not None, "CliBackend must pass an explicit env"
    assert "CLAUDE_CODE_SESSION_ID" not in captured["env"]
    assert "CLAUDECODE" not in captured["env"]
