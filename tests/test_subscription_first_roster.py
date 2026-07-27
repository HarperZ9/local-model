"""Subscription-first endpoint roster: the paid-for official CLI is the primary
tier, the public API is the fallback, and neither surfaces a credential value.

Regression guard for endpoint_roster()'s access-mode reporting: a provider whose
official CLI (claude max / codex plan) is on PATH must report access=subscription
even when no API key is set, so the desktop shows the tier the operator actually
has rather than the paid API path they may not."""
import json

import harness.gateway as gw

_BOGUS = "http://127.0.0.1:1"  # local probes fail fast; we assert on enterprise[]


def _roster(monkeypatch, *, cli_present, cred):
    import shutil
    monkeypatch.setattr(
        shutil, "which",
        lambda b: ("/x/" + b) if (cli_present and ("claude" in b or "codex" in b)) else None)
    monkeypatch.setattr(gw, "_resolve_credential", cred)
    return gw.endpoint_roster(_BOGUS, _BOGUS)


def test_subscription_cli_is_primary_over_absent_api(monkeypatch):
    r = _roster(monkeypatch, cli_present=True, cred=lambda env: "")  # CLIs on PATH, no keys
    by = {e["name"]: e for e in r["enterprise"]}
    assert by["claude"]["access"] == "subscription"
    assert by["claude"]["subscription_present"] is True
    assert by["claude"]["credential_present"] is False
    assert by["codex"]["access"] == "subscription"
    assert r["subscription_available"] >= 2
    assert r["enterprise_usable"] >= 2


def test_api_key_is_the_fallback_when_no_cli(monkeypatch):
    r = _roster(monkeypatch, cli_present=False,
                cred=lambda env: "sk-x" if env == "GEMINI_API_KEY" else "")
    by = {e["name"]: e for e in r["enterprise"]}
    assert by["gemini"]["access"] == "api"
    assert by["gemini"]["subscription_present"] is False
    assert by["deepseek"]["access"] == "none"  # neither CLI nor key


def test_roster_never_surfaces_a_credential_value(monkeypatch):
    r = _roster(monkeypatch, cli_present=True, cred=lambda env: "SECRET-TOKEN-VALUE")
    assert "SECRET-TOKEN-VALUE" not in json.dumps(r)  # presence only, never a value
