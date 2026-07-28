"""Routing to a subscription-CLI endpoint must bind the official-CLI backend,
not the paid API backend.

Regression for the _BUILD_ALIAS bug: claude-cli/codex-cli resolved to the
AnthropicBackend/OpenAICompatBackend (the API path, keyed by ANTHROPIC_API_KEY /
OPENAI_API_KEY) instead of the CliBackend that invokes the operator's own
subscription client. The roster advertised claude-cli/codex-cli as usable while
the live route silently hit the paid API."""
from harness.endpoint_registry import make_endpoint_proposer
from harness.endpoints import CliBackend


def test_claude_cli_binds_the_subscription_cli_backend():
    backend = getattr(make_endpoint_proposer("claude-cli"), "backend", None)
    assert isinstance(backend, CliBackend), \
        f"claude-cli must bind the subscription CliBackend, got {type(backend).__name__}"


def test_codex_cli_binds_the_subscription_cli_backend():
    backend = getattr(make_endpoint_proposer("codex-cli"), "backend", None)
    assert isinstance(backend, CliBackend), \
        f"codex-cli must bind the subscription CliBackend, got {type(backend).__name__}"
