"""Gateway hardening regressions (flagship assessment):

1. The static file server must never serve dotfiles/dotdirs (.keys/, .git/, .env)
   or non-web file types, so a secret under the repo root cannot leak as
   application/octet-stream. The path-traversal guard stays.
2. The chat endpoint must coerce a present-but-null or non-numeric field to its
   default and reject a non-finite number with a clean 400, rather than letting a
   bare float()/int() raise a TypeError that do_POST turns into an opaque 500.
"""
import io

import harness.gateway as gateway


def _static_handler(root):
    h = gateway._Handler.__new__(gateway._Handler)
    h.root = root
    rec = {"code": None, "ctype": None}
    h._json = lambda obj, code=200: rec.__setitem__("code", code)
    h.send_response = lambda c: rec.__setitem__("code", c)
    h.send_header = lambda k, v: (rec.__setitem__("ctype", v) if k == "Content-Type" else None)
    h.end_headers = lambda: None
    h.wfile = io.BytesIO()
    return h, rec


def test_static_blocks_path_traversal(tmp_path):
    h, rec = _static_handler(tmp_path)
    h._static("/../outside.txt")
    assert rec["code"] == 403


def test_static_blocks_dotdir_secrets(tmp_path):
    (tmp_path / ".keys").mkdir()
    (tmp_path / ".keys" / "ledger.key").write_bytes(b"SECRET")
    h, rec = _static_handler(tmp_path)
    h._static("/.keys/ledger.key")
    assert rec["code"] == 403


def test_static_refuses_unknown_extension(tmp_path):
    (tmp_path / "creds.db").write_bytes(b"SECRET")
    h, rec = _static_handler(tmp_path)
    h._static("/creds.db")
    assert rec["code"] == 404          # never served as octet-stream


def test_static_serves_a_legit_web_asset(tmp_path):
    site = tmp_path / "site"
    site.mkdir()
    (site / "index.html").write_text("<h1>hi</h1>", encoding="utf-8")
    h, rec = _static_handler(tmp_path)
    h._static("/site/index.html")
    assert rec["code"] == 200 and rec["ctype"] == "text/html"


class _StubProposer:
    model_ref = "stub"

    def generate(self, prompt, *, seed, temperature, max_new_tokens, system=""):
        from harness.proposer import ProposerOutput, prompt_hash
        return ProposerOutput(text="ok", model_ref="stub", seed=seed,
                              prompt_hash=prompt_hash(prompt), cache="stub")


_MSG = {"model": "flywheel", "messages": [{"role": "user", "content": "hi"}]}


def test_chat_rejects_nan_temperature_with_400():
    body, code, *_ = gateway.openai_chat(dict(_MSG, temperature=float("nan")), "http://x")
    assert code == 400 and body["error"]["type"] == "invalid_request_error"


def test_chat_degrades_null_numeric_instead_of_500(monkeypatch):
    monkeypatch.setattr(gateway, "_resolve_proposer",
                        lambda model, serve_url: (_StubProposer(), None, 200))
    body, code, *_ = gateway.openai_chat(
        dict(_MSG, temperature=None, max_tokens=None, seed=None), "http://x")
    assert code == 200          # null coerces to defaults; no TypeError -> 500
