"""Falsifier for harness/model_shim.py -- the harness-side server for
BuildLang's `Model` capability wire contract (see the module docstring in
harness/model_shim.py for the full contract restatement).

Covers: (a) echo mode end to end over a real socket, spawned as a subprocess
with --once --port 0, including the client-side trailing-newline-trim
semantics the wire contract specifies; (b) the two ways a connection can be
a protocol violation (overlong line, EOF before a newline) both close
without a reply; (c) the sanitizer collapses a \r/\n-bearing completion to
one line, exercised through handle_connection with urllib mocked at the
network boundary -- no live ollama call, ever; (d) --echo and --ollama are
refused together and refused absent, at the argparse level; (e) model
boundary receipt emission (`--receipt-dir`): the golden-fixture cross-repo
seal pin (docs/MODEL-RECEIPT.md and
docs/superpowers/specs/2026-07-29-model-boundary-receipts-design.md in
buildlang), an end-to-end emission round trip over a real echo-mode
connection, prompt/reply hash correctness against the exact wire bytes, the
FAILED_CLOSED and PROTOCOL_VIOLATION outcome shapes, and that omitting the
flag emits nothing.
"""
from __future__ import annotations

import hashlib
import json
import socket
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

from harness import model_shim
from harness.oracle import spawn_killable

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "model-receipt-golden.json"


def _spawn_shim(*extra_args: str) -> subprocess.Popen:
    cmd = [sys.executable, "-m", "harness.model_shim", "--port", "0", *extra_args]
    return spawn_killable(cmd, cwd=str(REPO_ROOT), stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, text=True, bufsize=1)


def _cleanup(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        try:
            proc.kill()
        except Exception:
            pass
    try:
        proc.wait(timeout=5)
    except Exception:
        pass


def _read_bound_port(proc: subprocess.Popen) -> int:
    line = proc.stdout.readline()
    assert line, f"shim produced no port line; stderr={proc.stderr.read()!r}"
    return int(line.strip())


def _read_to_close(sock: socket.socket) -> bytes:
    data = bytearray()
    while True:
        chunk = sock.recv(4096)
        if not chunk:
            return bytes(data)
        data += chunk


def _trim_trailing_newline(data: bytes) -> str:
    """Mirror the client-side trim the wire contract specifies: exactly one
    trailing \n, and a \r immediately before it, if present."""
    if data.endswith(b"\n"):
        data = data[:-1]
    if data.endswith(b"\r"):
        data = data[:-1]
    return data.decode("utf-8")


@pytest.fixture
def echo_server():
    proc = _spawn_shim("--echo", "--once")
    try:
        port = _read_bound_port(proc)
        yield port
    finally:
        _cleanup(proc)


def test_echo_mode_end_to_end_over_real_socket(echo_server):
    port = echo_server
    prompt = "what is the airspeed velocity of an unladen swallow"
    sock = socket.create_connection(("127.0.0.1", port), timeout=10)
    try:
        sock.sendall((prompt + "\n").encode("utf-8"))
        sock.shutdown(socket.SHUT_WR)
        raw = _read_to_close(sock)
    finally:
        sock.close()
    completion = _trim_trailing_newline(raw)
    assert completion == f"echo: {prompt}"
    # the wire write itself carries exactly one trailing \n, no \r
    assert raw == f"echo: {prompt}\n".encode("utf-8")


def test_overlong_prompt_line_closes_without_reply(echo_server):
    port = echo_server
    sock = socket.create_connection(("127.0.0.1", port), timeout=10)
    try:
        # one byte past the 1 MiB cap, no newline -- must be refused before
        # a newline could ever legally arrive
        sock.sendall(b"a" * (model_shim.MAX_PROMPT_BYTES + 1))
        raw = _read_to_close(sock)
    finally:
        sock.close()
    assert raw == b""


def test_eof_before_newline_closes_without_reply(echo_server):
    port = echo_server
    sock = socket.create_connection(("127.0.0.1", port), timeout=10)
    try:
        sock.sendall(b"partial prompt, never terminated")
        sock.shutdown(socket.SHUT_WR)  # EOF on the server's read side
        raw = _read_to_close(sock)
    finally:
        sock.close()
    assert raw == b""


def test_ollama_response_sanitized_to_one_line():
    """No network: urlopen is mocked at the urllib boundary. Confirms the
    \r/\n-bearing "response" field ollama_complete returns comes out of
    handle_connection as a single sanitized line on the wire."""
    fake_body = b'{"response": "hello\\nworld\\r\\nagain"}'

    class _FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return fake_body

    with mock.patch.object(model_shim.urllib.request, "urlopen",
                           return_value=_FakeResp()) as m:
        client_sock, server_sock = socket.socketpair()
        try:
            client_sock.sendall(b"prompt\n")
            model_shim.handle_connection(
                server_sock, mode="ollama", model="dummy-model",
                endpoint="http://127.0.0.1:99999", timeout=1.0)
            server_sock.close()
            raw = _read_to_close(client_sock)
        finally:
            client_sock.close()
    m.assert_called_once()
    assert raw == b"hello world  again\n"
    assert b"\r" not in raw
    assert raw.count(b"\n") == 1  # only the one trailing terminator


def test_ollama_http_failure_is_fail_closed_empty_reply():
    """An ollama error must write nothing -- read-to-close yields b"", never
    a fabricated completion. No network: urlopen raises directly."""
    import urllib.error

    with mock.patch.object(model_shim.urllib.request, "urlopen",
                           side_effect=urllib.error.URLError("connection refused")):
        client_sock, server_sock = socket.socketpair()
        try:
            client_sock.sendall(b"prompt\n")
            model_shim.handle_connection(
                server_sock, mode="ollama", model="dummy-model",
                endpoint="http://127.0.0.1:99999", timeout=1.0)
            server_sock.close()
            raw = _read_to_close(client_sock)
        finally:
            client_sock.close()
    assert raw == b""


def test_echo_and_ollama_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        model_shim.build_arg_parser().parse_args(
            ["--echo", "--ollama", "llama3", "--once"])


def test_exactly_one_mode_is_required():
    with pytest.raises(SystemExit):
        model_shim.build_arg_parser().parse_args(["--once"])


# ---------------------------------------------------------------------------
# Model boundary receipts (--receipt-dir). Contract: buildlang's
# docs/MODEL-RECEIPT.md and
# docs/superpowers/specs/2026-07-29-model-boundary-receipts-design.md. The
# golden fixture (tests/fixtures/model-receipt-golden.json here, the SAME
# bytes as compiler/tests/fixtures/model-receipt-golden.json in buildlang) is
# the cross-repo pin: if this file's tests and buildlang's
# `golden_fixture_reseals_to_its_pinned_seal` ever disagree, the
# cross-language canonicalization contract is broken.
# ---------------------------------------------------------------------------

GOLDEN_PINNED_SEAL_HEX = "6bb2a09c47f5eaa2e3208a5eadcd6d57d1faffa74a567e024e920571c3794035"


def _load_golden_bytes() -> bytes:
    return GOLDEN_FIXTURE_PATH.read_bytes()


def _load_golden_receipt() -> dict:
    return json.loads(_load_golden_bytes())


def test_golden_fixture_reseals_to_its_pinned_seal():
    """The Python sealer, applied to the golden fixture's UNSEALED body (seal
    blanked, same as the fixture had before it was sealed), must reproduce
    the golden's exact pinned seal hex. This is the Python half of the
    cross-language pin; compiler/src/model_receipt.rs's
    `golden_fixture_reseals_to_its_pinned_seal` is the Rust half."""
    receipt = _load_golden_receipt()
    assert receipt["seal"]["hex"] == GOLDEN_PINNED_SEAL_HEX

    unsealed = json.loads(_load_golden_bytes())  # fresh copy, key order preserved
    unsealed["seal"]["hex"] = ""
    recomputed = model_shim._seal_receipt(unsealed)
    assert recomputed == GOLDEN_PINNED_SEAL_HEX
    # _seal_receipt mutates in place too
    assert unsealed["seal"]["hex"] == GOLDEN_PINNED_SEAL_HEX


def test_golden_fixture_canonicalization_reproduces_its_exact_bytes():
    """Re-serializing the golden fixture through this module's canonical
    on-disk shape (indent=2, ensure_ascii=False, one trailing newline -- the
    exact shape `_emit_receipt` writes) reproduces the fixture file's exact
    bytes, key order included. This proves the fixture committed here is not
    just logically equivalent to buildlang's but was produced by (or is
    indistinguishable from) the same canonicalization this shim emits."""
    raw = _load_golden_bytes()
    receipt = json.loads(raw)  # Python dicts preserve JSON object key order
    reserialized = (json.dumps(receipt, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    assert reserialized == raw


def test_golden_fixture_field_order_matches_schema_and_verifies_shape():
    """Key order in the fixture is the sealed canonical order (design section
    2 / MODEL-RECEIPT.md): schema, source, shim, session, prompt, reply,
    model, seed, outcome, seal. Order is load-bearing for the seal, not just
    cosmetic, so pin it explicitly."""
    receipt = _load_golden_receipt()
    assert list(receipt.keys()) == [
        "schema", "source", "shim", "session", "prompt", "reply",
        "model", "seed", "outcome", "seal",
    ]
    assert receipt["schema"] == model_shim.MODEL_RECEIPT_SCHEMA
    assert receipt["outcome"] == "COMPLETED"
    assert receipt["model"] == {"name": "echo/v1"}  # echo: ollama-only keys OMITTED
    assert receipt["seed"] == {"status": "NOT_SENT"}


def test_hashed_bytes_is_sha256_and_byte_count_over_raw_bytes():
    data = "café".encode("utf-8")  # multi-byte utf-8, 5 bytes / 4 chars
    result = model_shim._hashed_bytes(data)
    assert result == {"sha256": hashlib.sha256(data).hexdigest(), "bytes": 5}
    assert result["bytes"] != len("café")


@pytest.fixture
def receipt_echo_server(tmp_path):
    """An --echo --once shim started with --receipt-dir pointed at a fresh
    tmp_path subdirectory the shim itself must create (mkdir(parents=True,
    exist_ok=True) in _emit_receipt)."""
    receipt_dir = tmp_path / "receipts"
    proc = _spawn_shim("--echo", "--once", "--receipt-dir", str(receipt_dir))
    try:
        port = _read_bound_port(proc)
        yield port, receipt_dir
    finally:
        _cleanup(proc)


def _one_receipt(receipt_dir: Path) -> dict:
    files = list(receipt_dir.iterdir())
    assert len(files) == 1, f"expected exactly one receipt file, found {files}"
    return json.loads(files[0].read_text(encoding="utf-8"))


def test_receipt_emitted_end_to_end_over_real_echo_connection(receipt_echo_server):
    port, receipt_dir = receipt_echo_server
    prompt = "ping"
    sock = socket.create_connection(("127.0.0.1", port), timeout=10)
    try:
        sock.sendall((prompt + "\n").encode("utf-8"))
        sock.shutdown(socket.SHUT_WR)
        raw = _read_to_close(sock)
    finally:
        sock.close()
    assert _trim_trailing_newline(raw) == "echo: ping"

    receipt = _one_receipt(receipt_dir)
    assert receipt["schema"] == model_shim.MODEL_RECEIPT_SCHEMA
    assert receipt["source"] == "model:echo:echo/v1"
    assert receipt["shim"] == {"name": "model_shim.py", "version": model_shim.SHIM_VERSION,
                               "mode": "echo"}
    assert receipt["outcome"] == "COMPLETED"
    assert receipt["model"] == {"name": "echo/v1"}
    assert receipt["seed"] == {"status": "NOT_SENT"}
    assert receipt["session"]["reply_written_utc"] is not None

    # This is the same prompt/reply pair as the golden fixture, so the
    # witnessed hashes must match the pin exactly.
    assert receipt["prompt"] == {
        "sha256": "758d61f26a44448384e5c4468a0dcb7a2abe456067b0f7b505bc28b9411fe931",
        "bytes": 4,
    }
    assert receipt["reply"] == {
        "sha256": "de2406a7ccdb9add6361bdf86cfd31dfaa95806f8d42f91102290ae3abe5afae",
        "bytes": 10,
    }

    # The emitted receipt must reseal to itself: the writer's own seal is
    # internally consistent (a live-emission analogue of the golden pin).
    original_hex = receipt["seal"]["hex"]
    recomputed_hex = model_shim._seal_receipt(json.loads(json.dumps(receipt)))
    assert recomputed_hex == original_hex


def test_receipt_prompt_and_reply_hashes_match_exact_wire_bytes(receipt_echo_server):
    """Hash correctness against the bytes the design names: prompt.sha256 is
    over the raw prompt-line bytes as received (terminator stripped, before
    utf-8 decode); reply.sha256 is over the sanitized completion bytes
    exactly as written, excluding the protocol-terminator \n."""
    port, receipt_dir = receipt_echo_server
    prompt = "hello café receipts"  # exercises multi-byte utf-8 in the prompt
    prompt_bytes = prompt.encode("utf-8")
    sock = socket.create_connection(("127.0.0.1", port), timeout=10)
    try:
        sock.sendall(prompt_bytes + b"\n")
        sock.shutdown(socket.SHUT_WR)
        raw = _read_to_close(sock)
    finally:
        sock.close()
    assert raw.endswith(b"\n")
    wire_reply_bytes = raw[:-1]  # exactly one trailing terminator per the wire contract

    receipt = _one_receipt(receipt_dir)
    assert receipt["prompt"]["sha256"] == hashlib.sha256(prompt_bytes).hexdigest()
    assert receipt["prompt"]["bytes"] == len(prompt_bytes)
    assert receipt["reply"]["sha256"] == hashlib.sha256(wire_reply_bytes).hexdigest()
    assert receipt["reply"]["bytes"] == len(wire_reply_bytes)


def test_protocol_violation_emits_receipt_with_null_prompt_and_reply(tmp_path):
    """An overlong/unterminated prompt line is OUTCOME PROTOCOL_VIOLATION:
    `prompt` is null (nothing was ever legally received) and `reply` is null
    (nothing was ever written)."""
    receipt_dir = tmp_path / "receipts"
    proc = _spawn_shim("--echo", "--once", "--receipt-dir", str(receipt_dir))
    try:
        port = _read_bound_port(proc)
        sock = socket.create_connection(("127.0.0.1", port), timeout=10)
        try:
            sock.sendall(b"partial prompt, never terminated")
            sock.shutdown(socket.SHUT_WR)
            raw = _read_to_close(sock)
        finally:
            sock.close()
        assert raw == b""
        proc.wait(timeout=5)
    finally:
        _cleanup(proc)

    receipt = _one_receipt(receipt_dir)
    assert receipt["outcome"] == "PROTOCOL_VIOLATION"
    assert receipt["prompt"] is None
    assert receipt["reply"] is None
    assert receipt["session"]["reply_written_utc"] is None
    # Still a validly sealed artifact even though nothing but a refusal
    # happened -- a refusal is a boundary fact too (design section 2 row 8).
    reloaded = json.loads(json.dumps(receipt))
    original_hex = reloaded["seal"]["hex"]
    assert model_shim._seal_receipt(reloaded) == original_hex


def test_ollama_failed_closed_emits_receipt_with_null_reply(tmp_path):
    """A fail-closed ollama error (network mocked, no live call -- see the
    module docstring) is outcome FAILED_CLOSED: `prompt` is present (the
    request WAS received), `reply` is null (nothing was ever written)."""
    receipt_dir = tmp_path / "receipts"
    receipt_dir.mkdir()
    import urllib.error

    with mock.patch.object(model_shim.urllib.request, "urlopen",
                           side_effect=urllib.error.URLError("connection refused")):
        client_sock, server_sock = socket.socketpair()
        try:
            client_sock.sendall(b"prompt\n")
            model_shim.handle_connection(
                server_sock, mode="ollama", model="dummy-model",
                endpoint="http://127.0.0.1:99999", timeout=1.0,
                receipt_dir=str(receipt_dir), listen="127.0.0.1:0")
            server_sock.close()
            raw = _read_to_close(client_sock)
        finally:
            client_sock.close()
    assert raw == b""

    receipt = _one_receipt(receipt_dir)
    assert receipt["outcome"] == "FAILED_CLOSED"
    assert receipt["reply"] is None
    assert receipt["prompt"] is not None
    assert receipt["prompt"]["sha256"] == hashlib.sha256(b"prompt").hexdigest()
    assert receipt["model"]["daemon_digest"] == {"status": "UNAVAILABLE"}


def test_no_receipt_dir_means_no_emission_attempted():
    """Design section 1 / MODEL-RECEIPT.md: 'No flag, no receipt, byte-
    identical behavior to today.' Proven at the unit level by making
    `_emit_receipt` a trap: if `receipt_dir=None` (the default -- what a
    command line omitting --receipt-dir produces) ever reaches emission, the
    trap fires and the test fails."""
    def _must_not_be_called(*args, **kwargs):
        raise AssertionError("_emit_receipt must not be called when receipt_dir is None")

    with mock.patch.object(model_shim, "_emit_receipt", side_effect=_must_not_be_called):
        client_sock, server_sock = socket.socketpair()
        try:
            client_sock.sendall(b"ping\n")
            model_shim.handle_connection(server_sock, mode="echo", model="", endpoint="",
                                         timeout=1.0)  # receipt_dir omitted -> None
            server_sock.close()
            raw = _read_to_close(client_sock)
        finally:
            client_sock.close()
    assert raw == b"echo: ping\n"


def test_no_receipt_dir_flag_leaves_directory_empty(tmp_path):
    """Same claim as above, exercised end to end over a real subprocess: a
    shim started WITHOUT --receipt-dir writes nothing to a directory it was
    never told about, and a normal reply still happens."""
    proc = _spawn_shim("--echo", "--once")
    try:
        port = _read_bound_port(proc)
        sock = socket.create_connection(("127.0.0.1", port), timeout=10)
        try:
            sock.sendall(b"ping\n")
            sock.shutdown(socket.SHUT_WR)
            raw = _read_to_close(sock)
        finally:
            sock.close()
        assert raw == b"echo: ping\n"
        proc.wait(timeout=5)
    finally:
        _cleanup(proc)
    assert not tmp_path.exists() or list(tmp_path.iterdir()) == []
