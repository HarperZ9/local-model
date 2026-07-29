"""model_shim.py: harness-side server for BuildLang's `Model` capability wire
contract (buildc branch feat/model-capability; contract source: the buildlang
repo's docs/SCIENTIFIC-RECEIPT.md Model paragraph and examples/model_propose.bld
header). Not a vendored copy of that doc -- this module IS the implementation,
so the contract is restated here as the thing this code must hold to.

Wire contract: a client connects over TCP, sends ONE prompt line terminated by
a single \n (the prompt itself never contains a newline -- a conforming client
aborts before sending one), then reads the reply until the server CLOSES the
connection. A conforming server writes ONE completion line and closes. The
client trims exactly one trailing \n (and a \r immediately before it, if any)
from what it read. Consequence: the line protocol has no way to carry an
embedded newline in the completion. Every completion is therefore sanitized to
a single line before it is written (see `_sanitize`) -- \r and \n each become
a space. This is not optional cleanup; an unsanitized multi-line completion
would be truncated or corrupted by every conforming client.

Modes (argparse-level, mutually exclusive, exactly one required -- an unknown
or ambiguous mode combination is refused before a socket is ever opened):
  --echo             deterministic completion "echo: " + prompt. For tests,
                      and for buildc-side offline development against a real
                      socket without a model.
  --ollama MODEL      proxy to an ollama endpoint (default --endpoint
                      http://127.0.0.1:11434): POST /api/generate with
                      {"model": MODEL, "prompt": PROMPT, "stream": false},
                      taking the "response" field as the completion.
                      UNTESTED-LIVE: as of this commit this path has not been
                      exercised against a running ollama instance (hardware
                      gated tonight). It is stdlib urllib and is unit tested
                      with the network mocked at the urllib boundary, but no
                      live call has been made -- treat it as unverified until
                      a hardware session confirms it end-to-end.

Fail-closed on ollama errors: a connection failure, a non-2xx status, a
malformed body, or a response missing the "response" field writes NOTHING to
the client. The connection is closed and the failure is logged to stderr
only. The client's read-to-close then yields an empty completion, never a
fabricated one -- a client that cannot tell "the model said nothing" from
"the model said something wrong" is worse than one that gets nothing back.

Serving shape: one connection at a time (the contract has no concurrency
requirement). --port 0 (the default) binds an ephemeral port; the actually
bound port is always printed as the first stdout line, flushed, so a caller
(a test, or buildc's offline harness) can read it back before connecting.
--once serves exactly one connection then exits 0; the default serves until
interrupted. Each connection gets its own socket timeout (--conn-timeout,
default 30s) so a client that connects and never sends its newline cannot
wedge the server open forever.

Model boundary receipts (`--receipt-dir PATH`, v1.1 of the shim contract):
when set, one sealed `buildlang-model-boundary-receipt/v0` JSON is written per
connection to PATH -- a provenance artifact witnessing the exact bytes that
crossed this boundary, never the model's quality or weights. No flag, no
receipt: behavior on the wire is byte-identical to before this feature
existed either way. Contract source: buildlang's docs/MODEL-RECEIPT.md and
docs/superpowers/specs/2026-07-29-model-boundary-receipts-design.md. The seal
is sha256 over the compact-JSON canonical body with `seal.hex` blanked,
computed to be byte-identical to buildlang's Rust sealer (same field order,
same compact separators, no floats anywhere in the schema) -- see
`_seal_receipt` and the golden fixture pinned in both repos
(tests/fixtures/model-receipt-golden.json here, compiler/tests/fixtures of
the same name in buildlang). Receipt emission never raises: any failure (a
bad directory, a permission error) is logged to stderr and swallowed, because
it must never block or break the reply path (see `_emit_receipt`).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

MAX_PROMPT_BYTES = 1024 * 1024  # 1 MiB cap on the incoming prompt line
DEFAULT_OLLAMA_ENDPOINT = "http://127.0.0.1:11434"
DEFAULT_CONN_TIMEOUT = 30.0
DEFAULT_HOST = "127.0.0.1"
RECV_CHUNK = 4096

# Model boundary receipt constants (docs/MODEL-RECEIPT.md in buildlang is the
# contract of record). SHIM_VERSION is sealed into every receipt's
# `shim.version` field and is pinned by the golden fixture in both repos --
# changing it changes the golden fixture's seal, so it must not be bumped
# without re-deriving that fixture in lockstep with buildlang.
MODEL_RECEIPT_SCHEMA = "buildlang-model-boundary-receipt/v0"
SHIM_VERSION = "0.1.0"
ECHO_MODEL_NAME = "echo/v1"


def _sanitize(text: str) -> str:
    """Collapse a completion to one line. The wire protocol trims exactly one
    trailing \n (and a preceding \r) and nothing else, so any \r or \n left
    inside the completion would be corrupted, not carried -- replace both
    with a space rather than let that happen silently."""
    return text.replace("\r", " ").replace("\n", " ")


def _read_prompt_line(conn: socket.socket, max_bytes: int) -> bytes | None:
    """Read one \n-terminated line, bounded at max_bytes.

    Returns the RAW prompt bytes (trailing \n and a preceding \r stripped,
    still undecoded), or None if the connection hit EOF before a newline
    arrived, or the line exceeded max_bytes before one did. Both are protocol
    violations: the caller must close the connection without writing a reply.
    Raw bytes, not a decoded string: a model boundary receipt's `prompt.sha256`
    is sealed over these exact bytes, BEFORE utf-8 decoding (which is lossy --
    the caller decodes separately with errors="replace" for the completion
    logic, but the receipt witnesses what actually crossed the wire).
    """
    buf = bytearray()
    while True:
        chunk = conn.recv(RECV_CHUNK)
        if not chunk:
            return None  # EOF before a newline arrived
        nl = chunk.find(b"\n")
        if nl != -1:
            buf += chunk[:nl]
            if len(buf) > max_bytes:
                return None  # overlong line
            break
        buf += chunk
        if len(buf) > max_bytes:
            return None  # overlong line
    if buf.endswith(b"\r"):
        buf = buf[:-1]
    return bytes(buf)


def echo_complete(prompt: str) -> str:
    return f"echo: {prompt}"


def _ollama_request_body_bytes(model: str, prompt: str) -> bytes:
    """The exact JSON bytes `ollama_complete` POSTs to /api/generate. Pulled
    out as its own pure function so a model boundary receipt's
    `model.request_body_sha256` can be computed from the SAME construction
    the real POST uses (calling this twice with the same args is
    byte-identical, since json.dumps over a fixed-key-order dict literal is
    deterministic), rather than risking two code paths drifting apart."""
    return json.dumps({"model": model, "prompt": prompt, "stream": False}).encode("utf-8")


def ollama_complete(prompt: str, model: str, endpoint: str,
                    timeout: float = DEFAULT_CONN_TIMEOUT) -> str | None:
    """POST prompt to an ollama /api/generate endpoint. Returns the raw
    "response" field (not yet sanitized -- the caller sanitizes uniformly
    before writing to the socket), or None on any failure. UNTESTED-LIVE:
    see the module docstring. Never called during this repo's own test run.
    """
    url = endpoint.rstrip("/") + "/api/generate"
    body = _ollama_request_body_bytes(model, prompt)
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
        print(f"[model_shim] ollama request failed: {e!r}", file=sys.stderr)
        return None
    response = payload.get("response") if isinstance(payload, dict) else None
    if not isinstance(response, str):
        print(f"[model_shim] ollama response missing 'response' field: {payload!r}",
              file=sys.stderr)
        return None
    return response


def _is_sha256_hex(value: str) -> bool:
    return len(value) == 64 and all(c in "0123456789abcdefABCDEF" for c in value)


def _fetch_ollama_daemon_digest(model: str, endpoint: str,
                                timeout: float = DEFAULT_CONN_TIMEOUT) -> dict:
    """GET <endpoint>/api/tags and extract the digest ollama declares for
    `model`, for the receipt's `model.daemon_digest` field.

    UNTESTED-LIVE (see the module docstring): the /api/tags response shape
    assumed here (`{"models": [{"name": ..., "digest": ...}, ...]}`, digest
    optionally prefixed `sha256:`) has not been confirmed against a running
    daemon; pin it during the hardware-gated live session the ollama path
    already needs. Fails closed to `{"status": "UNAVAILABLE"}` on ANY problem
    -- network failure, unexpected response shape, no matching model entry,
    or a digest that is not a well-formed 64-hex-char sha256 -- because a
    receipt claiming FETCHED must be right: "weights I could not identify" is
    honest where a guessed or malformed digest would not be. Even a FETCHED
    result only witnesses that the daemon reported this digest for this model
    name AT FETCH TIME; it is the daemon's own declaration about itself, not
    independently checked against the weights.
    """
    url = endpoint.rstrip("/") + "/api/tags"
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        models = payload.get("models") if isinstance(payload, dict) else None
        if not isinstance(models, list):
            return {"status": "UNAVAILABLE"}
        for entry in models:
            if not isinstance(entry, dict) or entry.get("name") != model:
                continue
            digest = entry.get("digest")
            if not isinstance(digest, str):
                break
            if digest.startswith("sha256:"):
                digest = digest[len("sha256:"):]
            if _is_sha256_hex(digest):
                return {"status": "FETCHED", "hex": digest.lower()}
            break
        return {"status": "UNAVAILABLE"}
    except (urllib.error.URLError, TimeoutError, OSError, ValueError,
            AttributeError, TypeError) as e:
        print(f"[model_shim] daemon digest fetch failed: {e!r}", file=sys.stderr)
        return {"status": "UNAVAILABLE"}


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hashed_bytes(data: bytes) -> dict:
    """`{ sha256, bytes }` over raw bytes -- the receipt's `prompt`/`reply`
    shape. `bytes` is a byte COUNT, never content: the receipt carries no
    plaintext (docs/MODEL-RECEIPT.md's deliberate exclusions)."""
    return {"sha256": _sha256_hex(data), "bytes": len(data)}


def _seal_receipt(receipt: dict) -> str:
    """Seal a receipt dict IN PLACE: sha256 over the canonical JSON with
    `seal.hex` blanked and `seal.algorithm` fixed to `"sha256"`. Returns the
    computed hex.

    This is the Python half of the cross-language canonicalization contract
    (docs/MODEL-RECEIPT.md in buildlang): compact separators (no whitespace,
    matching `serde_json::to_vec`), object keys in the schema's FIXED order
    (Python dicts preserve insertion order; the receipt dict is always built
    with keys inserted in that order -- see `_emit_receipt`), non-ASCII
    unescaped (`ensure_ascii=False`, matching serde_json's default), and no
    floats anywhere in the schema, which is what makes this agree byte-for-
    byte with the Rust sealer despite being two different JSON libraries.
    Mutates `receipt["seal"]` in place (rather than reassigning the `seal`
    key) so the key's ALREADY-CORRECT position in insertion order is
    preserved regardless of call order.
    """
    receipt["seal"]["algorithm"] = "sha256"
    receipt["seal"]["hex"] = ""
    canonical = json.dumps(receipt, separators=(",", ":"), ensure_ascii=False)
    hex_digest = _sha256_hex(canonical.encode("utf-8"))
    receipt["seal"]["hex"] = hex_digest
    return hex_digest


def _build_model_block(mode: str, model: str, endpoint: str,
                       request_body_sha256: str | None = None,
                       daemon_digest: dict | None = None) -> dict:
    """The receipt's `model` block. Echo mode carries only `name` (the three
    ollama-only keys are OMITTED, not null, on an echo receipt). For ollama,
    `request_body_sha256`/`daemon_digest` are each included only when a value
    was actually computed (never for a PROTOCOL_VIOLATION, where no request
    was ever constructed or sent -- there is nothing honest to claim)."""
    if mode == "echo":
        return {"name": ECHO_MODEL_NAME}
    block = {"name": model, "endpoint": endpoint}
    if request_body_sha256 is not None:
        block["request_body_sha256"] = request_body_sha256
    if daemon_digest is not None:
        block["daemon_digest"] = daemon_digest
    return block


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _utc_compact_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _emit_receipt(receipt_dir: str, *, mode: str, model: str, endpoint: str,
                  listen: str, shim_version: str, nonce: str,
                  request_received_utc: str, reply_written_utc: str | None,
                  prompt_raw: bytes | None, reply_bytes: bytes | None,
                  outcome: str, request_body_sha256: str | None = None,
                  daemon_digest: dict | None = None) -> None:
    """Build, seal, and write one model boundary receipt to `receipt_dir`.

    Never raises. Any failure here (a missing/unwritable directory, a
    serialization bug) is logged to stderr and swallowed: emission is
    opt-in and additive, so it must never block or break the reply path
    (the reply, when there is one, has already been sent to the client by
    the time this is called -- see `handle_connection`).
    """
    try:
        name_for_source = ECHO_MODEL_NAME if mode == "echo" else model
        receipt: dict = {
            "schema": MODEL_RECEIPT_SCHEMA,
            "source": f"model:{mode}:{name_for_source}",
            "shim": {"name": "model_shim.py", "version": shim_version, "mode": mode},
            "session": {
                "listen": listen,
                "nonce": nonce,
                "request_received_utc": request_received_utc,
                "reply_written_utc": reply_written_utc,
            },
            "prompt": _hashed_bytes(prompt_raw) if prompt_raw is not None else None,
            "reply": _hashed_bytes(reply_bytes) if reply_bytes is not None else None,
            "model": _build_model_block(mode, model, endpoint, request_body_sha256,
                                        daemon_digest),
            "seed": {"status": "NOT_SENT"},
            "outcome": outcome,
            "seal": {"algorithm": "sha256", "hex": ""},
        }
        _seal_receipt(receipt)

        receipt_dir_path = Path(receipt_dir)
        receipt_dir_path.mkdir(parents=True, exist_ok=True)
        path = receipt_dir_path / f"model-receipt-{_utc_compact_stamp()}-{nonce}.json"
        path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    except Exception as e:  # fail-closed: receipt emission must never crash
        print(f"[model_shim] receipt emission failed: {e!r}", file=sys.stderr)


def handle_connection(conn: socket.socket, *, mode: str, model: str, endpoint: str,
                      timeout: float = DEFAULT_CONN_TIMEOUT,
                      receipt_dir: str | None = None, listen: str = "",
                      shim_version: str = SHIM_VERSION) -> None:
    """Serve exactly one request on an already-accepted connection: read the
    prompt line, compute the completion for `mode`, write it sanitized with
    its trailing \n, and return (the caller closes the socket). Writes
    nothing and returns early on a protocol violation or a fail-closed
    ollama error -- the caller's close is then the entire reply.

    When `receipt_dir` is set, emits one sealed model boundary receipt per
    connection (see `_emit_receipt`), covering all three outcomes:
    PROTOCOL_VIOLATION (no prompt), FAILED_CLOSED (a prompt but no reply),
    and COMPLETED. `receipt_dir=None` (the default) skips all receipt work
    (no hashing, no daemon-digest fetch, no file I/O), so the wire behavior
    and cost are byte-identical to before this parameter existed.
    """
    request_received_utc = _utc_now_iso()
    nonce = os.urandom(4).hex()

    raw = _read_prompt_line(conn, MAX_PROMPT_BYTES)
    if raw is None:
        print("[model_shim] closing connection: overlong or unterminated prompt line",
              file=sys.stderr)
        if receipt_dir is not None:
            _emit_receipt(
                receipt_dir, mode=mode, model=model, endpoint=endpoint, listen=listen,
                shim_version=shim_version, nonce=nonce,
                request_received_utc=request_received_utc, reply_written_utc=None,
                prompt_raw=None, reply_bytes=None, outcome="PROTOCOL_VIOLATION",
            )
        return

    prompt = raw.decode("utf-8", errors="replace")
    request_body_sha256 = None
    if mode == "echo":
        completion = echo_complete(prompt)
    else:
        request_body_sha256 = _sha256_hex(_ollama_request_body_bytes(model, prompt))
        completion = ollama_complete(prompt, model, endpoint, timeout=timeout)
        if completion is None:
            if receipt_dir is not None:
                daemon_digest = _fetch_ollama_daemon_digest(model, endpoint, timeout)
                _emit_receipt(
                    receipt_dir, mode=mode, model=model, endpoint=endpoint, listen=listen,
                    shim_version=shim_version, nonce=nonce,
                    request_received_utc=request_received_utc, reply_written_utc=None,
                    prompt_raw=raw, reply_bytes=None, outcome="FAILED_CLOSED",
                    request_body_sha256=request_body_sha256, daemon_digest=daemon_digest,
                )
            return  # fail closed: nothing written, see module docstring

    sanitized_bytes = _sanitize(completion).encode("utf-8")
    conn.sendall(sanitized_bytes + b"\n")
    reply_written_utc = _utc_now_iso()

    if receipt_dir is not None:
        daemon_digest = (_fetch_ollama_daemon_digest(model, endpoint, timeout)
                         if mode == "ollama" else None)
        _emit_receipt(
            receipt_dir, mode=mode, model=model, endpoint=endpoint, listen=listen,
            shim_version=shim_version, nonce=nonce,
            request_received_utc=request_received_utc, reply_written_utc=reply_written_utc,
            prompt_raw=raw, reply_bytes=sanitized_bytes, outcome="COMPLETED",
            request_body_sha256=request_body_sha256, daemon_digest=daemon_digest,
        )


def serve(host: str, port: int, *, mode: str, model: str, endpoint: str,
          once: bool, conn_timeout: float, receipt_dir: str | None = None) -> int:
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        srv.bind((host, port))
        srv.listen(1)
        bound_port = srv.getsockname()[1]
        # First stdout line, flushed: the only way a caller (test, or
        # buildc's offline harness) learns an ephemeral --port 0 bind.
        print(bound_port, flush=True)
        listen = f"{host}:{bound_port}"
        while True:
            conn, _addr = srv.accept()
            conn.settimeout(conn_timeout)
            try:
                handle_connection(conn, mode=mode, model=model, endpoint=endpoint,
                                  timeout=conn_timeout, receipt_dir=receipt_dir,
                                  listen=listen)
            except socket.timeout:
                print("[model_shim] closing connection: timed out", file=sys.stderr)
            except OSError as e:
                print(f"[model_shim] connection error: {e!r}", file=sys.stderr)
            finally:
                conn.close()
            if once:
                return 0
    finally:
        srv.close()


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Harness-side server for BuildLang's Model capability wire contract.")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--echo", action="store_true",
                       help="deterministic completion: 'echo: ' + prompt")
    mode.add_argument("--ollama", metavar="MODEL", default=None,
                       help="proxy to an ollama /api/generate endpoint (UNTESTED-LIVE)")
    ap.add_argument("--endpoint", default=DEFAULT_OLLAMA_ENDPOINT,
                     help=f"ollama base URL (default {DEFAULT_OLLAMA_ENDPOINT})")
    ap.add_argument("--host", default=DEFAULT_HOST,
                     help=f"listen host (default {DEFAULT_HOST})")
    ap.add_argument("--port", type=int, default=0,
                     help="listen port; 0 = ephemeral (default)")
    ap.add_argument("--once", action="store_true",
                     help="serve exactly one connection then exit 0 "
                          "(default: serve until interrupted)")
    ap.add_argument("--conn-timeout", type=float, default=DEFAULT_CONN_TIMEOUT,
                     help=f"per-connection socket timeout in seconds "
                          f"(default {DEFAULT_CONN_TIMEOUT})")
    ap.add_argument("--receipt-dir", default=None, metavar="PATH",
                     help="write a sealed buildlang-model-boundary-receipt/v0 JSON "
                          "per connection to PATH (default: no receipts, wire "
                          "behavior byte-identical to before this flag existed)")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    mode = "echo" if args.echo else "ollama"
    try:
        return serve(args.host, args.port, mode=mode, model=args.ollama or "",
                    endpoint=args.endpoint, once=args.once,
                    conn_timeout=args.conn_timeout, receipt_dir=args.receipt_dir)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
