"""Zero-dependency demo shot recorder.

Runs a demo script (JSON list of titled shell steps), captures honest
stdout/stderr/exit-code/wall-time evidence per step, and emits:

* demos/<name>/transcript.json  (schema harness.demo-transcript/v1)
* demos/<name>/player.html      (self-contained offline terminal player)

Python 3.12 stdlib only. No external dependencies, no network of its own.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

try:
    from scripts.demo_player_html import render_player_html
except ImportError:  # direct invocation: python scripts/demo_recorder.py
    from demo_player_html import render_player_html

TRANSCRIPT_SCHEMA = "harness.demo-transcript/v1"
DEFAULT_STEP_TIMEOUT_SECONDS = 120.0
MAX_CAPTURED_CHARS = 100_000
SAFE_ENV_KEYS = {"COMSPEC", "PATH", "PATHEXT", "SYSTEMROOT", "TEMP", "TMP", "WINDIR"}
CAPTURE_TYPES = {"terminal-replay", "browser-video", "native-video"}
DRY_RUN_PLACEHOLDER = (
    "[dry-run] command not executed\n"
    "$ {command}\n"
    "(placeholder output so the demo pipeline can be tested with no side effects)"
)

_EMAIL_PATTERN = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_WINDOWS_PATH_PATTERN = re.compile(r"\b[A-Z]:\\+[^\r\n\"'<>|]*", re.IGNORECASE)
_BEARER_PATTERN = re.compile(r"\bBearer\s+[A-Z0-9._~+/=-]+", re.IGNORECASE)
_OPENAI_TOKEN_PATTERN = re.compile(r"\bsk-(?:proj-|svcacct-)?[A-Z0-9_-]{16,}", re.IGNORECASE)
_GITHUB_TOKEN_PATTERN = re.compile(r"\bgh[pousr]_[A-Z0-9]{20,}\b", re.IGNORECASE)
_CREDENTIAL_PATTERN = re.compile(
    r"\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|password|passwd|secret|token)\b"
    r"\s*[:=]\s*(?:[\"'][^\"'\r\n]+[\"']|[^\s,;]+)",
    re.IGNORECASE,
)
_BUILTIN_REDACTIONS = (
    (_EMAIL_PATTERN, "[redacted-email]"),
    (_WINDOWS_PATH_PATTERN, "[redacted-path]"),
    (_BEARER_PATTERN, "[redacted-token]"),
    (_OPENAI_TOKEN_PATTERN, "[redacted-token]"),
    (_GITHUB_TOKEN_PATTERN, "[redacted-token]"),
    (_CREDENTIAL_PATTERN, "[redacted-credential]"),
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _now_utc() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _truncate(text: str) -> str:
    if len(text) <= MAX_CAPTURED_CHARS:
        return text
    dropped = len(text) - MAX_CAPTURED_CHARS
    return text[:MAX_CAPTURED_CHARS] + f"\n[recorder] output truncated: {dropped} characters dropped"


def _concrete_private_values() -> tuple[list[str], list[str]]:
    home_values = {
        value
        for value in (str(Path.home()), os.environ.get("USERPROFILE"), os.environ.get("HOME"))
        if value
    }
    home_values |= {value.replace("\\", "/") for value in home_values}
    usernames = {
        value
        for value in (os.environ.get("USERNAME"), os.environ.get("USER"))
        if value and len(value) >= 3
    }
    return sorted(home_values, key=len, reverse=True), sorted(usernames, key=len, reverse=True)


def scrub_text(text: str, explicit_patterns: Sequence[str] = ()) -> tuple[str, int]:
    scrubbed = text
    replacement_count = 0
    home_values, usernames = _concrete_private_values()
    for value in home_values:
        scrubbed, count = re.subn(
            re.escape(value) + r"[^\r\n]*",
            "[redacted-home]",
            scrubbed,
            flags=re.IGNORECASE,
        )
        replacement_count += count
    for value in usernames:
        scrubbed, count = re.subn(
            re.escape(value),
            "[redacted-home]",
            scrubbed,
            flags=re.IGNORECASE,
        )
        replacement_count += count
    for pattern, replacement in _BUILTIN_REDACTIONS:
        scrubbed, count = pattern.subn(replacement, scrubbed)
        replacement_count += count
    for value in explicit_patterns:
        scrubbed, count = re.subn(re.escape(value), "[redacted-explicit]", scrubbed)
        replacement_count += count
    return scrubbed, replacement_count


def _has_forbidden_value(text: str, explicit_patterns: Sequence[str] = ()) -> bool:
    home_values, usernames = _concrete_private_values()
    lowered = text.casefold()
    if any(value.casefold() in lowered for value in (*home_values, *usernames)):
        return True
    if any(pattern.search(text) for pattern, _replacement in _BUILTIN_REDACTIONS):
        return True
    return any(value in text for value in explicit_patterns)


def _resolve_step_cwd(step: dict, default_cwd: Path) -> Path:
    configured = step.get("cwd")
    path = Path(configured).expanduser() if configured else default_cwd
    path = path if path.is_absolute() else default_cwd / path
    resolved = path.resolve()
    if not resolved.is_dir():
        raise ValueError(f"step cwd is not a directory: {configured or default_cwd}")
    return resolved


def _build_step_env(step: dict, inherited_env: Mapping[str, str]) -> dict[str, str]:
    names = SAFE_ENV_KEYS | set(step.get("env_allowlist", []))
    return {name: inherited_env[name] for name in sorted(names) if name in inherited_env}


def _evaluate_assertions(step: dict, exit_code: int, output: str) -> dict:
    expected = step.get("expected_exit_codes", [0])
    required = step.get("expect", [])
    missing = [value for value in required if value not in output]
    return {
        "expected_exit_codes": expected,
        "exit_code_ok": exit_code in expected,
        "required": required,
        "missing": missing,
        "passed": exit_code in expected and not missing,
    }


def load_demo_script(path: Path) -> list[dict]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    steps = raw.get("steps") if isinstance(raw, dict) else raw
    if not isinstance(steps, list) or not steps:
        raise ValueError(f"demo script must be a non-empty list of steps: {path}")
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            raise ValueError(f"step {index} is not an object")
        for key in ("title", "command", "narration"):
            if not isinstance(step.get(key), str) or not step[key].strip():
                raise ValueError(f"step {index} is missing a non-empty '{key}'")
        expected_exit_codes = step.get("expected_exit_codes", [0])
        if (
            not isinstance(expected_exit_codes, list)
            or not expected_exit_codes
            or any(type(value) is not int for value in expected_exit_codes)
        ):
            raise ValueError(f"step {index} expected_exit_codes must be a non-empty list of integers")
        for key in ("expect", "env_allowlist", "redact_patterns"):
            values = step.get(key, [])
            if (
                not isinstance(values, list)
                or any(not isinstance(value, str) or not value.strip() for value in values)
            ):
                raise ValueError(f"step {index} {key} must be a list of non-empty strings")
        capture = step.get("capture", "terminal-replay")
        if capture not in CAPTURE_TYPES:
            raise ValueError(f"step {index} capture must be one of {sorted(CAPTURE_TYPES)}")
    return steps


def execute_step(
    step: dict,
    *,
    index: int,
    dry_run: bool,
    timeout_seconds: float = DEFAULT_STEP_TIMEOUT_SECONDS,
    cwd: Path | None = None,
    run_temp: Path | None = None,
    inherited_env: Mapping[str, str] | None = None,
) -> dict:
    execution_step = dict(step)
    command = step["command"]
    if run_temp is not None:
        command = command.replace("{demo_temp}", str(run_temp))
        configured_cwd = execution_step.get("cwd")
        if isinstance(configured_cwd, str):
            execution_step["cwd"] = configured_cwd.replace("{demo_temp}", str(run_temp))
    display_command = step.get("display_command", step["command"])
    step_cwd = _resolve_step_cwd(execution_step, Path(cwd or _repo_root()))
    source_env = os.environ if inherited_env is None else inherited_env
    step_env = _build_step_env(step, source_env)
    if dry_run:
        stdout = DRY_RUN_PLACEHOLDER.format(command=display_command)
        stderr = ""
        exit_code = 0
        duration_ms = 0
        mode = "dry-run"
    else:
        mode = "live"
        started = time.perf_counter()
        try:
            completed = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                timeout=timeout_seconds,
                cwd=str(step_cwd),
                env=step_env,
            )
            stdout = completed.stdout.decode("utf-8", errors="replace")
            stderr = completed.stderr.decode("utf-8", errors="replace")
            exit_code = completed.returncode
        except subprocess.TimeoutExpired as exc:
            stdout = (exc.stdout or b"").decode("utf-8", errors="replace")
            stderr = (exc.stderr or b"").decode("utf-8", errors="replace")
            stderr += f"\n[recorder] step timed out after {timeout_seconds:g} seconds"
            exit_code = -1
        duration_ms = int(round((time.perf_counter() - started) * 1000))

    stdout = _truncate(stdout).replace("\r\n", "\n")
    stderr = _truncate(stderr).replace("\r\n", "\n")
    output = stdout if not stderr.strip() else (stdout + ("\n" if stdout else "") + stderr)
    assertions = _evaluate_assertions(step, exit_code, output)
    explicit_patterns = list(step.get("redact_patterns", []))
    if run_temp is not None:
        explicit_patterns.extend((str(run_temp), str(run_temp).replace("\\", "/")))

    redaction_count = 0

    def scrub(value: str) -> str:
        nonlocal redaction_count
        scrubbed, count = scrub_text(value, explicit_patterns)
        redaction_count += count
        return scrubbed

    safe_title = scrub(step["title"])
    safe_narration = scrub(step["narration"])
    safe_display_command = scrub(display_command)
    safe_stdout = scrub(stdout)
    safe_stderr = scrub(stderr)
    safe_output = scrub(output)
    assertions["required"] = [scrub(value) for value in assertions["required"]]
    assertions["missing"] = [scrub(value) for value in assertions["missing"]]
    return {
        "index": index,
        "title": safe_title,
        "narration": safe_narration,
        "command": safe_display_command,
        "display_command": safe_display_command,
        "capture": step.get("capture", "terminal-replay"),
        "mode": mode,
        "exit_code": exit_code,
        "duration_ms": duration_ms,
        "stdout": safe_stdout,
        "stderr": safe_stderr,
        "output": safe_output,
        "output_sha256": _sha256_text(safe_output),
        "assertions": assertions,
        "assertions_passed": assertions["passed"],
        "redaction_count": redaction_count,
    }


def transcript_receipt(steps: list[dict]) -> str:
    canonical = json.dumps(steps, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return _sha256_text(canonical)


def build_transcript(name: str, steps: list[dict], *, dry_run: bool) -> dict:
    return {
        "schema": TRANSCRIPT_SCHEMA,
        "name": name,
        "timestamp_utc": _now_utc(),
        "dry_run": dry_run,
        "step_count": len(steps),
        "total_duration_ms": sum(int(step.get("duration_ms", 0)) for step in steps),
        "steps": steps,
        "receipt_sha256": transcript_receipt(steps),
        "cleanup_ok": True,
        "publishable": all(step.get("assertions_passed", True) for step in steps),
    }


def record_demo(
    script_path: Path,
    name: str,
    *,
    out_root: Path | None = None,
    dry_run: bool = False,
    timeout_seconds: float = DEFAULT_STEP_TIMEOUT_SECONDS,
    cwd: Path | None = None,
) -> dict:
    script_steps = load_demo_script(script_path)
    results: list[dict] = []
    execution_finished = False
    cleanup_ok = False
    try:
        with tempfile.TemporaryDirectory(prefix=f"demo-{name}-") as temporary_path:
            run_temp = Path(temporary_path)
            results = [
                execute_step(
                    step,
                    index=index,
                    dry_run=dry_run,
                    timeout_seconds=timeout_seconds,
                    cwd=cwd,
                    run_temp=run_temp,
                )
                for index, step in enumerate(script_steps)
            ]
            execution_finished = True
    except Exception:
        if not execution_finished:
            raise
    else:
        cleanup_ok = True

    explicit_patterns = [
        pattern for step in script_steps for pattern in step.get("redact_patterns", [])
    ]
    safe_name, _name_redaction_count = scrub_text(name, explicit_patterns)
    transcript = build_transcript(safe_name, results, dry_run=dry_run)
    forbidden_value_survived = _has_forbidden_value(
        safe_name, explicit_patterns
    ) or any(
        _has_forbidden_value(
            json.dumps(result, ensure_ascii=False), step.get("redact_patterns", [])
        )
        for step, result in zip(script_steps, results, strict=True)
    )
    transcript["cleanup_ok"] = cleanup_ok
    transcript["publishable"] = (
        transcript["publishable"] and cleanup_ok and not forbidden_value_survived
    )

    demo_dir = Path(out_root or (_repo_root() / "demos")) / name
    demo_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = demo_dir / "transcript.json"
    player_path = demo_dir / "player.html"
    transcript_path.write_text(
        json.dumps(transcript, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    player_path.write_text(render_player_html(transcript), encoding="utf-8")
    return {
        "transcript": transcript,
        "transcript_path": str(transcript_path),
        "player_path": str(player_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script", required=True, help="path to the demo script JSON")
    parser.add_argument("--name", required=True, help="demo name; outputs land in demos/<name>/")
    parser.add_argument("--dry-run", action="store_true", help="execute nothing; emit placeholder output")
    parser.add_argument("--out-root", default="", help="override the demos/ output root")
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_STEP_TIMEOUT_SECONDS)
    args = parser.parse_args(argv)

    script_path = Path(args.script)
    console_redact_patterns = [
        pattern
        for step in load_demo_script(script_path)
        for pattern in step.get("redact_patterns", [])
    ]
    result = record_demo(
        script_path,
        args.name,
        out_root=Path(args.out_root) if args.out_root else None,
        dry_run=args.dry_run,
        timeout_seconds=args.timeout_seconds,
    )
    transcript = result["transcript"]
    print(f"recorded {transcript['step_count']} steps in {transcript['total_duration_ms']} ms")
    for step in transcript["steps"]:
        print(
            f"  [{step['index']}] {step['title']}: exit={step['exit_code']} "
            f"{step['duration_ms']} ms ({step['mode']})"
        )
    safe_transcript_path, _count = scrub_text(result["transcript_path"], console_redact_patterns)
    safe_player_path, _count = scrub_text(result["player_path"], console_redact_patterns)
    print(f"transcript: {safe_transcript_path}")
    print(f"player:     {safe_player_path}")
    print(f"receipt:    sha256:{transcript['receipt_sha256']}")
    print(f"publishable: {'yes' if transcript['publishable'] else 'no'}")
    return 0 if transcript["publishable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
