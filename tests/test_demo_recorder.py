"""Fast falsifiers for the zero-dependency demo recorder. No network."""

import copy
import hashlib
import json
import os
import re
from pathlib import Path

import pytest

import scripts.demo_recorder as demo_recorder
from scripts.demo_player_html import render_player_html
from scripts.demo_recorder import (
    TRANSCRIPT_SCHEMA,
    build_transcript,
    execute_step,
    load_demo_script,
    main,
    record_demo,
    transcript_receipt,
)

SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")
EXTERNAL_URL_ATTR = re.compile(r"""(?:src|href)\s*=\s*["']\s*(?:https?:)?//""", re.IGNORECASE)


def write_script(tmp_path: Path, steps: list[dict]) -> Path:
    path = tmp_path / "demo-script.json"
    path.write_text(json.dumps({"steps": steps}), encoding="utf-8")
    return path


def sample_steps() -> list[dict]:
    return [
        {
            "title": "Say hello",
            "command": 'python -c "print(\'hello from the demo\')"',
            "narration": "A tiny local machine says hello.",
        },
        {
            "title": "Count to three",
            "command": 'python -c "print(1); print(2); print(3)"',
            "narration": "It can count too.",
        },
    ]


def test_dry_run_produces_valid_transcript_and_player(tmp_path):
    script = write_script(tmp_path, sample_steps())
    result = record_demo(script, "dry-demo", out_root=tmp_path / "demos", dry_run=True)

    transcript_path = Path(result["transcript_path"])
    player_path = Path(result["player_path"])
    assert transcript_path.exists()
    assert player_path.exists()

    transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
    assert transcript["schema"] == TRANSCRIPT_SCHEMA
    assert transcript["dry_run"] is True
    assert transcript["step_count"] == 2
    assert SHA256_HEX.match(transcript["receipt_sha256"])
    for step in transcript["steps"]:
        assert step["mode"] == "dry-run"
        assert step["exit_code"] == 0
        assert "[dry-run] command not executed" in step["output"]
        assert SHA256_HEX.match(step["output_sha256"])
        assert step["output_sha256"] != transcript["receipt_sha256"]

    html_text = player_path.read_text(encoding="utf-8")
    assert "dry-demo" in html_text
    assert 'id="transcript-data"' in html_text
    assert "Play" in html_text and "Restart" in html_text


def test_transcript_receipt_changes_when_output_changes(tmp_path):
    script = write_script(tmp_path, sample_steps())
    baseline = record_demo(script, "hash-demo", out_root=tmp_path / "demos", dry_run=True)
    steps = baseline["transcript"]["steps"]

    mutated = copy.deepcopy(steps)
    mutated[0]["output"] = mutated[0]["output"] + " tampered"

    assert transcript_receipt(steps) == baseline["transcript"]["receipt_sha256"]
    assert transcript_receipt(mutated) != transcript_receipt(steps)
    assert (
        build_transcript("hash-demo", mutated, dry_run=True)["receipt_sha256"]
        != baseline["transcript"]["receipt_sha256"]
    )


def test_failing_command_records_exit_code_honestly():
    step = {
        "title": "Deliberate failure",
        "command": 'python -c "import sys; print(\'boom\'); sys.exit(3)"',
        "narration": "Failures get recorded, not hidden.",
    }
    result = execute_step(step, index=0, dry_run=False, timeout_seconds=30.0)

    assert result["exit_code"] == 3
    assert "boom" in result["output"]
    assert result["mode"] == "live"
    assert SHA256_HEX.match(result["output_sha256"])


def test_player_html_has_no_external_urls(tmp_path):
    script = write_script(tmp_path, sample_steps())
    result = record_demo(script, "offline-demo", out_root=tmp_path / "demos", dry_run=True)
    html_text = Path(result["player_path"]).read_text(encoding="utf-8")

    assert EXTERNAL_URL_ATTR.search(html_text) is None
    for attr_match in re.finditer(r"""(?:src|href)\s*=\s*["']([^"']*)["']""", html_text):
        assert not attr_match.group(1).lower().startswith(("http://", "https://", "//"))


def test_player_html_matches_telos_v2_accessibility_contract(tmp_path):
    script = write_script(tmp_path, sample_steps())
    result = record_demo(script, "telos-demo", out_root=tmp_path / "demos", dry_run=True)
    html_text = Path(result["player_path"]).read_text(encoding="utf-8")

    assert 'class="skip-link" href="#terminal"' in html_text
    assert 'aria-live="polite"' in html_text
    assert ":focus-visible" in html_text
    assert "outline:" in html_text
    assert "@media (prefers-reduced-motion: reduce)" in html_text
    assert 'font-family: "Hanken Grotesk", "Segoe UI", sans-serif' in html_text
    assert 'font-family: "Conso", "Cascadia Mono", Consolas, monospace' in html_text
    assert "border-left: 3px" not in html_text
    assert "border: 1px solid var(--edge); border-radius: 8px" in html_text
    assert ".cursor { animation: none; }" in html_text
    assert "if (steps.length) { paint(false); }" in html_text
    assert EXTERNAL_URL_ATTR.search(html_text) is None


def test_load_demo_script_rejects_incomplete_steps(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps([{"title": "no command", "narration": "x"}]), encoding="utf-8")
    try:
        load_demo_script(path)
    except ValueError as exc:
        assert "command" in str(exc)
    else:
        raise AssertionError("expected ValueError for a step without a command")


def test_undeclared_successful_step_uses_safe_publishable_defaults(tmp_path):
    script = write_script(tmp_path, [{
        "title": "Default behavior",
        "command": 'python -c "print(\'ready\')"',
        "narration": "Legacy scripts remain executable with safe defaults.",
    }])

    result = record_demo(script, "default-demo", out_root=tmp_path / "demos")
    step = result["transcript"]["steps"][0]

    assert step["assertions"] == {
        "expected_exit_codes": [0],
        "exit_code_ok": True,
        "required": [],
        "missing": [],
        "passed": True,
    }
    assert step["assertions_passed"] is True
    assert step["capture"] == "terminal-replay"
    assert result["transcript"]["cleanup_ok"] is True
    assert result["transcript"]["publishable"] is True


def test_expected_exit_and_output_assertions_control_publishability(tmp_path):
    script = write_script(tmp_path, [{
        "title": "Assert output",
        "command": 'python -c "print(\'actual\')"',
        "display_command": "tool inspect --fixture public.json",
        "narration": "The public fixture produces a checkable value.",
        "expected_exit_codes": [0],
        "expect": ["required"],
    }])
    result = record_demo(script, "assert-demo", out_root=tmp_path / "demos")
    step = result["transcript"]["steps"][0]
    assert step["command"] == "tool inspect --fixture public.json"
    assert step["display_command"] == "tool inspect --fixture public.json"
    assert step["assertions_passed"] is False
    assert step["assertions"]["missing"] == ["required"]
    assert result["transcript"]["publishable"] is False


def test_unexpected_exit_code_fails_assertions_and_publishability(tmp_path):
    script = write_script(tmp_path, [{
        "title": "Unexpected refusal",
        "command": 'python -c "import sys; sys.exit(4)"',
        "narration": "Only the declared exit codes are accepted.",
        "expected_exit_codes": [0, 2],
    }])

    result = record_demo(script, "exit-demo", out_root=tmp_path / "demos")
    step = result["transcript"]["steps"][0]

    assert step["exit_code"] == 4
    assert step["assertions"]["exit_code_ok"] is False
    assert step["assertions_passed"] is False
    assert result["transcript"]["publishable"] is False


def test_main_returns_one_when_publish_gate_fails(tmp_path):
    script = write_script(tmp_path, [{
        "title": "Expected refusal",
        "command": 'python -c "import sys; sys.exit(4)"',
        "narration": "Unexpected failure blocks publication.",
    }])
    assert main([
        "--script", str(script), "--name", "blocked",
        "--out-root", str(tmp_path / "demos"),
    ]) == 1


def test_main_redacts_builtin_and_explicit_values_before_printing(tmp_path, capsys):
    script = write_script(tmp_path, [{
        "title": "Safe console",
        "command": 'python -c "print(\'ready\')"',
        "narration": "Console paths are sanitized before display.",
        "redact_patterns": ["PRIVATE-PRINT"],
    }])

    assert main([
        "--script", str(script), "--name", "console-demo",
        "--out-root", str(tmp_path / "PRIVATE-PRINT"),
    ]) == 0
    printed = capsys.readouterr().out

    assert str(Path.home()) not in printed
    assert "PRIVATE-PRINT" not in printed
    assert "[redacted-home]" in printed


def test_step_cwd_and_environment_are_explicitly_scoped(tmp_path):
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    inherited_env = dict(os.environ)
    inherited_env.update({"DEMO_ALLOWED": "visible", "DEMO_BLOCKED": "hidden"})
    step = {
        "title": "Scoped execution",
        "command": (
            'python -c "import os; from pathlib import Path; print(Path.cwd().name); '
            "print(os.environ.get('DEMO_ALLOWED')); "
            "print(os.environ.get('DEMO_BLOCKED', 'missing'))\""
        ),
        "narration": "Only declared process context reaches the command.",
        "cwd": "work",
        "env_allowlist": ["DEMO_ALLOWED"],
    }

    result = execute_step(
        step,
        index=0,
        dry_run=False,
        cwd=tmp_path,
        inherited_env=inherited_env,
    )

    output_lines = result["stdout"].splitlines()
    assert output_lines[0] == work_dir.name
    assert output_lines[1:] == ["visible", "missing"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("expected_exit_codes", []),
        ("expected_exit_codes", [0, "1"]),
        ("expect", "ready"),
        ("expect", [""]),
        ("env_allowlist", "PATH"),
        ("env_allowlist", ["  "]),
        ("redact_patterns", "secret"),
        ("redact_patterns", [""]),
        ("capture", "screen-grab"),
    ],
)
def test_load_demo_script_rejects_invalid_execution_metadata(tmp_path, field, value):
    step = sample_steps()[0]
    step[field] = value
    script = write_script(tmp_path, [step])

    with pytest.raises(ValueError, match=field):
        load_demo_script(script)


def test_sensitive_values_are_redacted_before_hash_and_render(tmp_path, monkeypatch):
    monkeypatch.setenv("DEMO_SECRET", "ghp_abcdefghijklmnopqrstuvwxyz1234567890")
    script = write_script(tmp_path, [{
        "title": "Scrub output",
        "command": (
            'python -c "import os; print(\'person@example.com\'); '
            "print(os.environ['DEMO_SECRET'])\""
        ),
        "display_command": "tool inspect C:\\Users\\PrivateName\\fixture.json",
        "narration": "The private marker PRIVATE-CUSTOM never leaves the recorder.",
        "env_allowlist": ["DEMO_SECRET"],
        "redact_patterns": ["PRIVATE-CUSTOM"],
    }])
    result = record_demo(script, "scrub-demo", out_root=tmp_path / "demos")
    serialized = json.dumps(result["transcript"])
    html_text = Path(result["player_path"]).read_text(encoding="utf-8")
    for forbidden in ("person@example.com", "ghp_", "PrivateName", "PRIVATE-CUSTOM"):
        assert forbidden not in serialized
        assert forbidden not in html_text
    step = result["transcript"]["steps"][0]
    assert step["redaction_count"] >= 4
    assert step["output_sha256"] == hashlib.sha256(step["output"].encode()).hexdigest()


def test_scrub_text_covers_builtin_secret_shapes(monkeypatch):
    username = os.environ.get("USERNAME") or os.environ.get("USER")
    sensitive = "\n".join([
        str(Path.home() / "private" / "fixture.json"),
        username or "",
        "person@example.com",
        r"C:\\private\\fixture.json",
        "Bearer bearer-token-value",
        "sk-proj-abcdefghijklmnopqrstuvwxyz123456",
        "ghp_abcdefghijklmnopqrstuvwxyz1234567890",
        "api_key=credential-value",
        "CUSTOM-MARKER",
    ])

    scrubbed, count = demo_recorder.scrub_text(sensitive, ["CUSTOM-MARKER"])

    for forbidden in (
        str(Path.home()),
        username or "unused-username",
        "person@example.com",
        r"C:\\private",
        "bearer-token-value",
        "sk-proj-",
        "ghp_",
        "credential-value",
        "CUSTOM-MARKER",
    ):
        assert forbidden not in scrubbed
    assert {
        "[redacted-home]",
        "[redacted-email]",
        "[redacted-path]",
        "[redacted-token]",
        "[redacted-credential]",
        "[redacted-explicit]",
    } <= set(scrubbed.splitlines())
    assert count >= 8


def test_demo_temp_is_unique_execution_only_state_and_is_cleaned(tmp_path, monkeypatch):
    real_temporary_directory = demo_recorder.tempfile.TemporaryDirectory
    created_paths = []

    def tracking_temporary_directory(*args, **kwargs):
        temporary = real_temporary_directory(*args, **kwargs)
        created_paths.append(Path(temporary.name))
        return temporary

    monkeypatch.setattr(
        demo_recorder.tempfile,
        "TemporaryDirectory",
        tracking_temporary_directory,
    )
    script = write_script(tmp_path, [{
        "title": "Use isolated state",
        "command": 'python -c "from pathlib import Path; print(Path.cwd())"',
        "display_command": "tool inspect {demo_temp}/public.json",
        "narration": "The execution path stays private.",
        "cwd": "{demo_temp}",
    }])

    first = record_demo(script, "temp-demo", out_root=tmp_path / "first")
    second = record_demo(script, "temp-demo", out_root=tmp_path / "second")

    assert len(created_paths) == 2
    assert created_paths[0] != created_paths[1]
    assert all(not path.exists() for path in created_paths)
    for result, temp_path in zip((first, second), created_paths, strict=True):
        serialized = json.dumps(result)
        step = result["transcript"]["steps"][0]
        assert step["command"] == "tool inspect {demo_temp}/public.json"
        assert str(temp_path) not in serialized
        assert result["transcript"]["cleanup_ok"] is True


def test_cleanup_failure_is_recorded_without_losing_transcript(tmp_path, monkeypatch):
    real_temporary_directory = demo_recorder.tempfile.TemporaryDirectory
    created_paths = []

    class CleanupFailure:
        def __init__(self, *args, **kwargs):
            self._temporary = real_temporary_directory(*args, **kwargs)
            created_paths.append(Path(self._temporary.name))

        def __enter__(self):
            return self._temporary.__enter__()

        def __exit__(self, exc_type, exc_value, traceback):
            self._temporary.__exit__(exc_type, exc_value, traceback)
            raise OSError("simulated cleanup evidence failure")

    monkeypatch.setattr(demo_recorder.tempfile, "TemporaryDirectory", CleanupFailure)
    script = write_script(tmp_path, [sample_steps()[0]])

    result = record_demo(script, "cleanup-demo", out_root=tmp_path / "demos")

    assert result["transcript"]["cleanup_ok"] is False
    assert result["transcript"]["publishable"] is False
    assert Path(result["transcript_path"]).exists()
    assert Path(result["player_path"]).exists()
    assert str(created_paths[0]) not in json.dumps(result)


def test_surviving_forbidden_value_blocks_publication(tmp_path, monkeypatch):
    monkeypatch.setattr(demo_recorder, "scrub_text", lambda text, explicit_patterns=(): (text, 0))
    script = write_script(tmp_path, [{
        "title": "Post-scrub scan",
        "command": 'python -c "print(\'ready\')"',
        "narration": "The post-scrub scan covers transcript metadata.",
        "redact_patterns": ["SURVIVING-FORBIDDEN"],
    }])

    result = record_demo(script, "SURVIVING-FORBIDDEN", out_root=tmp_path / "demos")

    assert result["transcript"]["publishable"] is False


def test_redact_patterns_scrub_every_step_before_hash_and_render(tmp_path):
    secret = "CROSS-STEP-SECRET"
    script = write_script(tmp_path, [
        {
            "title": "Declare shared redaction",
            "command": 'python -c "print(\'safe\')"',
            "narration": "One declaration protects the complete transcript.",
            "redact_patterns": [secret],
        },
        {
            "title": "Use shared redaction",
            "command": f'python -c "print(\'{secret}\')"',
            "display_command": f"tool inspect {secret}.json",
            "narration": f"The shared marker {secret} never leaves this step.",
        },
    ])

    result = record_demo(script, "cross-step-demo", out_root=tmp_path / "demos")
    transcript = result["transcript"]
    serialized = json.dumps(transcript)
    html_text = Path(result["player_path"]).read_text(encoding="utf-8")
    protected_step = transcript["steps"][1]

    assert secret not in serialized
    assert secret not in html_text
    assert protected_step["redaction_count"] >= 3
    assert protected_step["output_sha256"] == hashlib.sha256(
        protected_step["output"].encode()
    ).hexdigest()
    assert transcript["publishable"] is True


def test_transcript_wide_patterns_govern_final_publishability_scan(tmp_path, monkeypatch):
    secret = "CROSS-STEP-SECRET"
    monkeypatch.setattr(demo_recorder, "scrub_text", lambda text, explicit_patterns=(): (text, 0))
    script = write_script(tmp_path, [
        {
            "title": "Declare shared redaction",
            "command": 'python -c "print(\'safe\')"',
            "narration": "The first step declares the transcript policy.",
            "redact_patterns": [secret],
        },
        {
            "title": "Surviving value",
            "command": 'python -c "print(\'ready\')"',
            "narration": f"A surviving {secret} blocks publication.",
        },
    ])

    result = record_demo(script, "cross-step-scan", out_root=tmp_path / "demos")

    assert result["transcript"]["publishable"] is False


def test_record_demo_returns_deduplicated_console_patterns_outside_transcript(tmp_path):
    script = write_script(tmp_path, [
        {
            **sample_steps()[0],
            "redact_patterns": ["FIRST-PATTERN", "SHARED-PATTERN"],
        },
        {
            **sample_steps()[1],
            "redact_patterns": ["SHARED-PATTERN", "SECOND-PATTERN"],
        },
    ])

    result = record_demo(script, "pattern-demo", out_root=tmp_path / "demos")

    assert result["_console_redact_patterns"] == (
        "FIRST-PATTERN",
        "SHARED-PATTERN",
        "SECOND-PATTERN",
    )
    assert "_console_redact_patterns" not in result["transcript"]


def test_main_loads_demo_script_once(tmp_path, monkeypatch):
    script = write_script(tmp_path, [sample_steps()[0]])
    real_load_demo_script = demo_recorder.load_demo_script
    loaded_paths = []

    def tracking_load_demo_script(path):
        loaded_paths.append(Path(path))
        return real_load_demo_script(path)

    monkeypatch.setattr(demo_recorder, "load_demo_script", tracking_load_demo_script)

    assert main([
        "--script", str(script), "--name", "single-load",
        "--out-root", str(tmp_path / "demos"),
    ]) == 0
    assert loaded_paths == [script]
