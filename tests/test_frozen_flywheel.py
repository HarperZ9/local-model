"""Standalone Desktop-engine falsifiers.

The frozen bundle is a deliberately tiny adjacent layout: one executable,
six site files, thirty physical parity witnesses, and one generated manifest.
Nothing else from the checkout is runtime data.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path, PurePosixPath

import pytest

import harness
from harness import gateway, lanes, parity
from scripts import build_local_harness_exes as builder


ROOT = Path(__file__).resolve().parents[1]

# The Desktop manifest lives in a separate repository whose checkout location is
# a property of the machine. Resolve it, and skip rather than fail when only the
# engine is checked out: a missing sibling repository is not an engine defect.
DESKTOP_MANIFEST = builder.find_desktop_release_manifest()
requires_desktop_checkout = pytest.mark.skipif(
    DESKTOP_MANIFEST is None,
    reason=(
        "Desktop checkout not found. Set $FLYWHEEL_DESKTOP_MANIFEST or "
        "$FLYWHEEL_DESKTOP_ROOT to run the cross-repository contract tests."
    ),
)

EXPECTED_RUNTIME_SOURCES = (
    "harness/compaction.py",
    "harness/consensus.py",
    "harness/endpoint_registry.py",
    "harness/envelope.py",
    "harness/fold_index.py",
    "harness/gateway.py",
    "harness/integrity.py",
    "harness/keychain.py",
    "harness/linter.py",
    "harness/local_mcp.py",
    "harness/loop_closure.py",
    "harness/lsp_bridge.py",
    "harness/lsp_diagnostics.py",
    "harness/marketplace.py",
    "harness/mcp_client.py",
    "harness/memory_api.py",
    "harness/plugins.py",
    "harness/profiles.py",
    "harness/router_stats.py",
    "harness/workflows.py",
    "harness/world.py",
    "site/assets/fonts/conso-medium.woff2",
    "site/assets/fonts/conso-regular.woff2",
    "site/assets/fonts/conso-semibold.woff2",
    "site/assets/fonts/hanken-grotesk-italic.woff2",
    "site/assets/fonts/hanken-grotesk.woff2",
    "site/index.html",
    "tests/test_keychain.py",
    "tests/test_linter.py",
    "tests/test_lsp_bridge.py",
    "tests/test_lsp_diagnostics.py",
    "tests/test_marketplace.py",
    "tests/test_memory_api.py",
    "tests/test_plugins.py",
    "tests/test_profiles_workflows.py",
    "tests/test_workspace_root.py",
)


def _expected_runtime_sources() -> tuple[str, ...]:
    return EXPECTED_RUNTIME_SOURCES


def _desktop_path_allowed(path: str, policy: dict) -> bool:
    pure = PurePosixPath(path)
    if not path or "\\" in path or pure.is_absolute() or ".." in pure.parts:
        return False
    lowered_parts = tuple(part.lower() for part in pure.parts)
    lowered_name = lowered_parts[-1]
    permitted = False
    for root in policy["permitted_roots"]:
        root_path = PurePosixPath(root["path"])
        if root["kind"] == "file" and pure == root_path:
            permitted = True
        if root["kind"] == "tree" and pure.parts[:len(root_path.parts)] == root_path.parts:
            permitted = True
    if not permitted:
        return False
    forbidden = policy["forbidden"]
    if any(lowered_name.startswith(value.lower()) for value in forbidden["name_prefixes"]):
        return False
    if any(value.lower() in lowered_name for value in forbidden["name_fragments"]):
        return False
    if lowered_name in {value.lower() for value in forbidden["names"]}:
        return False
    if any(part in {value.lower() for value in forbidden["segments"]} for part in lowered_parts):
        return False
    if any(lowered_name.endswith(value.lower()) for value in forbidden["suffixes"]):
        return False
    lowered_path = "/".join(lowered_parts)
    return not any(
        lowered_path == value.lower() or lowered_path.endswith("/" + value.lower())
        for value in forbidden["subpaths"]
    )


def test_frozen_runtime_root_is_adjacent_to_engine_bin(monkeypatch, tmp_path):
    executable = tmp_path / "candidate" / "engine" / "bin" / "flywheel.exe"
    monkeypatch.setattr(harness.sys, "frozen", True, raising=False)
    monkeypatch.setattr(harness.sys, "executable", str(executable))

    assert harness.runtime_root() == executable.parent.parent / "runtime"


def test_source_and_frozen_readiness_share_protocol_and_version(monkeypatch):
    source = gateway.runtime_record(10001, "source-token")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    frozen = gateway.runtime_record(10002, "frozen-token")

    assert set(source) == set(frozen) == {
        "protocol", "version", "pid", "port", "instance_token"
    }
    assert (source["protocol"], source["version"]) == (
        frozen["protocol"], frozen["version"]
    ) == ("flywheel.gateway/1", harness.__version__)


def test_absent_parity_gateway_witness_is_bounded_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(parity, "REPO", tmp_path)

    result = parity.parity_matrix()

    assert result == {
        "schema": "flywheel.parity/v1",
        "declared_on": parity.DECLARED_ON,
        "availability": "unavailable",
        "reason": "gateway-witness-absent",
        "rows": [],
        "summary": {
            "witnessed": 0,
            "absent": len(parity.ROWS),
            "uniquely_witnessed": [],
            "gaps": [],
        },
    }
    assert str(tmp_path) not in json.dumps(result)


def test_frozen_training_status_is_bounded_without_wsl(monkeypatch):
    def forbidden(_run_root):
        raise AssertionError("frozen Desktop runtime must not probe WSL training state")

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setitem(
        sys.modules,
        "harness.training_lane",
        type("TrainingLane", (), {"training_status": staticmethod(forbidden)}),
    )

    assert gateway._training_status("C:/unused") == {
        "schema": "flywheel.runtime-availability/1",
        "feature": "training-status",
        "availability": "unavailable",
        "reason": "not-in-desktop-engine",
    }


def test_frozen_lane_roster_never_runs_package_or_mcp_probes(monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(
        lanes.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("frozen roster must not spawn pip, npm, or MCP"),
    )

    roster = lanes.lane_roster(probe=True)

    local = next(row for row in roster["lanes"] if row["name"] == "local-model")
    external = [row for row in roster["lanes"] if row["name"] != "local-model"]
    assert local["status"] == lanes.LIVE
    assert all(row["availability"] == "unavailable" for row in external)


def test_desktop_runtime_source_allowlist_is_exact():
    source_fn = getattr(builder, "desktop_engine_runtime_sources", None)
    assert callable(source_fn), "Desktop engine runtime allowlist is not implemented"

    sources = source_fn(ROOT)

    assert sources == _expected_runtime_sources()
    assert len(sources) == 36
    assert tuple(sorted(set(sources))) == sources
    assert all((ROOT / PurePosixPath(path)).is_file() for path in sources)


def test_desktop_runtime_allowlist_does_not_expand_for_stray_site_file(tmp_path):
    for relative in EXPECTED_RUNTIME_SOURCES:
        path = tmp_path / PurePosixPath(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
    stray = tmp_path / "site" / "assets" / "cache-secret.log"
    stray.write_bytes(b"must never be packaged")

    assert builder.desktop_engine_runtime_sources(tmp_path) == EXPECTED_RUNTIME_SOURCES


def test_desktop_runtime_sources_must_all_be_tracked():
    validate = getattr(builder, "validate_desktop_runtime_source_tracking", None)
    assert callable(validate), "Desktop runtime tracking validation is not implemented"

    validate(EXPECTED_RUNTIME_SOURCES, EXPECTED_RUNTIME_SOURCES)
    with pytest.raises(RuntimeError, match="not tracked"):
        validate(EXPECTED_RUNTIME_SOURCES, EXPECTED_RUNTIME_SOURCES[:-1])


def test_runtime_source_provenance_uses_filter_aware_git_identity():
    provenance = getattr(builder, "_runtime_source_provenance", None)
    assert callable(provenance), "Runtime source provenance is not implemented"

    row = provenance(("site/index.html",))[0]

    assert row["path"] == "site/index.html"
    assert row["matches_head"] is True
    assert len(row["working_file_sha256"]) == 64
    assert len(row["head_blob_sha256"]) == 64


def test_runtime_manifest_hashes_only_the_exact_allowlist(tmp_path):
    stage_fn = getattr(builder, "stage_desktop_engine_runtime", None)
    assert callable(stage_fn), "Desktop engine runtime staging is not implemented"
    runtime_root = tmp_path / "engine" / "runtime"

    manifest_path = stage_fn(
        runtime_root,
        source_root=ROOT,
        source_commit="a" * 40,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["schema"] == "flywheel.engine-runtime-manifest/1"
    assert manifest["version"] == "1.0.0"
    assert manifest["protocol"] == "flywheel.gateway/1"
    assert manifest["source_commit"] == "a" * 40
    assert [row["path"] for row in manifest["files"]] == list(_expected_runtime_sources())
    for row in manifest["files"]:
        path = runtime_root / PurePosixPath(row["path"])
        assert row["bytes"] == path.stat().st_size
        assert row["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert row["role"] in {"site", "parity-witness"}
    actual = {
        path.relative_to(runtime_root).as_posix()
        for path in runtime_root.rglob("*")
        if path.is_file() and path != manifest_path
    }
    assert actual == set(_expected_runtime_sources())


@requires_desktop_checkout
def test_canonical_candidate_matches_desktop_payload_policy(tmp_path):
    stage_fn = getattr(builder, "stage_desktop_engine_runtime", None)
    assert callable(stage_fn), "Desktop engine runtime staging is not implemented"
    candidate = tmp_path / "candidate"
    executable = candidate / "engine" / "bin" / "flywheel.exe"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"frozen-executable")
    stage_fn(candidate / "engine" / "runtime", source_root=ROOT, source_commit="b" * 40)

    paths = tuple(sorted(
        path.relative_to(candidate).as_posix()
        for path in candidate.rglob("*")
        if path.is_file()
    ))
    policy = json.loads(DESKTOP_MANIFEST.read_text(encoding="utf-8"))["payload_policy"]

    assert len(paths) == 38
    assert paths[0] == "engine/bin/flywheel.exe"
    assert "engine/runtime/runtime-manifest.json" in paths
    assert all(_desktop_path_allowed(path, policy) for path in paths)


@requires_desktop_checkout
def test_builder_enforces_desktop_manifest_default_deny():
    validate = getattr(builder, "validate_desktop_candidate_paths", None)
    assert callable(validate), "Desktop payload-policy validation is not implemented"

    validate(
        (
            "engine/bin/flywheel.exe",
            "engine/runtime/runtime-manifest.json",
            "engine/runtime/site/index.html",
        ),
        manifest_path=DESKTOP_MANIFEST,
    )
    with pytest.raises(RuntimeError, match="payload policy"):
        validate(
            (
                "engine/bin/flywheel.exe",
                "engine/runtime/runtime-manifest.json",
                "engine/runtime/site/secret-model.gguf",
            ),
            manifest_path=DESKTOP_MANIFEST,
        )


def test_build_receipt_is_canonical_and_outside_candidate(tmp_path):
    resolve = getattr(builder, "desktop_engine_receipt_path", None)
    assert callable(resolve), "Desktop receipt-path validation is not implemented"
    output = tmp_path / "desktop-engine-candidate"
    canonical = tmp_path / "desktop-engine-candidate-build-receipt.json"

    assert resolve(output, None) == canonical.resolve()
    assert resolve(output, canonical) == canonical.resolve()
    with pytest.raises(ValueError, match="canonical adjacent path"):
        resolve(output, output / "receipt.json")
    with pytest.raises(ValueError, match="canonical adjacent path"):
        resolve(output, tmp_path / "renamed-receipt.json")


def test_desktop_builder_contract_is_pinned_and_spec_free(tmp_path):
    assert getattr(builder, "DESKTOP_ENGINE_PYTHON_VERSION", None) == "3.12.10"
    assert getattr(builder, "DESKTOP_ENGINE_PYINSTALLER_VERSION", None) == "6.21.0"
    command_fn = getattr(builder, "desktop_engine_pyinstaller_command", None)
    assert callable(command_fn), "Desktop PyInstaller command is not implemented"

    command = command_fn(
        "python.exe",
        dist_path=tmp_path / "dist",
        work_path=tmp_path / "work",
        spec_path=tmp_path / "generated-spec",
    )

    assert command[:3] == ["python.exe", "-m", "PyInstaller"]
    assert command[command.index("--name") + 1] == "flywheel"
    assert command[command.index("--specpath") + 1] == str(tmp_path / "generated-spec")
    assert command[-1] == str(ROOT / "scripts" / "flywheel_entry.py")
    assert not any(arg.lower().endswith(".spec") for arg in command)


def test_builder_script_ignores_ambient_checkout_on_import(tmp_path):
    fake_package = tmp_path / "harness"
    fake_package.mkdir()
    (fake_package / "__init__.py").write_text("# incompatible ambient harness\n", encoding="utf-8")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(tmp_path)

    proc = subprocess.run(
        [sys.executable, "scripts/build_local_harness_exes.py", "--help"],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert proc.returncode == 0, proc.stderr
    assert "--desktop-engine" in proc.stdout
