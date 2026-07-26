#!/usr/bin/env python3
"""Build one-file executables for local harness entrypoints."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent.parent
if not sys.path or sys.path[0] != str(ROOT):
    sys.path.insert(0, str(ROOT))

from harness import __version__  # noqa: E402
from harness.gateway import GATEWAY_PROTOCOL  # noqa: E402
from harness.parity import runtime_witness_paths  # noqa: E402

DIST = ROOT / "artifacts" / "exe"
WORK = ROOT / "artifacts" / ".pyinstaller"
DEFAULT_SERVE_PYTHON = "E:/local-model-run/venv/Scripts/python.exe"
DEFAULT_TOOLS = "index,forum,gather,crucible,telos,aleph,mneme,relay,plexus,pubscan,local-model"
DESKTOP_ENGINE_PYTHON_VERSION = "3.12.10"
DESKTOP_ENGINE_PYINSTALLER_VERSION = "6.21.0"
DESKTOP_ENGINE_CANDIDATE = ROOT / "artifacts" / "desktop-engine-candidate"

# The Desktop checkout is a separate repository, so its location is a property
# of the machine, not of this source tree. Pinning one worktree name made the
# engine suite fail for anyone not standing in that exact pair of directories.
DESKTOP_MANIFEST_ENV = "FLYWHEEL_DESKTOP_MANIFEST"
DESKTOP_ROOT_ENV = "FLYWHEEL_DESKTOP_ROOT"
DESKTOP_CHECKOUT_NAMES = (
    "flywheel-desktop-v1-rc",
    "flywheel-desktop",
)
_MANIFEST_RELPATH = Path("packaging") / "release-manifest.json"


def desktop_manifest_candidates() -> list[Path]:
    """Every path that could hold the Desktop release manifest, in order."""
    explicit = os.environ.get(DESKTOP_MANIFEST_ENV)
    if explicit:
        return [Path(explicit).expanduser()]
    roots: list[Path] = []
    configured = os.environ.get(DESKTOP_ROOT_ENV)
    if configured:
        roots.append(Path(configured).expanduser())
    roots.extend(ROOT.parent / name for name in DESKTOP_CHECKOUT_NAMES)
    roots.append(ROOT.parent.parent / "flywheel-desktop")
    return [root / _MANIFEST_RELPATH for root in roots]


def find_desktop_release_manifest() -> Path | None:
    """Resolve the Desktop manifest, or None when no checkout is present.

    Returning None lets callers skip rather than fail: an engine-only checkout
    is a legitimate state, and a missing sibling repository is not a defect in
    the engine.
    """
    for candidate in desktop_manifest_candidates():
        if candidate.is_file():
            return candidate.resolve()
    return None


def require_desktop_release_manifest() -> Path:
    """Resolve the Desktop manifest or fail with the paths that were tried."""
    found = find_desktop_release_manifest()
    if found is not None:
        return found
    tried = "\n  ".join(str(p) for p in desktop_manifest_candidates())
    raise FileNotFoundError(
        f"Desktop release manifest not found. Set ${DESKTOP_MANIFEST_ENV} or "
        f"${DESKTOP_ROOT_ENV}. Tried:\n  {tried}"
    )
DESKTOP_ENGINE_RUNTIME_SOURCES = (
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
DESKTOP_ENGINE_RUNTIME_SOURCE_COUNT = 36
DESKTOP_ENGINE_CANDIDATE_FILE_COUNT = 38


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def desktop_engine_runtime_sources(source_root: Path = ROOT) -> tuple[str, ...]:
    """Exact physical data allowlist for the frozen Desktop engine."""
    source_root = Path(source_root)
    sources = DESKTOP_ENGINE_RUNTIME_SOURCES
    if len(sources) != DESKTOP_ENGINE_RUNTIME_SOURCE_COUNT:
        raise RuntimeError("Desktop engine runtime source-count contract drifted")
    expected_witnesses = tuple(path for path in sources if not path.startswith("site/"))
    if runtime_witness_paths() != expected_witnesses:
        raise RuntimeError("Desktop engine parity-witness inventory drifted")
    missing = [path for path in sources if not (source_root / path).is_file()]
    if missing:
        raise FileNotFoundError(f"Desktop engine runtime source missing: {missing[0]}")
    return sources


def validate_desktop_runtime_source_tracking(
    source_paths: tuple[str, ...],
    tracked_paths: tuple[str, ...],
) -> None:
    """Fail closed if an allowlisted runtime source is not tracked by Git."""
    missing = sorted(set(source_paths) - set(tracked_paths))
    if missing:
        raise RuntimeError(f"Desktop engine runtime source is not tracked: {missing[0]}")


def _desktop_manifest(manifest_path: Path) -> dict:
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Desktop release manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1,
        "version": __version__,
        "protocol": GATEWAY_PROTOCOL,
        "python": DESKTOP_ENGINE_PYTHON_VERSION,
        "pyinstaller": DESKTOP_ENGINE_PYINSTALLER_VERSION,
        "executable": "engine/bin/flywheel.exe",
        "runtime_root": "engine/runtime",
    }
    observed = {
        "schema_version": manifest.get("schema_version"),
        "version": manifest.get("product", {}).get("version"),
        "protocol": manifest.get("gateway", {}).get("protocol"),
        "python": manifest.get("toolchain", {}).get("python"),
        "pyinstaller": manifest.get("toolchain", {}).get("pyinstaller"),
        "executable": manifest.get("engine_layout", {}).get("executable"),
        "runtime_root": manifest.get("engine_layout", {}).get("runtime_root"),
    }
    if observed != expected:
        raise RuntimeError("Desktop release manifest diverges from the engine contract")
    policy = manifest.get("payload_policy", {})
    if policy.get("default") != "reject" or policy.get("path_mode") != "literal-posix":
        raise RuntimeError("Desktop release manifest payload policy is not default-deny")
    return manifest


def _desktop_path_allowed(path: str, policy: dict) -> bool:
    if not path or "\\" in path:
        return False
    pure = PurePosixPath(path)
    if pure.is_absolute() or path != "/".join(pure.parts) or ".." in pure.parts:
        return False
    lowered_parts = tuple(part.lower() for part in pure.parts)
    lowered_name = lowered_parts[-1]
    permitted = False
    for root in policy.get("permitted_roots", ()):
        root_path = PurePosixPath(root.get("path", ""))
        if root.get("kind") == "file" and pure == root_path:
            permitted = True
        if (
            root.get("kind") == "tree"
            and pure.parts[:len(root_path.parts)] == root_path.parts
        ):
            permitted = True
    if not permitted:
        return False
    forbidden = policy.get("forbidden", {})
    if any(
        lowered_name.startswith(value.lower())
        for value in forbidden.get("name_prefixes", ())
    ):
        return False
    if any(
        value.lower() in lowered_name
        for value in forbidden.get("name_fragments", ())
    ):
        return False
    if lowered_name in {value.lower() for value in forbidden.get("names", ())}:
        return False
    forbidden_segments = {
        value.lower() for value in forbidden.get("segments", ())
    }
    if any(part in forbidden_segments for part in lowered_parts):
        return False
    if any(
        lowered_name.endswith(value.lower())
        for value in forbidden.get("suffixes", ())
    ):
        return False
    lowered_path = "/".join(lowered_parts)
    return not any(
        lowered_path == value.lower() or lowered_path.endswith("/" + value.lower())
        for value in forbidden.get("subpaths", ())
    )


def validate_desktop_candidate_paths(
    paths: tuple[str, ...] | set[str],
    *,
    manifest_path: Path | None = None,
) -> dict:
    """Apply the Desktop-owned default-deny payload policy to engine paths."""
    manifest = _desktop_manifest(
        manifest_path if manifest_path is not None
        else require_desktop_release_manifest()
    )
    policy = manifest["payload_policy"]
    rejected = sorted(path for path in paths if not _desktop_path_allowed(path, policy))
    if rejected:
        raise RuntimeError(f"Desktop payload policy rejected candidate path: {rejected[0]}")
    return manifest


def stage_desktop_engine_runtime(
    runtime_root: Path,
    *,
    source_root: Path = ROOT,
    source_commit: str,
) -> Path:
    """Copy only allowlisted runtime data and write its deterministic manifest."""
    runtime_root = Path(runtime_root)
    source_root = Path(source_root)
    runtime_root.mkdir(parents=True, exist_ok=True)
    if any(runtime_root.iterdir()):
        raise FileExistsError(f"Desktop engine runtime root is not empty: {runtime_root}")

    rows = []
    for rel in desktop_engine_runtime_sources(source_root):
        source = source_root / rel
        destination = runtime_root / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        rows.append({
            "path": rel,
            "bytes": destination.stat().st_size,
            "sha256": _sha256(destination),
            "role": "site" if rel.startswith("site/") else "parity-witness",
        })

    manifest = {
        "schema": "flywheel.engine-runtime-manifest/1",
        "version": __version__,
        "protocol": GATEWAY_PROTOCOL,
        "source_commit": source_commit,
        "files": rows,
    }
    manifest_path = runtime_root / "runtime-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest_path


def desktop_engine_pyinstaller_command(
    python: str,
    *,
    dist_path: Path,
    work_path: Path,
    spec_path: Path,
) -> list[str]:
    """Tracked, spec-free PyInstaller invocation for ``flywheel.exe``."""
    return [
        python,
        "-m",
        "PyInstaller",
        "--onefile",
        "--noconfirm",
        "--clean",
        "--distpath",
        str(dist_path),
        "--workpath",
        str(work_path),
        "--specpath",
        str(spec_path),
        "--name",
        "flywheel",
        "--hidden-import",
        "harness.gateway",
        "--hidden-import",
        "harness.lanes",
        "--hidden-import",
        "harness.parity",
        str(ROOT / "scripts" / "flywheel_entry.py"),
    ]


def _build(name: str, entry: str, *, python: str, hidden: list[str] | None = None) -> None:
    cmd = [
        python,
        "-m",
        "PyInstaller",
        "--onefile",
        "--noconfirm",
        "--clean",
        "--distpath",
        str(DIST),
        "--workpath",
        str(WORK / name),
        "--name",
        name,
        entry,
    ]
    for h in hidden or []:
        cmd.extend(["--hidden-import", h])
    print(f"[build] {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=ROOT)
    if proc.returncode != 0:
        raise RuntimeError(f"pyinstaller failed for {name} ({proc.returncode})")


def _py_path(raw: str | None) -> str:
    if not raw:
        return sys.executable
    explicit = str(Path(raw).expanduser().resolve())
    if not Path(explicit).exists():
        raise FileNotFoundError(f"python executable not found: {explicit}")
    return explicit


def _has_modules(python: str) -> bool:
    probe = (
        "import torch,transformers,peft,bitsandbytes; "
        "import importlib.util; "
        "print('ok')"
    )
    proc = subprocess.run([python, "-c", probe], capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"[warn] torch stack unavailable via {python}:")
        print(proc.stderr.strip())
        return False
    return True


def _has_pyinstaller(python: str) -> bool:
    probe = "import PyInstaller"
    proc = subprocess.run([python, "-c", probe], capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"[warn] PyInstaller unavailable via {python}:")
        print(proc.stderr.strip())
        return False
    return True


def _python_probe(python: str, code: str, label: str) -> str:
    proc = subprocess.run(
        [python, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"unable to identify {label}: {proc.stderr.strip()}")
    return proc.stdout.strip()


def _git_value(*args: str) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


def _git_file_at_head_sha256(path: str) -> str:
    proc = subprocess.run(
        ["git", "show", f"HEAD:{path}"],
        cwd=ROOT,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"git show HEAD:{path} failed: {proc.stderr.decode(errors='replace').strip()}"
        )
    return hashlib.sha256(proc.stdout).hexdigest()


def _git_path_matches_head(path: str) -> bool:
    """Use Git's clean/smudge filters when comparing a worktree file to HEAD."""
    proc = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", path],
        cwd=ROOT,
    )
    if proc.returncode not in {0, 1}:
        raise RuntimeError(f"git diff failed while checking runtime source: {path}")
    return proc.returncode == 0


def _desktop_source_status() -> tuple[str, ...]:
    """Return source dirt while excluding only known generated engine outputs."""
    value = _git_value(
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        ".",
        ":(exclude)artifacts/.pyinstaller/**",
        ":(exclude)artifacts/desktop-engine-candidate/**",
        ":(exclude)artifacts/desktop-engine-candidate-build-receipt.json",
        ":(exclude)artifacts/desktop-engine-focused-tests-receipt.json",
    )
    return tuple(line for line in value.splitlines() if line)


def _runtime_source_provenance(paths: tuple[str, ...]) -> list[dict]:
    rows = []
    for relative in paths:
        working_sha256 = _sha256(ROOT / relative)
        head_sha256 = _git_file_at_head_sha256(relative)
        rows.append({
            "path": relative,
            "working_file_sha256": working_sha256,
            "head_blob_sha256": head_sha256,
            "matches_head": _git_path_matches_head(relative),
        })
    return rows


def _candidate_file_rows(candidate_root: Path) -> list[dict]:
    rows = []
    for path in sorted(p for p in candidate_root.rglob("*") if p.is_file()):
        rows.append({
            "path": path.relative_to(candidate_root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        })
    return rows


def _replace_candidate(staged: Path, output: Path) -> None:
    output = output.resolve()
    default = DESKTOP_ENGINE_CANDIDATE.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if output == default:
            shutil.rmtree(output)
        elif output.is_dir() and not any(output.iterdir()):
            output.rmdir()
        else:
            raise FileExistsError(f"Desktop engine output must be empty: {output}")
    shutil.move(str(staged), str(output))


def desktop_engine_receipt_path(output: Path, receipt_path: Path | None) -> Path:
    """Resolve the sole receipt location, adjacent to and outside the candidate."""
    output = Path(output).resolve()
    canonical = (output.parent / f"{output.name}-build-receipt.json").resolve()
    if receipt_path is not None and Path(receipt_path).resolve() != canonical:
        raise ValueError(
            f"Desktop engine receipt must use canonical adjacent path: {canonical}"
        )
    return canonical


def build_desktop_engine(
    *,
    python: str,
    output: Path,
    receipt_path: Path | None = None,
) -> tuple[Path, Path]:
    """Build and stage the canonical standalone Desktop engine candidate."""
    python = str(Path(python).resolve())
    output = Path(output).resolve()
    # Resolve once and reuse: the existing sha256 re-check below still catches a
    # manifest edited mid-build, and pinning the path keeps a checkout that
    # appears or moves during the freeze from silently switching manifests.
    desktop_release_manifest = require_desktop_release_manifest()
    receipt_path = desktop_engine_receipt_path(output, receipt_path)
    observed_python = _python_probe(
        python,
        "import platform; print(platform.python_version())",
        "Python version",
    )
    observed_pyinstaller = _python_probe(
        python,
        "import PyInstaller; print(PyInstaller.__version__)",
        "PyInstaller version",
    )
    if observed_python != DESKTOP_ENGINE_PYTHON_VERSION:
        raise RuntimeError(
            f"Desktop engine requires Python {DESKTOP_ENGINE_PYTHON_VERSION}; "
            f"observed {observed_python}"
        )
    if observed_pyinstaller != DESKTOP_ENGINE_PYINSTALLER_VERSION:
        raise RuntimeError(
            f"Desktop engine requires PyInstaller {DESKTOP_ENGINE_PYINSTALLER_VERSION}; "
            f"observed {observed_pyinstaller}"
        )

    source_commit = _git_value("rev-parse", "HEAD")
    source_epoch = _git_value("show", "-s", "--format=%ct", "HEAD")
    source_status = _desktop_source_status()
    runtime_sources = desktop_engine_runtime_sources(ROOT)
    tracked_sources = tuple(
        line for line in _git_value("ls-files", "--", *runtime_sources).splitlines() if line
    )
    validate_desktop_runtime_source_tracking(runtime_sources, tracked_sources)
    runtime_source_provenance = _runtime_source_provenance(runtime_sources)
    expected = {
        "engine/bin/flywheel.exe",
        "engine/runtime/runtime-manifest.json",
        *(f"engine/runtime/{path}" for path in runtime_sources),
    }
    if len(expected) != DESKTOP_ENGINE_CANDIDATE_FILE_COUNT:
        raise RuntimeError("Desktop engine candidate file-count contract drifted")
    desktop_manifest = validate_desktop_candidate_paths(
        expected,
        manifest_path=desktop_release_manifest,
    )
    desktop_manifest_sha256 = _sha256(desktop_release_manifest)

    desktop_work = WORK / "desktop-engine"
    if desktop_work.exists():
        shutil.rmtree(desktop_work)
    dist_path = desktop_work / "dist"
    work_path = desktop_work / "work"
    spec_path = desktop_work / "generated-spec"
    staged = desktop_work / "candidate"
    for path in (dist_path, work_path, spec_path, staged / "engine" / "bin"):
        path.mkdir(parents=True, exist_ok=True)

    command = desktop_engine_pyinstaller_command(
        python,
        dist_path=dist_path,
        work_path=work_path,
        spec_path=spec_path,
    )
    environment = dict(os.environ)
    environment["PYTHONHASHSEED"] = "0"
    environment["SOURCE_DATE_EPOCH"] = source_epoch
    print(f"[desktop-engine] {' '.join(command)}")
    proc = subprocess.run(command, cwd=ROOT, env=environment)
    if proc.returncode != 0:
        raise RuntimeError(f"Desktop engine PyInstaller build failed ({proc.returncode})")

    built_executable = dist_path / "flywheel.exe"
    if not built_executable.is_file():
        raise FileNotFoundError(f"PyInstaller did not produce {built_executable}")
    staged_executable = staged / "engine" / "bin" / "flywheel.exe"
    shutil.copyfile(built_executable, staged_executable)
    staged_manifest = stage_desktop_engine_runtime(
        staged / "engine" / "runtime",
        source_root=ROOT,
        source_commit=source_commit,
    )

    observed = {row["path"] for row in _candidate_file_rows(staged)}
    if observed != expected:
        raise RuntimeError(
            "Desktop engine candidate path set diverged from the runtime allowlist"
        )
    validate_desktop_candidate_paths(
        observed,
        manifest_path=desktop_release_manifest,
    )
    if _sha256(desktop_release_manifest) != desktop_manifest_sha256:
        raise RuntimeError("Desktop release manifest changed during the engine build")

    _replace_candidate(staged, output)
    manifest_path = output / "engine" / "runtime" / staged_manifest.name
    receipt = {
        "schema": "flywheel.engine-build-receipt/1",
        "version": __version__,
        "protocol": GATEWAY_PROTOCOL,
        "builder": "scripts/build_local_harness_exes.py:--desktop-engine",
        "source": {
            "commit": source_commit,
            "dirty": bool(source_status),
            "status": list(source_status),
            "runtime_files": runtime_source_provenance,
        },
        "desktop_release_manifest": {
            "path": str(desktop_release_manifest.resolve()),
            "sha256": desktop_manifest_sha256,
            "schema_version": desktop_manifest["schema_version"],
            "manifest_state": desktop_manifest["manifest_state"],
        },
        "toolchain": {
            "python_required": DESKTOP_ENGINE_PYTHON_VERSION,
            "python_observed": observed_python,
            "python_implementation": platform.python_implementation(),
            "python_executable_sha256": _sha256(Path(python)),
            "pyinstaller_required": DESKTOP_ENGINE_PYINSTALLER_VERSION,
            "pyinstaller_observed": observed_pyinstaller,
            "platform": platform.platform(),
        },
        "determinism": {
            "PYTHONHASHSEED": "0",
            "SOURCE_DATE_EPOCH": source_epoch,
        },
        "command": command,
        "candidate_root": str(output),
        "files": _candidate_file_rows(output),
        "executable_sha256": _sha256(output / "engine" / "bin" / "flywheel.exe"),
        "runtime_manifest_sha256": _sha256(manifest_path),
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"[ok] Desktop engine candidate {output}")
    print(f"[ok] Desktop engine build receipt {receipt_path}")
    return output, receipt_path


def _write_cmd_wrapper(name: str) -> Path:
    path = DIST / f"{name}.cmd"
    path.write_text(
        "\n".join(
            [
                "@echo off",
                "setlocal",
                'if not defined LOCAL_HARNESS_REPO set "LOCAL_HARNESS_REPO=%~dp0..\\.."',
                f'"%~dp0{name}.exe" %*',
                "exit /b %ERRORLEVEL%",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _emit_endpoint_profiles(args: argparse.Namespace) -> Path:
    path = DIST / "model_endpoint_profiles.local.json"
    markdown = DIST / "model_endpoint_profiles.local.md"
    command = [
        sys.executable,
        "scripts/run_model_endpoint_profiles.py",
        "--models",
        "14B,32B",
        "--serve-url-14b",
        args.serve_url_14b,
        "--serve-url-32b",
        args.serve_url_32b,
        "--serve-runtime-32b",
        args.serve_runtime_32b,
        "--out",
        str(path),
        "--markdown-out",
        str(markdown),
    ]
    print(f"[profiles] {' '.join(command)}")
    proc = subprocess.run(command, cwd=ROOT)
    if proc.returncode != 0:
        raise RuntimeError(f"endpoint profile generation failed ({proc.returncode})")
    return path


def _emit_model_release_readiness(args: argparse.Namespace, *, profiles_path: Path) -> Path:
    path = DIST / "model_release_readiness.local.json"
    markdown = DIST / "model_release_readiness.local.md"
    command = [
        sys.executable,
        "scripts/run_model_release_readiness.py",
        "--models",
        args.model_release_models,
        "--base-root",
        args.model_run_root,
        "--artifact-roots",
        args.model_release_artifact_roots,
        "--endpoint-profile-artifacts",
        str(profiles_path),
        "--max-entries",
        str(args.model_release_max_entries),
        "--out",
        str(path),
        "--markdown-out",
        str(markdown),
    ]
    endpoint_gate_path = DIST / "model_endpoint_gate.local.json"
    if endpoint_gate_path.exists():
        command.extend(["--endpoint-gate-artifacts", str(endpoint_gate_path)])
    print(f"[model-release] {' '.join(command)}")
    proc = subprocess.run(command, cwd=ROOT)
    if proc.returncode != 0:
        raise RuntimeError(f"model release readiness generation failed ({proc.returncode})")
    return path


def _emit_model_publish_plan(args: argparse.Namespace, *, release_readiness_path: Path) -> Path:
    path = DIST / "model_publish_plan.local.json"
    markdown = DIST / "model_publish_plan.local.md"
    command = [
        sys.executable,
        "scripts/run_model_publish_plan.py",
        "--release-readiness-artifact",
        str(release_readiness_path),
        "--name-prefix",
        args.model_publish_name_prefix,
        "--out",
        str(path),
        "--markdown-out",
        str(markdown),
    ]
    print(f"[model-publish] {' '.join(command)}")
    proc = subprocess.run(command, cwd=ROOT)
    if proc.returncode != 0:
        raise RuntimeError(f"model publish plan generation failed ({proc.returncode})")
    return path


def _emit_model_repo_stage(
    args: argparse.Namespace,
    *,
    release_readiness_path: Path,
    publish_plan_path: Path,
) -> Path:
    path = DIST / "model_repo_stage.local.json"
    markdown = DIST / "model_repo_stage.local.md"
    command = [
        sys.executable,
        "scripts/run_model_repo_stage.py",
        "--release-readiness-artifact",
        str(release_readiness_path),
        "--publish-plan-artifact",
        str(publish_plan_path),
        "--docs-root",
        args.model_repo_docs_root,
        "--stage-root",
        str(DIST / "model_repositories"),
        "--namespace",
        args.huggingface_namespace,
        "--out",
        str(path),
        "--markdown-out",
        str(markdown),
    ]
    print(f"[model-repo-stage] {' '.join(command)}")
    proc = subprocess.run(command, cwd=ROOT)
    if proc.returncode != 0:
        raise RuntimeError(f"model repository staging failed ({proc.returncode})")
    return path


def _emit_huggingface_release_stage(
    args: argparse.Namespace,
    *,
    release_readiness_path: Path,
    publish_plan_path: Path,
) -> Path:
    path = DIST / "huggingface_release_stage.local.json"
    markdown = DIST / "huggingface_release_stage.local.md"
    command = [
        sys.executable,
        "scripts/run_huggingface_release_stage.py",
        "--release-readiness-artifact",
        str(release_readiness_path),
        "--publish-plan-artifact",
        str(publish_plan_path),
        "--namespace",
        args.huggingface_namespace,
        "--out",
        str(path),
        "--markdown-out",
        str(markdown),
    ]
    if args.huggingface_private:
        command.append("--private")
    print(f"[huggingface] {' '.join(command)}")
    proc = subprocess.run(command, cwd=ROOT)
    if proc.returncode != 0:
        raise RuntimeError(f"Hugging Face release stage generation failed ({proc.returncode})")
    return path


def _emit_executable_manifest(args: argparse.Namespace) -> Path:
    path = DIST / "harness_executable_manifest.local.json"
    markdown = DIST / "harness_executable_manifest.local.md"
    command = [
        sys.executable,
        "scripts/run_harness_cli.py",
        "manifest",
        "--store-root",
        args.store_root,
        "--out",
        str(path),
        "--markdown-out",
        str(markdown),
    ]
    print(f"[manifest] {' '.join(command)}")
    proc = subprocess.run(command, cwd=ROOT)
    if proc.returncode != 0:
        raise RuntimeError(f"executable manifest generation failed ({proc.returncode})")
    return path


def _emit_context_inventory(args: argparse.Namespace) -> Path:
    path = DIST / "context_inventory.local.json"
    markdown = DIST / "context_inventory.local.md"
    command = [
        sys.executable,
        "scripts/run_context_inventory.py",
        "--roots",
        args.context_roots,
        "--max-depth",
        str(args.context_max_depth),
        "--max-entries-per-root",
        str(args.context_max_entries_per_root),
        "--out",
        str(path),
        "--markdown-out",
        str(markdown),
    ]
    print(f"[context] {' '.join(command)}")
    proc = subprocess.run(command, cwd=ROOT)
    if proc.returncode != 0:
        raise RuntimeError(f"context inventory generation failed ({proc.returncode})")
    return path


def _emit_pubscan_resource_profiles(args: argparse.Namespace) -> Path:
    path = DIST / "pubscan_resource_profiles.local.json"
    markdown = DIST / "pubscan_resource_profiles.local.md"
    command = [
        sys.executable,
        "scripts/run_pubscan_resource_profiles.py",
        "--pubscan-root",
        args.pubscan_root,
        "--render-roots",
        args.pubscan_render_roots,
        "--storage-roots",
        args.pubscan_storage_roots,
        "--max-depth",
        str(args.pubscan_max_depth),
        "--max-entries",
        str(args.pubscan_max_entries),
        "--out",
        str(path),
        "--markdown-out",
        str(markdown),
    ]
    print(f"[pubscan] {' '.join(command)}")
    proc = subprocess.run(command, cwd=ROOT)
    if proc.returncode != 0:
        raise RuntimeError(f"pubscan resource profile generation failed ({proc.returncode})")
    return path


def _emit_tool_readiness_receipt(args: argparse.Namespace) -> Path:
    path = DIST / "tool_readiness.local.json"
    markdown = DIST / "tool_readiness.local.md"
    command = [
        sys.executable,
        "scripts/run_tool_readiness_receipts.py",
        "--tools",
        args.tools,
        "--base-root",
        args.tool_base_root,
        "--out",
        str(path),
        "--markdown-out",
        str(markdown),
    ]
    for tool_root in args.tool_root:
        command.extend(["--tool-root", tool_root])
    print(f"[tool-readiness] {' '.join(command)}")
    proc = subprocess.run(command, cwd=ROOT)
    if proc.returncode != 0:
        raise RuntimeError(f"tool readiness receipt generation failed ({proc.returncode})")
    return path


def _emit_tool_hardening_plan(args: argparse.Namespace, *, readiness_path: Path) -> Path:
    path = DIST / "tool_hardening_plan.local.json"
    markdown = DIST / "tool_hardening_plan.local.md"
    command = [
        sys.executable,
        "scripts/run_tool_hardening_plan.py",
        "--readiness-artifact",
        str(readiness_path),
        "--out",
        str(path),
        "--markdown-out",
        str(markdown),
    ]
    print(f"[tool-hardening] {' '.join(command)}")
    proc = subprocess.run(command, cwd=ROOT)
    if proc.returncode != 0:
        raise RuntimeError(f"tool hardening plan generation failed ({proc.returncode})")
    return path


def _emit_tool_contract(args: argparse.Namespace) -> Path:
    path = DIST / "tool_integration_contract.local.json"
    markdown = DIST / "tool_integration_contract.local.md"
    command = [
        sys.executable,
        "scripts/run_tool_integration_contract.py",
        "--tools",
        args.tools,
        "--base-root",
        args.tool_base_root,
        "--package-root",
        str(DIST),
        "--out",
        str(path),
        "--markdown-out",
        str(markdown),
    ]
    for tool_root in args.tool_root:
        command.extend(["--tool-root", tool_root])
    print(f"[tools] {' '.join(command)}")
    proc = subprocess.run(command, cwd=ROOT)
    if proc.returncode != 0:
        raise RuntimeError(f"tool integration contract generation failed ({proc.returncode})")
    return path


def _emit_tool_operator_guide(*, tool_contract_path: Path) -> Path:
    path = DIST / "tool_operator_guide.local.json"
    markdown = DIST / "tool_operator_guide.local.md"
    command = [
        sys.executable,
        "scripts/run_tool_operator_guide.py",
        "--tool-contract",
        str(tool_contract_path),
        "--out",
        str(path),
        "--markdown-out",
        str(markdown),
    ]
    print(f"[tool-guide] {' '.join(command)}")
    proc = subprocess.run(command, cwd=ROOT)
    if proc.returncode != 0:
        raise RuntimeError(f"tool operator guide generation failed ({proc.returncode})")
    return path


def _emit_runtime_contract(args: argparse.Namespace) -> Path:
    path = DIST / "runtime_activation_contract.local.json"
    markdown = DIST / "runtime_activation_contract.local.md"
    command = [
        sys.executable,
        "scripts/run_runtime_activation_contract.py",
        "--package-root",
        str(DIST),
        "--repo-root",
        str(ROOT),
        "--store-root",
        args.store_root,
        "--model-run-root",
        args.model_run_root,
        "--log-root",
        args.log_root,
        "--out",
        str(path),
        "--markdown-out",
        str(markdown),
    ]
    print(f"[runtime] {' '.join(command)}")
    proc = subprocess.run(command, cwd=ROOT)
    if proc.returncode != 0:
        raise RuntimeError(f"runtime activation contract generation failed ({proc.returncode})")
    return path


def _emit_codex_mcp_contract(args: argparse.Namespace) -> Path:
    path = DIST / "codex_mcp_launch_contract.local.json"
    markdown = DIST / "codex_mcp_launch_contract.local.md"
    command = [
        sys.executable,
        "scripts/run_codex_mcp_launch_contract.py",
        "--codex-config",
        args.codex_config,
        "--tools",
        args.codex_mcp_tools,
        "--observation",
        "index=TRANSPORT_CLOSED|active Codex MCP wrapper may require host reload after source/config repair",
        "--out",
        str(path),
        "--markdown-out",
        str(markdown),
    ]
    print(f"[codex-mcp] {' '.join(command)}")
    proc = subprocess.run(command, cwd=ROOT)
    if proc.returncode != 0:
        raise RuntimeError(f"Codex MCP launch contract generation failed ({proc.returncode})")
    return path


def _emit_enterprise_readiness_report(args: argparse.Namespace) -> Path:
    path = DIST / "enterprise_readiness_report.local.json"
    markdown = DIST / "enterprise_readiness_report.local.md"
    command = [
        sys.executable,
        "scripts/run_enterprise_readiness_report.py",
        "--tool-contract",
        str(DIST / "tool_integration_contract.local.json"),
        "--tools",
        args.enterprise_tools,
        "--out",
        str(path),
        "--markdown-out",
        str(markdown),
    ]
    print(f"[enterprise] {' '.join(command)}")
    proc = subprocess.run(command, cwd=ROOT)
    if proc.returncode != 0:
        raise RuntimeError(f"enterprise readiness report generation failed ({proc.returncode})")
    return path


def _emit_architecture_report(args: argparse.Namespace, *, release_manifest_path: Path) -> Path:
    path = DIST / "harness_architecture_report.local.json"
    markdown = DIST / "harness_architecture_report.local.md"
    command = [
        sys.executable,
        "scripts/run_harness_architecture_report.py",
        "--dist",
        str(DIST),
        "--release-manifest",
        str(release_manifest_path),
        "--package-doctor",
        str(DIST / "package-doctor-generated-after-bundle.json"),
        "--out",
        str(path),
        "--markdown-out",
        str(markdown),
    ]
    print(f"[architecture] {' '.join(command)}")
    proc = subprocess.run(command, cwd=ROOT)
    if proc.returncode != 0:
        raise RuntimeError(f"harness architecture report generation failed ({proc.returncode})")
    return path


def _write_release_manifest(args: argparse.Namespace, *, profiles_path: Path, built: list[str], skipped: list[str]) -> Path:
    manifest = {
        "schema": "harness.local-executable-release/v1",
        "created_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "repo_root": str(ROOT),
        "dist_root": str(DIST),
        "dependency_posture": {
            "runtime": "zero mandatory hosted services; local model serving uses configured local Python/runtime",
            "build": "PyInstaller is a build-time dependency only",
        },
        "executables": [
            {
                "name": name,
                "path": str(DIST / f"{name}.exe"),
                "cmd_wrapper": str(DIST / f"{name}.cmd") if name == "local-harness" else "",
                "exists": (DIST / f"{name}.exe").exists(),
            }
            for name in built
        ],
        "skipped": skipped,
        "local_models": {
            "endpoint_profiles": str(profiles_path),
            "serve_python": args.serve_python,
            "serve_url_14b": args.serve_url_14b,
            "serve_url_32b": args.serve_url_32b,
            "serve_runtime_32b": args.serve_runtime_32b,
            "offload_runtime": args.serve_runtime_32b == "cpu-offload",
            "release_readiness": str(DIST / "model_release_readiness.local.json"),
            "publish_plan": str(DIST / "model_publish_plan.local.json"),
            "model_repo_stage": str(DIST / "model_repo_stage.local.json"),
            "model_repo_stage_root": str(DIST / "model_repositories"),
            "huggingface_release_stage": str(DIST / "huggingface_release_stage.local.json"),
            "huggingface_namespace": args.huggingface_namespace,
            "huggingface_private": args.huggingface_private,
            "publish_name_prefix": args.model_publish_name_prefix,
        },
        "tool_integration": {
            "contract": str(DIST / "tool_integration_contract.local.json"),
            "tools": args.tools,
            "base_root": args.tool_base_root,
            "root_overrides": args.tool_root,
        },
        "tool_readiness": {
            "json": str(DIST / "tool_readiness.local.json"),
            "markdown": str(DIST / "tool_readiness.local.md"),
        },
        "tool_hardening_plan": {
            "json": str(DIST / "tool_hardening_plan.local.json"),
            "markdown": str(DIST / "tool_hardening_plan.local.md"),
        },
        "tool_operator_guide": {
            "json": str(DIST / "tool_operator_guide.local.json"),
            "markdown": str(DIST / "tool_operator_guide.local.md"),
        },
        "executable_manifest": {
            "json": str(DIST / "harness_executable_manifest.local.json"),
            "markdown": str(DIST / "harness_executable_manifest.local.md"),
        },
        "context_inventory": {
            "json": str(DIST / "context_inventory.local.json"),
            "markdown": str(DIST / "context_inventory.local.md"),
            "roots": args.context_roots,
            "max_depth": args.context_max_depth,
            "max_entries_per_root": args.context_max_entries_per_root,
        },
        "pubscan_resource_profiles": {
            "json": str(DIST / "pubscan_resource_profiles.local.json"),
            "markdown": str(DIST / "pubscan_resource_profiles.local.md"),
            "pubscan_root": args.pubscan_root,
            "render_roots": args.pubscan_render_roots,
            "storage_roots": args.pubscan_storage_roots,
            "max_depth": args.pubscan_max_depth,
            "max_entries": args.pubscan_max_entries,
        },
        "architecture_report": {
            "json": str(DIST / "harness_architecture_report.local.json"),
            "markdown": str(DIST / "harness_architecture_report.local.md"),
        },
        "enterprise_readiness": {
            "json": str(DIST / "enterprise_readiness_report.local.json"),
            "markdown": str(DIST / "enterprise_readiness_report.local.md"),
            "tools": args.enterprise_tools,
        },
        "codex_mcp": {
            "contract": str(DIST / "codex_mcp_launch_contract.local.json"),
            "tools": args.codex_mcp_tools,
            "codex_config": args.codex_config,
        },
        "runtime_activation": {
            "contract": str(DIST / "runtime_activation_contract.local.json"),
            "store_root": args.store_root,
            "model_run_root": args.model_run_root,
            "log_root": args.log_root,
        },
        "operator_notes": [
            "Run local-harness.cmd manifest to inspect the packaged command surface.",
            "Set LOCAL_HARNESS_REPO if the artifacts/exe folder is moved away from the repo checkout.",
            "Use local-harness.cmd readiness model-endpoints with the emitted profile settings before starting local serve.",
        ],
    }
    path = DIST / "local-harness-release.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--desktop-engine", action="store_true",
                    help="build only the standalone Desktop engine candidate")
    ap.add_argument("--desktop-engine-output", default=str(DESKTOP_ENGINE_CANDIDATE),
                    help="empty/default-replaceable root receiving engine/bin and engine/runtime")
    ap.add_argument("--desktop-engine-receipt",
                    help="build-command receipt path (defaults beside the candidate)")
    ap.add_argument("--skip-harness", action="store_true",
                    help="skip the full local-harness executable")
    ap.add_argument("--skip-agent", action="store_true",
                    help="skip the local-agent executable")
    ap.add_argument("--skip-serve", action="store_true",
                    help="skip the optional heavy local-serve executable")
    ap.add_argument("--serve-python", default=DEFAULT_SERVE_PYTHON,
                    help="python interpreter used for the torch-backed serve executable")
    ap.add_argument("--serve-url-14b", default="http://127.0.0.1:8765")
    ap.add_argument("--serve-url-32b", default="http://127.0.0.1:8768")
    ap.add_argument("--serve-runtime-32b", default="cpu-offload")
    ap.add_argument("--tools", default=DEFAULT_TOOLS)
    ap.add_argument("--tool-base-root", default="C:/dev/public")
    ap.add_argument("--tool-root", action="append", default=["aleph=C:/dev/aleph", "local-model=C:/dev/local-model"])
    ap.add_argument("--codex-config", default="C:/Users/Zain/.codex/config.toml")
    ap.add_argument("--codex-mcp-tools", default="index,forum,gather,crucible,telos")
    ap.add_argument("--enterprise-tools", default="mneme,relay,plexus")
    ap.add_argument("--store-root", default="C:/tmp/harness_file_store")
    ap.add_argument("--model-run-root", default="E:/local-model-run")
    ap.add_argument("--log-root", default="C:/tmp/local_model_serve_logs")
    ap.add_argument("--model-release-models", default="14B,32B")
    ap.add_argument("--model-release-artifact-roots", default="C:/dev/local-model/artifacts;C:/tmp")
    ap.add_argument("--model-release-max-entries", type=int, default=200)
    ap.add_argument("--model-publish-name-prefix", default="Flywheel-Local-Coder")
    ap.add_argument("--model-repo-docs-root", default="C:/dev/local-model/project-docs/releases")
    ap.add_argument("--huggingface-namespace", default="zaindanaharper")
    ap.add_argument("--huggingface-private", action="store_true")
    ap.add_argument(
        "--context-roots",
        default=(
            "C:/dev/local-model/.scratch;"
            "C:/dev/local-model/scratch;"
            "C:/dev/local-model/artifacts;"
            "C:/tmp;"
            "C:/Users/Zain/.codex;"
            "C:/Users/Zain/.claude;"
            "C:/Users/Zain/AppData/Roaming/opencode"
        ),
    )
    ap.add_argument("--context-max-depth", type=int, default=3)
    ap.add_argument("--context-max-entries-per-root", type=int, default=500)
    ap.add_argument("--pubscan-root", default="C:/dev/public/pubscan")
    ap.add_argument("--pubscan-render-roots", default="C:/dev/public;C:/dev/tools;C:/dev/local-model")
    ap.add_argument("--pubscan-storage-roots", default="C:/tmp;C:/dev;E:/local-model-run")
    ap.add_argument("--pubscan-max-depth", type=int, default=3)
    ap.add_argument("--pubscan-max-entries", type=int, default=2000)
    ap.add_argument("--package", action="store_true",
                    help="assemble a local release bundle after building")
    ap.add_argument("--package-version", default=datetime.now(UTC).strftime("%Y%m%d-%H%M%S"))
    args = ap.parse_args(argv)

    if args.desktop_engine:
        build_desktop_engine(
            python=sys.executable,
            output=Path(args.desktop_engine_output),
            receipt_path=(Path(args.desktop_engine_receipt)
                          if args.desktop_engine_receipt else None),
        )
        return 0

    DIST.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    serve_python = _py_path(args.serve_python)
    built: list[str] = []
    skipped: list[str] = []

    if not _has_pyinstaller(sys.executable):
        raise RuntimeError(f"PyInstaller unavailable via {sys.executable}")
    if not args.skip_harness:
        _build("local-harness", str(ROOT / "scripts" / "local_harness_entry.py"), python=sys.executable)
        _write_cmd_wrapper("local-harness")
        built.append("local-harness")
    else:
        skipped.append("local-harness")
    if not args.skip_agent:
        _build("local-agent", str(ROOT / "scripts" / "local_agent_entry.py"), python=sys.executable)
        built.append("local-agent")
    else:
        skipped.append("local-agent")
    if not args.skip_serve:
        if not _has_pyinstaller(serve_python):
            print("[warn] serve skipped: PyInstaller unavailable for serve-python interpreter")
            skipped.append("local-serve:missing_pyinstaller")
        elif not _has_modules(serve_python):
            print("[warn] serve skipped: required serve stack not available in that interpreter")
            skipped.append("local-serve:missing_modules")
        else:
            _build("local-serve", str(ROOT / "scripts" / "local_serve_entry.py"),
                   python=serve_python, hidden=[
                       "transformers",
                       "bitsandbytes",
                       "torch",
                       "peft",
                   ])
            built.append("local-serve")
    else:
        skipped.append("local-serve")

    _emit_executable_manifest(args)
    _emit_context_inventory(args)
    _emit_pubscan_resource_profiles(args)
    tool_readiness_path = _emit_tool_readiness_receipt(args)
    _emit_tool_hardening_plan(args, readiness_path=tool_readiness_path)
    profiles_path = _emit_endpoint_profiles(args)
    release_readiness_path = _emit_model_release_readiness(args, profiles_path=profiles_path)
    publish_plan_path = _emit_model_publish_plan(args, release_readiness_path=release_readiness_path)
    _emit_model_repo_stage(
        args,
        release_readiness_path=release_readiness_path,
        publish_plan_path=publish_plan_path,
    )
    _emit_huggingface_release_stage(
        args,
        release_readiness_path=release_readiness_path,
        publish_plan_path=publish_plan_path,
    )
    tool_contract_path = _emit_tool_contract(args)
    _emit_tool_operator_guide(tool_contract_path=tool_contract_path)
    _emit_runtime_contract(args)
    _emit_codex_mcp_contract(args)
    _emit_enterprise_readiness_report(args)
    manifest_path = _write_release_manifest(args, profiles_path=profiles_path, built=built, skipped=skipped)
    _emit_architecture_report(args, release_manifest_path=manifest_path)
    print(f"[ok] executables in {DIST}")
    print(f"[ok] release manifest {manifest_path}")
    if args.package:
        package_command = [
            sys.executable,
            "scripts/package_local_harness_release.py",
            "--version",
            args.package_version,
        ]
        if not args.skip_serve and "local-serve" in built:
            package_command.append("--include-serve")
        print(f"[package] {' '.join(package_command)}")
        proc = subprocess.run(package_command, cwd=ROOT)
        if proc.returncode != 0:
            raise RuntimeError(f"release package assembly failed ({proc.returncode})")
    if not args.skip_serve:
        print("[note] local-serve bundle is intentionally heavy because it includes torch/transformers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
