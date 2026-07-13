"""Fast falsifiers for the zero-dependency demo package manifest. No network."""

import hashlib
import importlib
import json
from pathlib import Path

import pytest


def package_api():
    try:
        module = importlib.import_module("scripts.demo_package")
    except ModuleNotFoundError:
        pytest.fail("scripts.demo_package is not implemented")
    assert hasattr(module, "PACKAGE_SCHEMA")
    assert hasattr(module, "build_demo_package")
    return module


def write_package_fixture(tmp_path: Path, *, publishable: bool = True):
    demo_dir = tmp_path / "demo"
    demo_dir.mkdir()
    transcript = {
        "schema": "harness.demo-transcript/v1",
        "name": "package-demo",
        "publishable": publishable,
        "receipt_sha256": "a" * 64,
        "steps": [
            {"capture": "terminal-replay"},
            {"capture": "native-video"},
        ],
    }
    artifacts = {
        "transcript": demo_dir / "transcript.json",
        "player": demo_dir / "player.html",
        "full_video": demo_dir / "full.webm",
        "short_video": demo_dir / "short.webm",
        "captions": demo_dir / "captions.vtt",
        "poster": demo_dir / "poster.png",
    }
    artifacts["transcript"].write_text(json.dumps(transcript), encoding="utf-8")
    artifacts["player"].write_text("<!doctype html><title>Demo</title>", encoding="utf-8")
    artifacts["full_video"].write_bytes(b"full-video")
    artifacts["short_video"].write_bytes(b"short-video")
    artifacts["captions"].write_text("WEBVTT\n", encoding="utf-8")
    artifacts["poster"].write_bytes(b"\x89PNG\r\n\x1a\nposter")
    return demo_dir, artifacts, transcript


def test_build_demo_package_emits_verified_relative_manifest(tmp_path):
    module = package_api()
    demo_dir, artifacts, transcript = write_package_fixture(tmp_path)

    manifest = module.build_demo_package(
        demo_dir,
        source_revision="0123456789abcdef",
        artifacts=artifacts,
    )

    assert manifest["schema"] == "harness.demo-package/v1"
    assert manifest["name"] == "package-demo"
    assert manifest["source_revision"] == "0123456789abcdef"
    assert manifest["capture"] == ["native-video", "terminal-replay"]
    assert manifest["transcript_receipt_sha256"] == transcript["receipt_sha256"]
    assert set(manifest["artifacts"]) == set(artifacts)
    for role, info in manifest["artifacts"].items():
        relative_path = Path(info["path"])
        assert not relative_path.is_absolute()
        assert ".." not in relative_path.parts
        assert info["bytes"] == artifacts[role].stat().st_size
        assert info["sha256"] == hashlib.sha256(artifacts[role].read_bytes()).hexdigest()
    written = json.loads((demo_dir / "demo-package.json").read_text(encoding="utf-8"))
    assert written == manifest


def test_build_demo_package_rejects_non_publishable_transcript(tmp_path):
    module = package_api()
    demo_dir, artifacts, _transcript = write_package_fixture(tmp_path, publishable=False)

    with pytest.raises(ValueError, match="transcript is not publishable"):
        module.build_demo_package(demo_dir, source_revision="rev", artifacts=artifacts)


def test_build_demo_package_requires_exact_artifact_roles(tmp_path):
    module = package_api()
    demo_dir, artifacts, _transcript = write_package_fixture(tmp_path)
    artifacts.pop("poster")

    with pytest.raises(ValueError, match="artifact roles must be exactly"):
        module.build_demo_package(demo_dir, source_revision="rev", artifacts=artifacts)


def test_build_demo_package_rejects_missing_files(tmp_path):
    module = package_api()
    demo_dir, artifacts, _transcript = write_package_fixture(tmp_path)
    artifacts["poster"].unlink()

    with pytest.raises(ValueError, match="poster must be a file"):
        module.build_demo_package(demo_dir, source_revision="rev", artifacts=artifacts)


def test_build_demo_package_rejects_artifacts_outside_demo_dir(tmp_path):
    module = package_api()
    demo_dir, artifacts, _transcript = write_package_fixture(tmp_path)
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"outside")
    artifacts["poster"] = outside

    with pytest.raises(ValueError, match="poster must be a file inside demo"):
        module.build_demo_package(demo_dir, source_revision="rev", artifacts=artifacts)
