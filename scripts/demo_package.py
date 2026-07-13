"""Build a verified, portable manifest for offline demo media."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

PACKAGE_SCHEMA = "harness.demo-package/v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_demo_package(
    demo_dir: Path,
    *,
    source_revision: str,
    artifacts: Mapping[str, Path],
) -> dict:
    root = demo_dir.resolve()
    required = {
        "transcript",
        "player",
        "full_video",
        "short_video",
        "captions",
        "poster",
    }
    if set(artifacts) != required:
        raise ValueError(f"artifact roles must be exactly {sorted(required)}")

    resolved: dict[str, Path] = {}
    for role, configured in artifacts.items():
        path = configured.resolve()
        if not path.is_file() or not path.is_relative_to(root):
            raise ValueError(f"{role} must be a file inside {root.name}")
        resolved[role] = path

    transcript = json.loads(resolved["transcript"].read_text(encoding="utf-8"))
    if transcript.get("publishable") is not True:
        raise ValueError("transcript is not publishable")

    manifest = {
        "schema": PACKAGE_SCHEMA,
        "name": transcript.get("name", root.name),
        "source_revision": source_revision,
        "capture": sorted(
            {step.get("capture", "terminal-replay") for step in transcript["steps"]}
        ),
        "transcript_receipt_sha256": transcript["receipt_sha256"],
        "artifacts": {
            role: {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
            for role, path in sorted(resolved.items())
        },
    }
    (root / "demo-package.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return manifest
