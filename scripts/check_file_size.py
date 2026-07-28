#!/usr/bin/env python3
"""File-size ratchet gate.

The project standard is 'no file > 300 lines'. 61 tracked .py files already exceed
it; splitting them all at once is out of scope. This gate instead RATCHETS: it
fails only when a NEW .py file crosses 300 lines without being in the baseline,
so the debt cannot grow, and it flags baseline entries that have dropped back
under the limit so the list only shrinks.

    python scripts/check_file_size.py            # exit 1 on a new violation
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

LIMIT = 300
ROOT = Path(__file__).resolve().parent.parent
BASELINE = ROOT / "scripts" / "file_size_baseline.txt"


def _tracked_py() -> list[str]:
    out = subprocess.run(["git", "ls-files", "*.py"], cwd=ROOT,
                         capture_output=True, text=True)
    return [line for line in out.stdout.splitlines() if line.strip()]


def _line_count(rel: str) -> int:
    try:
        with (ROOT / rel).open("rb") as fh:
            return sum(1 for _ in fh)
    except OSError:
        return 0


def _load_baseline() -> set[str]:
    if not BASELINE.exists():
        return set()
    return {line.strip() for line in BASELINE.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")}


def evaluate(tracked, baseline):
    """Return (new_violations, dropped_below) as sorted lists. Pure for testing."""
    over = {rel for rel in tracked if _line_count(rel) > LIMIT}
    new = sorted(over - baseline)
    dropped = sorted(baseline - over)
    return new, dropped, sorted(over)


def main() -> int:
    baseline = _load_baseline()
    new, dropped, over = evaluate(_tracked_py(), baseline)
    if new:
        print(f"FAIL: {len(new)} file(s) over {LIMIT} lines and not in the baseline:")
        for rel in new:
            print(f"  {rel}: {_line_count(rel)} lines")
        print("Split the file into modules under the limit, or (last resort) add it to "
              "scripts/file_size_baseline.txt with a note.")
        return 1
    if dropped:
        print(f"OK, and {len(dropped)} baseline entr(y/ies) are now under {LIMIT} lines — "
              f"remove them from the baseline to ratchet the debt down:")
        for rel in dropped:
            print(f"  {rel}")
    print(f"file-size gate PASS: {len(over)} known over-limit, 0 new violations.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
