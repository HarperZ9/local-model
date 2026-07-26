"""harness — Layer B verified-inference harness (HARNESS-ROADMAP.md M0-M7).

M0 (proposer) is serve.py. This package implements M1: the minimal witnessed
loop (task -> retrieve -> propose -> oracle-verify -> envelope -> witness).

Invariants carried through every module (HARNESS.md §reward-shaping, §envelope):
  - no receipt -> no accept
  - no learned model in the accept path (only the real oracle accepts)
  - the harness never authors the criterion (operator names the oracle)
"""
from __future__ import annotations

import sys
from pathlib import Path


__version__ = "1.0.0"


def runtime_root() -> Path:
    """Return the physical root used by gateway runtime resources."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent.parent / "runtime"
    return Path(__file__).resolve().parent.parent
