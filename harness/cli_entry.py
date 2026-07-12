"""cli_entry.py -- the `flywheel` command dispatcher.

Flywheel is the one platform: routing + verification + the lane layer + the
closed verified-inference loop. This module is the single console-script entry
(``flywheel = harness.cli_entry:main`` in pyproject.toml).

Design: it is a thin layer over the existing ``scripts/run_harness_cli.py``
front controller. Every existing subcommand (app, manifest, registry,
benchmarks, mcp-health, ...) passes through unchanged. The new umbrella
subcommands -- ``lanes``, ``loop-status``, ``install``, ``up`` -- are handled
here once their modules land (Phase 2: lanes.py; Phase 3: loop-closure
updates). Until then they report a clear "not yet implemented" rather than
silently falling through.

Repo-root resolution mirrors ``scripts/local_harness_entry.py`` so the command
works identically as a console-script, from a checkout, and from a frozen exe.
"""
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

# The new umbrella subcommands. Handled in cli_entry; everything else is
# delegated to the existing run_harness_cli front controller.
_UMBRELLA_COMMANDS = {"lanes", "loop-status", "install", "up", "down", "corpus-export"}


def _candidate_roots() -> list[Path]:
    candidates: list[Path] = []
    explicit = os.environ.get("FLYWHEEL_REPO", "").strip() or os.environ.get("LOCAL_HARNESS_REPO", "").strip()
    if explicit:
        candidates.append(Path(explicit))
    candidates.append(Path.cwd())
    if getattr(sys, "frozen", False):
        exe = Path(sys.executable).resolve()
        candidates.extend([exe.parent, exe.parent.parent, exe.parent.parent.parent])
    candidates.append(Path(__file__).resolve().parent.parent)
    return candidates


def find_repo_root() -> Path:
    """Locate the flywheel/local-model checkout containing scripts/ and harness/."""
    seen: set[Path] = set()
    for candidate in _candidate_roots():
        try:
            resolved = candidate.expanduser().resolve()
        except OSError:
            continue
        for root in [resolved, *resolved.parents]:
            if root in seen:
                continue
            seen.add(root)
            if (root / "scripts" / "run_harness_cli.py").exists() and (root / "harness").is_dir():
                return root
    raise FileNotFoundError(
        "could not locate the flywheel repo root; set FLYWHEEL_REPO to the "
        "checkout containing scripts/run_harness_cli.py and harness/"
    )


def _dispatch_umbrella(command: str, argv: list[str]) -> int:
    """Handle the new umbrella subcommands. Phase 2/3 implement these fully."""
    repo_root = find_repo_root()
    if command == "loop-status":
        from harness.loop_closure import measure_loop, loop_report
        import tempfile
        m = measure_loop(tempfile.mkdtemp())
        print(loop_report(m))
        print()
        for h in m["handoffs"]:
            mark = "CLOSED" if h["closed"] else "OPEN"
            print(f"  {h['frm']:>10} -> {h['to']:<10} [{mark}]  {h['carries']}")
            print(f"             {h['evidence']}")
        return 0
    if command == "lanes":
        from harness.lanes import lane_roster, lane_report
        roster = lane_roster()
        print(lane_report(roster))
        return 0
    if command in {"install", "up", "down"}:
        print(f"`flywheel {command}` is part of the lane layer (Phase 2/3).", file=sys.stderr)
        print("Not yet wired in this build. See the umbrella plan.", file=sys.stderr)
        return 2
    if command == "corpus-export":
        # Gap E: export verified envelopes to a training shard (operator-gated).
        import json as _json
        import sys as _sys
        from harness.corpus_export import export_corpus
        args = [a for a in argv if not a.startswith("-")]
        if len(args) < 2:
            print("usage: flywheel corpus-export <envelopes_dir> <out.jsonl> [verdict_filter]", file=_sys.stderr)
            return 2
        verdict = args[2] if len(args) > 2 else "PASS"
        r = export_corpus(args[0], args[1], verdict_filter=verdict)
        print(_json.dumps(r, indent=2))
        return 0
    return 2


def main(argv: list[str] | None = None) -> int:
    raw = list(argv if argv is not None else sys.argv[1:])
    # Peek at the first positional to decide umbrella-vs-passthrough. The
    # existing run_harness_cli parser requires a subcommand, so the first
    # non-flag token is the command name.
    command = next((a for a in raw if not a.startswith("-")), None)
    if command in _UMBRELLA_COMMANDS:
        rest = [a for a in raw if a is not command]
        return _dispatch_umbrella(command, rest)
    # Passthrough: re-invoke scripts/run_harness_cli.py from the repo root so
    # its cwd-relative subprocess dispatch (build_command) resolves correctly.
    repo_root = find_repo_root()
    os.chdir(repo_root)
    script = repo_root / "scripts" / "run_harness_cli.py"
    sys.argv = [str(script), *raw]
    try:
        runpy.run_path(str(script), run_name="__main__")
    except SystemExit as exc:
        return int(exc.code or 0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
