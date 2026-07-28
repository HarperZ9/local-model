"""The file-size ratchet gate blocks NEW over-limit files while tolerating the
baselined pre-existing ones, and flags baseline entries that have shrunk."""
import importlib.util
from pathlib import Path

_GATE = Path(__file__).resolve().parent.parent / "scripts" / "check_file_size.py"
_spec = importlib.util.spec_from_file_location("check_file_size", _GATE)
gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate)


def test_new_over_limit_file_is_a_violation(monkeypatch):
    counts = {"new_big.py": 350, "small.py": 100, "old_big.py": 900}
    monkeypatch.setattr(gate, "_line_count", lambda rel: counts.get(rel, 0))
    new, dropped, over = gate.evaluate(list(counts), {"old_big.py"})
    assert new == ["new_big.py"]        # over 300 and not baselined -> fails
    assert "small.py" not in new        # under the limit
    assert "old_big.py" not in new      # over the limit but baselined -> tolerated
    assert dropped == []


def test_baseline_entry_that_shrank_is_flagged(monkeypatch):
    monkeypatch.setattr(gate, "_line_count", lambda rel: 120)  # now under the limit
    new, dropped, over = gate.evaluate(["shrunk.py"], {"shrunk.py"})
    assert new == []
    assert dropped == ["shrunk.py"]     # ratchet: remove it from the baseline
