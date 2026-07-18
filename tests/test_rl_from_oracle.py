"""Falsifier for rl_from_oracle.py -- the GRPO-from-oracle training signal.

The component must: reward each candidate by the oracle (1/0), compute
group-relative advantages that are ZERO when the group has no spread (all pass or
all fail) and correctly signed when mixed, flag reward hacking via a held-out
oracle, emit a re-derivable receipt (same generations + same oracle -> same hash),
and refuse a group too small to carry a relative signal.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from harness.oracle import OracleResult
from harness.proposer import ProposerOutput
from harness.task import Task
from harness.adaptive_select import SCHEDULE_CAPACITY
from harness.rl_from_oracle import RLFromOracle, RLItem, grpo_advantages


def _task(tid="t1", prompt="p"):
    return Task(task_id=tid, prompt=prompt, oracle="stub", oracle_cmd="",
                workdir=".", candidate_path="c.py", max_new_tokens=16)


class TempProposer:
    """Distinct output per temperature. budget_schedule holds seed constant for the
    first nine candidates and varies temperature, so a small group differs by temp."""
    model_ref = "temp"

    def generate(self, prompt, *, seed, temperature, max_new_tokens, system=""):
        return ProposerOutput(text=f"c{temperature}", model_ref="temp",
                              seed=seed, prompt_hash="h", cache="stub")


class SetOracle:
    def __init__(self, passing, oracle_type="set"):
        self.passing = set(passing)
        self.oracle_type = oracle_type

    def verify(self, candidate, task):
        p = candidate in self.passing
        return OracleResult(passed=p, cmd="set", output_hash="h",
                            stdout_excerpt="", rc=0 if p else 1)


def test_grpo_advantages_zero_when_no_spread():
    assert grpo_advantages([1, 1, 1]) == [0.0, 0.0, 0.0]
    assert grpo_advantages([0, 0, 0]) == [0.0, 0.0, 0.0]
    assert grpo_advantages([]) == []


def test_grpo_advantages_signs_and_zero_sum():
    a = grpo_advantages([1, 0, 0, 0])
    assert a[0] > 0 and all(x < 0 for x in a[1:])
    assert abs(sum(a)) < 1e-6


def test_group_size_floor():
    with pytest.raises(ValueError):
        RLFromOracle(TempProposer(), group_size=1)


def test_group_size_over_capacity_rejected():
    with pytest.raises(ValueError):
        RLFromOracle(TempProposer(), group_size=SCHEDULE_CAPACITY + 1)


def test_all_pass_group_not_learnable():
    rl = RLFromOracle(TempProposer(), group_size=4)
    g = rl.collect(_task(), SetOracle({"c0.0", "c0.2", "c0.35", "c0.5"}))
    assert g.n_pass == 4 and g.learnable is False
    assert all(r.advantage == 0.0 for r in g.rollouts)


def test_mixed_group_learnable_and_signed():
    rl = RLFromOracle(TempProposer(), group_size=4)
    g = rl.collect(_task(), SetOracle({"c0.0", "c0.2"}))
    assert g.n_pass == 2 and g.learnable is True
    passers = [r for r in g.rollouts if r.reward >= 1.0]
    failers = [r for r in g.rollouts if r.reward < 1.0]
    assert passers and failers
    assert all(r.advantage > 0 for r in passers)
    assert all(r.advantage < 0 for r in failers)


def test_reward_hacking_flagged_by_held_out():
    rl = RLFromOracle(TempProposer(), group_size=4)
    g = rl.collect(_task(), SetOracle({"c0.0", "c0.2"}), held_out=SetOracle(set()))
    assert g.reward_hacks == 2
    hacked = [r for r in g.rollouts if r.reward_hacked]
    assert len(hacked) == 2
    assert all(r.reward >= 1.0 and r.held_out_reward == 0.0 for r in hacked)


def test_no_false_hack_when_held_out_agrees():
    rl = RLFromOracle(TempProposer(), group_size=4)
    g = rl.collect(_task(), SetOracle({"c0.0", "c0.2"}),
                   held_out=SetOracle({"c0.0", "c0.2"}))
    assert g.reward_hacks == 0


def test_receipt_hash_is_rederivable():
    items = [RLItem(_task("a"), SetOracle({"c0.0", "c0.2"})),
             RLItem(_task("b"), SetOracle({"c0.0"}))]
    r1 = RLFromOracle(TempProposer(), group_size=4).run(items)
    r2 = RLFromOracle(TempProposer(), group_size=4).run(items)
    assert r1.receipt_hash == r2.receipt_hash
    assert r1.n_groups == 2


def test_optimizer_receives_only_learnable_groups():
    seen = []

    class StubOpt:
        def update(self, groups):
            seen.extend(g.task_id for g in groups)
            return {"loss": 0.123, "groups": len(groups)}

    rl = RLFromOracle(TempProposer(), group_size=4)
    items = [
        RLItem(_task("mixed"), SetOracle({"c0.0", "c0.2"})),
        RLItem(_task("allpass"), SetOracle({"c0.0", "c0.2", "c0.35", "c0.5"})),
    ]
    r = rl.run(items, optimizer=StubOpt())
    assert seen == ["mixed"]
    assert r.optimizer_stats["loss"] == 0.123
    assert r.n_learnable == 1
