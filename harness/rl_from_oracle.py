"""rl_from_oracle.py -- zero-label RL signal from the flywheel's own verifier.

The frontier direction (Ring-Zero, arXiv 2607.12395) is RL with no human labels,
where the reward comes from a verifier rather than an annotator. The flywheel
already owns the scarce half of that setup: a non-self-authored oracle whose
verdict a third party can re-derive. This module turns the SAME best-of-N group
the inference loop uses for selection into a GRPO training signal, with the oracle
as the reward. Training and inference share one generation-plus-verification
substrate; only the use differs.

GRPO (group-relative policy optimization, no value model): sample a group of G
candidates for one task, reward each, and score each by how far its reward sits
from the group mean, normalized by the group's spread. A group that is all-pass
or all-fail teaches nothing (zero advantage); a mixed group carries the gradient.
No critic model, so it fits a single-GPU, small-RAM box.

Two properties make this flywheel-native rather than a generic RL loop:

  1. The reward is a re-derivable receipt. Anyone can re-run the oracle on a
     candidate and reproduce its reward, so the training SIGNAL is auditable in a
     way a black-box reward model is not. Each group carries a content hash.
  2. The held-out oracle catches reward hacking. When a task carries a command the
     model never sees, a candidate that passes the visible oracle but fails the
     held-out one is flagged: it gamed the test instead of solving it. Training on
     that reward would teach the gaming; the flag lets the caller drop it.

The heavy weight update (logprobs plus optimizer) sits behind the PolicyOptimizer
protocol, so this core stays pure, tests with stubs, and runs with zero GPU. A
signal-only run IS the gradable-RL-data export (the forum bridge).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from statistics import fmean, pstdev
from typing import Protocol

from .task import Task
from .oracle import Oracle
from .proposer import Proposer, prompt_hash
from .adaptive_select import budget_schedule

EPS = 1e-8


def grpo_advantages(rewards: list[float]) -> list[float]:
    """Group-relative advantages: (r - mean) / (std + EPS).

    A group with no spread (all pass or all fail) returns all-zero advantages:
    there is no relative signal to learn from, and we report that honestly rather
    than manufacturing a gradient. Output length matches the input.
    """
    if not rewards:
        return []
    mean = fmean(rewards)
    std = pstdev(rewards)
    if std <= EPS:
        return [0.0 for _ in rewards]
    return [(r - mean) / (std + EPS) for r in rewards]


@dataclass
class Rollout:
    text: str
    seed: int
    temperature: float
    reward: float                       # visible-oracle reward: 1.0 pass, 0.0 fail
    advantage: float = 0.0
    held_out_reward: float | None = None
    reward_hacked: bool = False         # passed the visible oracle, failed held-out
    text_hash: str = ""


@dataclass
class RLGroup:
    task_id: str
    prompt_hash: str
    reward_source: str                  # e.g. "oracle:PytestOracle"
    rollouts: list[Rollout]
    group_mean: float
    group_std: float
    n_pass: int
    learnable: bool                     # spread > 0: the group carries a gradient
    reward_hacks: int
    signal_hash: str                    # content hash, re-derivable from the rollouts

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "prompt_hash": self.prompt_hash,
            "reward_source": self.reward_source,
            "group_mean": round(self.group_mean, 6),
            "group_std": round(self.group_std, 6),
            "n_pass": self.n_pass,
            "learnable": self.learnable,
            "reward_hacks": self.reward_hacks,
            "signal_hash": self.signal_hash,
            "rollouts": [
                {"seed": r.seed, "temperature": r.temperature, "reward": r.reward,
                 "advantage": round(r.advantage, 6), "held_out_reward": r.held_out_reward,
                 "reward_hacked": r.reward_hacked, "text_hash": r.text_hash}
                for r in self.rollouts
            ],
        }


@dataclass
class RLReceipt:
    group_size: int
    groups: list[RLGroup]
    n_groups: int
    n_learnable: int                    # groups with a gradient (mixed rewards)
    mean_reward: float                  # over every rollout
    pass_rate: float
    reward_hacks: int
    optimizer_stats: dict | None        # PolicyOptimizer.update output, or None (signal-only)
    receipt_hash: str
    schema: str = "flywheel.rl-from-oracle/v1"

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "group_size": self.group_size,
            "n_groups": self.n_groups,
            "n_learnable": self.n_learnable,
            "mean_reward": round(self.mean_reward, 6),
            "pass_rate": round(self.pass_rate, 6),
            "reward_hacks": self.reward_hacks,
            "optimizer_stats": self.optimizer_stats,
            "receipt_hash": self.receipt_hash,
            "groups": [g.to_dict() for g in self.groups],
        }


@dataclass
class RLItem:
    """One training item: a task, the oracle that rewards it, and an optional
    held-out oracle (a second command the model never sees) for hack detection."""
    task: Task
    oracle: Oracle
    held_out: Oracle | None = None


class PolicyOptimizer(Protocol):
    """The weight-update side: consume scored groups, run the GRPO gradient (with a
    KL term to a reference policy) on a QLoRA adapter, return training stats. Kept
    behind a protocol so the loop above is pure and GPU-free."""

    def update(self, groups: list[RLGroup]) -> dict: ...


def _hash_group(task: Task, rollouts: list[Rollout]) -> str:
    payload = {
        "task_id": task.task_id,
        "prompt_hash": prompt_hash(task.prompt),
        "rollouts": sorted(
            [[r.text_hash, r.reward, r.held_out_reward] for r in rollouts],
            key=lambda x: (x[0], x[1]),
        ),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _receipt_hash(groups: list[RLGroup]) -> str:
    blob = json.dumps(sorted(g.signal_hash for g in groups)).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


class RLFromOracle:
    """Generate a GRPO group per task, reward it with the oracle, compute
    group-relative advantages, and (optionally) hand the scored groups to a policy
    optimizer. Reuses the proposer, the oracle, and the index-stable (temperature,
    seed) grid that the inference loop already uses, so a training group has the
    same diversity guarantee as a best-of-N selection pool.
    """

    def __init__(self, proposer: Proposer, *, group_size: int = 8,
                 max_new_tokens: int | None = None):
        if group_size < 2:
            raise ValueError("GRPO needs a group of at least 2 to have relative signal")
        # index-stable grid must be able to supply group_size unique (temp, seed) pairs
        budget_schedule(group_size)
        self.proposer = proposer
        self.group_size = group_size
        self.max_new_tokens = max_new_tokens

    def collect(self, task: Task, oracle: Oracle, *, held_out: Oracle | None = None) -> RLGroup:
        schedule = budget_schedule(self.group_size)
        rollouts: list[Rollout] = []
        max_tokens = self.max_new_tokens or task.max_new_tokens
        for (temp, seed) in schedule:
            out = self.proposer.generate(task.prompt, seed=seed, temperature=temp,
                                         max_new_tokens=max_tokens, system=task.system)
            text = getattr(out, "text", "")
            reward = 1.0 if oracle.verify(text, task).passed else 0.0
            held_reward: float | None = None
            hacked = False
            if held_out is not None:
                held_reward = 1.0 if held_out.verify(text, task).passed else 0.0
                hacked = reward >= 1.0 and held_reward < 1.0
            rollouts.append(Rollout(
                text=text, seed=seed, temperature=temp, reward=reward,
                held_out_reward=held_reward, reward_hacked=hacked,
                text_hash=prompt_hash(text)))

        advs = grpo_advantages([r.reward for r in rollouts])
        for r, a in zip(rollouts, advs):
            r.advantage = a

        rewards = [r.reward for r in rollouts]
        mean = fmean(rewards) if rewards else 0.0
        std = pstdev(rewards) if len(rewards) > 1 else 0.0
        source = getattr(oracle, "oracle_type", type(oracle).__name__)
        return RLGroup(
            task_id=task.task_id, prompt_hash=prompt_hash(task.prompt),
            reward_source=f"oracle:{source}", rollouts=rollouts,
            group_mean=mean, group_std=std,
            n_pass=sum(1 for x in rewards if x >= 1.0),
            learnable=std > EPS,
            reward_hacks=sum(1 for r in rollouts if r.reward_hacked),
            signal_hash=_hash_group(task, rollouts))

    def run(self, items: list[RLItem], *, optimizer: PolicyOptimizer | None = None) -> RLReceipt:
        groups = [self.collect(it.task, it.oracle, held_out=it.held_out) for it in items]
        all_rewards = [r.reward for g in groups for r in g.rollouts]
        n_pass = sum(1 for x in all_rewards if x >= 1.0)
        stats = optimizer.update([g for g in groups if g.learnable]) if optimizer else None
        return RLReceipt(
            group_size=self.group_size, groups=groups, n_groups=len(groups),
            n_learnable=sum(1 for g in groups if g.learnable),
            mean_reward=fmean(all_rewards) if all_rewards else 0.0,
            pass_rate=(n_pass / len(all_rewards)) if all_rewards else 0.0,
            reward_hacks=sum(g.reward_hacks for g in groups),
            optimizer_stats=stats, receipt_hash=_receipt_hash(groups))
