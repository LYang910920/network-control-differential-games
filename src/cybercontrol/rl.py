"""Replay, rollout, and advantage utilities shared by learning examples."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class ReplayTransition:
    """One off-policy transition with explicit state and terminal fields."""

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """Bounded replay storage with a local deterministic NumPy generator."""

    def __init__(self, capacity: int = 50_000, *, seed: int = 0):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._data: deque[ReplayTransition] = deque(maxlen=int(capacity))
        self._rng = np.random.default_rng(seed)

    def push(self, state, action, reward, next_state, done) -> None:
        self._data.append(
            ReplayTransition(
                np.asarray(state, dtype=np.float32).copy(),
                int(action),
                float(reward),
                np.asarray(next_state, dtype=np.float32).copy(),
                bool(done),
            )
        )

    def sample_numpy(self, batch_size: int) -> tuple[np.ndarray, ...]:
        """Return a without-replacement NumPy minibatch."""

        if batch_size > len(self._data):
            raise ValueError("batch_size exceeds the number of stored transitions")
        index = self._rng.choice(len(self._data), size=batch_size, replace=False)
        batch = [self._data[int(i)] for i in index]
        return (
            np.stack([row.state for row in batch]),
            np.asarray([row.action for row in batch], dtype=np.int64)[:, None],
            np.asarray([row.reward for row in batch], dtype=np.float32)[:, None],
            np.stack([row.next_state for row in batch]),
            np.asarray([row.done for row in batch], dtype=np.float32)[:, None],
        )

    def sample_torch(self, batch_size: int, *, device: str = "cpu"):
        """Return a Torch minibatch on ``device``."""

        import torch

        state, action, reward, next_state, done = self.sample_numpy(batch_size)
        return (
            torch.as_tensor(state, device=device),
            torch.as_tensor(action, device=device),
            torch.as_tensor(reward, device=device),
            torch.as_tensor(next_state, device=device),
            torch.as_tensor(done, device=device),
        )

    def __len__(self) -> int:
        return len(self._data)


@dataclass
class RolloutBuffer:
    """On-policy storage; callers decide the observation/action tensor shapes."""

    observations: list[Any] = field(default_factory=list)
    actions: list[Any] = field(default_factory=list)
    log_probabilities: list[Any] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    dones: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)

    def add(self, observation, action, log_probability, reward, done, value) -> None:
        self.observations.append(observation)
        self.actions.append(action)
        self.log_probabilities.append(log_probability)
        self.rewards.append(float(reward))
        self.dones.append(float(done))
        self.values.append(float(value))

    def __len__(self) -> int:
        return len(self.rewards)


def compute_gae(
    rewards: Iterable[float],
    values: Iterable[float],
    dones: Iterable[float],
    last_value: float,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute generalized advantages and bootstrapped returns."""

    reward = np.asarray(list(rewards), dtype=np.float32)
    value = np.asarray(list(values), dtype=np.float32)
    done = np.asarray(list(dones), dtype=np.float32)
    if not (reward.shape == value.shape == done.shape):
        raise ValueError("rewards, values, and dones must have the same one-dimensional shape")
    advantages = np.zeros_like(reward)
    running = 0.0
    for index in range(len(reward) - 1, -1, -1):
        next_value = float(last_value) if index == len(reward) - 1 else float(value[index + 1])
        nonterminal = 1.0 - float(done[index])
        delta = float(reward[index]) + gamma * next_value * nonterminal - float(value[index])
        running = delta + gamma * gae_lambda * nonterminal * running
        advantages[index] = running
    return advantages, advantages + value
