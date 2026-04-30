"""Agent protocols — define the interface each framework must satisfy.

Any model registered in the model registry should implement exactly one of:

TorchAgent
----------
  select_action(obs, mask) -> (action, log_prob, value)
  • obs   : np.ndarray, shape (n²,)  — flattened known-relation matrix
  • mask  : np.ndarray, shape (n*(n-1)//2,)  — boolean legal-actions mask
  • Returns a 3-tuple:
      action   : int
      log_prob : torch.Tensor  (scalar, differentiable)
      value    : torch.Tensor  (scalar, differentiable)

JaxAgent
--------
  act(obs, mask, rng_key) -> action
  • obs     : np.ndarray, shape (n²,)
  • mask    : np.ndarray, shape (n*(n-1)//2,)
  • rng_key : jax.Array  — JAX PRNG key
  • Returns:
      action : int

The training loops detect which protocol a model satisfies via:
  is_jax_agent(model)   — True if model has .act but not .select_action
  is_torch_agent(model) — True if model has .select_action
"""
from __future__ import annotations

from typing import TYPE_CHECKING, runtime_checkable, Protocol

import numpy as np

try:
    from typing import Protocol, runtime_checkable
except ImportError:  # Python < 3.8
    from typing_extensions import Protocol, runtime_checkable  # type: ignore

__all__ = [
    "TorchAgent",
    "JaxAgent",
    "is_torch_agent",
    "is_jax_agent",
]


@runtime_checkable
class TorchAgent(Protocol):
    """Protocol for PyTorch actor-critic models."""

    def select_action(self, obs: np.ndarray, mask: np.ndarray):
        """Return (action: int, log_prob: Tensor, value: Tensor)."""
        ...


@runtime_checkable
class JaxAgent(Protocol):
    """Protocol for JAX/Flax actor models."""

    def act(self, obs: np.ndarray, mask: np.ndarray, rng_key):
        """Return action: int."""
        ...


# ---------------------------------------------------------------------------
# Helpers used by train.py, train_jax.py, bench.py, sweep.py
# ---------------------------------------------------------------------------

def is_torch_agent(model) -> bool:
    """True if *model* implements the TorchAgent protocol."""
    return isinstance(model, TorchAgent)


def is_jax_agent(model) -> bool:
    """True if *model* implements the JaxAgent protocol (and NOT TorchAgent).

    A model that has both .act and .select_action is treated as a TorchAgent.
    """
    return isinstance(model, JaxAgent) and not isinstance(model, TorchAgent)
