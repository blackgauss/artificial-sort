"""Backward-compatibility shim."""
from poset_rl.models.mlp import ActorCritic  # noqa: F401
import numpy as np

TORCH_AVAILABLE = True


class PolicyNet(ActorCritic):
    """PolicyNet: same as ActorCritic but select_action returns (action, log_prob: float)."""

    def select_action(self, obs: np.ndarray, mask: np.ndarray):
        action, log_prob, _value = super().select_action(obs, mask)
        return action, float(log_prob)


__all__ = ["PolicyNet", "ActorCritic", "TORCH_AVAILABLE"]
