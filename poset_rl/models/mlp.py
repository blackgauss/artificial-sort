"""PyTorch MLP ActorCritic — fixed n.  Registered as 'mlp'."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from poset_rl.models import register


@register("mlp")
class ActorCritic(nn.Module):
    """Shared-trunk MLP. Observation is the n² flattened known-relation matrix."""

    def __init__(self, n: int, hidden: int = 64, **_):
        super().__init__()
        obs_size  = n * n
        n_actions = n * (n - 1) // 2
        self.trunk  = nn.Sequential(
            nn.Linear(obs_size, hidden), nn.ReLU(),
            nn.Linear(hidden,  hidden), nn.ReLU(),
        )
        self.policy = nn.Linear(hidden, n_actions)
        self.value  = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor, mask: torch.Tensor | None = None):
        h      = self.trunk(obs)
        logits = self.policy(h)
        if mask is not None:
            logits = logits.masked_fill(~mask.bool(), float("-inf"))
        value  = self.value(h).squeeze(-1)
        return logits, value

    def select_action(self, obs: np.ndarray, mask: np.ndarray):
        device = next(self.parameters()).device
        x = torch.tensor(obs,  dtype=torch.float32, device=device).unsqueeze(0)
        m = torch.tensor(mask, dtype=torch.bool,    device=device).unsqueeze(0)
        logits, v = self(x, m)
        dist   = torch.distributions.Categorical(logits=logits.squeeze(0))
        action = dist.sample()
        return int(action), dist.log_prob(action), v.squeeze(0)
