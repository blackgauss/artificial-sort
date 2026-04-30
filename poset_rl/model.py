"""Policy models for the Poset RL agent.

Two implementations are provided with identical public interfaces:
  - Torch (preferred): PolicyNet and ActorCritic are proper nn.Module subclasses
    that can be trained with gradient-based optimisers.
  - NumPy fallback: lightweight stubs that sample uniformly over legal actions;
    useful for running tests and smoke checks without installing PyTorch.

TORCH_AVAILABLE is exported so callers can branch on availability.
"""
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

import numpy as np

__all__ = ["PolicyNet", "ActorCritic", "TORCH_AVAILABLE"]


if TORCH_AVAILABLE:
    class PolicyNet(nn.Module):
        def __init__(self, n: int, hidden: int = 128):
            super().__init__()
            self.n = n
            obs_dim = n * n
            self.fc1 = nn.Linear(obs_dim, hidden)
            self.fc2 = nn.Linear(hidden, hidden)
            # output logits for each unordered pair i<j
            self.out_dim = n * (n - 1) // 2
            self.head = nn.Linear(hidden, self.out_dim)

        def forward(self, obs: torch.Tensor, mask: torch.Tensor = None):
            # obs: batch x obs_dim
            x = F.relu(self.fc1(obs))
            x = F.relu(self.fc2(x))
            logits = self.head(x)
            if mask is not None:
                # mask shape batch x out_dim with 1 for legal
                inf_mask = (mask <= 0).float() * -1e9
                logits = logits + inf_mask
            return logits

        def select_action(self, obs: np.ndarray, mask: np.ndarray):
            obs_t = torch.from_numpy(obs).unsqueeze(0)
            mask_t = torch.from_numpy(mask).unsqueeze(0)
            logits = self.forward(obs_t.float(), mask_t.float())
            probs = F.softmax(logits, dim=-1)
            m = torch.distributions.Categorical(probs)
            action = m.sample()
            return int(action.item()), float(m.log_prob(action).item())

    class ActorCritic(nn.Module):
        """Combined actor-critic: shared trunk, separate policy and value heads."""

        def __init__(self, n: int, hidden: int = 128):
            super().__init__()
            self.n = n
            obs_dim = n * n
            self.out_dim = n * (n - 1) // 2
            self.fc1 = nn.Linear(obs_dim, hidden)
            self.fc2 = nn.Linear(hidden, hidden)
            self.policy_head = nn.Linear(hidden, self.out_dim)
            self.value_head = nn.Linear(hidden, 1)

        def _trunk(self, obs: torch.Tensor):
            x = F.relu(self.fc1(obs))
            return F.relu(self.fc2(x))

        def forward(self, obs: torch.Tensor, mask: torch.Tensor = None):
            x = self._trunk(obs)
            logits = self.policy_head(x)
            if mask is not None:
                logits = logits + (mask <= 0).float() * -1e9
            value = self.value_head(x).squeeze(-1)
            return logits, value

        def select_action(self, obs: np.ndarray, mask: np.ndarray):
            device = next(self.parameters()).device
            obs_t  = torch.from_numpy(obs).unsqueeze(0).float().to(device)
            mask_t = torch.from_numpy(mask).unsqueeze(0).float().to(device)
            logits, value = self.forward(obs_t, mask_t)
            probs = F.softmax(logits, dim=-1)
            m = torch.distributions.Categorical(probs)
            action = m.sample()
            # return tensors with grad so train_step can back-prop through them
            return int(action.item()), m.log_prob(action), value.squeeze(0)
else:
    # ------------------------------------------------------------------
    # NumPy-only fallbacks — uniform random policy, no learning.
    # Identical public interface so tests and smoke checks run without torch.
    # ------------------------------------------------------------------
    class PolicyNet:
        def __init__(self, n: int, hidden: int = 128):
            self.n = n
            self.out_dim = n * (n - 1) // 2

        def select_action(self, obs: np.ndarray, mask: np.ndarray):
            legal = np.where(mask > 0)[0]
            if len(legal) == 0:
                raise RuntimeError("No legal actions")
            a = int(np.random.choice(legal))
            return a, float(-np.log(len(legal)))

    class ActorCritic:
        def __init__(self, n: int, hidden: int = 128):
            self.n = n
            self.out_dim = n * (n - 1) // 2

        def select_action(self, obs: np.ndarray, mask: np.ndarray):
            legal = np.where(mask > 0)[0]
            if len(legal) == 0:
                raise RuntimeError("No legal actions")
            a = int(np.random.choice(legal))
            return a, float(-np.log(len(legal))), 0.0
