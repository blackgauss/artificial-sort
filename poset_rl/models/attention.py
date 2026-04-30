"""PyTorch Attention ActorCritic — permutation-equivariant, any n.

Registered as 'attention'.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from poset_rl.models import register

PAIR_FEAT_DIM = 6


def _pair_features(known: np.ndarray) -> np.ndarray:
    """(n*(n-1)/2, 6) symmetric pair feature matrix.

    Features: [is_resolved, out_min, out_max, in_min, in_max, trans_gain]
    Symmetric in the two endpoints — equivariance is preserved under relabelling.
    """
    n      = known.shape[0]
    out_deg = (known == 1).sum(axis=1).astype(np.float32)
    in_deg  = (known == -1).sum(axis=1).astype(np.float32)
    rows = []
    for i in range(n):
        for j in range(i + 1, n):
            resolved = float(known[i, j] != 0)
            gain = 0.0
            if known[i, j] == 0:
                gain_ij = int((known[:, i] == 1).sum()) * int((known[j, :] == 1).sum())
                gain_ji = int((known[:, j] == 1).sum()) * int((known[i, :] == 1).sum())
                gain = float(max(gain_ij, gain_ji))
            rows.append([
                resolved,
                min(out_deg[i], out_deg[j]), max(out_deg[i], out_deg[j]),
                min(in_deg[i],  in_deg[j]),  max(in_deg[i],  in_deg[j]),
                gain,
            ])
    return np.array(rows, dtype=np.float32)


class _TransformerBlock(nn.Module):
    def __init__(self, hidden: int, nhead: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.attn  = nn.MultiheadAttention(hidden, nhead, batch_first=True, dropout=0.0)
        self.ff    = nn.Sequential(
            nn.Linear(hidden, hidden * 4), nn.ReLU(),
            nn.Linear(hidden * 4, hidden),
        )

    def forward(self, x):
        # pre-norm self-attention
        xn = self.norm1(x)
        x  = x + self.attn(xn, xn, xn, need_weights=False)[0]
        x  = x + self.ff(self.norm2(x))
        return x


@register("attention")
class AttentionActorCritic(nn.Module):
    """Variable-n transformer over pair-level features."""

    def __init__(self, hidden: int = 64, nhead: int = 2, nlayers: int = 1, **_):
        super().__init__()
        self.embed   = nn.Linear(PAIR_FEAT_DIM, hidden)
        self.encoder = nn.Sequential(
            *[_TransformerBlock(hidden, nhead) for _ in range(nlayers)]
        )
        self.policy  = nn.Linear(hidden, 1)
        self.value   = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )

    def forward(self, feats: torch.Tensor, mask: torch.Tensor | None = None):
        """feats: (B, P, FEAT_DIM)  mask: (B, P) bool."""
        x      = self.encoder(self.embed(feats))
        logits = self.policy(x).squeeze(-1)
        if mask is not None:
            logits = logits.masked_fill(~mask.bool(), float("-inf"))
        value  = self.value(x.mean(dim=1)).squeeze(-1)
        return logits, value

    def select_action(self, obs: np.ndarray, mask: np.ndarray):
        """Accept either a flat n² array or an (n,n) known matrix."""
        obs = np.asarray(obs)
        if obs.ndim == 2:
            known = obs
        else:
            n     = int(round(len(obs) ** 0.5))
            known = obs.reshape(n, n)
        feats = _pair_features(known)
        device = next(self.parameters()).device
        ft = torch.tensor(feats, device=device).unsqueeze(0)
        mt = torch.tensor(mask,  dtype=torch.bool, device=device).unsqueeze(0)
        logits, v = self(ft, mt)
        dist   = torch.distributions.Categorical(logits=logits.squeeze(0))
        action = dist.sample()
        return int(action), dist.log_prob(action), v.squeeze(0)
