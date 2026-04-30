"""Permutation-equivariant AttentionActorCritic for the Poset RL problem.

Architecture
------------
Each candidate pair (i, j) with i < j is represented as a feature vector:

  pair_feat(i,j) = [
      known_ij,          # 1 if i>j known, -1 if j>i known, 0 unknown
      out_deg_i,         # how many elements i is known to beat
      in_deg_i,          # how many elements beat i
      out_deg_j,
      in_deg_j,
      transitive_gain,   # number of NEW inferences that would follow if we
                         # queried (i,j) right now (1-step lookahead)
  ]

A shared linear projects each pair feature to a hidden dim.  A small
Transformer encoder (self-attention over all pairs) lets pairs communicate —
e.g. a pair that bridges two known chains learns to score higher.

Policy head : per-pair scalar logit → masked softmax
Value head  : mean-pool over pairs → MLP → scalar V(s)

Key property — permutation equivariance
---------------------------------------
The features are purely relational (degrees, known status).  No index-based
positional encodings are used.  Renaming element 0 ↔ element 1 throughout
produces identical scores for the corresponding pairs.  This lets one model
generalise across all n without retraining.
"""
from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

PAIR_FEAT_DIM = 6   # see docstring above


def _pair_features(known: np.ndarray) -> np.ndarray:
    """Compute a (n*(n-1)/2, PAIR_FEAT_DIM) feature matrix from known[][].

    Features are *symmetric* in the two endpoints so that relabelling elements
    (applying any permutation to rows and columns of `known`) produces the same
    set of feature vectors, just in a different row order.

    Feature layout
    --------------
    0  is_resolved    abs(known[i,j])         1 if pair already known, else 0
    1  out_min        min(out_deg_i, out_deg_j)
    2  out_max        max(out_deg_i, out_deg_j)
    3  in_min         min(in_deg_i,  in_deg_j)
    4  in_max         max(in_deg_i,  in_deg_j)
    5  trans_gain     max transitive inferences from querying this pair (0 if resolved)
    """
    n = known.shape[0]
    out_deg = (known == 1).sum(axis=1).astype(np.float32)
    in_deg  = (known == -1).sum(axis=1).astype(np.float32)

    rows = []
    for i in range(n):
        for j in range(i + 1, n):
            is_resolved = float(known[i, j] != 0)

            out_a, out_b = out_deg[i], out_deg[j]
            in_a,  in_b  = in_deg[i],  in_deg[j]

            gain = 0.0
            if known[i, j] == 0:
                gain_ij = int((known[:, i] == 1).sum()) * int((known[j, :] == 1).sum())
                gain_ji = int((known[:, j] == 1).sum()) * int((known[i, :] == 1).sum())
                gain = float(max(gain_ij, gain_ji))

            rows.append([
                is_resolved,
                min(out_a, out_b), max(out_a, out_b),
                min(in_a,  in_b),  max(in_a,  in_b),
                gain,
            ])
    return np.array(rows, dtype=np.float32)


if TORCH_AVAILABLE:
    class _PairTransformer(nn.Module):
        """Projects pair features → hidden, runs nhead×nlayers self-attention."""

        def __init__(self, hidden: int = 64, nhead: int = 4, nlayers: int = 2):
            super().__init__()
            self.proj = nn.Linear(PAIR_FEAT_DIM, hidden)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden, nhead=nhead,
                dim_feedforward=hidden * 2,
                dropout=0.0, batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)

        def forward(self, pair_feats: torch.Tensor) -> torch.Tensor:
            # pair_feats: (batch, n_pairs, PAIR_FEAT_DIM)
            x = self.proj(pair_feats)          # (batch, n_pairs, hidden)
            return self.encoder(x)             # (batch, n_pairs, hidden)

    class AttentionActorCritic(nn.Module):
        """Variable-n actor-critic using pair-level self-attention.

        Works for any n at inference time — the transformer processes whatever
        number of pairs the current episode has.
        """

        def __init__(self, hidden: int = 64, nhead: int = 4, nlayers: int = 2):
            super().__init__()
            self.hidden = hidden
            self.transformer = _PairTransformer(hidden, nhead, nlayers)
            # policy: per-pair score
            self.policy_head = nn.Linear(hidden, 1)
            # value: pool → scalar
            self.value_head = nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1),
            )

        def forward(
            self,
            pair_feats: torch.Tensor,   # (batch, n_pairs, PAIR_FEAT_DIM)
            mask: torch.Tensor,         # (batch, n_pairs)  1=legal
        ):
            x = self.transformer(pair_feats)            # (batch, n_pairs, hidden)
            logits = self.policy_head(x).squeeze(-1)    # (batch, n_pairs)
            logits = logits + (mask <= 0).float() * -1e9
            value = self.value_head(x.mean(dim=1)).squeeze(-1)  # (batch,)
            return logits, value

        def select_action(self, known: np.ndarray, mask: np.ndarray):
            """
            Parameters
            ----------
            known : (n, n) int8 array — current known-relations matrix
            mask  : (n*(n-1)/2,) float32 — legal action mask

            Returns
            -------
            action   : int
            log_prob : torch.Tensor  (keeps grad for backprop)
            value    : torch.Tensor  (keeps grad for backprop)
            """
            device = next(self.parameters()).device
            feats = _pair_features(known)
            feats_t = torch.from_numpy(feats).unsqueeze(0).to(device)
            mask_t  = torch.from_numpy(mask).unsqueeze(0).to(device)

            logits, value = self.forward(feats_t, mask_t)
            probs = F.softmax(logits, dim=-1)
            m = torch.distributions.Categorical(probs)
            action = m.sample()
            return int(action.item()), m.log_prob(action), value.squeeze(0)

else:
    # NumPy fallback — uniform random, same interface
    class AttentionActorCritic:
        def __init__(self, hidden: int = 64, nhead: int = 4, nlayers: int = 2):
            pass

        def select_action(self, known: np.ndarray, mask: np.ndarray):
            legal = np.where(mask > 0)[0]
            if len(legal) == 0:
                raise RuntimeError("No legal actions")
            a = int(np.random.choice(legal))
            return a, float(-np.log(len(legal))), 0.0
