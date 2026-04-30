"""JAX/Flax Attention ActorCritic — permutation-equivariant, any n.

Registered as 'jax_attention'.

Architecture
------------
Pair features (same symmetric 6-dim as the PyTorch version) are embedded and
processed by a stack of pre-norm transformer blocks implemented from scratch
with NNX primitives — no dependency on flax.nnx.MultiHeadAttention whose API
has changed across minor releases.

Install:  pip install "artificial-sort[jax]"
"""
from __future__ import annotations
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from flax import nnx
    _JAX_OK = True
except ImportError:
    _JAX_OK = False

# _pair_features is pure numpy — but attention.py imports torch at module level,
# so we duplicate the small helper here to avoid a hard torch dependency.
PAIR_FEAT_DIM = 6

def _pair_features(known: np.ndarray) -> np.ndarray:
    """Symmetric 6-dim feature vector per unordered pair (i,j), i<j."""
    n = known.shape[0]
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    feats = np.zeros((len(pairs), PAIR_FEAT_DIM), dtype=np.float32)
    for k, (i, j) in enumerate(pairs):
        resolved   = float(known[i, j] != 0)
        out_i      = int((known[i, :] == 1).sum())
        out_j      = int((known[j, :] == 1).sum())
        in_i       = int((known[:, i] == -1).sum())
        in_j       = int((known[:, j] == -1).sum())
        gain_ij    = int((known[:, i] == 1).sum()) * int((known[j, :] == 1).sum())
        gain_ji    = int((known[:, j] == 1).sum()) * int((known[i, :] == 1).sum())
        gain       = max(gain_ij, gain_ji) / max(n * n, 1)
        feats[k]   = [resolved,
                      out_i / n, out_j / n,
                      in_i  / n, in_j  / n,
                      gain]
    return feats

if _JAX_OK:
    from poset_rl.models import register

    # ------------------------------------------------------------------ blocks

    class _MHSelfAttn(nnx.Module):
        """Multi-head self-attention implemented with Linear projections."""

        def __init__(self, hidden: int, nhead: int, *, rngs: nnx.Rngs):
            assert hidden % nhead == 0, "hidden must be divisible by nhead"
            self.nhead  = nhead
            self.hidden = hidden
            self.head_dim = hidden // nhead
            self.q = nnx.Linear(hidden, hidden, use_bias=False, rngs=rngs)
            self.k = nnx.Linear(hidden, hidden, use_bias=False, rngs=rngs)
            self.v = nnx.Linear(hidden, hidden, use_bias=False, rngs=rngs)
            self.o = nnx.Linear(hidden, hidden, rngs=rngs)

        def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
            B, S, H = x.shape
            nh, hd  = self.nhead, self.head_dim

            def split_heads(t):
                return t.reshape(B, S, nh, hd).transpose(0, 2, 1, 3)  # (B,nh,S,hd)

            q, k, v = split_heads(self.q(x)), split_heads(self.k(x)), split_heads(self.v(x))
            scale   = hd ** -0.5
            attn    = jax.nn.softmax(jnp.einsum("bhsd,bhtd->bhst", q, k) * scale, axis=-1)
            out     = jnp.einsum("bhst,bhtd->bhsd", attn, v)
            out     = out.transpose(0, 2, 1, 3).reshape(B, S, H)
            return self.o(out)

    class _TransformerBlock(nnx.Module):
        def __init__(self, hidden: int, nhead: int, *, rngs: nnx.Rngs):
            self.norm1 = nnx.LayerNorm(hidden, rngs=rngs)
            self.norm2 = nnx.LayerNorm(hidden, rngs=rngs)
            self.attn  = _MHSelfAttn(hidden, nhead, rngs=rngs)
            self.ff1   = nnx.Linear(hidden, hidden * 4, rngs=rngs)
            self.ff2   = nnx.Linear(hidden * 4, hidden, rngs=rngs)

        def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
            x = x + self.attn(self.norm1(x))
            x = x + self.ff2(nnx.relu(self.ff1(self.norm2(x))))
            return x

    # ------------------------------------------------------------------ model

    @register("jax_attention")
    class JaxAttention(nnx.Module):
        """Variable-n transformer over symmetric pair-level features."""

        def __init__(
            self,
            hidden:  int = 64,
            nhead:   int = 2,
            nlayers: int = 1,
            *,
            rngs: nnx.Rngs,
            **_,
        ):
            self.embed   = nnx.Linear(PAIR_FEAT_DIM, hidden, rngs=rngs)
            self.blocks  = [
                _TransformerBlock(hidden, nhead, rngs=rngs)
                for _ in range(nlayers)
            ]
            self.policy  = nnx.Linear(hidden, 1, rngs=rngs)
            self.value_h = nnx.Linear(hidden, hidden, rngs=rngs)
            self.value_o = nnx.Linear(hidden, 1, rngs=rngs)

        def __call__(
            self,
            feats: jnp.ndarray,             # (B, P, FEAT_DIM)
            mask:  jnp.ndarray | None = None,  # (B, P) bool
        ):
            x = self.embed(feats)
            for blk in self.blocks:
                x = blk(x)
            logits = self.policy(x).squeeze(-1)          # (B, P)
            if mask is not None:
                logits = jnp.where(mask, logits, -1e9)
            pooled = x.mean(axis=1)                       # (B, hidden)
            value  = self.value_o(nnx.relu(self.value_h(pooled))).squeeze(-1)
            return logits, value

        def act(
            self,
            obs:     np.ndarray,   # flat n² array
            mask:    np.ndarray,   # (n*(n-1)/2,) float32
            rng_key: jax.Array,
        ) -> int:
            n     = int(round(len(obs) ** 0.5))
            known = obs.reshape(n, n)
            feats = _pair_features(known)
            ft = jnp.array(feats, dtype=jnp.float32)[None]
            m  = jnp.array(mask,  dtype=bool)[None]
            logits, _ = self(ft, m)
            action = jax.random.categorical(rng_key, logits[0])
            return int(action)
