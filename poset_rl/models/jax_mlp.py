"""JAX/Flax MLP ActorCritic — fixed n.  Registered as 'jax_mlp'.

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

if _JAX_OK:
    from poset_rl.models import register

    @register("jax_mlp")
    class JaxMLP(nnx.Module):
        """Two-layer MLP.  obs = flat n² known-relation matrix."""

        def __init__(self, n: int, hidden: int = 64, *, rngs: nnx.Rngs, **_):
            obs_size  = n * n
            n_actions = n * (n - 1) // 2
            self.fc1    = nnx.Linear(obs_size,  hidden,    rngs=rngs)
            self.fc2    = nnx.Linear(hidden,     hidden,    rngs=rngs)
            self.policy = nnx.Linear(hidden,     n_actions, rngs=rngs)
            self.value  = nnx.Linear(hidden,     1,         rngs=rngs)

        def __call__(
            self,
            obs:  jnp.ndarray,             # (B, obs_size)
            mask: jnp.ndarray | None = None,  # (B, n_actions) bool
        ):
            x      = nnx.relu(self.fc1(obs))
            x      = nnx.relu(self.fc2(x))
            logits = self.policy(x)
            if mask is not None:
                logits = jnp.where(mask, logits, -1e9)
            value  = self.value(x).squeeze(-1)
            return logits, value

        def act(
            self,
            obs:      np.ndarray,   # flat n² array
            mask:     np.ndarray,   # (n*(n-1)/2,) float32
            rng_key:  jax.Array,
        ) -> int:
            x = jnp.array(obs,  dtype=jnp.float32)[None]
            m = jnp.array(mask, dtype=bool)[None]
            logits, _ = self(x, m)
            action = jax.random.categorical(rng_key, logits[0])
            return int(action)
