"""Tests for JaxAttention — JIT-safety and forward pass shape correctness.

These tests catch the class of bug where numpy / Python concrete ops are
accidentally called inside a jax.jit traced context.
"""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("flax")

import jax.numpy as jnp
from flax import nnx

from poset_rl.env import PosetEnv
from poset_rl.models.jax_attention import JaxAttention, _pair_features_jax, _pair_features, PAIR_FEAT_DIM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_obs(n: int = 5, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    rel = PosetEnv.total_order_from_perm(perm)
    env = PosetEnv(rel)
    obs = env.reset()
    for _ in range(2):
        mask = env.legal_actions_mask()
        action = int(np.where(mask)[0][0])
        obs, _, done, _ = env.step(action)
        if done:
            break
    return obs  # flat (n²,)


def make_model(n: int = 5, hidden: int = 16, nhead: int = 2, nlayers: int = 1):
    rngs = nnx.Rngs(0)
    return JaxAttention(hidden=hidden, nhead=nhead, nlayers=nlayers, rngs=rngs)


# ---------------------------------------------------------------------------
# _pair_features_jax matches numpy reference
# ---------------------------------------------------------------------------

def test_pair_features_jax_matches_numpy():
    n = 5
    obs = make_obs(n)
    known = obs.reshape(n, n)

    ref = _pair_features(known)                              # numpy reference
    got = np.array(_pair_features_jax(jnp.array(obs)[None]))[0]  # JAX, batch=1

    np.testing.assert_allclose(got, ref, atol=1e-5,
                               err_msg="_pair_features_jax diverges from numpy reference")


@pytest.mark.parametrize("n", [4, 5, 6])
def test_pair_features_jax_shape(n):
    obs = make_obs(n)
    B = 3
    obs_batch = jnp.tile(jnp.array(obs), (B, 1))
    feats = _pair_features_jax(obs_batch)
    P = n * (n - 1) // 2
    assert feats.shape == (B, P, PAIR_FEAT_DIM), f"Expected ({B},{P},{PAIR_FEAT_DIM}), got {feats.shape}"


# ---------------------------------------------------------------------------
# Forward pass shape
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [4, 5, 6])
def test_forward_shape(n):
    model = make_model(n)
    obs = make_obs(n)
    mask = np.ones(n * (n - 1) // 2, dtype=bool)
    P = n * (n - 1) // 2

    obs_j  = jnp.array(obs)[None]          # (1, n²)
    mask_j = jnp.array(mask)[None]         # (1, P)
    logits, value = model(obs_j, mask_j)

    assert logits.shape == (1, P),  f"logits shape wrong: {logits.shape}"
    assert value.shape  == (1,),    f"value shape wrong: {value.shape}"


# ---------------------------------------------------------------------------
# JIT-safety — this is the regression test for the ConcretizationTypeError
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [4, 5, 6])
def test_forward_jit_safe(n):
    """Model forward pass must not raise under jax.jit."""
    model = make_model(n)
    obs = make_obs(n)
    mask = np.ones(n * (n - 1) // 2, dtype=bool)

    obs_j  = jnp.array(obs)[None]
    mask_j = jnp.array(mask)[None]

    # This is what train_jax._jit_grad_step does — if _pair_features uses
    # Python float() or numpy inside JIT it raises ConcretizationTypeError.
    @jax.jit
    def fwd(obs_j, mask_j):
        logits, value = model(obs_j, mask_j)
        return logits.sum() + value.sum()

    result = fwd(obs_j, mask_j)
    assert np.isfinite(float(result)), "JIT forward pass returned non-finite value"


# ---------------------------------------------------------------------------
# .act() interface
# ---------------------------------------------------------------------------

def test_act_returns_valid_action():
    n = 5
    model = make_model(n)
    obs = make_obs(n)
    env = PosetEnv(PosetEnv.total_order_from_perm(np.arange(n)))
    env.reset()
    mask = env.legal_actions_mask()

    rng = jax.random.PRNGKey(0)
    action = model.act(obs, mask, rng)

    assert isinstance(action, int)
    assert 0 <= action < n * (n - 1) // 2
    assert mask[action], "Model selected an illegal action"
