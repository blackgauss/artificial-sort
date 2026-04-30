"""JAX/Flax training loop for Poset RL.

Public API
----------
run_episode_jax(model, n, rng_key) -> (obs_list, mask_list, actions, rewards, steps)
discount_returns_jax(rewards, gamma) -> jnp.ndarray
train_step_jax(model, optimizer, batch) -> float
train_jax(model, n_or_range, ...) -> List[dict]
"""
from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import List, Union

import numpy as np

import jax
import jax.numpy as jnp
from flax import nnx
import optax

from poset_rl.env import PosetEnv

__all__ = [
    "run_episode_jax",
    "run_episode_jax_padded",
    "discount_returns_jax",
    "train_step_jax",
    "train_jax",
]


# ---------------------------------------------------------------------------
# Episode collection
# ---------------------------------------------------------------------------

def run_episode_jax(model, n: int, rng_key: jax.Array):
    """Run one episode, storing (obs, mask, action) for gradient re-computation.

    Returns
    -------
    obs_arr     : np.ndarray  (T, n*n)
    mask_arr    : np.ndarray  (T, n*(n-1)//2)
    actions     : np.ndarray  (T,) int32
    rewards     : np.ndarray  (T,) float32
    steps       : int
    """
    perm = np.random.permutation(n)
    rel  = PosetEnv.total_order_from_perm(perm)
    env  = PosetEnv(rel)
    obs  = env.reset()

    obs_list: list    = []
    mask_list: list   = []
    action_list: list = []
    reward_list: list = []

    done = False
    while not done:
        mask           = env.legal_actions_mask()
        rng_key, subk  = jax.random.split(rng_key)
        action         = model.act(obs, mask, subk)
        obs_list.append(obs.copy())
        mask_list.append(mask.copy())
        action_list.append(action)
        obs, reward, done, _ = env.step(action)
        reward_list.append(float(reward))

    return (
        np.stack(obs_list).astype(np.float32),
        np.stack(mask_list).astype(bool),
        np.array(action_list, dtype=np.int32),
        np.array(reward_list, dtype=np.float32),
        env.steps,
    )


# ---------------------------------------------------------------------------
# Returns
# ---------------------------------------------------------------------------

def discount_returns_jax(
    rewards: np.ndarray,
    gamma:   float = 0.99,
) -> jnp.ndarray:
    G, returns = 0.0, []
    for r in reversed(rewards.tolist()):
        G = r + gamma * G
        returns.insert(0, G)
    t = jnp.array(returns, dtype=jnp.float32)
    return (t - t.mean()) / (t.std() + 1e-8)


# ---------------------------------------------------------------------------
# Train step  (differentiate through a full episode)
# ---------------------------------------------------------------------------

def _make_loss_fn(value_coef: float = 0.5):
    """Return a JIT-able loss function closed over value_coef.

    Separating the closure from the gradient call lets JAX compile the XLA
    kernel once per (model_structure, value_coef) pair rather than re-tracing
    on every episode.
    """
    def loss_fn(model, obs_arr, mask_arr, actions, returns):
        logits, values = model(obs_arr, mask_arr)        # (T, A), (T,)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        lp        = log_probs[jnp.arange(len(actions)), actions]
        advantage = jax.lax.stop_gradient(returns - values)
        actor_loss  = -(lp * advantage).mean()
        critic_loss = value_coef * jnp.mean((returns - values) ** 2)
        return actor_loss + critic_loss
    return loss_fn


# JIT-compiled grad function — compiled once, reused every episode.
# The model (graph module) is differentiated; all arrays are traced.
@nnx.jit
def _jit_grad_step(model, optimizer, obs_arr, mask_arr, actions, returns,
                   value_coef=0.5):
    loss_fn = _make_loss_fn(value_coef)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=0)(
        model, obs_arr, mask_arr, actions, returns
    )
    optimizer.update(grads)
    return loss


def train_step_jax(
    model,
    optimizer,
    obs_arr:   jnp.ndarray,   # (T, obs_size)
    mask_arr:  jnp.ndarray,   # (T, n_pairs)  bool
    actions:   jnp.ndarray,   # (T,)          int32
    returns:   jnp.ndarray,   # (T,)          float32  (normalised)
    value_coef: float = 0.5,
) -> float:
    """One REINFORCE-with-baseline gradient step (JIT-compiled)."""
    loss = _jit_grad_step(model, optimizer, obs_arr, mask_arr, actions, returns,
                          value_coef)
    return float(loss)


# ---------------------------------------------------------------------------
# Padded episode  — fixed-length tensors required for jax.vmap
# ---------------------------------------------------------------------------

def run_episode_jax_padded(model, n: int, rng_key: jax.Array):
    """Like run_episode_jax but pads every tensor to max_steps = n*(n-1)//2.

    Returns arrays of **static shape** so they can be stacked across seeds and
    passed to jax.vmap / jax.lax.map without recompilation.

    Returns
    -------
    obs_arr   : np.ndarray  (max_steps, n*n)
    mask_arr  : np.ndarray  (max_steps, n*(n-1)//2)
    actions   : np.ndarray  (max_steps,) int32
    returns   : np.ndarray  (max_steps,) float32  (padded positions = 0)
    valid     : np.ndarray  (max_steps,) bool      — True for real steps
    steps     : int
    """
    obs_arr, mask_arr, actions, rewards, steps = run_episode_jax(model, n, rng_key)
    T        = len(rewards)
    max_T    = n * (n - 1) // 2

    ret_arr  = np.array(discount_returns_jax(rewards), dtype=np.float32)
    valid    = np.zeros(max_T, dtype=bool)
    valid[:T] = True

    def pad(arr, fill=0.0):
        pad_rows = max_T - T
        if arr.ndim == 1:
            return np.concatenate([arr, np.full(pad_rows, fill, dtype=arr.dtype)])
        return np.concatenate([arr, np.zeros((pad_rows, arr.shape[1]), dtype=arr.dtype)])

    return (
        pad(obs_arr),
        pad(mask_arr),
        pad(actions, fill=0).astype(np.int32),
        pad(ret_arr),
        valid,
        steps,
    )


# ---------------------------------------------------------------------------
# High-level train loop
# ---------------------------------------------------------------------------

def train_jax(
    model,
    n_or_range:   Union[int, List[int]],
    *,
    episodes:     int   = 3000,
    lr:           float = 3e-3,
    gamma:        float = 0.99,
    value_coef:   float = 0.5,
    log_interval: int   = 500,
    out_csv:      str   = "training.csv",
    seed:         int   = 42,
) -> List[dict]:
    """Train a JAX/Flax model with REINFORCE + baseline.

    Parameters
    ----------
    model       : any registered jax_* model (nnx.Module with .act())
    n_or_range  : int for fixed-n; list for curriculum (round-robin)
    """
    optimizer = nnx.Optimizer(model, optax.adam(lr))
    ns = [n_or_range] if isinstance(n_or_range, int) else list(n_or_range)

    rng = jax.random.PRNGKey(seed)
    np.random.seed(seed)

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    history: List[dict] = []

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "n", "steps", "loss"])
        writer.writeheader()

        for ep in range(1, episodes + 1):
            n = ns[(ep - 1) % len(ns)]

            rng, ep_key = jax.random.split(rng)
            obs_arr, mask_arr, actions, rewards, steps = run_episode_jax(
                model, n, ep_key
            )

            returns = discount_returns_jax(rewards, gamma)

            # convert to jax arrays for the forward/backward pass
            obs_j  = jnp.array(obs_arr)
            mask_j = jnp.array(mask_arr)
            act_j  = jnp.array(actions)

            loss = train_step_jax(
                model, optimizer, obs_j, mask_j, act_j, returns, value_coef
            )

            row = dict(episode=ep, n=n, steps=steps, loss=loss)
            history.append(row)
            writer.writerow(row)

            if ep % log_interval == 0:
                recent = [r["steps"] for r in history[-log_interval:]]
                print(f"  ep {ep:>6}  n={n}  "
                      f"mean_steps={np.mean(recent):.2f}  loss={loss:.4f}")

    import math
    for n in ns:
        lb   = math.ceil(math.log2(math.factorial(n)))
        last = [r["steps"] for r in history if r["n"] == n][-200:]
        if last:
            print(f"  n={n}  final mean={np.mean(last):.2f}  lb={lb}")

    return history
