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

def train_step_jax(
    model,
    optimizer,
    obs_arr:   jnp.ndarray,   # (T, obs_size)
    mask_arr:  jnp.ndarray,   # (T, n_pairs)  bool
    actions:   jnp.ndarray,   # (T,)          int32
    returns:   jnp.ndarray,   # (T,)          float32  (normalised)
    value_coef: float = 0.5,
) -> float:
    """One REINFORCE-with-baseline gradient step.

    Re-evaluates the model on the stored (obs, mask) batch so gradients flow
    through the policy and value heads.
    """
    def loss_fn(model):
        logits, values = model(obs_arr, mask_arr)   # (T, A), (T,)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        # gather log-prob of the action taken at each timestep
        lp = log_probs[jnp.arange(len(actions)), actions]

        advantage   = jax.lax.stop_gradient(returns - values)
        actor_loss  = -(lp * advantage).mean()
        critic_loss = value_coef * (returns - values).pow(2).mean() \
                      if False else value_coef * jnp.mean((returns - values) ** 2)
        return actor_loss + critic_loss

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(grads)
    return float(loss)


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
