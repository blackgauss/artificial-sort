"""Parallel experiment sweeps.

Two modes
---------
run_sweep(configs, ...)
    Runs a list of ExperimentConfigs concurrently using ProcessPoolExecutor.
    Each config gets its own process — works for both PyTorch and JAX models.
    Returns a list of (ExperimentConfig, model, history) tuples in the order
    configs were submitted.

run_sweep_jax_batched(cfg, seeds, ...)
    JAX-only.  Uses jax.vmap to run *len(seeds)* independent episodes in
    parallel on a single device (GPU/TPU).  All seeds share the same model
    weights — useful for variance-reduced gradient estimates (DiCE / REINFORCE
    with multiple rollouts per step).

Usage
-----
>>> from poset_rl.sweep import run_sweep, run_sweep_jax_batched
>>> from poset_rl.config import ExperimentConfig, ModelConfig, TrainConfig, EvalConfig
>>> configs = [
...     ExperimentConfig("mlp_n4", ModelConfig("mlp"), TrainConfig(n_or_range=4)),
...     ExperimentConfig("mlp_n5", ModelConfig("mlp"), TrainConfig(n_or_range=5)),
...     ExperimentConfig("attn",   ModelConfig("attention"), TrainConfig([3,4,5,6])),
... ]
>>> results = run_sweep(configs, max_workers=3)
"""
from __future__ import annotations

import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Optional, Sequence, Tuple

import numpy as np

from poset_rl.config import ExperimentConfig
from poset_rl.train import train_from_config


# ---------------------------------------------------------------------------
# Multi-process sweep
# ---------------------------------------------------------------------------

def _worker(cfg_and_device: tuple):
    """Top-level function so it is pickle-able by ProcessPoolExecutor."""
    cfg, device = cfg_and_device
    model, history = train_from_config(cfg, device=device)
    return cfg, history


def run_sweep(
    configs: Sequence[ExperimentConfig],
    max_workers: Optional[int] = None,
    device: Optional[str] = None,
    verbose: bool = True,
) -> List[Tuple[ExperimentConfig, list]]:
    """Run multiple experiments concurrently.

    Parameters
    ----------
    configs     : sequence of ExperimentConfig to run
    max_workers : number of parallel workers (default: len(configs))
    device      : torch device string for torch models; JAX ignores this
    verbose     : print progress as experiments complete

    Returns
    -------
    List of (cfg, history) tuples, in completion order.
    Use ``sorted(..., key=lambda r: configs.index(r[0]))`` for submission order.
    """
    max_workers = max_workers or len(configs)
    args = [(cfg, device) for cfg in configs]
    results = []
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_worker, a): a[0] for a in args}
        for fut in as_completed(futures):
            cfg_done, history = fut.result()
            elapsed = time.time() - t0
            final_mean = float(np.mean([r["steps"] for r in history[-100:]]))
            if verbose:
                print(f"  ✓  {cfg_done.name:<35} "
                      f"mean_steps={final_mean:.2f}  [{elapsed:.0f}s]")
            results.append((cfg_done, history))

    return results


# ---------------------------------------------------------------------------
# JAX vmap sweep  — true on-device parallelism via padded fixed-length episodes
# ---------------------------------------------------------------------------

def run_sweep_jax_vmapped(
    cfg: ExperimentConfig,
    n_seeds: int = 8,
    log_interval: int = 200,
):
    """Train one JAX model with ``jax.vmap`` over *n_seeds* parallel rollouts.

    Unlike ``run_sweep_jax_batched`` (which loops over seeds in Python),
    this function pads every episode to ``max_steps = n*(n-1)//2`` so that
    all seed tensors have the same static shape and can be stacked into a
    single ``(n_seeds, max_steps, ...)`` batch.  The loss function is then
    mapped over the batch dimension with ``jax.vmap``, giving a single XLA
    kernel launch per gradient step — ideal for a GB10 Blackwell GPU with
    128 GB unified memory.

    Recommended for
    ---------------
    * NVIDIA Blackwell (GB10, B100, B200) — high-bandwidth unified memory
    * Google TPU v4/v5 — native batch-first layout
    * Any device where kernel launch overhead >> compute

    Parameters
    ----------
    cfg      : ExperimentConfig  (must have a ``jax_*`` model)
    n_seeds  : number of parallel rollouts per gradient step
    """
    if cfg.model.framework != "jax":
        raise ValueError(
            f"run_sweep_jax_vmapped requires a jax_* model, got '{cfg.model.name}'"
        )

    import inspect
    import jax
    import jax.numpy as jnp
    from flax import nnx
    import optax

    from poset_rl.models import get_model
    from poset_rl.train_jax import (
        run_episode_jax_padded,
        discount_returns_jax,
        _make_loss_fn,
    )

    mc, tc = cfg.model, cfg.train

    # ── build model ──────────────────────────────────────────────────────────
    rngs = nnx.Rngs(tc.seed)
    cls  = get_model(mc.name)
    sig  = inspect.signature(cls.__init__)
    if "n" in sig.parameters:
        n_max = tc.n_or_range if isinstance(tc.n_or_range, int) else max(tc.n_or_range)
        model = cls(n=n_max, hidden=mc.hidden, rngs=rngs, **mc.kwargs)
    else:
        model = cls(hidden=mc.hidden, nhead=mc.nhead,
                    nlayers=mc.nlayers, rngs=rngs, **mc.kwargs)

    optimizer = nnx.Optimizer(model, optax.adam(tc.lr), wrt=nnx.Param)
    ns = [tc.n_or_range] if isinstance(tc.n_or_range, int) else list(tc.n_or_range)

    rng_keys = [jax.random.PRNGKey(s) for s in range(n_seeds)]
    np.random.seed(tc.seed)

    # ── JIT-compiled vmapped grad step ───────────────────────────────────────
    loss_fn = _make_loss_fn(tc.value_coef)

    @nnx.jit
    def _vmapped_grad_step(model, optimizer,
                           obs_b, mask_b, acts_b, rets_b, valid_b):
        # obs_b  : (S, max_T, obs_size)   S = n_seeds
        # valid_b: (S, max_T)  bool — mask out padding in the loss

        def per_seed_loss(obs, mask, acts, rets, valid):
            logits, values = model(obs, mask)          # (max_T, A), (max_T,)
            lp  = jax.nn.log_softmax(logits, axis=-1)
            lp  = lp[jnp.arange(obs.shape[0]), acts]
            adv = jax.lax.stop_gradient(rets - values)
            # zero out padded positions
            w   = valid.astype(jnp.float32)
            actor_loss  = -(lp * adv * w).sum() / (w.sum() + 1e-8)
            critic_loss = tc.value_coef * (((rets - values) ** 2) * w).sum() / (w.sum() + 1e-8)
            return actor_loss + critic_loss

        # vmap over the seeds axis — single XLA kernel
        batched_loss = jax.vmap(per_seed_loss)(obs_b, mask_b, acts_b, rets_b, valid_b)
        mean_loss    = batched_loss.mean()

        # gradient w.r.t. model parameters (averaged across seeds automatically
        # because we differentiate the mean)
        _, grads = nnx.value_and_grad(
            lambda m: jax.vmap(per_seed_loss)(obs_b, mask_b, acts_b, rets_b, valid_b).mean()
        )(model)
        optimizer.update(model, grads)
        return mean_loss

    # ── training loop ────────────────────────────────────────────────────────
    history: list = []
    t0 = time.time()

    for ep in range(1, tc.episodes + 1):
        n = ns[(ep - 1) % len(ns)]
        max_T = n * (n - 1) // 2

        # collect n_seeds padded episodes (Python loop — CPU side only)
        obs_b, mask_b, acts_b, rets_b, valid_b, ep_steps = [], [], [], [], [], []
        for i, rk in enumerate(rng_keys):
            rk, subk = jax.random.split(rk)
            rng_keys[i] = rk
            o, m, a, r, v, steps = run_episode_jax_padded(model, n, subk)
            obs_b.append(o);  mask_b.append(m);  acts_b.append(a)
            rets_b.append(r); valid_b.append(v); ep_steps.append(steps)

        # stack → (n_seeds, max_T, ...) and ship to device
        obs_j  = jnp.array(np.stack(obs_b),   dtype=jnp.float32)
        mask_j = jnp.array(np.stack(mask_b),  dtype=bool)
        acts_j = jnp.array(np.stack(acts_b),  dtype=jnp.int32)
        rets_j = jnp.array(np.stack(rets_b),  dtype=jnp.float32)
        vld_j  = jnp.array(np.stack(valid_b), dtype=bool)

        loss = _vmapped_grad_step(model, optimizer,
                                  obs_j, mask_j, acts_j, rets_j, vld_j)

        mean_steps = float(np.mean(ep_steps))
        row = dict(episode=ep, n=n, steps=mean_steps, loss=float(loss))
        history.append(row)

        if ep % log_interval == 0:
            recent = [r["steps"] for r in history[-log_interval:]]
            print(f"  ep {ep:>6}  n={n}  seeds={n_seeds}"
                  f"  mean_steps={np.mean(recent):.2f}"
                  f"  loss={float(loss):.4f}"
                  f"  [{time.time()-t0:.0f}s]")

    return model, history


# ---------------------------------------------------------------------------
# JAX batched-episode sweep  (sequential seeds, averaged gradients)
# ---------------------------------------------------------------------------

def run_sweep_jax_batched(
    cfg: ExperimentConfig,
    seeds: Sequence[int],
    log_interval: int = 500,
) -> List[dict]:
    """Train one JAX model using multiple parallel rollouts per gradient step.

    At each episode index, *len(seeds)* independent episodes are collected via
    ``jax.vmap``.  The returns from all rollouts are averaged into a single
    gradient update — this is the multi-sample REINFORCE estimator, which has
    lower variance than single-rollout at the cost of len(seeds)× more env
    steps per update.

    Only works with ``cfg.model.framework == 'jax'``.

    Parameters
    ----------
    cfg     : ExperimentConfig with a jax_* model name
    seeds   : list of RNG seeds for the parallel rollouts
    """
    if cfg.model.framework != "jax":
        raise ValueError(
            f"run_sweep_jax_batched requires a jax_* model, got '{cfg.model.name}'"
        )

    import jax
    import jax.numpy as jnp
    from flax import nnx
    import optax

    from poset_rl.models import get_model
    from poset_rl.train_jax import discount_returns_jax, run_episode_jax

    import inspect
    mc, tc = cfg.model, cfg.train

    rngs = nnx.Rngs(tc.seed)
    cls  = get_model(mc.name)
    sig  = inspect.signature(cls.__init__)
    if "n" in sig.parameters:
        n = tc.n_or_range if isinstance(tc.n_or_range, int) else max(tc.n_or_range)
        model = cls(n=n, hidden=mc.hidden, rngs=rngs, **mc.kwargs)
    else:
        model = cls(hidden=mc.hidden, nhead=mc.nhead,
                    nlayers=mc.nlayers, rngs=rngs, **mc.kwargs)

    optimizer = nnx.Optimizer(model, optax.adam(tc.lr), wrt=nnx.Param)
    ns = [tc.n_or_range] if isinstance(tc.n_or_range, int) else list(tc.n_or_range)

    rng_keys  = [jax.random.PRNGKey(s) for s in seeds]
    np.random.seed(tc.seed)

    history: list = []
    t0 = time.time()

    for ep in range(1, tc.episodes + 1):
        n = ns[(ep - 1) % len(ns)]

        # collect one episode per seed
        all_obs, all_mask, all_actions, all_returns = [], [], [], []
        ep_steps = []
        for rk in rng_keys:
            rk, subk = jax.random.split(rk)
            obs_a, mask_a, acts_a, rews_a, steps = run_episode_jax(model, n, subk)
            all_obs.append(jnp.array(obs_a))
            all_mask.append(jnp.array(mask_a))
            all_actions.append(jnp.array(acts_a))
            all_returns.append(discount_returns_jax(rews_a, tc.gamma))
            ep_steps.append(steps)
            rng_keys[rng_keys.index(rk)] = rk  # update in place

        # average gradient across rollouts
        def loss_fn(model):
            total = jnp.array(0.0)
            for obs_j, mask_j, act_j, ret_j in zip(
                all_obs, all_mask, all_actions, all_returns
            ):
                logits, values = model(obs_j, mask_j)
                lp  = jax.nn.log_softmax(logits, axis=-1)
                lp  = lp[jnp.arange(len(act_j)), act_j]
                adv = jax.lax.stop_gradient(ret_j - values)
                total = total + (-(lp * adv).mean() +
                                 tc.value_coef * jnp.mean((ret_j - values) ** 2))
            return total / len(seeds)

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)

        mean_steps = float(np.mean(ep_steps))
        row = dict(episode=ep, n=n, steps=mean_steps, loss=float(loss))
        history.append(row)

        if ep % log_interval == 0:
            recent = [r["steps"] for r in history[-log_interval:]]
            print(f"  ep {ep:>6}  n={n}  mean_steps={np.mean(recent):.2f}  "
                  f"loss={float(loss):.4f}  [{time.time()-t0:.0f}s]")

    return model, history


# ---------------------------------------------------------------------------
# Summary table helper
# ---------------------------------------------------------------------------

def sweep_table(results: List[Tuple[ExperimentConfig, list]], window: int = 200):
    """Print a comparison table from run_sweep results."""
    print(f"\n{'experiment':<35}  {'model':<16}  {'n':<12}  "
          f"{'final_mean':>10}  {'episodes':>8}")
    print("-" * 88)
    for cfg, history in sorted(results, key=lambda r: r[0].name):
        final_mean = float(np.mean([r["steps"] for r in history[-window:]]))
        n_str = str(cfg.train.n_or_range)
        print(f"  {cfg.name:<33}  {cfg.model.name:<16}  {n_str:<12}  "
              f"{final_mean:>10.2f}  {len(history):>8}")
