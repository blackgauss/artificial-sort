"""Benchmark helpers for Poset RL agents.

Public API
----------
agent_random(env)           -> int  (action)
agent_greedy(env)           -> int  (action)
run_baseline(agent_fn, env) -> int  (steps)
evaluate_model(model, ns, eval_eps, dataset, device) -> {n: mean_steps}
evaluate_baseline(agent_fn, ns, eval_eps, dataset)   -> {n: mean_steps}
benchmark(ns, ...)          -> dict  (full results table)
main()                      CLI entry point
"""
from __future__ import annotations

import argparse
import math
import time
from typing import Callable, Dict, List, Optional, Sequence, Union

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from poset_rl.env import PosetEnv
from poset_rl.datasets import make_sampler
from poset_rl.train import run_episode, train

__all__ = [
    "agent_random",
    "agent_greedy",
    "run_baseline",
    "evaluate_model",
    "evaluate_baseline",
    "benchmark",
    "main",
]


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def agent_random(env: PosetEnv) -> int:
    """Pick a uniformly random legal pair."""
    mask = env.legal_actions_mask()
    legal = np.where(mask > 0)[0]
    return int(np.random.choice(legal))


def agent_greedy(env: PosetEnv) -> int:
    """Pick the pair with the highest 1-step transitive gain (ties broken randomly)."""
    n = env.n
    known = env.known
    best_gain = -1
    best_pairs: list = []

    for i in range(n):
        for j in range(i + 1, n):
            if known[i, j] != 0:
                continue
            gain_ij = int((known[:, i] == 1).sum()) * int((known[j, :] == 1).sum())
            gain_ji = int((known[:, j] == 1).sum()) * int((known[i, :] == 1).sum())
            gain = max(gain_ij, gain_ji)
            if gain > best_gain:
                best_gain = gain
                best_pairs = [(i, j)]
            elif gain == best_gain:
                best_pairs.append((i, j))

    i, j = best_pairs[np.random.randint(len(best_pairs))]
    idx = 0
    for a in range(n):
        for b in range(a + 1, n):
            if a == i and b == j:
                return idx
            idx += 1
    raise RuntimeError("pair not found")


def run_baseline(agent_fn: Callable[[PosetEnv], int], env: PosetEnv) -> int:
    env.reset()
    done = False
    while not done:
        action = agent_fn(env)
        _, _, done, _ = env.step(action)
    return env.steps


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model,
    ns: Sequence[int],
    eval_eps: int = 200,
    dataset: str = "uniform",
    device: Optional[str] = None,
) -> Dict[int, float]:
    """Return ``{n: mean_steps}`` for each n in *ns*."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required to evaluate learned models")

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    model.to(torch.device(device))

    results: Dict[int, float] = {}
    model.eval()
    with torch.no_grad():
        for n in ns:
            sampler = make_sampler(dataset, n)
            steps_list = [run_episode(sampler(), model)[1] for _ in range(eval_eps)]
            results[n] = float(np.mean(steps_list))
    model.train()
    return results


def evaluate_baseline(
    agent_fn: Callable,
    ns: Sequence[int],
    eval_eps: int = 200,
    dataset: str = "uniform",
) -> Dict[int, float]:
    """Return ``{n: mean_steps}`` for a non-learning baseline agent."""
    results: Dict[int, float] = {}
    for n in ns:
        sampler = make_sampler(dataset, n)
        steps_list = [run_baseline(agent_fn, sampler()) for _ in range(eval_eps)]
        results[n] = float(np.mean(steps_list))
    return results


# ---------------------------------------------------------------------------
# Full benchmark run
# ---------------------------------------------------------------------------

def benchmark(
    ns: Sequence[int] = (4, 5, 6, 7, 8),
    train_eps: int = 3000,
    eval_eps: int = 300,
    hidden: int = 64,
    nhead: int = 2,
    nlayers: int = 1,
    lr: float = 3e-3,
    dataset: str = "uniform",
    device: Optional[str] = None,
    include_mlp: bool = True,
    include_attention: bool = True,
    seed: int = 42,
    log_interval: int = 500,
) -> dict:
    """Train and evaluate all agents; return a results dict.

    Returns
    -------
    {
      "ns":        list[int],
      "lower_bd":  {n: int},
      "worst":     {n: int},
      "random":    {n: float},
      "greedy":    {n: float},
      "mlp":       {n: float},      # only if include_mlp
      "attention": {n: float},      # only if include_attention
    }
    """
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
    np.random.seed(seed)
    ns = list(ns)

    results = {
        "ns": ns,
        "lower_bd": {n: math.ceil(math.log2(math.factorial(n))) for n in ns},
        "worst":    {n: n * (n - 1) // 2 for n in ns},
    }

    print("Evaluating baselines...")
    t0 = time.time()
    results["random"] = evaluate_baseline(agent_random, ns, eval_eps, dataset)
    results["greedy"] = evaluate_baseline(agent_greedy, ns, eval_eps, dataset)
    print(f"  done in {time.time()-t0:.1f}s")

    if include_mlp and TORCH_AVAILABLE:
        from poset_rl.model import ActorCritic
        results["mlp"] = {}
        for n in ns:
            print(f"\nTraining MLP (n={n})  {train_eps} eps...")
            mlp = ActorCritic(n, hidden=hidden)
            t0 = time.time()
            train(mlp, n, episodes=train_eps, lr=lr, dataset=dataset,
                  device=device, log_interval=log_interval)
            res = evaluate_model(mlp, [n], eval_eps, dataset, device)
            results["mlp"][n] = res[n]
            print(f"  n={n}  trained in {time.time()-t0:.1f}s  "
                  f"mean_steps={results['mlp'][n]:.2f}")

    if include_attention and TORCH_AVAILABLE:
        from poset_rl.attention_model import AttentionActorCritic
        print(f"\nTraining AttentionActorCritic  n={ns}  {train_eps} eps (curriculum)...")
        attn = AttentionActorCritic(hidden=hidden, nhead=nhead, nlayers=nlayers)
        t0 = time.time()
        train(attn, ns, episodes=train_eps, lr=lr, dataset=dataset,
              device=device, log_interval=log_interval)
        print(f"  trained in {time.time()-t0:.1f}s")
        results["attention"] = evaluate_model(attn, ns, eval_eps, dataset, device)

    return results


def print_table(results: dict) -> None:
    """Pretty-print a results dict from :func:`benchmark`."""
    ns = results["ns"]
    agents = ["random", "greedy"]
    if "mlp" in results:
        agents.append("mlp")
    if "attention" in results:
        agents.append("attention")

    col_w = 9
    header = (f"{'n':>3}  {'lower_bd':>8}  {'worst':>5}  " +
              "  ".join(f"{a:>{col_w}}" for a in agents))
    print("\n" + "=" * len(header))
    print("Results — mean comparisons to sort n items")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for n in ns:
        row = (f"{n:>3}  {results['lower_bd'][n]:>8}  {results['worst'][n]:>5}  " +
               "  ".join(f"{results[a].get(n, float('nan')):>{col_w}.2f}"
                         for a in agents))
        print(row)
    print("\nlower_bd = ⌈log₂(n!)⌉  (information-theoretic minimum)")
    print("worst    = n*(n-1)/2  (no transitivity exploited)")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Poset RL agents against baselines."
    )
    parser.add_argument("--ns", type=int, nargs="+", default=[4, 5, 6, 7, 8])
    parser.add_argument("--train_eps", type=int, default=3000)
    parser.add_argument("--eval_eps", type=int, default=300)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=2)
    parser.add_argument("--nlayers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--dataset", type=str, default="uniform",
                        choices=["uniform", "zipf"])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no_mlp", action="store_true")
    parser.add_argument("--no_attention", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_interval", type=int, default=500)
    args = parser.parse_args()

    results = benchmark(
        ns=args.ns,
        train_eps=args.train_eps,
        eval_eps=args.eval_eps,
        hidden=args.hidden,
        nhead=args.nhead,
        nlayers=args.nlayers,
        lr=args.lr,
        dataset=args.dataset,
        device=args.device,
        include_mlp=not args.no_mlp,
        include_attention=not args.no_attention,
        seed=args.seed,
        log_interval=args.log_interval,
    )
    print_table(results)


if __name__ == "__main__":
    main()
