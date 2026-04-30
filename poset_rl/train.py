"""Training loop for Poset RL agents.

Public API
----------
run_episode(env, model)          -> (transitions, steps)
discount_returns(rewards, gamma) -> list[float]
train_step(transitions, optimizer, gamma, value_coef) -> (actor_loss, critic_loss)
train(model, n_or_range, episodes, lr, dataset, device, callbacks) -> history
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Union

import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from poset_rl.env import PosetEnv
from poset_rl.datasets import make_sampler

__all__ = [
    "run_episode",
    "discount_returns",
    "train_step",
    "train",
    "main",
]


# ---------------------------------------------------------------------------
# Core training primitives
# ---------------------------------------------------------------------------

def discount_returns(rewards: List[float], gamma: float = 1.0) -> List[float]:
    R = 0.0
    returns: List[float] = []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.append(R)
    returns.reverse()
    return returns


def run_episode(env: PosetEnv, model):
    """Run one episode with *model* on *env*.

    Supports both MLP ``ActorCritic`` and ``AttentionActorCritic``.

    Returns
    -------
    transitions : list of (log_prob, value, reward)
    steps       : int  — number of explicit comparisons made
    """
    from poset_rl.attention_model import AttentionActorCritic
    use_attention = isinstance(model, AttentionActorCritic)

    obs = env.reset()
    done = False
    transitions = []
    while not done:
        mask = env.legal_actions_mask()
        if use_attention:
            action, log_prob, value = model.select_action(env.known, mask)
        else:
            action, log_prob, value = model.select_action(obs, mask)
        obs, reward, done, _ = env.step(action)
        transitions.append((log_prob, value, reward))
    return transitions, env.steps


def train_step(
    transitions,
    optimizer,
    gamma: float = 1.0,
    value_coef: float = 0.5,
):
    """One REINFORCE-with-baseline update.

    Tensors in *transitions* already carry the right device (they come straight
    from ``model.select_action``), so no explicit device argument is needed.

    Returns
    -------
    actor_loss  : float
    critic_loss : float
    """
    rewards = [t[2] for t in transitions]
    returns = discount_returns(rewards, gamma)

    log_probs = torch.stack([t[0] for t in transitions])
    values    = torch.stack([t[1] for t in transitions])

    # returns_t lives on the same device as the model's outputs
    device = log_probs.device
    returns_t = torch.tensor(returns, dtype=torch.float32, device=device)

    advantages = returns_t - values.detach()
    if len(advantages) > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    actor_loss  = -(log_probs * advantages).sum()
    critic_loss = value_coef * ((returns_t - values) ** 2).sum()
    loss = actor_loss + critic_loss

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]["params"], 1.0)
    optimizer.step()
    return float(actor_loss.item()), float(critic_loss.item())


# ---------------------------------------------------------------------------
# High-level training function
# ---------------------------------------------------------------------------

def train(
    model,
    n_or_range: Union[int, Sequence[int]],
    episodes: int = 2000,
    lr: float = 3e-3,
    gamma: float = 1.0,
    dataset: str = "uniform",
    device: Optional[str] = None,
    log_interval: int = 100,
    callbacks: Optional[List[Callable]] = None,
    out_csv: Optional[Union[str, Path]] = None,
) -> List[dict]:
    """Train *model* for *episodes* episodes.

    Parameters
    ----------
    model       : ActorCritic or AttentionActorCritic (must be an nn.Module)
    n_or_range  : single int → fixed n;  list/range → curriculum over those values
    episodes    : total training episodes
    lr          : Adam learning rate
    gamma       : discount factor (1.0 = undiscounted)
    dataset     : ``"uniform"`` or ``"zipf"``
    device      : ``"cuda"``, ``"mps"``, ``"cpu"``, or ``None`` (auto-detect)
    log_interval: print progress every this many episodes
    callbacks   : list of callables invoked after each episode as
                  ``cb(ep, n, steps, actor_loss, critic_loss)``
    out_csv     : if given, append one row per episode to this CSV file

    Returns
    -------
    history : list of dicts with keys
              ``episode, n, steps, total_reward, actor_loss, critic_loss``
    """
    # ---- device ----
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    dev = torch.device(device)
    model.to(dev)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    is_curriculum = not isinstance(n_or_range, int)
    n_range = list(n_or_range) if is_curriculum else None

    if out_csv is not None:
        log_path = Path(out_csv)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", newline="") as f:
            csv.writer(f).writerow(
                ["episode", "n", "steps", "total_reward", "actor_loss", "critic_loss"]
            )

    if not is_curriculum:
        sampler = make_sampler(dataset, n_or_range)

    history: List[dict] = []
    window_steps: List[int] = []

    for ep in range(1, episodes + 1):
        if is_curriculum:
            n = n_range[(ep - 1) % len(n_range)]
            env = make_sampler(dataset, n)()
        else:
            env = sampler()
            n = n_or_range

        transitions, steps = run_episode(env, model)
        actor_loss, critic_loss = train_step(transitions, optimizer, gamma)
        total_reward = float(sum(t[2] for t in transitions))
        window_steps.append(steps)

        row = dict(
            episode=ep, n=n, steps=steps, total_reward=total_reward,
            actor_loss=actor_loss, critic_loss=critic_loss,
        )
        history.append(row)

        if out_csv is not None:
            with log_path.open("a", newline="") as f:
                csv.writer(f).writerow(
                    [ep, n, steps, total_reward,
                     f"{actor_loss:.4f}", f"{critic_loss:.4f}"]
                )

        if callbacks:
            for cb in callbacks:
                cb(ep, n, steps, actor_loss, critic_loss)

        if ep % log_interval == 0:
            avg = float(np.mean(window_steps[-log_interval:]))
            print(
                f"ep={ep:5d}  n={n}  avg_steps={avg:.2f}  "
                f"actor={actor_loss:.4f}  critic={critic_loss:.4f}"
            )

    return history


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train a Poset RL agent to minimise comparison count."
    )
    parser.add_argument("--n", type=int, default=5,
                        help="Poset size (ignored when --curriculum is set)")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--out_csv", type=str, default="training_log.csv")
    parser.add_argument("--dataset", type=str, default="uniform",
                        choices=["uniform", "zipf"])
    parser.add_argument("--vocab_size", type=int, default=1000)
    parser.add_argument("--zipf_s", type=float, default=1.0)
    parser.add_argument("--model", type=str, default="mlp",
                        choices=["mlp", "attention"])
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--nlayers", type=int, default=2)
    parser.add_argument("--curriculum", action="store_true")
    parser.add_argument("--n_min", type=int, default=3)
    parser.add_argument("--n_max", type=int, default=10)
    parser.add_argument("--device", type=str, default=None,
                        help="Force device: cpu / cuda / mps  (default: auto)")
    args = parser.parse_args()

    from poset_rl.model import ActorCritic
    from poset_rl.attention_model import AttentionActorCritic

    if args.model == "attention":
        model = AttentionActorCritic(hidden=args.hidden,
                                     nhead=args.nhead, nlayers=args.nlayers)
        print(f"Model   : AttentionActorCritic  hidden={args.hidden} "
              f"nhead={args.nhead} nlayers={args.nlayers}")
    else:
        if args.curriculum:
            raise ValueError("--curriculum requires --model attention")
        model = ActorCritic(args.n, hidden=args.hidden)
        print(f"Model   : MLP ActorCritic  n={args.n} hidden={args.hidden}")

    n_or_range = (
        range(args.n_min, args.n_max + 1) if args.curriculum else args.n
    )

    history = train(
        model,
        n_or_range=n_or_range,
        episodes=args.episodes,
        lr=args.lr,
        gamma=args.gamma,
        dataset=args.dataset,
        device=args.device,
        log_interval=args.log_interval,
        out_csv=args.out_csv,
    )

    lb = math.ceil(math.log2(math.factorial(
        args.n if not args.curriculum else args.n_max
    )))
    final_avg = float(np.mean([r["steps"] for r in history[-args.log_interval:]]))
    print(f"\nTraining complete. Log → {args.out_csv}")
    if args.curriculum:
        from collections import defaultdict
        per_n: dict = defaultdict(list)
        for r in history[-args.log_interval:]:
            per_n[r["n"]].append(r["steps"])
        print(f"\nPer-n breakdown (final {args.log_interval} episodes):")
        print(f"  {'n':>4}  {'lower_bound':>11}  {'mean_steps':>10}")
        for ni in sorted(per_n):
            lb_i = math.ceil(math.log2(math.factorial(ni)))
            print(f"  {ni:>4}  {lb_i:>11}  {np.mean(per_n[ni]):>10.2f}")
    else:
        print(f"Info-theoretic lower bound (n={args.n}): {lb} comparisons")
        print(f"Final {args.log_interval}-ep average steps: {final_avg:.2f}")


if __name__ == "__main__":
    main()
