"""Train an ActorCritic agent to minimise comparison count on small total orders.

Algorithm: REINFORCE with baseline (actor-critic).
  Advantage = G_t - V(s_t)
  Actor loss = -log π(a|s) * advantage
  Critic loss = (G_t - V(s_t))^2

Reward: -1 per comparison (agent wants to infer as much as possible via
        transitivity, so every explicit query costs one unit).
"""
import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch

from poset_rl.env import PosetEnv
from poset_rl.model import ActorCritic
from poset_rl.datasets import make_sampler


def discount_returns(rewards, gamma: float = 1.0) -> list:
    R = 0.0
    returns = []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.append(R)
    returns.reverse()
    return returns


def run_episode(env: PosetEnv, model: ActorCritic):
    obs = env.reset()
    done = False
    transitions = []   # (log_prob, value_est, reward)
    while not done:
        mask = env.legal_actions_mask()
        action, log_prob, value = model.select_action(obs, mask)
        obs, reward, done, _ = env.step(action)
        transitions.append((log_prob, value, reward))
    return transitions, env.steps


def train_step(transitions, optimizer, gamma: float = 1.0, value_coef: float = 0.5):
    rewards = [t[2] for t in transitions]
    returns = discount_returns(rewards, gamma)
    returns_t = torch.tensor(returns, dtype=torch.float32)

    # transitions store tensors (with grad_fn) from ActorCritic.select_action
    log_probs = torch.stack([t[0] for t in transitions])
    values    = torch.stack([t[1] for t in transitions])

    advantages = returns_t - values.detach()
    if len(advantages) > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    actor_loss = -(log_probs * advantages).sum()
    critic_loss = value_coef * ((returns_t - values) ** 2).sum()
    loss = actor_loss + critic_loss

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'], 1.0)
    optimizer.step()
    return float(actor_loss.item()), float(critic_loss.item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5, help="Poset size")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--out_csv", type=str, default="training_log.csv")
    parser.add_argument("--dataset", type=str, default="uniform",
                        choices=["uniform", "zipf"],
                        help="Episode distribution: 'uniform' or 'zipf' (text-like)")
    parser.add_argument("--vocab_size", type=int, default=1000,
                        help="Vocabulary size for zipf dataset")
    parser.add_argument("--zipf_s", type=float, default=1.0,
                        help="Zipf exponent (1.0 ≈ English text)")
    args = parser.parse_args()

    n = args.n
    sampler = make_sampler(args.dataset, n,
                           **({} if args.dataset == "uniform" else
                              {"vocab_size": args.vocab_size, "zipf_s": args.zipf_s}))
    print(f"Dataset : {sampler.name}")
    if args.dataset == "zipf":
        print(f"  entropy        : {sampler.episode_entropy():.2f} bits")
        print(f"  top-10 coverage: {sampler.top_k_coverage(10)*100:.1f}% of mass")
    model = ActorCritic(n, hidden=args.hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    log_path = Path(args.out_csv)
    with log_path.open("w", newline="") as f:
        csv.writer(f).writerow(["episode", "steps", "total_reward",
                                 "actor_loss", "critic_loss"])

    window_steps = []

    for ep in range(1, args.episodes + 1):
        env = sampler()
        transitions, steps = run_episode(env, model)
        actor_loss, critic_loss = train_step(transitions, optimizer, args.gamma)
        total_reward = sum(t[2] for t in transitions)
        window_steps.append(steps)

        with log_path.open("a", newline="") as f:
            csv.writer(f).writerow([ep, steps, total_reward,
                                     f"{actor_loss:.4f}", f"{critic_loss:.4f}"])

        if ep % args.log_interval == 0:
            avg = np.mean(window_steps[-args.log_interval:])
            print(f"ep={ep:5d}  avg_steps={avg:.2f}  last_steps={steps}"
                  f"  actor_loss={actor_loss:.4f}  critic_loss={critic_loss:.4f}")

    lb = math.ceil(math.log2(math.factorial(n)))
    final_avg = np.mean(window_steps[-args.log_interval:])
    print(f"\nTraining complete. Log → {log_path}")
    print(f"Info-theoretic lower bound (n={n}): {lb} comparisons")
    print(f"Final {args.log_interval}-ep average steps: {final_avg:.2f}")


if __name__ == "__main__":
    main()
