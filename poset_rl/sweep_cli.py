"""CLI entry point for sweep commands.

Usage
-----
  python -m poset_rl.sweep --config configs/gb10_large.yaml --n-seeds 64
  python -m poset_rl.sweep --config configs/gb10_large.yaml --mode batched --n-seeds 8
  python -m poset_rl.sweep --config configs/mlp_n5.yaml --mode multi   # multi-process
"""
from __future__ import annotations

import argparse
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run a poset-RL sweep")
    parser.add_argument("--config",  required=True, help="Path to ExperimentConfig YAML")
    parser.add_argument(
        "--mode", choices=["vmapped", "batched", "multi"], default="vmapped",
        help=(
            "vmapped  — jax.vmap over seeds, single XLA kernel (GB10/TPU, default)\n"
            "batched  — sequential seeds, averaged gradients (JAX fallback)\n"
            "multi    — ProcessPoolExecutor over a list of configs (torch or jax)"
        ),
    )
    parser.add_argument("--n-seeds",    type=int, default=8,
                        help="Number of parallel seeds (vmapped / batched modes)")
    parser.add_argument("--max-workers", type=int, default=None,
                        help="Max worker processes (multi mode)")
    parser.add_argument("--device",     default=None,
                        help="Device override for multi mode (cuda / cpu)")
    args = parser.parse_args(argv)

    from poset_rl.config import ExperimentConfig

    cfg = ExperimentConfig.from_yaml(args.config)
    print(f"Loaded config: {cfg.name}  model={cfg.model.name}  "
          f"n={cfg.train.n_or_range}  episodes={cfg.train.episodes}")

    if args.mode == "vmapped":
        from poset_rl.sweep import run_sweep_jax_vmapped, sweep_table
        model, history = run_sweep_jax_vmapped(cfg, n_seeds=args.n_seeds)
        sweep_table([(cfg, history)])

    elif args.mode == "batched":
        from poset_rl.sweep import run_sweep_jax_batched, sweep_table
        seeds = list(range(args.n_seeds))
        model, history = run_sweep_jax_batched(cfg, seeds=seeds)
        sweep_table([(cfg, history)])

    elif args.mode == "multi":
        from poset_rl.sweep import run_sweep, sweep_table
        results = run_sweep([cfg], max_workers=args.max_workers, device=args.device)
        sweep_table(results)


if __name__ == "__main__":
    main()
