Poset RL — Artificial Sort
==========================

An RL prototype that learns an efficient sorting / poset-evaluation strategy by
minimising the number of explicit pairwise comparisons.

## Key ideas

| Concept | Implementation |
|---|---|
| **State** | Flattened `n×n` matrix of known relations (1 = i>j known, −1 = j>i known, 0 = unknown) |
| **Action** | Pick unordered pair `(i,j)` to compare — `n(n−1)/2` discrete actions |
| **Reward** | −1 per comparison (proxy for compute cost) |
| **Transitivity** | After each comparison the env propagates the transitive closure for free — the agent is rewarded for picking pairs that maximise free inferences |
| **Model** | `ActorCritic` — shared trunk, separate policy and value heads; trained with REINFORCE + baseline (advantage = G_t − V(s_t)) |

## Quick start

Install `pytest` and `numpy` using `uv` (no torch required for smoke tests and env tests):

```bash
uv tool install --with numpy --with pytest pytest
```

Install the full runtime (torch + numpy) into a project venv:

```bash
./setup_uv.sh          # tries uv pip, falls back to pip, then creates .venv if needed
source .venv/bin/activate   # if setup_uv.sh created a venv
```

## Run the tests

```bash
pytest tests/ -v
```

Tests that require torch are automatically skipped when torch is absent and activated when it is present.

## Train the agent

```bash
python train.py --n 5 --episodes 2000 --log_interval 100
```

Metrics are written to `training_log.csv` (episode, steps, reward, actor_loss, critic_loss).

Optional flags:

| Flag | Default | Description |
|---|---|---|
| `--n` | 5 | Poset size |
| `--episodes` | 2000 | Training episodes |
| `--hidden` | 128 | Hidden layer width |
| `--lr` | 3e-3 | Adam learning rate |
| `--gamma` | 1.0 | Discount factor |
| `--out_csv` | `training_log.csv` | Metrics output file |

## Smoke test (no torch needed)

```bash
python smoke_test.py
```

## Repository layout

```
poset_rl/
  __init__.py       exports PosetEnv, PolicyNet, ActorCritic, TORCH_AVAILABLE
  env.py            Poset environment with transitive closure propagation
  model.py          PolicyNet + ActorCritic (torch); numpy fallback when torch absent
train.py            Actor-critic training loop with metrics CSV
smoke_test.py       Quick single-episode sanity check
tests/
  test_env.py       22 pytest tests for the environment
  test_model.py     7 pytest tests for models (4 skipped without torch)
setup_uv.sh         Install helper: tries uv pip, then pip, then creates .venv
requirements.txt    torch, numpy
```

