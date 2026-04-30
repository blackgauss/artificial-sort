"""Smoke test: one full episode with the numpy fallback (no torch needed)."""
import numpy as np
from poset_rl.env import PosetEnv
from poset_rl.model import ActorCritic


def test_smoke_episode_completes():
    n = 4
    perm = np.random.permutation(n)
    rel = PosetEnv.total_order_from_perm(perm)
    env = PosetEnv(rel)
    model = ActorCritic(n)

    obs = env.reset()
    done = False
    steps = 0
    while not done:
        mask = env.legal_actions_mask()
        action, *_ = model.select_action(obs, mask)
        obs, _, done, _ = env.step(action)
        steps += 1
        assert steps <= n * (n - 1) // 2, "episode exceeded max possible steps"

    assert done
    assert steps >= 1
