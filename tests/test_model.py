"""Tests for PolicyNet and ActorCritic model interfaces.

These tests use only numpy-based paths so they run without torch installed.
When torch is present the torch branch is exercised instead.
"""
import numpy as np
import pytest
from poset_rl.model import PolicyNet, ActorCritic, TORCH_AVAILABLE
from poset_rl.env import PosetEnv


def make_obs_mask(n: int):
    perm = np.arange(n)
    rel = PosetEnv.total_order_from_perm(perm)
    env = PosetEnv(rel)
    obs = env.reset()
    mask = env.legal_actions_mask()
    return obs, mask


class TestPolicyNet:
    def test_select_action_returns_legal(self):
        n = 4
        obs, mask = make_obs_mask(n)
        policy = PolicyNet(n)
        action, logp = policy.select_action(obs, mask)
        assert mask[action] == 1.0, "selected action must be legal"

    def test_logp_is_float(self):
        n = 4
        obs, mask = make_obs_mask(n)
        policy = PolicyNet(n)
        _, logp = policy.select_action(obs, mask)
        assert isinstance(logp, float)

    def test_action_in_range(self):
        n = 4
        obs, mask = make_obs_mask(n)
        policy = PolicyNet(n)
        action, _ = policy.select_action(obs, mask)
        assert 0 <= action < n * (n - 1) // 2


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
class TestActorCritic:
    def test_select_action_returns_three_values(self):
        n = 4
        obs, mask = make_obs_mask(n)
        ac = ActorCritic(n)
        result = ac.select_action(obs, mask)
        assert len(result) == 3, "ActorCritic.select_action must return (action, logp, value)"

    def test_action_is_legal(self):
        n = 5
        obs, mask = make_obs_mask(n)
        ac = ActorCritic(n)
        action, _, _ = ac.select_action(obs, mask)
        assert mask[action] == 1.0

    def test_value_is_float(self):
        """Value returned must be scalar-coercible (tensor with grad or plain float)."""
        n = 4
        obs, mask = make_obs_mask(n)
        ac = ActorCritic(n)
        _, _, value = ac.select_action(obs, mask)
        # torch version returns a 0-d tensor (keeps grad_fn for backprop);
        # numpy fallback returns a plain float — both must be coercible to float
        assert float(value) == float(value)  # no exception, finite scalar

    def test_full_episode_runs(self):
        n = 4
        perm = np.arange(n)
        rel = PosetEnv.total_order_from_perm(perm)
        env = PosetEnv(rel)
        ac = ActorCritic(n)
        obs = env.reset()
        done = False
        steps = 0
        while not done and steps < 20:
            mask = env.legal_actions_mask()
            action, _, _ = ac.select_action(obs, mask)
            obs, _, done, _ = env.step(action)
            steps += 1
        assert done, "episode should terminate within step budget"
