"""Tests for PosetEnv."""
import numpy as np
import pytest
from poset_rl.env import PosetEnv


def make_env(n: int = 4):
    perm = np.arange(n)  # deterministic: 0 > 1 > 2 > 3
    rel = PosetEnv.total_order_from_perm(perm)
    return PosetEnv(rel), rel


def action_for_pair(n: int, i: int, j: int) -> int:
    """Return the action index corresponding to pair (i, j) with i < j."""
    assert i < j
    idx = 0
    for a in range(n):
        for b in range(a + 1, n):
            if a == i and b == j:
                return idx
            idx += 1
    raise ValueError(f"Pair ({i},{j}) not found for n={n}")


class TestReset:
    def test_obs_shape(self):
        env, _ = make_env(4)
        obs = env.reset()
        assert obs.shape == (16,), "obs should be n*n flat"

    def test_known_all_zero_after_reset(self):
        env, _ = make_env(4)
        env.reset()
        assert env.known.sum() == 0

    def test_steps_zero(self):
        env, _ = make_env(4)
        env.reset()
        assert env.steps == 0


class TestTotalOrderRelation:
    def test_relation_shape(self):
        perm = np.arange(4)
        rel = PosetEnv.total_order_from_perm(perm)
        assert rel.shape == (4, 4)

    def test_antisymmetry(self):
        perm = np.arange(5)
        rel = PosetEnv.total_order_from_perm(perm)
        for i in range(5):
            for j in range(5):
                if i != j:
                    assert rel[i, j] == -rel[j, i], "antisymmetry violated"

    def test_total(self):
        """All off-diagonal pairs must be comparable (total order)."""
        perm = np.arange(4)
        rel = PosetEnv.total_order_from_perm(perm)
        for i in range(4):
            for j in range(4):
                if i != j:
                    assert rel[i, j] != 0, f"pair ({i},{j}) is not comparable"


class TestStep:
    def test_reward_minus_one(self):
        env, _ = make_env(4)
        env.reset()
        mask = env.legal_actions_mask()
        action = int(np.where(mask > 0)[0][0])
        _, reward, _, _ = env.step(action)
        assert reward == -1.0

    def test_known_updated(self):
        env, _ = make_env(4)
        env.reset()
        env.step(0)  # compare pair (0, 1)
        # at least 2 cells updated (symmetric pair)
        assert env.known[0, 1] != 0
        assert env.known[1, 0] != 0
        assert env.known[0, 1] == -env.known[1, 0]

    def test_steps_incremented(self):
        env, _ = make_env(4)
        env.reset()
        env.step(0)
        assert env.steps == 1

    def test_done_when_no_legal_actions(self):
        env, _ = make_env(3)
        obs = env.reset()
        done = False
        steps = 0
        while not done:
            mask = env.legal_actions_mask()
            action = int(np.where(mask > 0)[0][0])
            obs, r, done, _ = env.step(action)
            steps += 1
            assert steps <= 10, "episode should terminate quickly for n=3"
        assert done


class TestTransitiveClosure:
    def test_transitive_inference_skips_known_pair(self):
        """After learning 0>1 and 1>2, the pair (0,2) should be inferred
        automatically and must not appear as a legal action."""
        n = 4
        perm = np.array([0, 1, 2, 3])
        rel = PosetEnv.total_order_from_perm(perm)
        env = PosetEnv(rel)
        env.reset()
        env.step(action_for_pair(n, 0, 1))   # reveals 0 > 1
        env.step(action_for_pair(n, 1, 2))   # reveals 1 > 2 → infers 0 > 2
        mask = env.legal_actions_mask()
        a02 = action_for_pair(n, 0, 2)
        assert mask[a02] == 0, "inferred pair (0,2) should not be legal after transitivity"

    def test_chain_completion_saves_comparisons(self):
        """Comparing the chain 0>1>2>3 in order requires only 3 steps,
        leaving all cross-pairs inferred — total < n*(n-1)/2."""
        n = 4
        perm = np.array([0, 1, 2, 3])
        rel = PosetEnv.total_order_from_perm(perm)
        env = PosetEnv(rel)
        env.reset()
        env.step(action_for_pair(n, 0, 1))
        env.step(action_for_pair(n, 1, 2))
        env.step(action_for_pair(n, 2, 3))
        mask = env.legal_actions_mask()
        assert mask.sum() == 0, (
            "After revealing the chain 0>1>2>3, all pairs must be inferred"
        )
        assert env.steps == 3, f"Only 3 explicit comparisons needed, got {env.steps}"

    def test_known_inferred_entry_not_counted_as_step(self):
        """After comparing 0>1 and 1>2, env should infer 0>2 for free."""
        n = 4
        perm = np.array([0, 1, 2, 3])
        rel = PosetEnv.total_order_from_perm(perm)
        env = PosetEnv(rel)
        env.reset()
        env.step(action_for_pair(n, 0, 1))
        assert env.steps == 1
        env.step(action_for_pair(n, 1, 2))
        assert env.steps == 2
        assert env.known[0, 2] == 1, "transitivity should have inferred 0 > 2"


class TestMask:
    def test_mask_length(self):
        env, _ = make_env(5)
        env.reset()
        mask = env.legal_actions_mask()
        assert len(mask) == 5 * 4 // 2

    def test_already_revealed_action_not_legal(self):
        env, _ = make_env(4)
        env.reset()
        env.step(0)  # reveal pair (0,1)
        mask = env.legal_actions_mask()
        assert mask[0] == 0, "already-revealed pair should not be legal"
