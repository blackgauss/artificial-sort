"""Tests for AttentionActorCritic.

Key property under test: permutation equivariance.
If we relabel all elements (apply a permutation to both rows and columns of
known[][] and reorder the mask accordingly), the model must assign the same
probability to the corresponding pair.
"""
import numpy as np
import pytest
from poset_rl.env import PosetEnv
from poset_rl.attention_model import AttentionActorCritic, _pair_features, TORCH_AVAILABLE


def make_known(n: int = 5, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    rel = PosetEnv.total_order_from_perm(perm)
    env = PosetEnv(rel)
    env.reset()
    # reveal a couple of pairs so known is non-trivial
    mask = env.legal_actions_mask()
    for _ in range(2):
        legal = np.where(mask > 0)[0]
        if len(legal):
            env.step(int(legal[0]))
            mask = env.legal_actions_mask()
    return env.known.copy()


# -----------------------------------------------------------------------
# Feature extraction
# -----------------------------------------------------------------------

class TestPairFeatures:
    def test_shape(self):
        n = 5
        known = make_known(n)
        feats = _pair_features(known)
        assert feats.shape == (n * (n - 1) // 2, 6)

    def test_dtype(self):
        feats = _pair_features(make_known(4))
        assert feats.dtype == np.float32

    def test_known_status_values(self):
        """First feature (known status) must be in {-1, 0, 1}."""
        feats = _pair_features(make_known(5))
        assert set(feats[:, 0].tolist()).issubset({-1.0, 0.0, 1.0})

    def test_transitive_gain_nonneg(self):
        """Transitive gain must be ≥ 0."""
        feats = _pair_features(make_known(6))
        assert (feats[:, 5] >= 0).all()


# -----------------------------------------------------------------------
# Interface
# -----------------------------------------------------------------------

class TestAttentionActorCriticInterface:
    def test_select_action_returns_three_values(self):
        model = AttentionActorCritic()
        known = make_known(5)
        n = known.shape[0]
        mask = np.ones(n * (n - 1) // 2, dtype=np.float32)
        result = model.select_action(known, mask)
        assert len(result) == 3

    def test_action_is_legal(self):
        model = AttentionActorCritic()
        known = make_known(5)
        n = known.shape[0]
        mask = np.zeros(n * (n - 1) // 2, dtype=np.float32)
        mask[2] = 1.0
        mask[4] = 1.0
        action, _, _ = model.select_action(known, mask)
        assert mask[action] == 1.0

    def test_different_n(self):
        """Single model must handle n=4, n=7, n=10 without error."""
        model = AttentionActorCritic()
        for n in [4, 7, 10]:
            known = make_known(n)
            mask = np.ones(n * (n - 1) // 2, dtype=np.float32)
            action, _, _ = model.select_action(known, mask)
            assert 0 <= action < n * (n - 1) // 2


# -----------------------------------------------------------------------
# Permutation equivariance  (torch only — needs real forward pass)
# -----------------------------------------------------------------------

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
class TestEquivariance:
    """Relabelling elements must not change the policy's pair probabilities."""

    def _pair_index(self, n: int, i: int, j: int) -> int:
        assert i < j
        idx = 0
        for a in range(n):
            for b in range(a + 1, n):
                if a == i and b == j:
                    return idx
                idx += 1
        raise ValueError

    def test_relabel_gives_same_probs(self):
        import torch
        import torch.nn.functional as F

        torch.manual_seed(0)
        model = AttentionActorCritic(hidden=16, nhead=2, nlayers=1)
        model.eval()

        n = 5
        known = make_known(n)

        # ---- original forward ----
        from poset_rl.attention_model import _pair_features
        feats_orig = _pair_features(known)
        mask = np.ones(n * (n - 1) // 2, dtype=np.float32)

        import torch
        feats_t = torch.from_numpy(feats_orig).unsqueeze(0)
        mask_t  = torch.from_numpy(mask).unsqueeze(0)
        with torch.no_grad():
            logits_orig, _ = model(feats_t, mask_t)
        probs_orig = F.softmax(logits_orig, dim=-1).squeeze(0).numpy()

        # ---- permuted forward ----
        rng = np.random.default_rng(42)
        p = rng.permutation(n)
        known_perm = known[np.ix_(p, p)]

        feats_perm = _pair_features(known_perm)
        feats_pt = torch.from_numpy(feats_perm).unsqueeze(0)
        with torch.no_grad():
            logits_perm, _ = model(feats_pt, mask_t)
        probs_perm = F.softmax(logits_perm, dim=-1).squeeze(0).numpy()

        # known_perm[a,b] = known[p[a], p[b]], so element `a` in the permuted
        # problem corresponds to element p[a] in the original.  Pair (i,j) in
        # the original therefore corresponds to pair (p_inv[i], p_inv[j]) in
        # the permuted problem, where p_inv = argsort(p).
        p_inv = np.argsort(p)
        for i in range(n):
            for j in range(i + 1, n):
                pi_inv, pj_inv = int(p_inv[i]), int(p_inv[j])
                orig_idx = self._pair_index(n, i, j)
                perm_idx = self._pair_index(n, min(pi_inv, pj_inv), max(pi_inv, pj_inv))

                np.testing.assert_allclose(
                    probs_orig[orig_idx], probs_perm[perm_idx],
                    atol=1e-4,
                    err_msg=(
                        f"Equivariance failed: orig pair ({i},{j}) idx={orig_idx} "
                        f"vs perm pair ({min(pi_inv,pj_inv)},{max(pi_inv,pj_inv)}) "
                        f"idx={perm_idx}"
                    )
                )

    def test_full_episode_any_n(self):
        """Attention model must complete an episode for n=4,8,12 without error."""
        import torch
        torch.manual_seed(1)
        model = AttentionActorCritic(hidden=32, nhead=2, nlayers=1)
        for n in [4, 8, 12]:
            perm = np.random.permutation(n)
            rel = PosetEnv.total_order_from_perm(perm)
            env = PosetEnv(rel)
            env.reset()
            done = False
            steps = 0
            while not done:
                mask = env.legal_actions_mask()
                action, _, _ = model.select_action(env.known, mask)
                _, _, done, _ = env.step(action)
                steps += 1
                assert steps <= n * (n - 1) // 2
            assert done
