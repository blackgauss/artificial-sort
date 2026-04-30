"""Episode generators for different element distributions.

Uniform:
  Elements are drawn uniformly at random — baseline.

Zipf (text-like):
  A vocabulary of V words is ranked 1..V with frequencies ~ 1/rank (Zipf's law),
  matching how words appear in natural text.  Each episode samples n distinct words
  proportional to their frequency, so high-frequency words co-occur more often —
  just as they do in a corpus.  The ground-truth total order is the frequency rank
  (rank 1 = most frequent = "greatest").

This is the key structural difference from uniform:
  * The agent will see the same high-frequency words over and over in the same
    relative positions.  A good policy should learn to recognise them and skip
    redundant comparisons even faster than on uniform data.
  * The long tail of rare words still appears occasionally, keeping the problem
    from collapsing to a trivial fixed-set sort.
"""
from __future__ import annotations

import numpy as np
from poset_rl.env import PosetEnv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _zipf_probs(vocab_size: int, s: float = 1.0) -> np.ndarray:
    """Return normalised Zipf probabilities for ranks 1..vocab_size."""
    ranks = np.arange(1, vocab_size + 1, dtype=np.float64)
    probs = 1.0 / (ranks ** s)
    return probs / probs.sum()


# ---------------------------------------------------------------------------
# Samplers  (callable: () -> PosetEnv)
# ---------------------------------------------------------------------------

class UniformSampler:
    """Each episode: random permutation of 0..n-1 (uniform total order)."""

    def __init__(self, n: int):
        self.n = n
        self.name = "uniform"

    def __call__(self) -> PosetEnv:
        perm = np.random.permutation(self.n)
        rel = PosetEnv.total_order_from_perm(perm)
        return PosetEnv(rel)


class ZipfSampler:
    """Each episode: sample n distinct words from a Zipf-ranked vocabulary,
    then sort them by frequency rank (most frequent = greatest element).

    Parameters
    ----------
    n          : number of elements per episode
    vocab_size : total vocabulary size (default 1000, like a small corpus)
    zipf_s     : Zipf exponent (1.0 ≈ English text; higher = more skewed)
    seed       : optional RNG seed for reproducibility
    """

    def __init__(self, n: int, vocab_size: int = 1000,
                 zipf_s: float = 1.0, seed: int | None = None):
        if n > vocab_size:
            raise ValueError(f"n={n} must be ≤ vocab_size={vocab_size}")
        self.n = n
        self.vocab_size = vocab_size
        self.name = f"zipf(V={vocab_size},s={zipf_s})"
        self._probs = _zipf_probs(vocab_size, s=zipf_s)
        self._rng = np.random.default_rng(seed)

        # word_rank[i] = frequency rank of word i  (0 = most frequent)
        # Since probs are already sorted descending, rank == index
        self._word_rank = np.arange(vocab_size, dtype=np.int32)

    def __call__(self) -> PosetEnv:
        # Sample n distinct words proportional to their Zipf frequency
        chosen = self._rng.choice(
            self.vocab_size, size=self.n, replace=False, p=self._probs
        )
        # Ground-truth order: lower rank (more frequent) = greater element
        # Build perm such that perm[0] is the most frequent chosen word
        local_ranks = np.argsort(self._word_rank[chosen])   # ascending freq rank
        perm = local_ranks[::-1].copy()                      # descending → greatest first
        rel = PosetEnv.total_order_from_perm(perm)
        return PosetEnv(rel)

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def episode_entropy(self) -> float:
        """Shannon entropy of the sampling distribution (bits).
        Higher = more uniform; lower = more skewed."""
        p = self._probs
        return float(-np.sum(p * np.log2(p + 1e-12)))

    def top_k_coverage(self, k: int) -> float:
        """Fraction of probability mass in the top-k words."""
        return float(self._probs[:k].sum())


def make_sampler(dataset: str, n: int, **kwargs) -> UniformSampler | ZipfSampler:
    """Factory: 'uniform' or 'zipf'."""
    if dataset == "uniform":
        return UniformSampler(n)
    elif dataset == "zipf":
        return ZipfSampler(n, **kwargs)
    else:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose 'uniform' or 'zipf'.")
